# -*- coding: utf-8 -*-
"""
根据 1.1.generate_points.py 保存的原始六轴角度，可靠地复现姿态并采集
手眼标定图像和 T_end2base。

与 2.generate_images_and_T.py 保持相同的数据输出：
• calibration_images/<原始点编号>.jpg
• calibration_images/T_end2base.yaml

额外输出：
• actual_degrees.npy：每张有效图像对应的实际到位角度
• capture_indices.npy：有效图像对应的原始点编号
• capture_log.yaml：目标/实际角度、逐轴误差和采集状态

本程序不使用 calculate_T_based_on_degrees()，也不执行 q5=110-q5；保存与复现
均使用机器人服务端定义的原始电机角度。
"""

from datetime import datetime
import glob
import os
import shutil
import time

import cv2
import numpy as np
import pyrealsense2 as rs
import yaml

from episodeApp import EpisodeAPP


def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


config = load_config()
SAVE_DIR = config["paths"]["save_dir"]

# 闭环到位判定参数。第五轴和其他轴使用同样严格的原始电机角度容差。
JOINT_TOLERANCE_DEG = np.full(6, 0.80, dtype=np.float64)
FEEDBACK_INTERVAL = 0.20
REQUIRED_STABLE_READS = 5
MOVE_TIMEOUT = 35.0
FRESH_FRAME_COUNT = 8


class CameraCalibration:
    def __init__(
        self,
        degrees_list_path,
        corner_point_long=config["checkerboard"]["pattern_size"][0],
        corner_point_short=config["checkerboard"]["pattern_size"][1],
        corner_point_size=config["checkerboard"]["square_size"],
    ):
        self.degrees_list_path = degrees_list_path
        self.corner_point_long = corner_point_long
        self.corner_point_short = corner_point_short
        self.corner_point_size = corner_point_size
        self.criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            30,
            1e-3,
        )
        self.cb_flags = (
            cv2.CALIB_CB_ADAPTIVE_THRESH
            | cv2.CALIB_CB_FAST_CHECK
            | cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        self.app = EpisodeAPP("localhost", 12345)
        self.pipeline = rs.pipeline()
        self.config_rs = rs.config()
        self.align = rs.align(rs.stream.color)
        self.pipeline_started = False

        self.degrees_list = np.load(self.degrees_list_path, allow_pickle=True)
        if self.degrees_list.ndim != 2 or self.degrees_list.shape[1] != 6:
            raise ValueError(
                f"角度文件格式错误，应为 N×6，实际为 {self.degrees_list.shape}"
            )
        self.degrees_list = np.asarray(self.degrees_list, dtype=np.float64)
        print(f"角度列表加载完成，共 {len(self.degrees_list)} 个目标点")

        self.T_list = []
        self.actual_degrees_list = []
        self.capture_indices = []
        self.capture_log = []
        self.results_saved = False

    @staticmethod
    def _valid_angles(angles):
        if angles is None or len(angles) != 6:
            return False
        try:
            values = np.asarray(angles, dtype=np.float64)
        except (TypeError, ValueError):
            return False
        return values.shape == (6,) and np.all(np.isfinite(values))

    def start_camera(self):
        self.config_rs.enable_stream(
            rs.stream.depth,
            *config["camera"]["resolution"],
            rs.format.z16,
            config["camera"]["fps"],
        )
        self.config_rs.enable_stream(
            rs.stream.color,
            *config["camera"]["resolution"],
            rs.format.bgr8,
            config["camera"]["fps"],
        )
        self.pipeline.start(self.config_rs)
        self.pipeline_started = True

    def archive_previous_results(self):
        """备份旧图像和采集结果，保留 degrees_list.npy。"""
        patterns = [
            "*.jpg",
            "T_end2base.yaml",
            "actual_degrees.npy",
            "capture_indices.npy",
            "capture_log.yaml",
        ]
        old_paths = []
        for pattern in patterns:
            old_paths.extend(glob.glob(os.path.join(SAVE_DIR, pattern)))
        if not old_paths:
            return

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_dir = os.path.join(SAVE_DIR, f"previous_capture_{stamp}")
        suffix = 1
        while os.path.exists(archive_dir):
            archive_dir = os.path.join(
                SAVE_DIR, f"previous_capture_{stamp}_{suffix}"
            )
            suffix += 1
        os.makedirs(archive_dir)
        for path in old_paths:
            shutil.move(path, os.path.join(archive_dir, os.path.basename(path)))
        print(f"旧采集结果已备份到: {archive_dir}")

    def wait_until_reached(self, target, expected_time):
        """闭环等待所有关节到位并连续稳定，返回实际角度和逐轴误差。"""
        deadline = time.monotonic() + max(
            MOVE_TIMEOUT, float(expected_time or 0.0) + 10.0
        )
        stable_samples = []
        last_actual = None
        last_error = None

        while time.monotonic() < deadline:
            actual = self.app.get_motor_angles()
            if not self._valid_angles(actual):
                stable_samples.clear()
                time.sleep(FEEDBACK_INTERVAL)
                continue

            actual = np.asarray(actual, dtype=np.float64)
            error = actual - target
            last_actual = actual
            last_error = error

            if np.all(np.abs(error) <= JOINT_TOLERANCE_DEG):
                stable_samples.append(actual.copy())
                if len(stable_samples) >= REQUIRED_STABLE_READS:
                    stable_samples = np.asarray(
                        stable_samples[-REQUIRED_STABLE_READS:]
                    )
                    spread = np.ptp(stable_samples, axis=0)
                    if np.all(spread <= JOINT_TOLERANCE_DEG):
                        actual_median = np.median(stable_samples, axis=0)
                        return True, actual_median, actual_median - target
            else:
                stable_samples.clear()

            time.sleep(FEEDBACK_INTERVAL)

        return False, last_actual, last_error

    def get_fresh_frames(self):
        """丢弃运动期间缓存帧，只返回稳定后的新帧。"""
        frameset = None
        for _ in range(FRESH_FRAME_COUNT):
            frameset = self.align.process(self.pipeline.wait_for_frames())
        return frameset

    def capture_images_and_calibrate(self):
        """逐点复现；只有严格到位并成功检测棋盘格时才保存图像和 T。"""
        # 防止上一次示教异常退出后仍处于自由模式。
        self.app.set_free_mode(0)
        self.start_camera()
        cv2.namedWindow("RealSense", cv2.WINDOW_AUTOSIZE)

        for index, target in enumerate(self.degrees_list):
            target = np.asarray(target, dtype=np.float64)
            log_item = {
                "index": int(index),
                "target_degrees": target.tolist(),
                "status": "pending",
            }

            if not self._valid_angles(target):
                log_item["status"] = "invalid_target"
                self.capture_log.append(log_item)
                print(f"点 {index}：目标角度无效，跳过")
                continue

            print(
                f"\n点 {index}/{len(self.degrees_list) - 1}，目标角度："
                f"{np.round(target, 3).tolist()}"
            )
            expected_time = self.app.angle_mode(
                target.tolist(), speed_ratio=config["robot"]["speed_ratio"]
            )
            if expected_time is None:
                log_item["status"] = "command_failed"
                self.capture_log.append(log_item)
                print(f"点 {index}：运动命令失败，跳过且不拍照")
                continue
            if float(expected_time) < 0:
                log_item["status"] = "ik_or_command_rejected"
                self.capture_log.append(log_item)
                print(f"点 {index}：运动命令被拒绝，跳过且不拍照")
                continue

            reached, actual, error = self.wait_until_reached(target, expected_time)
            if actual is not None:
                log_item["actual_degrees"] = actual.tolist()
            if error is not None:
                log_item["joint_error_degrees"] = error.tolist()

            if not reached:
                log_item["status"] = "reach_timeout"
                self.capture_log.append(log_item)
                print(
                    f"点 {index}：到位超时，最后逐轴误差："
                    f"{None if error is None else np.round(error, 3).tolist()}；"
                    "跳过且不拍照"
                )
                continue

            print(
                f"点 {index} 已稳定到位，逐轴误差："
                f"{np.round(error, 3).tolist()}，第五轴误差 {error[4]:.3f}°"
            )

            frameset = self.get_fresh_frames()
            color_frame = frameset.get_color_frame()
            depth_frame = frameset.get_depth_frame()
            if not color_frame or not depth_frame:
                log_item["status"] = "missing_camera_frame"
                self.capture_log.append(log_item)
                print(f"点 {index}：相机帧无效，跳过")
                continue

            raw_color_image = np.asanyarray(color_frame.get_data()).copy()
            color_image = raw_color_image.copy()
            depth_image = np.asanyarray(depth_frame.get_data())
            gray = cv2.cvtColor(raw_color_image, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(
                gray,
                (self.corner_point_long, self.corner_point_short),
                self.cb_flags,
            )

            if not ret:
                log_item["status"] = "checkerboard_not_found"
                self.capture_log.append(log_item)
                print(f"点 {index}：未检测到棋盘格，不保存图像和 T")
            else:
                corners = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1), self.criteria
                )
                cv2.drawChessboardCorners(
                    color_image,
                    (self.corner_point_long, self.corner_point_short),
                    corners,
                    ret,
                )

                quality = cv2.Laplacian(gray, cv2.CV_64F).var()
                log_item["image_quality"] = float(quality)
                if quality < config["image_quality"]["calibration_threshold"]:
                    log_item["status"] = "low_image_quality"
                    self.capture_log.append(log_item)
                    print(
                        f"点 {index}：图像质量过低（{quality:.2f}），"
                        "不保存图像和 T"
                    )
                else:
                    # 在机器人稳定状态下尽量相邻地读取实际角度与服务端 T。
                    actual_at_capture = self.app.get_motor_angles()
                    T = self.app.get_T()
                    if not self._valid_angles(actual_at_capture) or T is None:
                        log_item["status"] = "robot_feedback_failed"
                        self.capture_log.append(log_item)
                        print(f"点 {index}：拍摄时机器人反馈失败，不保存")
                    else:
                        actual_at_capture = np.asarray(
                            actual_at_capture, dtype=np.float64
                        )
                        capture_error = actual_at_capture - target
                        T = np.asarray(T, dtype=np.float64)
                        if T.shape != (4, 4) or not np.all(np.isfinite(T)):
                            log_item["status"] = "invalid_transform"
                            self.capture_log.append(log_item)
                            print(f"点 {index}：服务端 T 无效，不保存")
                        elif not np.all(
                            np.abs(capture_error) <= JOINT_TOLERANCE_DEG
                        ):
                            log_item["status"] = "moved_during_capture"
                            log_item["actual_at_capture"] = (
                                actual_at_capture.tolist()
                            )
                            log_item["capture_error_degrees"] = (
                                capture_error.tolist()
                            )
                            self.capture_log.append(log_item)
                            print(f"点 {index}：拍摄时关节已偏离目标，不保存")
                        else:
                            file_name = os.path.join(SAVE_DIR, f"{index}.jpg")
                            if not cv2.imwrite(file_name, raw_color_image):
                                log_item["status"] = "image_write_failed"
                                self.capture_log.append(log_item)
                                print(f"点 {index}：图像写入失败，不保存 T")
                            else:
                                self.T_list.append(T.copy())
                                self.actual_degrees_list.append(
                                    actual_at_capture.copy()
                                )
                                self.capture_indices.append(index)
                                log_item["status"] = "captured"
                                log_item["actual_at_capture"] = (
                                    actual_at_capture.tolist()
                                )
                                log_item["capture_error_degrees"] = (
                                    capture_error.tolist()
                                )
                                self.capture_log.append(log_item)
                                print(
                                    f"点 {index} 已捕获；第五轴目标/实际："
                                    f"{target[4]:.3f}°/{actual_at_capture[4]:.3f}°"
                                )

            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.2), cv2.COLORMAP_JET
            )
            images = np.hstack((color_image, depth_colormap))
            cv2.putText(
                images,
                f"Current Point: {index}  Captured: {len(self.T_list)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )
            cv2.imshow("RealSense", images)
            if cv2.waitKey(300) & 0xFF == ord("q"):
                print("收到 Q，提前结束采集。")
                break

        self.save_results()

    def save_results(self):
        """原格式输出 T，并额外保存目标/实际角度配对日志。"""
        os.makedirs(SAVE_DIR, exist_ok=True)
        if self.T_list:
            data = {"T_end2base": [T.tolist() for T in self.T_list]}
            with open(
                os.path.join(SAVE_DIR, "T_end2base.yaml"), "w"
            ) as f:
                yaml.safe_dump(data, f, sort_keys=False)
            np.save(
                os.path.join(SAVE_DIR, "actual_degrees.npy"),
                np.asarray(self.actual_degrees_list, dtype=np.float64),
            )
            np.save(
                os.path.join(SAVE_DIR, "capture_indices.npy"),
                np.asarray(self.capture_indices, dtype=np.int64),
            )

        with open(os.path.join(SAVE_DIR, "capture_log.yaml"), "w") as f:
            yaml.safe_dump(
                {"captures": self.capture_log}, f, sort_keys=False
            )
        self.results_saved = True
        print(
            f"\n采集结束：目标 {len(self.degrees_list)} 个，"
            f"严格配对成功 {len(self.T_list)} 个。"
        )

    def stop(self):
        """任何退出路径都停止相机并确保自由模式关闭。"""
        try:
            self.app.set_free_mode(0)
        except Exception as exc:
            print(f"警告：退出自由模式失败：{exc}")
        if self.pipeline_started:
            try:
                self.pipeline.stop()
            except RuntimeError:
                pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    calibration = CameraCalibration(
        degrees_list_path=os.path.join(SAVE_DIR, "degrees_list.npy"),
        corner_point_long=config["checkerboard"]["pattern_size"][0],
        corner_point_short=config["checkerboard"]["pattern_size"][1],
        corner_point_size=config["checkerboard"]["square_size"],
    )
    try:
        calibration.archive_previous_results()
        calibration.capture_images_and_calibrate()
    except KeyboardInterrupt:
        print("\n收到 Ctrl+C，正在保存已完成的数据。")
        calibration.save_results()
    finally:
        if not calibration.results_saved:
            calibration.save_results()
        calibration.stop()
        print("采集程序已安全退出。")
