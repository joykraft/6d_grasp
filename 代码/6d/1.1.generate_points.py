# -*- coding: utf-8 -*-
"""
使用 RealSense D435 相机和 Episode 机械臂生成可靠的手眼标定点。

与 1.generate_points.py 保持相同的操作方式：
• python 1.1.generate_points.py prepare
• python 1.1.generate_points.py generate
• Space：保存当前稳定点（仅棋盘格检测成功且图像质量合格）
• S：保存并退出

改进：
• 限制关节反馈读取频率，避免持续占用 CAN/机器人服务。
• 只保存连续多次稳定的关节反馈，并对样本取中位数。
• 保存的是独立快照，不保存后台线程正在更新的列表引用。
• 正常退出、异常退出和 Ctrl+C 都会退出自由模式并保存已有点。
• 原标定目录会改名备份，不直接删除。
"""

import argparse
from collections import deque
from datetime import datetime
import os
import threading
import time

import cv2
import numpy as np
import pyrealsense2 as rs
import yaml

from episodeApp import EpisodeAPP


def load_config():
    """加载与原程序相同的配置文件。"""
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


config = load_config()
CHECKERBOARD = tuple(config["checkerboard"]["pattern_size"])
SAVE_DIR = config["paths"]["save_dir"]

# 示教反馈参数。六轴均使用电机原始角度，不做 DH 零点换算。
FEEDBACK_INTERVAL = 0.10
STABILITY_WINDOW = 8
STABILITY_MAX_SPREAD_DEG = 0.50


class GeneratePoints:
    def __init__(
        self,
        corner_point_long=CHECKERBOARD[0],
        corner_point_short=CHECKERBOARD[1],
    ):
        self.episode_app = EpisodeAPP("localhost", 12345)
        self.in_free_mode = False
        self.motors_degrees = None
        self.corner_point_long = corner_point_long
        self.corner_point_short = corner_point_short

        self._feedback_lock = threading.Lock()
        self._feedback_history = deque(maxlen=STABILITY_WINDOW)
        self._feedback_thread = None

    @staticmethod
    def _valid_angles(angles):
        if angles is None or len(angles) != 6:
            return False
        try:
            values = np.asarray(angles, dtype=np.float64)
        except (TypeError, ValueError):
            return False
        return values.shape == (6,) and np.all(np.isfinite(values))

    def prepare_save_dir(self, folder_path):
        """备份旧目录并创建全新的标定目录。"""
        if os.path.exists(folder_path):
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"{folder_path}_backup_{stamp}"
            suffix = 1
            while os.path.exists(backup_path):
                backup_path = f"{folder_path}_backup_{stamp}_{suffix}"
                suffix += 1
            os.replace(folder_path, backup_path)
            print(f"旧标定数据已备份到: {backup_path}")

        os.makedirs(folder_path, exist_ok=False)
        print(f"已创建标定目录: {folder_path}")

    def get_degrees(self):
        """以有限频率读取六轴原始电机角度。"""
        while self.in_free_mode:
            angles = self.episode_app.get_motor_angles()
            if self._valid_angles(angles):
                snapshot = np.asarray(angles, dtype=np.float64).copy()
                with self._feedback_lock:
                    self.motors_degrees = snapshot
                    self._feedback_history.append(snapshot)
            time.sleep(FEEDBACK_INTERVAL)

    def get_stable_snapshot(self):
        """返回稳定窗口的中位数角度；不稳定或数据不足时返回原因。"""
        with self._feedback_lock:
            samples = np.asarray(list(self._feedback_history), dtype=np.float64)

        if len(samples) < STABILITY_WINDOW:
            return None, f"关节反馈不足（{len(samples)}/{STABILITY_WINDOW}）"

        spread = np.ptp(samples, axis=0)
        if np.any(spread > STABILITY_MAX_SPREAD_DEG):
            worst_axis = int(np.argmax(spread)) + 1
            return (
                None,
                f"机械臂尚未稳定：第{worst_axis}轴窗口波动 "
                f"{spread[worst_axis - 1]:.2f}°，允许值 "
                f"{STABILITY_MAX_SPREAD_DEG:.2f}°",
            )

        return np.median(samples, axis=0).copy(), None

    @staticmethod
    def assess_image_quality(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    @staticmethod
    def save_degrees(degrees_list):
        if not degrees_list:
            return
        save_path = os.path.join(SAVE_DIR, "degrees_list.npy")
        np.save(save_path, np.asarray(degrees_list, dtype=np.float64))
        print(f"已保存 {len(degrees_list)} 组数据到: {save_path}")

    def prepare(self):
        """准备阶段：退出自由模式并移动到原程序指定的初始位姿。"""
        self.episode_app.set_free_mode(0)
        self.episode_app.move_xyz_rotation(
            config["robot"]["initial_position"],
            config["robot"]["initial_rotation"],
            rotation_order=config["robot"]["rotation_order"],
            speed_ratio=config["robot"]["speed_ratio"],
        )
        print("已到达初始位姿。请安装并紧固相机，然后运行 generate。")

    def generate(self):
        """进入自由示教模式并记录稳定的原始六轴电机角度。"""
        pipeline = None
        degrees_list = []

        try:
            # 无论上一次程序如何退出，本次都先明确退出自由模式。
            self.episode_app.set_free_mode(0)
            self.episode_app.move_xyz_rotation(
                config["robot"]["initial_position"],
                config["robot"]["initial_rotation"],
                rotation_order=config["robot"]["rotation_order"],
                speed_ratio=config["robot"]["speed_ratio"],
            )

            print("机械臂 10 秒后进入自由模式，请持续托住机械臂，尤其是第五轴。")
            self.prepare_save_dir(SAVE_DIR)
            time.sleep(10)

            self.episode_app.set_free_mode(1)
            self.in_free_mode = True
            self._feedback_thread = threading.Thread(
                target=self.get_degrees, daemon=True
            )
            self._feedback_thread.start()

            pipeline = rs.pipeline()
            config_rs = rs.config()
            config_rs.enable_stream(
                rs.stream.depth,
                *config["camera"]["resolution"],
                rs.format.z16,
                config["camera"]["fps"],
            )
            config_rs.enable_stream(
                rs.stream.color,
                *config["camera"]["resolution"],
                rs.format.bgr8,
                config["camera"]["fps"],
            )
            align = rs.align(rs.stream.color)
            pipeline.start(config_rs)

            criteria = (
                cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                30,
                1e-3,
            )
            cb_flags = (
                cv2.CALIB_CB_ADAPTIVE_THRESH
                | cv2.CALIB_CB_FAST_CHECK
                | cv2.CALIB_CB_NORMALIZE_IMAGE
            )

            cv2.namedWindow("RealSense", cv2.WINDOW_AUTOSIZE)
            while self.in_free_mode:
                frameset = align.process(pipeline.wait_for_frames())
                color_frame = frameset.get_color_frame()
                depth_frame = frameset.get_depth_frame()
                if not color_frame or not depth_frame:
                    continue

                raw_color_image = np.asanyarray(color_frame.get_data()).copy()
                color_image = raw_color_image.copy()
                depth_image = np.asanyarray(depth_frame.get_data())
                gray = cv2.cvtColor(raw_color_image, cv2.COLOR_BGR2GRAY)

                ret, corners = cv2.findChessboardCorners(
                    gray,
                    (self.corner_point_long, self.corner_point_short),
                    cb_flags,
                )
                if ret:
                    corners = cv2.cornerSubPix(
                        gray, corners, (11, 11), (-1, -1), criteria
                    )
                    cv2.drawChessboardCorners(
                        color_image,
                        (self.corner_point_long, self.corner_point_short),
                        corners,
                        ret,
                    )

                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_image, alpha=0.2), cv2.COLORMAP_JET
                )
                images = np.hstack((color_image, depth_colormap))
                stable_angles, stability_error = self.get_stable_snapshot()
                stable_text = "Stable: YES" if stable_angles is not None else "Stable: NO"
                cv2.putText(
                    images,
                    f"Saved Points: {len(degrees_list)}  {stable_text}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 180, 0) if stable_angles is not None else (0, 0, 255),
                    2,
                )
                cv2.imshow("RealSense", images)

                key = cv2.waitKey(1) & 0xFF
                if key == 32:  # Space
                    stable_angles, stability_error = self.get_stable_snapshot()
                    if stable_angles is None:
                        print(f"未保存：{stability_error}")
                        continue
                    if not ret:
                        print("未保存：未检测到棋盘格角点")
                        continue

                    quality_score = self.assess_image_quality(raw_color_image)
                    if quality_score < config["image_quality"]["threshold"]:
                        print(
                            f"未保存：图像质量较低（{quality_score:.2f} < "
                            f"{config['image_quality']['threshold']}）"
                        )
                        continue

                    degrees_list.append(stable_angles.copy())
                    print(
                        f"已保存点 #{len(degrees_list)}，原始六轴角度："
                        f"{np.round(stable_angles, 3).tolist()}"
                    )

                if key == ord("s"):
                    break

                # 用户点击窗口关闭按钮时也安全退出并保存。
                if cv2.getWindowProperty("RealSense", cv2.WND_PROP_VISIBLE) < 1:
                    break

        except KeyboardInterrupt:
            print("\n收到 Ctrl+C，正在安全保存并退出。")
        finally:
            self.in_free_mode = False
            if self._feedback_thread is not None:
                self._feedback_thread.join(timeout=2.0)
            try:
                self.episode_app.set_free_mode(0)
            except Exception as exc:
                print(f"警告：退出自由模式失败：{exc}")
            self.save_degrees(degrees_list)
            if pipeline is not None:
                try:
                    pipeline.stop()
                except RuntimeError:
                    pass
            cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="稳定的机械臂手眼标定点生成工具")
    parser.add_argument(
        "mode",
        choices=["prepare", "generate"],
        help="prepare：移动到初始位姿；generate：进入自由示教并采集",
    )
    args = parser.parse_args()

    generator = GeneratePoints(
        corner_point_long=CHECKERBOARD[0],
        corner_point_short=CHECKERBOARD[1],
    )
    if args.mode == "prepare":
        generator.prepare()
    else:
        generator.generate()
