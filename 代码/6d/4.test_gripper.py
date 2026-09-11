# -*- coding: utf-8 -*-
"""
使用 RealSense D435 相机和 ArUco 标记进行物体抓取测试

• 使用 ArUco 标记进行物体定位
• 支持自动抓取和放置功能
• 包含坐标转换和可视化功能

使用方法:
    python 5.test_gripper.py  # 运行抓取测试程序
"""

import sys
import os
import pyrealsense2 as rs
import numpy as np
import cv2
import cv2.aruco as aruco
import time
import random
from spatialmath import *
from episodeApp import EpisodeAPP
import yaml


class Calibration:
    def __init__(self):
        """
        初始化抓取测试类
        """
        # ───── 基本参数设置 ─────
        self.sucker_length = 60  # 吸嘴长度（单位：mm）
        self.robot = EpisodeAPP()
        self.marker_size = 0.05  # ArUco标记尺寸（单位：m）
        self.last_drop_position = None  # 记录上一次放置位置

        # ───── RealSense 初始化 ─────
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)  # 宽、高、数据格式、帧率
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        self.align = rs.align(rs.stream.color)
        
        # 启动相机流
        self.pipeline.start(config)

        # ───── ArUco 设置 ─────
        self.dictionary = aruco.getPredefinedDictionary(aruco.DICT_5X5_50)
        self.parameters = aruco.DetectorParameters()

    def get_aruco_center(self, calib=True):
        """
        检测 ArUco 标记并获取中心点坐标
        
        Args:
            calib (bool): 标定模式标志
            
        Returns:
            tuple: (图像, 中心点坐标)
                - 图像: 包含颜色帧和深度帧的组合图像
                - 中心点坐标: 如检测到则返回[x, y, z]列表，否则为None
        """
        # 获取相机帧
        frames = self.pipeline.wait_for_frames()
        frames = self.align.process(frames)
        depth = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())

        # 获取相机内参
        intr = color_frame.profile.as_video_stream_profile().intrinsics
        cam_matrix = np.array([
            [intr.fx, 0, intr.ppx], 
            [0, intr.fy, intr.ppy], 
            [0, 0, 1]
        ])
        dist = np.array(intr.coeffs)

        # 检测 ArUco 标记
        detector = aruco.ArucoDetector(self.dictionary, self.parameters)
        corners, ids, rejected = detector.detectMarkers(color_image)
        
        center = None
        # 如果检测到标记
        if ids is not None and len(ids) > 0:
            # 估计标记位姿
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, self.marker_size, cam_matrix, dist)
            # 绘制标记边框
            aruco.drawDetectedMarkers(color_image, corners, ids)
            
            # 处理第一个检测到的标记
            for rvec, tvec, corner in zip(rvecs, tvecs, corners):
                # 绘制坐标轴
                cv2.drawFrameAxes(color_image, cam_matrix, dist, rvec, tvec, self.marker_size)
                
                # 计算标记中心点
                x = float((corner[0][0][0] + corner[0][2][0]) / 2)
                y = float((corner[0][0][1] + corner[0][2][1]) / 2)
                
                # 获取深度信息
                d = depth.get_distance(int(x), int(y))
                # 转换为相机坐标系下的3D坐标
                xyz = rs.rs2_deproject_pixel_to_point(intr, [x, y], d)
                center = list(xyz)
                
                # 在图像上标记中心点
                cv2.circle(color_image, (int(x), int(y)), 5, (0, 0, 255), -1)
                
                # 在图像上显示坐标信息
                txt = f"x:{xyz[0]:.3f} y:{xyz[1]:.3f} z:{xyz[2]:.3f}"
                cv2.putText(color_image, txt, (int(x)+5, int(y)-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
                # 只处理第一个标记
                break

        # 处理深度图像
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(np.asanyarray(depth.get_data()), alpha=0.14),
            cv2.COLORMAP_JET
        )
        
        # 组合图像
        images = np.hstack((color_image, depth_colormap))
        
        return images, center
        
    def run_recog(self):
        """
        运行物体识别和抓取主循环
        """
        # ───── 加载标定参数 ─────
        with open('./T_camera2end.yaml', 'r') as f:
            T_data = yaml.safe_load(f)
            T_camera2end = np.array(T_data['T_camera2end'])
        
        print("相机到末端的变换矩阵：")
        print(T_camera2end)

        # ───── 初始化机器人状态 ─────
        self.robot.gripper_off()
        # 移动到初始位置
        self.robot.move_xyz_rotation([260, 0, 400], [180, 0, 90], rotation_order="xyz", speed_ratio=1)
        time.sleep(1)
        
        # 获取当前末端到基座的变换矩阵
        T_end2base = self.robot.get_T()
        print("当前末端到基座的变换矩阵：")
        print(T_end2base)
        
        # ───── 主循环 ─────
        while True:
            # 移动到观察位置
            self.robot.move_xyz_rotation([260, 0, 400], [180, 0, 90], rotation_order="xyz", speed_ratio=1)
            time.sleep(1)

            # 检测标记
            images, center = self.get_aruco_center(calib=False)
            if center is not None:
                center = np.array(center) * 1000  # 转换为毫米
                
                cv2.imshow("image", images)
                cv2.waitKey(1)
                
                # ───── 坐标转换 ─────
                # 构建相机坐标系下的齐次坐标
                P_camera = np.ones(4)
                P_camera[0:3] = center
                print(f'相对于相机的坐标：{P_camera[:3]}')

                # 转换到工具末端坐标系
                P_end = T_camera2end @ P_camera
                print(f'相对于末端的坐标：{P_end[:3]}')

                # 转换到机器人基座坐标系
                P_base = T_end2base @ P_end
                print(f'相对于基座的坐标：{P_base[:3]}')
                print("--------------------------------")

                
                # 移动到物体上方
                self.robot.move_xyz_rotation(
                    [P_base[0], P_base[1], P_base[2] + self.sucker_length + 100], 
                    [180, 0, 90], 
                    rotation_order="xyz", 
                    speed_ratio=1
                )

                # ───── 执行抓取动作 ─────
                self.robot.gripper_on()
                
                # 移动到抓取位置
                self.robot.move_xyz_rotation(
                    [P_base[0], P_base[1], P_base[2] + self.sucker_length-15], 
                    [180, 0, 90], 
                    rotation_order="xyz", 
                    speed_ratio=1
                )

                # 向上移动20mm
                self.robot.move_xyz_rotation(
                    [P_base[0], P_base[1], P_base[2] + self.sucker_length + 20], 
                    [180, 0, 90], 
                    rotation_order="xyz", 
                    speed_ratio=1
                )
                
                # ───── 随机放置 ─────
                # 生成随机放置位置，确保与上一次放置位置有足够距离
                min_distance = 150  # 最小距离要求（毫米）
                while True:
                    dx, dy = random.randint(250,380), random.randint(-220,220)
                    if self.last_drop_position is None:
                        break
                    # 计算与上一次放置位置的距离
                    distance = np.sqrt((dx - self.last_drop_position[0])**2 + 
                                     (dy - self.last_drop_position[1])**2)
                    if distance >= min_distance:
                        break
                
                drop = [dx, dy, self.sucker_length + 100]
                self.last_drop_position = [dx, dy]  # 更新上一次放置位置
                self.robot.move_xyz_rotation(
                    drop,
                    [180, 0, 90], 
                    rotation_order="xyz", 
                    speed_ratio=1
                )
                
                # 释放物体
                self.robot.gripper_off()
                time.sleep(1)
                
            else:
                print("no marker detected")
                time.sleep(1)
                
    def cleanup(self):
        """
        清理资源：停止相机流、关闭窗口、关闭夹爪
        """
        self.pipeline.stop()
        cv2.destroyAllWindows()
        self.robot.gripper_off()
        print("已释放所有资源。")

# 主程序入口
if __name__ == "__main__":
    cali = Calibration()
    try:
        cali.run_recog()
    except KeyboardInterrupt:
        print("程序被用户中断")
    finally:
        cali.cleanup()