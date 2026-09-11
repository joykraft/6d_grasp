# -*- coding: utf-8 -*-
"""
使用 RealSense D435 相机和 Episode 机械臂生成标定图像并计算变换矩阵T

• 根据之前保存的角度列表自动移动机械臂
• 捕获棋盘格图像并计算相应的变换矩阵
• 保存旋转矩阵和平移向量用于后续标定

使用方法:
    python 2.generate_images_and_T.py
"""

import sys
import os
import numpy as np
import pyrealsense2 as rs
import cv2
import time
from scipy.spatial.transform import Rotation
import glob
from episodeApp import EpisodeAPP
import yaml

def load_config():
    """Load configuration from YAML file"""
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

# Load configuration
config = load_config()

class CameraCalibration:
    def __init__(self, degrees_list_path, corner_point_long=config['checkerboard']['pattern_size'][0], 
                 corner_point_short=config['checkerboard']['pattern_size'][1], 
                 corner_point_size=config['checkerboard']['square_size']):
        """
        初始化相机标定类
        :param degrees_list_path: 角度列表文件路径
        :param corner_point_long: 棋盘格长边内角点数
        :param corner_point_short: 棋盘格短边内角点数
        :param corner_point_size: 棋盘格格子尺寸(mm)
        """
        self.degrees_list_path = degrees_list_path
        self.corner_point_long = corner_point_long
        self.corner_point_short = corner_point_short
        self.corner_point_size = corner_point_size
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
        self.cb_flags = (cv2.CALIB_CB_ADAPTIVE_THRESH |
                        cv2.CALIB_CB_FAST_CHECK |
                        cv2.CALIB_CB_NORMALIZE_IMAGE)

        self.app = EpisodeAPP('localhost', 12345)
        self.pipeline = rs.pipeline()
        self.config_rs = rs.config()
        self.align = rs.align(rs.stream.color)

        # 配置深度和彩色图像流
        self.config_rs.enable_stream(rs.stream.depth, *config['camera']['resolution'], rs.format.z16, config['camera']['fps'])
        self.config_rs.enable_stream(rs.stream.color, *config['camera']['resolution'], rs.format.bgr8, config['camera']['fps'])
        self.pipeline.start(self.config_rs)

        # 加载角度列表
        self.degrees_list = np.load(self.degrees_list_path, allow_pickle=True)
        print('角度列表加载完成')
        print(self.degrees_list)

        # 存储结果的列表
        self.R_list = []
        self.t_list = []
        self.euler_angles_list = []

    def capture_images_and_calibrate(self):
        """捕获图像并进行标定"""
        for index, degrees in enumerate(self.degrees_list):
            if None in degrees:
                continue

            # 移动机械臂
            sleep_t = self.app.angle_mode(degrees.tolist())
            time.sleep(sleep_t)
            time.sleep(1)

            # 捕获图像帧
            frameset = self.pipeline.wait_for_frames()
            frameset = self.align.process(frameset)
            color_frame = frameset.get_color_frame()
            depth_frame = frameset.get_depth_frame()

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            raw_color_image = color_image.copy()

            # 棋盘格角点检测
            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(
                gray, 
                (self.corner_point_long, self.corner_point_short), 
                self.cb_flags
            )

            if ret:
                # 亚像素优化 + 角点可视化
                corners2 = cv2.cornerSubPix(
                    gray, 
                    corners, 
                    (11, 11), 
                    (-1, -1), 
                    self.criteria
                )
                # 绘制检测到的角点
                cv2.drawChessboardCorners(
                    color_image, 
                    (self.corner_point_long, self.corner_point_short), 
                    corners2, 
                    ret
                )
                # 保存原始图像
                file_name = f'{config["paths"]["save_dir"]}/{index}.jpg'
                cv2.imwrite(file_name, raw_color_image)

                # 计算变换矩阵T
                T = self.app.get_T()
                if T is not None:
                    # 从T矩阵中提取旋转和平移部分
                    T_array = T
                    R = T_array[:3, :3]  # 提取旋转矩阵
                    t = T_array[:3, 3]   # 提取平移向量
                    self.R_list.append(R)
                    self.t_list.append(t)
               
                print(f'点 {index} 已捕获并保存。')

            # 生成深度图的彩色映射
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.2), 
                cv2.COLORMAP_JET
            )
            # 水平拼接图像
            images = np.hstack((color_image, depth_colormap))
            cv2.putText(
                images, 
                f"Current Point: {index}", 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (0, 0, 255), 
                2
            )
            cv2.imshow('RealSense', images)
            cv2.waitKey(300)

        self.save_results()

    def save_results(self):
        """保存标定结果"""
        if len(self.R_list) > 0:
            R_list = np.asarray(self.R_list)
            t_list = np.asarray(self.t_list)
            euler_angles_list = np.asarray(self.euler_angles_list)

            # 构建完整的变换矩阵列表
            T_list = []
            for R, t in zip(R_list, t_list):
                T = np.eye(4)
                T[:3, :3] = R
                T[:3, 3] = t
                T_list.append(T)

            # 保存为YAML文件
            T_list_dict = {'T_end2base': [T.tolist() for T in T_list]}
            with open(f'{config["paths"]["save_dir"]}/T_end2base.yaml', 'w') as f:
                yaml.dump(T_list_dict, f)

    def stop(self):
        """停止相机流并关闭窗口"""
        self.pipeline.stop()
        cv2.destroyAllWindows()

    def remove_files(self):
        """删除之前的标定图像和数据"""
        # 删除之前的图像
        images = glob.glob(f'{config["paths"]["save_dir"]}/*.jpg')
        for image in images:
            os.remove(image)
        # 删除之前的数据
        if os.path.exists(f'{config["paths"]["save_dir"]}/T_end2base.yaml'):
            os.remove(f'{config["paths"]["save_dir"]}/T_end2base.yaml')

if __name__ == "__main__":
    # 初始化相机标定类
    calibration = CameraCalibration(
        degrees_list_path=f'{config["paths"]["save_dir"]}/degrees_list.npy',
        corner_point_long=config['checkerboard']['pattern_size'][0], 
        corner_point_short=config['checkerboard']['pattern_size'][1], 
        corner_point_size=config['checkerboard']['square_size']
    )
    try:
        calibration.remove_files()
        calibration.capture_images_and_calibrate()
    finally:
        calibration.stop()
        print('采集完成。')
