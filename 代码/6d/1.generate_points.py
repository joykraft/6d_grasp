# -*- coding: utf-8 -*-
"""
使用 RealSense D435 相机和 Episode 机械臂生成标定点

• 分辨率：1280 × 720 @ 30 FPS
• 按 <Space> 保存当前机械臂角度（仅棋盘格成功检测时）
• 按 <s> 保存所有数据并退出

使用方法:
    python 1.generate_points.py prepare  # 准备阶段：移动机械臂到初始位置
    python 1.generate_points.py generate # 开始采集数据
"""

import sys
import os
import numpy as np
import pyrealsense2 as rs
import cv2
import threading
import shutil
import argparse
from episodeApp import EpisodeAPP
import time
import yaml

def load_config():
    """Load configuration from YAML file"""
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

# Load configuration
config = load_config()
CHECKERBOARD = tuple(config['checkerboard']['pattern_size'])
SAVE_DIR = config['paths']['save_dir']

class GeneratePoints:
    def __init__(self, corner_point_long=CHECKERBOARD[0], corner_point_short=CHECKERBOARD[1]):
        """
        初始化生成点类
        :param corner_point_long: 棋盘格长边内角点数
        :param corner_point_short: 棋盘格短边内角点数
        """
        self.episode_app = EpisodeAPP('localhost', 12345)
        self.in_free_mode = True
        self.motors_degrees = None
        self.corner_point_long = corner_point_long
        self.corner_point_short = corner_point_short

    def recreate_folder(self, folder_path):
        """
        删除并重新创建文件夹
        :param folder_path: 文件夹路径
        """
        if os.path.exists(folder_path):
            try:
                shutil.rmtree(folder_path)
                print(f"已删除文件夹: {folder_path}")
            except Exception as e:
                print(f"删除文件夹 {folder_path} 时出错: {e}")
                return

        try:
            os.makedirs(folder_path)
            print(f"已重新创建文件夹: {folder_path}")
        except Exception as e:
            print(f"创建文件夹 {folder_path} 时出错: {e}")

    def get_degrees(self):
        """持续获取机械臂关节角度的线程函数"""
        while self.in_free_mode:
            self.motors_degrees = self.episode_app.get_motor_angles()
            
            

    @staticmethod
    def assess_image_quality(image):
        """
        评估图像质量
        :param image: 输入图像
        :return: 图像质量分数（拉普拉斯方差）
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian_var

    def prepare(self):
        """准备阶段：移动机械臂到初始位置"""
        self.episode_app.set_free_mode(0)
        # 设置固定姿态
        self.episode_app.move_xyz_rotation(
            config['robot']['initial_position'],  # 位置 [x, y, z]
            config['robot']['initial_rotation'],  # 旋转角度 [rx, ry, rz]
            rotation_order=config['robot']['rotation_order'],  # 旋转顺序
            speed_ratio=config['robot']['speed_ratio']  # 速度比例
        )
        print('请安装深度相机，然后修改参数后重新运行')

    def generate(self):
        """生成标定点主函数"""
        self.episode_app.move_xyz_rotation(
            config['robot']['initial_position'],  # 位置 [x, y, z]
            config['robot']['initial_rotation'],  # 旋转角度 [rx, ry, rz]
            rotation_order=config['robot']['rotation_order'],  # 旋转顺序
            speed_ratio=config['robot']['speed_ratio']  # 速度比例
        )
        print('机械臂10S后即将进入自由运动模式，请用手托举')
        self.recreate_folder(SAVE_DIR)
        time.sleep(10)
        # 设置自由模式并启动角度获取线程
        self.episode_app.set_free_mode(1)
        thread = threading.Thread(target=self.get_degrees, daemon=True)
        thread.start()

        saved_points = 0
        degrees_list = []

        # ───── RealSense 初始化 ─────
        pipeline = rs.pipeline()
        config_rs = rs.config()
        config_rs.enable_stream(rs.stream.depth, *config['camera']['resolution'], rs.format.z16, config['camera']['fps'])
        config_rs.enable_stream(rs.stream.color, *config['camera']['resolution'], rs.format.bgr8, config['camera']['fps'])
        align = rs.align(rs.stream.color)
        pipeline.start(config_rs)

        # 棋盘格角点检测参数
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
        cb_flags = (cv2.CALIB_CB_ADAPTIVE_THRESH |
                   cv2.CALIB_CB_FAST_CHECK |
                   cv2.CALIB_CB_NORMALIZE_IMAGE)

        try:
            while self.in_free_mode:
                # --------------- 采帧并对齐 ---------------
                frameset = pipeline.wait_for_frames()
                frameset = align.process(frameset)
                color_frame = frameset.get_color_frame()
                depth_frame = frameset.get_depth_frame()
                if not color_frame or not depth_frame:
                    continue

                # 转换为numpy数组
                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())
                gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

                # ---------- 棋盘格检测 ----------
                ret, corners = cv2.findChessboardCorners(
                    gray, 
                    (self.corner_point_long, self.corner_point_short), 
                    cb_flags
                )
                
                if ret:
                    # 亚像素优化 + 角点可视化
                    cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                    cv2.drawChessboardCorners(
                        color_image, 
                        (self.corner_point_long, self.corner_point_short), 
                        corners, 
                        ret
                    )

                # 生成深度图的彩色映射
                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_image, alpha=0.2), 
                    cv2.COLORMAP_JET
                )

                # 水平拼接图像
                images = np.hstack((color_image, depth_colormap))
                status_txt = f"Saved Points: {saved_points}"
                cv2.putText(images, status_txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # 显示图像
                cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('RealSense', images)

                key = cv2.waitKey(1)
                # -------- 保存逻辑 --------
                if key == 32:  # SPACE
                    # 判断self.motors_degrees是否含有None
                    if None in self.motors_degrees:
                        print('机械臂角度读取失败，请重新托举')
                        continue
                    
                    # 检查是否检测到角点
                    if not ret:
                        print('未检测到棋盘格角点，请调整位置')
                        # 在图像上显示警告
                        cv2.putText(images, "No Chessboard Detected", (10, 70), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        cv2.imshow('RealSense', images)
                        cv2.waitKey(1000)  # 显示1秒
                        continue
                    
                    # 评估图像质量
                    quality_score = self.assess_image_quality(color_image)
                    if quality_score < config['image_quality']['threshold']:
                        print(f'图像质量较低 ({quality_score:.2f})，请调整角度后重试')
                        # 在图像上显示警告
                        cv2.putText(images, f"Low Quality: {quality_score:.2f}", (10, 70), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        cv2.imshow('RealSense', images)
                        cv2.waitKey(1000)  # 显示1秒
                        continue
                    
                    saved_points += 1
                    degrees_list.append(self.motors_degrees)
                    print(f'存储角度：{self.motors_degrees}')

                # -------- 退出并保存 --------
                if key & 0xFF == ord('s'):
                    if len(degrees_list) != 0:
                        np.save(f'{SAVE_DIR}/degrees_list.npy', np.asarray(degrees_list))
                        self.in_free_mode = False
                        thread.join()  # 等待线程结束
                        self.episode_app.set_free_mode(0)
                        print(f'共保存{len(degrees_list)}组数据')
                        cv2.destroyAllWindows()

        finally:
            # 停止相机流
            pipeline.stop()

if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='机械臂标定点生成工具')
    parser.add_argument('mode', choices=['prepare', 'generate'],
                      help='运行模式: prepare(准备阶段) 或 generate(采集数据)')
    
    args = parser.parse_args()
    
    # 初始化生成点类
    G = GeneratePoints(corner_point_long=CHECKERBOARD[0], corner_point_short=CHECKERBOARD[1])
    
    # 根据命令行参数选择执行模式
    if args.mode == 'prepare':
        G.prepare()
    else:  # generate
        G.generate()
