# -*- coding: utf-8 -*-
"""
Episode 机械臂教学模式工具

通过教学模式记录机械臂轨迹，并支持复现功能。
适用于快速创建机械臂运动轨迹，无需编程。

操作说明:
• 在教学模式下，手动移动机械臂到目标位置，系统自动检测稳定状态并记录
• 记录完成后，可以让机械臂自动复现所记录的运动轨迹
• 按 'q' 结束教学模式并保存轨迹数据

使用方法:
    python 0.teach_mode.py prepare    # 进入教学模式，记录机械臂运动轨迹
    python 0.teach_mode.py replicate  # 复现已记录的机械臂运动轨迹
"""

import sys
import os
import numpy as np
import time
from episodeApp import EpisodeAPP
import threading
import cv2
import argparse

# ─────────── 用户可调参数 ────────────
SAVE_FILE = './motors_degrees.npy'  # 轨迹数据保存路径
STABILITY_THRESHOLD = 0.5           # 稳定性判断阈值，值越小要求越稳定
STABILITY_WINDOW = 50               # 稳定性判断窗口大小
POSITION_DIFF_THRESHOLD = 20        # 位置差异阈值，控制记录点的密度
# ────────────────────────────────────

class GeneratePoints:
    def __init__(self):
        """
        初始化教学模式类
        连接到机械臂服务端并准备教学环境
        """
        self.episode_app = EpisodeAPP('localhost', 12345)
        
        self.motors_degrees = None
        self.in_free_mode = True
        self.prev_degrees = []  # 用于保存最近的角度变化记录
        self.last_saved_degrees = None  # 保存上一次存储的角度值
        
    def get_degrees(self):
        """
        持续获取机械臂关节角度的线程函数
        在自由模式下循环读取关节角度
        """
        while self.in_free_mode:
            self.motors_degrees = self.episode_app.get_motor_angles()
            time.sleep(0.1)  # 减少读取频率，避免过高的资源消耗

    def is_motor_stable(self, threshold=STABILITY_THRESHOLD, window=STABILITY_WINDOW):
        """
        判断机械臂是否处于稳定状态
        
        参数:
            threshold: 稳定判断的角度变化阈值
            window: 判断稳定所需的连续帧数量
            
        返回:
            是否稳定(True/False)
        """
        if self.motors_degrees is None:
            return False
        
        self.prev_degrees.append(self.motors_degrees)
        if len(self.prev_degrees) > window:
            self.prev_degrees.pop(0)
        
        if len(self.prev_degrees) < window:
            return False  # 数据不足时无法判断
        
        # 计算每一帧的变化量
        deltas = [np.linalg.norm(np.array(self.prev_degrees[i]) - np.array(self.prev_degrees[i-1])) 
                  for i in range(1, len(self.prev_degrees))]
        
        # 如果所有变化量都小于阈值，则认为稳定
        return all(delta < threshold for delta in deltas)

    def prepare(self):
        """
        教学模式: 进入自由模式，记录机械臂的运动轨迹
        
        用户可手动移动机械臂，当机械臂保持稳定状态时自动记录关节角度
        按 'q' 键结束并保存数据
        """
        print("\n================ 教学模式 ================")
        print("• 10s后进入自由模式，请手动移动机械臂...")
        print("• 机械臂稳定时会自动记录位置")
        print("• 按 'q' 键结束并保存轨迹数据")
        print("=========================================\n")
        time.sleep(10)
        self.episode_app.set_free_mode(1)  # 进入自由模式
        thread = threading.Thread(target=self.get_degrees, daemon=True)  # 创建守护线程
        thread.start()
        degrees_list = []
        
        while True:
            # 创建显示界面
            image = np.zeros((400, 800, 3), dtype=np.uint8)
            cv2.putText(image, "Teach Mode: Points recorded when stable", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(image, f"Recorded points: {len(degrees_list)}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(image, "Press 'q' to save and exit", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            if self.motors_degrees is not None:
                cv2.putText(image, f"Current angles: {self.motors_degrees}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("Teach Mode", image)
            key = cv2.waitKey(1)
            
            # 检测机械臂是否稳定，稳定时记录位置
            if self.is_motor_stable(threshold=STABILITY_THRESHOLD, window=STABILITY_WINDOW):
                # 检查是否与上一次保存的角度相似
                if (self.last_saved_degrees is None or 
                    np.linalg.norm(np.array(self.motors_degrees) - np.array(self.last_saved_degrees)) > POSITION_DIFF_THRESHOLD):
                    degrees_list.append(self.motors_degrees)
                    self.last_saved_degrees = self.motors_degrees  # 更新最后保存的角度
                    print(f'[INFO] 位置点 #{len(degrees_list)} 已记录: {self.motors_degrees}')
            
            # 按 'q' 键退出
            if key == ord('q'):
                cv2.destroyAllWindows()
                self.episode_app.set_free_mode(0)  # 退出自由模式
                self.in_free_mode = False
                print(f"\n[INFO] 教学完成，共记录 {len(degrees_list)} 个位置点")
                
                # 保存轨迹数据
                np.save(SAVE_FILE, degrees_list)
                print(f"[INFO] 轨迹数据已保存至 {SAVE_FILE}")
                break

    def replicate(self):
        """
        复现模式: 重现机械臂之前记录的运动轨迹
        
        按照记录的顺序，依次移动到保存的位置点
        """
        print("\n================ 轨迹复现 ================")
        print(f"• 尝试加载轨迹数据: {SAVE_FILE}")
        
        # 读取角度数据
        try:
            degrees_list = np.load(SAVE_FILE)
            print(f"• 已加载 {len(degrees_list)} 个位置点")
            print("• 开始复现轨迹...")
            print("=========================================\n")
            
            # 依次移动到各个位置点
            for i, degree in enumerate(degrees_list):
                print(f"[INFO] 移动到位置点 #{i+1}/{len(degrees_list)}: {degree}")
                tt = self.episode_app.angle_mode(degree.tolist())
                time.sleep(tt)
            
            print("\n[INFO] 轨迹复现完成")
            
        except FileNotFoundError:
            print("\n[ERROR] 未找到轨迹数据文件!")
            print(f"[ERROR] 文件 '{SAVE_FILE}' 不存在")
            print("[INFO] 请先使用 'prepare' 模式记录轨迹")
            print("=========================================\n")

if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='机械臂教学模式工具')
    parser.add_argument('mode', choices=['prepare', 'replicate'],
                      help='运行模式: prepare(教学阶段，记录轨迹) 或 replicate(复现轨迹)')
    
    args = parser.parse_args()
    
    # 初始化类
    G = GeneratePoints()
    
    # 根据命令行参数选择执行模式
    if args.mode == 'prepare':
        G.prepare()
    else:  # replicate
        G.replicate()
