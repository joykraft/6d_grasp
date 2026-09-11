
import os
import sys
import numpy as np
import open3d as o3d
import argparse
import importlib
import scipy.io as scio
from PIL import Image
import cv2
import time
import yaml

import torch
from graspnetAPI import GraspGroup
# https://graspnetapi.readthedocs.io/en/latest/about.html

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from graspnet import GraspNet, pred_decode
from graspnet_dataset import GraspNetDataset
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo, create_point_cloud_from_depth_image

from episodeApp import EpisodeAPP
from scipy.spatial.transform import Rotation 

class GraspNetDemo:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--checkpoint_path',  default="./checkpoint-rs.tar", help='模型检查点路径')
        parser.add_argument('--num_point', type=int, default=20000, help='点云数量 [默认: 20000]')
        parser.add_argument('--num_view', type=int, default=300, help='视角数量 [默认: 300]')
        parser.add_argument('--collision_thresh', type=float, default=0.01, help='碰撞检测阈值 [默认: 0.01]')
        parser.add_argument('--voxel_size', type=float, default=0.01, help='点云体素大小 [默认: 0.01]')
        self.cfgs = parser.parse_args()

        # 初始化调整参数
        self.extra_degree = 83
        self.extra_height = 10
        self.filter_max_angle = 45
        self.filter_min_height = 5
        self.select_threshold = 30

        # 初始化机器人
        self.EpRobot = EpisodeAPP('localhost', 12345)

        # 加载相机到末端的变换矩阵
        with open('./T_camera2end.yaml', 'r') as f:
            T_data = yaml.safe_load(f)
            self.T_camera2end = np.array(T_data['T_camera2end'])

        # 关闭夹爪
        self.EpRobot.servo_gripper(0)
        # 移动到默认位置
        result = self.EpRobot.move_xyz_rotation([260, 0, 400], [180, 0, 90], rotation_order="xyz", speed_ratio=1)
        time.sleep(result)
        # 获取当前末端到基座的变换矩阵
        self.T_end2base = self.EpRobot.get_T()

        # 构造舵机夹爪到末端的变换矩阵
        theta = 0
        alpha = 0
        d = 120 
        a = 0
        self.T_servo2end = np.array(
            [
                [np.cos(theta), -np.sin(theta)*np.cos(alpha), np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
                [np.sin(theta), np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
                [0, np.sin(alpha), np.cos(alpha), d],
                [0, 0, 0, 1]
            ]
        )
        
        # 计算相机到基座的坐标，用来筛选抓取点
        T_camera2base = self.T_end2base @ self.T_camera2end
        # 提取t
        self.t_camera2base = T_camera2base[:3, 3]
        # print(f'相机到基座的坐标：{self.t_camera2base}')
    

    def get_net(self):
        # 初始化模型
        net = GraspNet(input_feature_dim=0, num_view=self.cfgs.num_view, num_angle=12, num_depth=4,
                cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net.to(device)
        # 加载检查点
        checkpoint = torch.load(self.cfgs.checkpoint_path)
        net.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        print("-> 加载检查点 %s (轮次: %d)"%(self.cfgs.checkpoint_path, start_epoch))
        # 设置为评估模式
        net.eval()
        return net

    def get_camera_paras(self):
        frames = self.pipeline.wait_for_frames()
        frames = self.align.process(frames)
        
        depth = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())

        intr = color_frame.profile.as_video_stream_profile().intrinsics
        intr_matrix = np.array([
            [intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]
        ]) 
        intr_coeffs = np.array(intr.coeffs)

        return intr_matrix, intr_coeffs

    def get_one_frame(self):
        frames = self.pipeline.wait_for_frames()
        frames = self.align.process(frames)
        
        depth = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth.get_data())
        
        return color_image, depth_image

    def get_and_process_data(self,camera,bbox):
        # 加载数据
        color = np.load('./VLM_related/realsense_captured/color.npy')
        depth = np.load('./VLM_related/realsense_captured/depth.npy')
        # bgr转rgb
        color = color[:, :, [2, 1, 0]]
        # 转换为float32类型
        color = color.astype(np.float32) / 255.0
        depth = depth.astype(np.float32)

        # workspace_mask = np.array(Image.open('doc/example_data/workspace_mask.png'))
        l,t,r,b = bbox
        workspace_mask =  np.zeros((720,1280), dtype=bool)
        if l < 280:
            l = 280
        if r > (280+720):
            r = 280+720
        workspace_mask[t:b,l:r] = True

        # 生成点云
        cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

        # 获取有效点
        mask = (workspace_mask & (depth > 0) )  
        cloud_masked = cloud[mask]
        color_masked = color[mask]

        # 采样点
        if len(cloud_masked) >= self.cfgs.num_point:
            idxs = np.random.choice(len(cloud_masked), self.cfgs.num_point, replace=False)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(len(cloud_masked), self.cfgs.num_point-len(cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        cloud_sampled = cloud_masked[idxs]
        color_sampled = color_masked[idxs]

        # 转换数据
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
        cloud.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))
        end_points = dict()
        cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        cloud_sampled = cloud_sampled.to(device)
        end_points['point_clouds'] = cloud_sampled
        end_points['cloud_colors'] = color_sampled

        return end_points, cloud

    def get_grasps(self, net, end_points):
        # 前向传播
        with torch.no_grad():
            end_points = net(end_points)
            grasp_preds = pred_decode(end_points)
        gg_array = grasp_preds[0].detach().cpu().numpy()
        gg = GraspGroup(gg_array)
        return gg

    def collision_detection(self, gg, cloud):
        mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=self.cfgs.voxel_size)
        collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=self.cfgs.collision_thresh)
        gg = gg[~collision_mask]
        return gg

    def filter_grasps_from_inliers(self, grasp_positions, cloud, distance_thresh=0.003):
        plane_model, inliers = cloud.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
        distance_matrix = np.linalg.norm(grasp_positions[:, np.newaxis] - np.asarray(cloud.points)[inliers], axis=2)
        is_close_to_inlier = np.any(distance_matrix < distance_thresh, axis=1)
        return np.where(~is_close_to_inlier)[0]
    
    def filterT(self):
        pass

    def filter_vertical_grasps(self, gg, max_angle=30, min_height=10):
        """筛选竖直向上的抓取姿态，并过滤掉位置太低的抓取点
        参数:
            gg: GraspGroup对象
            max_angle: 抓取方向与世界坐标系z轴的最大允许夹角（度）
            min_height: 抓取点的最小高度阈值（毫米）
        返回:
            筛选后的GraspGroup
        """
        # 获取旋转矩阵
        rotation_matrices = gg.rotation_matrices
        
        # 世界坐标系z轴（竖直向上）
        world_z = np.array([0, 0, 1])
        
        # 计算抓取方向（x轴）与世界坐标系z轴的夹角
        grasp_approach = rotation_matrices[:, :, 0]  # 获取每个抓取的x轴
        angles = np.arccos(np.clip(np.dot(grasp_approach, world_z), -1.0, 1.0))
        angles_deg = np.degrees(angles)
        
        # 根据夹角筛选抓取
        angle_mask = angles_deg <= max_angle
        
        # 根据高度筛选抓取（从米转换为毫米）
        # gg.translations[:, 2]是抓取点在相机坐标系下的z坐标
        height_mask = gg.translations[:, 2] * 1000 <= (self.t_camera2base[2] - min_height)
        
        # 组合两个条件
        valid_indices = np.where(angle_mask & height_mask)[0]
        return gg[valid_indices]

    def vis_grasps(self,  gg, cloud):

        gg.nms()
        gg.sort_by_score()

        # 挑选出30个结果
        gg = gg[:self.select_threshold]

        print(f"过滤前：{len(gg)}")
        # 保存过滤前的gg
        gg_unfiltered = GraspGroup(gg.grasp_group_array.copy())
        # 筛选竖直抓取
        gg = self.filter_vertical_grasps(gg, max_angle=self.filter_max_angle, min_height=self.filter_min_height)
        print(f"过滤后：{len(gg)}")

        if len(gg) == 0:
            print("没有检测到抓取位置，跳过")
            return

        # 显示在open3d窗口中
        grippers_unfiltered = gg_unfiltered.to_open3d_geometry_list()
        grippers_filtered = gg.to_open3d_geometry_list()
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

        # 创建可视化器
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="抓取点可视化", width=1920, height=1080)

        # 创建左右两侧的点云
        cloud_left = o3d.geometry.PointCloud()
        cloud_left.points = cloud.points
        cloud_left.colors = cloud.colors
        cloud_left.translate([-0.2, 0, 0])  # 向左平移

        cloud_right = o3d.geometry.PointCloud()
        cloud_right.points = cloud.points
        cloud_right.colors = cloud.colors
        cloud_right.translate([0.2, 0, 0])  # 向右平移

        # 创建左右两侧的坐标系
        frame_left = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[-0.2, 0, 0])
        frame_right = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0.2, 0, 0])

        # 添加几何体
        vis.add_geometry(cloud_left)
        vis.add_geometry(cloud_right)
        vis.add_geometry(frame_left)
        vis.add_geometry(frame_right)

        # 添加未过滤的抓取点（左侧）
        grippers_left = []
        for gripper in grippers_unfiltered:
            gripper.translate([-0.2, 0, 0])
            grippers_left.append(gripper)
            vis.add_geometry(gripper)

        # 添加过滤后的抓取点（右侧）
        grippers_right = []
        for gripper in grippers_filtered:
            gripper.translate([0.2, 0, 0])
            grippers_right.append(gripper)
            vis.add_geometry(gripper)

        # 设置视角
        view_control = vis.get_view_control()
        view_control.set_zoom(0.35)
        view_control.set_front([0, 0.5, -0.5])  # Camera looking along the Y-axis
        view_control.set_lookat([0, 0, 0.2])  # Looking at the origin
        view_control.set_up([0, 0, -1])     # Z-axis is "down"

        # 计算变换矩阵和欧拉角的函数
        def calculate_transforms(grasp_idx=0):
            nonlocal gg
            
            if len(gg) == 0:
                print("没有合适的抓取点")
                return None, None, None
                
            if grasp_idx >= len(gg):
                print(f"索引 {grasp_idx} 超出范围，共有 {len(gg)} 个抓取点")
                return None, None, None
            
            # 保存旋转矩阵和平移向量
            rotation_matrix = gg.rotation_matrices
            translation = gg.translations
            widths = gg.widths

            # 构建抓取坐标系到相机坐标系的变换矩阵
            T_grasp2camera = np.eye(4)
            T_grasp2camera[:3, :3] = rotation_matrix[grasp_idx]
            # 调整坐标系方向
            T_grasp2camera[:, [0, 2]] = T_grasp2camera[:, [2, 0]]
            T_grasp2camera[:, 0] = -T_grasp2camera[:, 0]
            # 设置平移向量（单位：毫米）
            T_grasp2camera[:3, 3] = translation[grasp_idx] * 1000

            # 计算抓取坐标系到末端坐标系的变换
            T_grasp2end = self.T_camera2end @ T_grasp2camera

            # 计算抓取坐标系到基座坐标系的变换
            T_grasp2base = self.T_end2base @ T_grasp2end

            # 计算新的末端到基座的变换
            T_end2base_new = T_grasp2base @ np.linalg.inv(self.T_servo2end)
            P_base = T_end2base_new[:3, 3]

            # 计算欧拉角
            rotation = Rotation.from_matrix(T_end2base_new[:3, :3])
            euler = rotation.as_euler('XYZ', degrees=True)
            
            return P_base, euler, T_end2base_new
        
        # 初始化计算
        P_base, euler, T_end2base_new = calculate_transforms()
        
        # 更新渲染器以显示场景
        vis.update_renderer()
        
        # 渲染视图而不阻塞
        vis.poll_events()
        vis.update_renderer()
        time.sleep(2)  # 给用户一些时间查看可视化
        
        # 关闭可视化器
        vis.destroy_window()
        
        # 自动执行抓取（前M键的功能）
        if len(gg) > 0:
            print(f"开始尝试移动，共有 {len(gg)} 个候选抓取点")
            
            # 遍历所有抓取点，直到找到一个有效的
            for grasp_idx in range(len(gg)):
                # 计算当前抓取点的变换
                P_base, euler, _ = calculate_transforms(grasp_idx)
                
                if P_base is None or euler is None:
                    continue
                
                # 尝试移动机器人
                result = self.EpRobot.move_xyz_rotation(
                    [P_base[0], P_base[1], P_base[2] + self.extra_height], 
                    [euler[2] + self.extra_degree, euler[1], euler[0]], 
                    'zyx'
                )
                
                # 检查结果
                if result != -1:
                    print(f"成功移动到抓取点 {grasp_idx}")
                    time.sleep(result)
                    break
                else:
                    print(f"抓取点 {grasp_idx} 没有逆运动学解，尝试下一个")
            
            if result == -1:
                print("所有抓取点都没有有效的逆运动学解")

 
        # 抓取物体
        self.EpRobot.servo_gripper(100)
        time.sleep(0.5)
        # 移动到物体上方
        # t = self.EpRobot.move_xyz_rotation([P_base[0], P_base[1], P_base[2] + self.extra_height + 100], 
        #                              [euler[2]  + self.extra_degree, euler[1], euler[0]], 'zyx')
        # time.sleep(t)
        t = self.EpRobot.move_xyz_rotation([140, -180, 340], [180, 0, 90], 'xyz')
        time.sleep(t)
        # 打开夹爪
        self.EpRobot.servo_gripper(0)

        # 返回初始位置
        t = self.EpRobot.move_xyz_rotation([260, 0, 400], [180, 0, 90], rotation_order="xyz", speed_ratio=1)
        time.sleep(t)
        
        

    def demo(self):

        net = self.get_net()
        result_box_file = './VLM_related/exchange/result_box.npy'
        if os.path.exists(result_box_file):
            os.remove(result_box_file)

        while True:
            # 检查文件是否存在
            if os.path.exists(result_box_file):
                # 获取相机参数
                intrinsic = np.load('./VLM_related/exchange/intr_matrix.npy')
                factor_depth = [[1000.]]
                camera = CameraInfo(1280, 720, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)
                # 获取点云数据
                bbox = np.load(result_box_file)
                end_points, cloud = self.get_and_process_data(camera, bbox)
                # 获取抓取点
                gg = self.get_grasps(net, end_points)
                if self.cfgs.collision_thresh > 0:
                    self.vis_grasps(gg, cloud)
            
                # 删除文件
                os.remove(result_box_file)
            else:
                print("等待中...")
                time.sleep(1)




if __name__ == '__main__':
    demo = GraspNetDemo()
    demo.demo()
