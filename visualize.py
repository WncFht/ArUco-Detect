import plotly.graph_objects as go
import numpy as np
import os
from scipy.spatial.transform import Rotation

class CameraTrajectoryVisualizer:
    def __init__(self):
        self.video_poses = []
        
    def add_video_pose(self, pose_data):
        """添加一个视频的相机位姿数据"""
        self.video_poses.append(pose_data)
        
    def visualize_trajectory(self, show_cameras=True, show_path=True):
        """可视化相机轨迹"""
        fig = go.Figure()
        
        # 为不同视频选择不同颜色
        colors = ['blue', 'red', 'green', 'purple', 'orange', 'cyan', 'magenta', 'yellow']
        
        for i, pose_data in enumerate(self.video_poses):
            color = colors[i % len(colors)]
            video_id = pose_data['video_id']
            positions = pose_data['positions']
            
            # 添加相机轨迹
            if show_path and len(positions) > 1:
                fig.add_trace(go.Scatter3d(
                    x=positions[:, 0],
                    y=positions[:, 1],
                    z=positions[:, 2],
                    mode='lines+markers',
                    line=dict(color=color, width=2),
                    marker=dict(size=4),
                    name=f'Camera {video_id}'
                ))
            
            # 添加相机视锥体
            if show_cameras:
                # 只为每个视频的第一帧和最后一帧添加视锥体
                for idx in [0, -1]:
                    pose = np.eye(4)
                    pose[:3, :3] = pose_data['rotations'][idx]
                    pose[:3, 3] = positions[idx]
                    
                    # 计算相机视锥体
                    cone_points = self._calculate_camera_cone(pose)
                    
                    # 添加视锥体线条
                    self._add_camera_cone_to_figure(fig, cone_points, color)
                    
        # 设置坐标系和视角
        fig.update_layout(
            scene=dict(
                aspectmode='data',
                camera=dict(
                    up=dict(x=0, y=0, z=1),
                    eye=dict(x=1.5, y=1.5, z=1.5)
                ),
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            ),
            title='Multi-Camera Trajectory Visualization'
        )
        
        return fig
    
    def _calculate_camera_cone(self, pose, fov=60, scale=0.3):
        """计算相机视锥体的顶点"""
        fov_rad = np.deg2rad(fov)
        z = -scale
        xy = np.tan(fov_rad/2) * abs(z)
        
        points = np.array([
            [0, 0, 0],  # 相机中心
            [xy, xy, z],  # 右上
            [-xy, xy, z],  # 左上
            [-xy, -xy, z],  # 左下
            [xy, -xy, z]   # 右下
        ])
        
        points_world = []
        for p in points:
            p_world = pose[:3, :3] @ p + pose[:3, 3]
            points_world.append(p_world)
            
        return np.array(points_world)
    
    def _add_camera_cone_to_figure(self, fig, points, color):
        """将相机视锥体添加到图中"""
        edges = [
            (0, 1), (0, 2), (0, 3), (0, 4),  # 从中心到角点的线
            (1, 2), (2, 3), (3, 4), (4, 1)   # 底面的线
        ]
        
        for start, end in edges:
            fig.add_trace(go.Scatter3d(
                x=[points[start, 0], points[end, 0]],
                y=[points[start, 1], points[end, 1]],
                z=[points[start, 2], points[end, 2]],
                mode='lines',
                line=dict(color=color, width=1),
                showlegend=False
            ))

def visualize_camera_trajectories(data_dir, output_html='camera_trajectories.html'):
    """加载所有相机位姿数据并创建可视化"""
    # 查找所有位姿数据文件
    pose_files = sorted([f for f in os.listdir(data_dir) if f.startswith('camera_poses_') and f.endswith('.npy')])
    
    # 创建可视化器
    visualizer = CameraTrajectoryVisualizer()
    
    # 加载每个位姿数据文件
    for pose_file in pose_files:
        pose_data = np.load(os.path.join(data_dir, pose_file), allow_pickle=True).item()
        visualizer.add_video_pose(pose_data)
    
    # 创建可视化图像
    fig = visualizer.visualize_trajectory()
    
    # 保存为HTML文件
    fig.write_html(output_html)
    
    return fig

if __name__ == '__main__':
    # 使用当前目录的所有位姿数据创建可视化
    fig = visualize_camera_trajectories('.')
    fig.show()