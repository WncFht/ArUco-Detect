import plotly.graph_objects as go
import numpy as np
from scipy.spatial.transform import Rotation

class CameraTrajectoryVisualizer:
    def __init__(self):
        self.poses = []
        self.timestamps = []
        
    def add_camera_pose(self, position, rotation, timestamp=None):
        """添加一个相机位姿"""
        # 构建4x4变换矩阵
        pose = np.eye(4)
        pose[:3, :3] = rotation
        pose[:3, 3] = position
        self.poses.append(pose)
        self.timestamps.append(timestamp if timestamp else len(self.poses))
        
    def visualize_trajectory(self, show_cameras=True, show_path=True):
        """可视化相机轨迹"""
        fig = go.Figure()
        
        # 添加相机轨迹
        if show_path and len(self.poses) > 1:
            positions = np.array([pose[:3, 3] for pose in self.poses])
            fig.add_trace(go.Scatter3d(
                x=positions[:, 0],
                y=positions[:, 1],
                z=positions[:, 2],
                mode='lines+markers',
                line=dict(color='blue', width=2),
                marker=dict(size=4),
                name='Camera Path'
            ))
        
        # 添加相机视锥体
        if show_cameras:
            for i, pose in enumerate(self.poses):
                # 计算相机视锥体
                cone_points = self._calculate_camera_cone(pose)
                
                # 添加相机中心点
                fig.add_trace(go.Scatter3d(
                    x=[pose[0, 3]],
                    y=[pose[1, 3]],
                    z=[pose[2, 3]],
                    mode='markers',
                    marker=dict(size=8, color='red'),
                    name=f'Camera {i}'
                ))
                
                # 添加视锥体线条
                self._add_camera_cone_to_figure(fig, cone_points)
                
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
                zaxis_title='Z',
                xaxis=dict(range=[-2, 2]),
                yaxis=dict(range=[-2, 2]),
                zaxis=dict(range=[-2, 2])
            ),
            title='Camera Trajectory Visualization'
        )
        
        return fig
    
    def _calculate_camera_cone(self, pose, fov=60, scale=0.3):
        """计算相机视锥体的顶点"""
        # 视锥体顶点在相机坐标系中的位置
        fov_rad = np.deg2rad(fov)
        z = -scale
        xy = np.tan(fov_rad/2) * abs(z)
        
        # 视锥体的5个顶点（相机中心点和四个角点）
        points = np.array([
            [0, 0, 0],  # 相机中心
            [xy, xy, z],  # 右上
            [-xy, xy, z],  # 左上
            [-xy, -xy, z],  # 左下
            [xy, -xy, z]   # 右下
        ])
        
        # 将点转换到世界坐标系
        points_world = []
        for p in points:
            p_world = pose[:3, :3] @ p + pose[:3, 3]
            points_world.append(p_world)
            
        return np.array(points_world)
    
    def _add_camera_cone_to_figure(self, fig, points):
        """将相机视锥体添加到图中"""
        # 视锥体边线的连接关系
        edges = [
            (0, 1), (0, 2), (0, 3), (0, 4),  # 从中心到角点的线
            (1, 2), (2, 3), (3, 4), (4, 1)   # 底面的线
        ]
        
        # 添加每条边线
        for start, end in edges:
            fig.add_trace(go.Scatter3d(
                x=[points[start, 0], points[end, 0]],
                y=[points[start, 1], points[end, 1]],
                z=[points[start, 2], points[end, 2]],
                mode='lines',
                line=dict(color='gray', width=1),
                showlegend=False
            ))

def visualize_camera_trajectory(pose_data_path, output_html='camera_trajectory.html'):
    """主函数：加载位姿数据并创建可视化"""
    # 加载位姿数据
    pose_data = np.load(pose_data_path, allow_pickle=True).item()
    
    # 创建可视化器
    visualizer = CameraTrajectoryVisualizer()
    
    # 添加所有相机位姿
    for pos, rot in zip(pose_data['positions'], pose_data['rotations']):
        visualizer.add_camera_pose(pos, rot)
    
    # 创建可视化图像
    fig = visualizer.visualize_trajectory()
    
    # 保存为HTML文件
    fig.write_html(output_html)
    
    return fig

# 使用示例
if __name__ == '__main__':
    # 假设我们已经有了保存的相机位姿数据
    fig = visualize_camera_trajectory('camera_poses.npy')
    fig.show()