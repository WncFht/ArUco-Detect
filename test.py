import os
import cv2
import numpy as np
from cv2 import aruco
import math
from tqdm import tqdm
import visualize

class ArucoDetector:
    def __init__(self):
        # 相机内参
        self.camera_matrix = np.array([
            [906.16424561, 0.00000000e+00, 649.84283447],
            [0.00000000e+00, 905.67376709, 374.98306274],
            [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
        ])
        # 假设无畸变
        self.dist_coeffs = np.zeros((5,1))
        # ArUco字典
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
        self.detector = aruco.ArucoDetector(self.aruco_dict, aruco.DetectorParameters())
        # ArUco标记尺寸(米)
        self.marker_size = 0.03
        
    def detect_and_estimate(self, frame):
        """检测ArUco码并估计位姿"""
        # 转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 检测ArUco标记
        corners, ids, rejected = self.detector.detectMarkers(gray)
        
        # 绘制结果的图像
        frame_out = frame.copy()
        
        if ids is not None and len(ids) > 0:
            # 绘制检测到的标记
            aruco.drawDetectedMarkers(frame_out, corners)
            
            for i in range(len(ids)):
                # 估计每个标记的位姿
                rvec, tvec, _ = aruco.estimatePoseSingleMarkers(
                    corners[i], self.marker_size, 
                    self.camera_matrix, self.dist_coeffs
                )
                
                # 将旋转向量转换为旋转矩阵
                R, _ = cv2.Rodrigues(rvec)
                
                # 计算相机在标记坐标系中的位置
                R_inv = np.transpose(R)
                tvec_reshape = tvec.reshape(3, 1)
                cam_pos = -R_inv @ tvec_reshape
                
                # 计算距离和角度
                distance = np.linalg.norm(cam_pos)
                angle_rad = math.atan2(cam_pos[0], cam_pos[2])
                angle_deg = math.degrees(angle_rad)
                
                # 绘制坐标轴
                cv2.drawFrameAxes(frame_out, self.camera_matrix, 
                                self.dist_coeffs, rvec, tvec, 0.05)
                
                # 添加文本信息
                font = cv2.FONT_HERSHEY_SIMPLEX
                pos_str = f"Position: X: {cam_pos[0,0]:.2f} Y: {cam_pos[1,0]:.2f} Z: {cam_pos[2,0]:.2f}"
                cv2.putText(frame_out, pos_str, (10, 30), font, 0.7, (0,255,0), 2)
                cv2.putText(frame_out, f"Distance: {distance:.3f} m", (10, 60), font, 0.7, (0,255,0), 2)
                cv2.putText(frame_out, f"Angle: {angle_deg:.1f} degree", (10, 90), font, 0.7, (0,255,0), 2)
                
                return frame_out, {
                    'position': cam_pos.flatten(),
                    'rotation': R,
                    'distance': distance,
                    'angle': angle_deg,
                    'rvec': rvec,
                    'tvec': tvec
                }
        
        # 如果没有检测到标记
        cv2.putText(frame_out, "No ArUco markers detected", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        return frame_out, None

def process_video(video_path, output_path=None, save_poses=True, save_frames=True):
    """处理视频文件并保存结果"""
    detector = ArucoDetector()
    
    # 创建frames目录
    if save_frames:
        os.makedirs('frames', exist_ok=True)
    
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    
    # 获取视频信息
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 设置输出视频
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 存储所有位姿数据
    poses = []
    frame_idx = 0
    saved_frames_count = 0
    
    # 计算需要保存的帧间隔(假设我们要保存5张关键帧)
    save_interval = frame_count // 5
    
    # 使用tqdm显示处理进度
    for _ in tqdm(range(frame_count), desc="Processing video"):
        ret, frame = cap.read()
        if not ret:
            break
            
        # 处理帧
        frame_viz, pose_data = detector.detect_and_estimate(frame)
        
        if pose_data is not None:
            poses.append(pose_data)
            print(f"Frame {frame_idx}: Position = {pose_data['position']}")
            
            # 保存关键帧
            if save_frames and (frame_idx % save_interval == 0) and saved_frames_count < 5:
                frame_path = os.path.join('frames', f'frame_{frame_idx:04d}.jpg')
                cv2.imwrite(frame_path, frame_viz)
                print(f"Saved frame to {frame_path}")
                saved_frames_count += 1
        
        # 保存处理后的帧到视频
        if output_path:
            out.write(frame_viz)
            
        frame_idx += 1
    
    # 释放资源
    cap.release()
    if output_path:
        out.release()
    
    # 保存位姿数据
    if save_poses and poses:
        pose_data = {
            'positions': np.array([p['position'] for p in poses]),
            'rotations': np.array([p['rotation'] for p in poses]),
            'distances': np.array([p['distance'] for p in poses]),
            'angles': np.array([p['angle'] for p in poses])
        }
        np.save('camera_poses.npy', pose_data)
        print(f"Saved {len(poses)} poses to camera_poses.npy")
    
    return poses

def main():
    # 指定视频文件路径
    video_path = 'rgb_10.mp4'
    output_path = 'output_video.mp4'
    
    # 处理视频
    poses = process_video(video_path, output_path, save_poses=True, save_frames=True)
    
    print(f"Processed {len(poses)} frames with valid ArUco detections")

    fig = visualize.visualize_camera_trajectory('camera_poses.npy')

    print("Visualized camera trajectory done!")

if __name__ == '__main__':
    main()