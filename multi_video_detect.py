import os
import cv2
import numpy as np
from cv2 import aruco
import math
from tqdm import tqdm
import visualize

class ArucoDetector:
    def __init__(self):
        # 相机内参保持不变
        self.camera_matrix = np.array([
            [906.16424561, 0.00000000e+00, 649.84283447],
            [0.00000000e+00, 905.67376709, 374.98306274],
            [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
        ])
        self.dist_coeffs = np.zeros((5,1))
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
        self.detector = aruco.ArucoDetector(self.aruco_dict, aruco.DetectorParameters())
        self.marker_size = 0.03

    def detect_and_estimate(self, frame):
        """检测ArUco码并估计位姿"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = self.detector.detectMarkers(gray)
        frame_out = frame.copy()
        
        if ids is not None and len(ids) > 0:
            aruco.drawDetectedMarkers(frame_out, corners)
            
            for i in range(len(ids)):
                rvec, tvec, _ = aruco.estimatePoseSingleMarkers(
                    corners[i], self.marker_size, 
                    self.camera_matrix, self.dist_coeffs
                )
                
                R, _ = cv2.Rodrigues(rvec)
                R_inv = np.transpose(R)
                tvec_reshape = tvec.reshape(3, 1)
                cam_pos = -R_inv @ tvec_reshape
                
                distance = np.linalg.norm(cam_pos)
                angle_rad = math.atan2(cam_pos[0], cam_pos[2])
                angle_deg = math.degrees(angle_rad)
                
                cv2.drawFrameAxes(frame_out, self.camera_matrix, 
                                self.dist_coeffs, rvec, tvec, 0.05)
                
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
        
        cv2.putText(frame_out, "No ArUco markers detected", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        return frame_out, None

def process_video(video_path, save_video=True, save_poses=True, save_frames=True):
    """处理视频文件并保存结果"""
    # 从文件名中提取视频编号
    video_name = os.path.basename(video_path)
    video_id = int(video_name.split('_')[1].split('.')[0])  # 从rgb_1.mp4提取1
    
    detector = ArucoDetector()
    # 创建frames目录
    if save_frames:
        os.makedirs('frames', exist_ok=True)
    if save_video:
        os.makedirs('output', exist_ok=True)

    # 确保视频路径正确
    print(f"Opening video file: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return []
        
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    output_video_path = os.path.join('output', f'video_{video_id:02d}.mp4')
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    poses = []
    frame_idx = 0
    representative_frame_saved = False
    
    print(f"Processing {frame_count} frames from video {video_id}")
    for _ in tqdm(range(frame_count), desc=f"Processing video {video_id}"):
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_viz, pose_data = detector.detect_and_estimate(frame)
        
        if pose_data is not None:
            poses.append(pose_data)
            
            # 只保存第一个有效检测的帧
            if save_frames and not representative_frame_saved:
                frame_path = os.path.join('frames', f'rgb_{video_id:02d}_frame.jpg')
                cv2.imwrite(frame_path, frame_viz)
                print(f"Saved representative frame to {frame_path}")
                representative_frame_saved = True
        
        out.write(frame_viz)
            
        frame_idx += 1
    
    cap.release()
    out.release()
    
    if save_poses and poses:
        pose_data = {
            'video_id': video_id,
            'positions': np.array([p['position'] for p in poses]),
            'rotations': np.array([p['rotation'] for p in poses]),
            'distances': np.array([p['distance'] for p in poses]),
            'angles': np.array([p['angle'] for p in poses])
        }
        pose_file_path = os.path.join('output', f'camera_poses_{video_id:02d}.npy')
        np.save(pose_file_path, pose_data)
        print(f"Saved {len(poses)} poses to camera_poses_{video_id:02d}.npy")
    
    return poses

def main():
    # 获取所有视频文件
    video_dir = './video'  # 视频目录
    all_poses = []

    # 确保视频目录存在
    if not os.path.exists(video_dir):
        print(f"Error: Video directory {video_dir} does not exist!")
        return
        
    # 获取视频文件列表
    video_files = sorted([f for f in os.listdir(video_dir) if f.startswith('rgb_') and f.endswith('.mp4')])
    if not video_files:
        print(f"Error: No video files found in {video_dir}")
        return
    
    print(f"Found {len(video_files)} video files")
    
    # 处理每个视频
    for video_file in video_files:
        video_path = os.path.join(video_dir, video_file)
        poses = process_video(video_path, save_poses=True)
        if poses:
            all_poses.extend(poses)

    np.save('all_camera_poses.npy', all_poses)

    # 可视化所有轨迹
    fig = visualize.visualize_camera_trajectories('./output')
    print("Visualized all camera trajectories!")

if __name__ == '__main__':
    main()