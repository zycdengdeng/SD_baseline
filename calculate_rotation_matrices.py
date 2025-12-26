#!/usr/bin/env python3
"""
计算各个相机的旋转矩阵
基于你的系统中的get_K_R函数
"""
import numpy as np
import cv2

def get_K_R(FOV, THETA, PHI, height, width):
    """
    从你的utils.py复制的函数
    """
    f = 0.5 * width * 1 / np.tan(0.5 * FOV / 180.0 * np.pi)
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    K = np.array([
        [f, 0, cx],
        [0, f, cy],
        [0, 0,  1],
    ], np.float32)

    y_axis = np.array([0.0, 1.0, 0.0], np.float32)
    x_axis = np.array([1.0, 0.0, 0.0], np.float32)
    R1, _ = cv2.Rodrigues(y_axis * np.radians(THETA))
    R2, _ = cv2.Rodrigues(np.dot(R1, x_axis) * np.radians(PHI))
    R = R2 @ R1
    return K, R

def print_rotation_matrix(R, angle, camera_name=""):
    """
    格式化打印旋转矩阵
    """
    print(f"\n{'='*50}")
    print(f"相机角度: {angle}° {camera_name}")
    print(f"{'='*50}")
    print("旋转矩阵 R:")
    for i in range(3):
        print(f"[{R[i,0]:8.5f}, {R[i,1]:8.5f}, {R[i,2]:8.5f}]")
    
    # 计算欧拉角 (仅供参考)
    try:
        # 从旋转矩阵提取欧拉角 (ZYX顺序)
        sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
        singular = sy < 1e-6
        
        if not singular:
            x = np.arctan2(R[2,1], R[2,2])
            y = np.arctan2(-R[2,0], sy)
            z = np.arctan2(R[1,0], R[0,0])
        else:
            x = np.arctan2(-R[1,2], R[1,1])
            y = np.arctan2(-R[2,0], sy)
            z = 0
            
        euler_x = np.degrees(x)
        euler_y = np.degrees(y) 
        euler_z = np.degrees(z)
        print(f"欧拉角 (X, Y, Z): ({euler_x:.2f}°, {euler_y:.2f}°, {euler_z:.2f}°)")
    except:
        print("欧拉角计算失败")

def main():
    # 参数设置
    width = 512
    height = 512
    FOV = 90
    PHI = 0  # 通常设为0，只考虑水平旋转
    
    print("="*80)
    print("计算各个相机的旋转矩阵")
    print("="*80)
    print(f"图像分辨率: {width}×{height}")
    print(f"视场角 FOV: {FOV}°")
    print(f"垂直角度 PHI: {PHI}°")
    
    # 1. nuScenes标准6相机配置
    print("\n" + "="*80)
    print("1. nuScenes标准6相机配置")
    print("="*80)
    
    nuscenes_cameras = {
        'CAM_FRONT_RIGHT': -55,
        'CAM_FRONT': 0,
        'CAM_FRONT_LEFT': 55,
        'CAM_BACK_LEFT': 110,
        'CAM_BACK': 180,
        'CAM_BACK_RIGHT': -110
    }
    
    nuscenes_Rs = {}
    for camera_name, angle in nuscenes_cameras.items():
        K, R = get_K_R(FOV, angle, PHI, height, width)
        nuscenes_Rs[camera_name] = R
        print_rotation_matrix(R, angle, f"({camera_name})")
    
