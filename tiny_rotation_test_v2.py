#!/usr/bin/env python3
"""
测试微小角度旋转的几何变换效果（改进版）
使用更高精度的数值计算
"""
import cv2
import numpy as np
import os
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt

# 添加源代码路径
sys.path.append('src')
from dataset.utils import get_K_R

def force_geometric_transform_high_precision(img_source, fov, theta, phi, img_shape):
    """
    强制进行几何变换，使用高精度计算
    """
    h, w = img_shape[:2]
    
    # 转换为弧度并使用64位浮点数
    theta_rad = np.float64(theta) * np.pi / 180.0
    phi_rad = np.float64(phi) * np.pi / 180.0
    fov_rad = np.float64(fov) * np.pi / 180.0
    
    # 获取变换矩阵
    K, R = get_K_R(fov, -theta, phi, h, w)
    
    # 转换为64位浮点数
    K = K.astype(np.float64)
    R = R.astype(np.float64)
    
    # 计算单应性矩阵
    H_geometric = K @ R @ np.linalg.inv(K)
    
    # 打印变换矩阵的偏离单位矩阵的程度
    I = np.eye(3, dtype=np.float64)
    diff_from_identity = np.abs(H_geometric - I)
    max_diff = np.max(diff_from_identity)
    print(f"    变换矩阵与单位矩阵的最大差异: {max_diff:.10f}")
    print(f"    变换矩阵:\n{H_geometric}")
    
    # 应用变换
    warped = cv2.warpPerspective(img_source, H_geometric, (w, h), 
                                flags=cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS)
    return warped, H_geometric

def analyze_pixel_differences(img1, img2):
    """
    详细分析两个图像之间的差异
    """
    diff = cv2.absdiff(img1, img2)
    
    # 计算每个通道的差异统计
    stats = []
    for channel in range(3):
        channel_diff = diff[:,:,channel]
        non_zero = channel_diff[channel_diff > 0]
        if len(non_zero) > 0:
            stats.append({
                'channel': channel,
                'mean': np.mean(non_zero),
                'max': np.max(channel_diff),
                'num_diff_pixels': len(non_zero),
                'percent_diff': len(non_zero) / (diff.shape[0] * diff.shape[1]) * 100
            })
    
    return stats

def test_tiny_rotation_v2():
    """
    测试微小角度旋转的效果（改进版）
    """
    print("=== 测试微小角度旋转的几何变换（高精度版）===")
    
    # 数据路径
    real_data_path = "/mnt/vdb1/lyt/localdata/nuscenes/processed_data"
    output_dir = "tiny_rotation_results_v2"
    
    if not os.path.exists(real_data_path):
        print(f"错误: 找不到数据路径 {real_data_path}")
        return
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载一个场景的数据进行详细分析
    print("\n加载测试数据...")
    folders = [f for f in os.listdir(real_data_path) if os.path.isdir(os.path.join(real_data_path, f))]
    if not folders:
        print("错误: 没有找到场景")
        return
    
    test_folder = os.path.join(real_data_path, folders[0])
    
    # 加载前视相机图像
    front_image_files = [f for f in os.listdir(test_folder) 
                        if f.startswith('CAM_FRONT_n') and f.endswith('.jpg')]
    if not front_image_files:
        print("错误: 找不到前视相机图像")
        return
    
    # 读取图像
    image_path = os.path.join(test_folder, front_image_files[0])
    img = cv2.imread(image_path)
    if img is None:
        print("错误: 无法读取图像")
        return
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 测试不同精度的旋转角度
    rotation_angles = [
        0.000335,  # 原始角度
        0.001,     # 稍大的角度
        0.01,      # 更大的角度
    ]
    
    fig, axes = plt.subplots(len(rotation_angles) + 1, 3, figsize=(15, 5*(len(rotation_angles)+1)))
    fig.suptitle('微小角度旋转效果分析（高精度版）', fontsize=16)
    
    # 显示原始图像
    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title('原始图像', fontsize=10)
    axes[0, 0].axis('off')
    axes[0, 1].axis('off')
    axes[0, 2].axis('off')
    
    for idx, angle in enumerate(rotation_angles):
        print(f"\n测试旋转角度: {angle}°")
        
        # 进行变换
        rotated_img, H = force_geometric_transform_high_precision(img_rgb, 90, angle, 0, img_rgb.shape)
        
        # 计算差异
        diff = cv2.absdiff(img_rgb, rotated_img)
        diff_stats = analyze_pixel_differences(img_rgb, rotated_img)
        
        # 显示结果
        row = idx + 1
        axes[row, 0].imshow(rotated_img)
        axes[row, 0].set_title(f'旋转 {angle}°', fontsize=10)
        axes[row, 0].axis('off')
        
        # 显示差异图（增强对比度以便观察）
        diff_display = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
        axes[row, 1].imshow(diff_display)
        axes[row, 1].set_title('差异图（增强显示）', fontsize=10)
        axes[row, 1].axis('off')
        
        # 显示差异热图
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
        axes[row, 2].imshow(diff_gray, cmap='hot')
        axes[row, 2].set_title('差异热图', fontsize=10)
        axes[row, 2].axis('off')
        
        # 打印详细的差异统计
        print("差异统计:")
        for stat in diff_stats:
            print(f"  通道 {stat['channel']}:")
            print(f"    平均差异: {stat['mean']:.10f}")
            print(f"    最大差异: {stat['max']:.10f}")
            print(f"    不同像素数: {stat['num_diff_pixels']}")
            print(f"    不同像素占比: {stat['percent_diff']:.6f}%")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'tiny_rotation_analysis.png'), 
                dpi=150, bbox_inches='tight')
    print(f"\n分析结果已保存: {output_dir}/tiny_rotation_analysis.png")

if __name__ == "__main__":
    test_tiny_rotation_v2()