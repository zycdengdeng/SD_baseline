#!/usr/bin/env python3
"""
测试旋转和平移的组合几何变换
支持类似以下数据的变换：
位置： [ 835.66081529 1795.62225487    0.        ]
位置变化 (x, y, z)： [ 0.96428304 -1.01574434  0.        ]
位置变化距离： 1.400564 米
旋转四元数： [ 0.9196665  0.00867151  0.01555967 -0.39229611]
旋转变化大小： 0.001065
瞬时速度： 17.40 米/秒 (62.66 km/h)
"""
import cv2
import numpy as np
import os
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# 添加源代码路径
sys.path.append('src')
from dataset.utils import get_K_R

# 导入优化的拼接函数
from warp_img_final_optimized import warp_img_final_optimized

def quaternion_to_euler(quaternion):
    """
    将四元数转换为欧拉角（角度制）
    """
    # 创建旋转对象
    rot = R.from_quat(quaternion)
    # 获取欧拉角（角度制）
    euler_angles = rot.as_euler('xyz', degrees=True)
    return euler_angles

def apply_rotation_translation_transform(img_source, fov, translation, quaternion, img_shape):
    """
    应用旋转和平移的组合变换
    
    参数:
    - img_source: 源图像
    - fov: 视场角
    - translation: [x, y, z] 平移向量（米）
    - quaternion: [w, x, y, z] 四元数表示的旋转
    - img_shape: 图像尺寸
    
    返回:
    - 变换后的图像
    - 变换矩阵
    """
    h, w = img_shape[:2]
    
    # 1. 从四元数计算旋转角度（欧拉角）
    euler_angles = quaternion_to_euler([quaternion[1], quaternion[2], quaternion[3], quaternion[0]])
    theta, phi, psi = euler_angles  # 绕x, y, z轴的旋转
    
    # 2. 计算基础旋转变换
    K, R = get_K_R(fov, -theta, phi, h, w)
    H_rotation = K @ R @ np.linalg.inv(K)
    
    # 3. 计算平移变换（像素坐标）
    # 假设1米 = 100像素（这个比例需要根据实际情况调整）
    pixel_scale = 100.0  
    tx = translation[0] * pixel_scale
    ty = translation[1] * pixel_scale
    
    # 创建平移矩阵
    T = np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1]
    ], dtype=np.float64)
    
    # 4. 组合旋转和平移
    H_combined = T @ H_rotation
    
    # 5. 应用变换
    warped = cv2.warpPerspective(img_source, H_combined, (w, h), 
                               flags=cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS)
    
    return warped, H_combined, euler_angles

def test_rotation_translation():
    """
    测试旋转和平移的组合变换
    """
    print("=== 测试旋转和平移的组合几何变换 ===")
    
    # 数据路径
    real_data_path = "/mnt/vdb1/lyt/localdata/nuscenes/processed_data"
    output_dir = "rotation_translation_results"
    
    if not os.path.exists(real_data_path):
        print(f"错误: 找不到数据路径 {real_data_path}")
        return
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载测试数据
    print("\n加载测试数据...")
    folders = [f for f in os.listdir(real_data_path) if os.path.isdir(os.path.join(real_data_path, f))]
    if not folders:
        print("错误: 没有找到场景")
        return
    
    # 选择前3个场景进行测试
    selected_folders = folders[:3]
    
    # 示例变换参数（从图片中提取）
    example_transforms = [
        {
            'position': np.array([835.66081529, 1795.62225487, 0.0]),
            'translation': np.array([0.96428304, -1.01574434, 0.0]),
            'distance': 1.400564,  # 米
            'quaternion': np.array([0.9196665, 0.00867151, 0.01555967, -0.39229611]),
            'rotation_magnitude': 0.001065,
            'speed': 17.40  # 米/秒
        },
        # 可以添加更多变换参数
        {
            'position': np.array([835.66081529, 1795.62225487, 0.0]),
            'translation': np.array([0.5, -0.5, 0.0]),
            'distance': 0.7071,  # 米
            'quaternion': np.array([0.9996, 0.0173, 0.0, 0.0]),
            'rotation_magnitude': 0.002,
            'speed': 10.0  # 米/秒
        },
        {
            'position': np.array([835.66081529, 1795.62225487, 0.0]),
            'translation': np.array([-0.3, 0.8, 0.0]),
            'distance': 0.854,  # 米
            'quaternion': np.array([0.9993, 0.0, 0.0, -0.0367]),
            'rotation_magnitude': 0.005,
            'speed': 5.0  # 米/秒
        }
    ]
    
    # 为每个场景测试变换
    for scene_idx, folder_name in enumerate(selected_folders):
        print(f"\n处理场景 {scene_idx+1}: {folder_name}")
        test_folder = os.path.join(real_data_path, folder_name)
        
        # 加载前视相机图像
        front_image_files = [f for f in os.listdir(test_folder) 
                            if f.startswith('CAM_FRONT_n') and f.endswith('.jpg')]
        if not front_image_files:
            print("  跳过: 找不到前视相机图像")
            continue
        
        # 读取图像
        image_path = os.path.join(test_folder, front_image_files[0])
        img = cv2.imread(image_path)
        if img is None:
            print("  跳过: 无法读取图像")
            continue
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 对每个变换参数进行测试
        for transform_idx, transform in enumerate(example_transforms):
            print(f"\n  应用变换 {transform_idx+1}:")
            print(f"    位置: {transform['position']}")
            print(f"    平移: {transform['translation']} (距离: {transform['distance']}米)")
            print(f"    四元数: {transform['quaternion']}")
            
            # 应用变换
            warped_img, H, euler_angles = apply_rotation_translation_transform(
                img_rgb, 
                90,  # 假设90度FOV
                transform['translation'], 
                transform['quaternion'], 
                img_rgb.shape
            )
            
            print(f"    欧拉角 (度): {euler_angles}")
            
            # 创建对比图
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # 原始图像
            axes[0].imshow(img_rgb)
            axes[0].set_title(f'原始图像\n场景: {folder_name}', fontsize=10)
            axes[0].axis('off')
            
            # 变换后图像
            axes[1].imshow(warped_img)
            axes[1].set_title(f'旋转+平移变换\n平移: [{transform["translation"][0]:.2f}, {transform["translation"][1]:.2f}]米\n'
                             f'欧拉角: [{euler_angles[0]:.2f}, {euler_angles[1]:.2f}, {euler_angles[2]:.2f}]度', 
                             fontsize=10)
            axes[1].axis('off')
            
            # 差异图
            diff = cv2.absdiff(img_rgb, warped_img)
            diff_normalized = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
            axes[2].imshow(diff_normalized)
            axes[2].set_title('差异图', fontsize=10)
            axes[2].axis('off')
            
            # 保存结果
            plt.tight_layout()
            output_file = os.path.join(output_dir, f'scene_{scene_idx+1}_transform_{transform_idx+1}.png')
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"    结果已保存: {output_file}")
    
    print(f"\n{'='*60}")
    print("旋转和平移组合变换测试完成！")
    print(f"结果保存在: {output_dir}/")
    print(f"{'='*60}")

def create_combined_transformation_function():
    """
    创建一个可以集成到warp_img_final_optimized.py中的组合变换函数
    """
    # 这个函数只是为了展示如何集成到现有代码中
    def combined_geometric_transform(img_source, img_target, geometric_theta, geometric_phi, 
                                   translation, quaternion, fov, img_shape):
        """
        结合几何旋转、平移和四元数旋转的变换函数
        可以替代原始的combine_geometric_and_feature_transform函数
        """
        h, w = img_shape[:2]
        
        # 1. 从四元数计算旋转角度（欧拉角）
        if quaternion is not None:
            rot = R.from_quat([quaternion[1], quaternion[2], quaternion[3], quaternion[0]])
            euler_angles = rot.as_euler('xyz', degrees=True)
            theta, phi, psi = euler_angles  # 绕x, y, z轴的旋转
        else:
            theta, phi = geometric_theta, geometric_phi
        
        # 2. 计算基础旋转变换
        K, R = get_K_R(fov, -theta, phi, h, w)
        H_rotation = K @ R @ np.linalg.inv(K)
        
        # 3. 如果有平移，添加平移变换
        if translation is not None:
            # 假设1米 = 100像素（这个比例需要根据实际情况调整）
            pixel_scale = 100.0  
            tx = translation[0] * pixel_scale
            ty = translation[1] * pixel_scale
            
            # 创建平移矩阵
            T = np.array([
                [1, 0, tx],
                [0, 1, ty],
                [0, 0, 1]
            ], dtype=np.float64)
            
            # 组合旋转和平移
            H_geometric = T @ H_rotation
        else:
            H_geometric = H_rotation
        
        # 4. 尝试特征匹配（与原函数相同）
        H_feature, matches = enhanced_sift_matching(img_source, img_target)
        
        # 5. 检测物体（与原函数相同）
        objects = detect_meaningful_objects(img_source, img_target, min_match_count=8)
        
        # 6. 如果特征匹配成功，检查其有效性（与原函数相同）
        if H_feature is not None and len(matches) > 30:
            # 检查特征匹配的单应性矩阵是否合理
            if check_homography_validity(H_feature, img_shape):
                # 计算几何变换和特征匹配的差异
                corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
                
                try:
                    geo_corners = cv2.perspectiveTransform(corners, H_geometric).reshape(-1, 2)
                    feat_corners = cv2.perspectiveTransform(corners, H_feature).reshape(-1, 2)
                    
                    # 计算角点差异
                    corner_diff = np.mean(np.linalg.norm(geo_corners - feat_corners, axis=1))
                    
                    # 如果差异不太大，可以使用特征匹配结果
                    if corner_diff < min(w, h) * 0.3:  # 差异小于图像尺寸的30%
                        H_base = H_feature
                        quality = min(0.9, len(matches) / 80.0)  # 高质量
                    else:
                        # 使用加权混合
                        weight_feature = 0.3  # 特征匹配权重较小
                        weight_geometric = 0.7  # 几何变换权重较大
                        
                        # 简单的矩阵加权混合
                        H_base = weight_geometric * H_geometric + weight_feature * H_feature
                        quality = 0.7
                        
                        if not check_homography_validity(H_base, img_shape):
                            H_base = H_geometric
                            quality = 0.5
                except:
                    H_base = H_geometric
                    quality = 0.5
            else:
                H_base = H_geometric
                quality = 0.5
        else:
            H_base = H_geometric
            quality = 0.4
        
        # 7. 应用物体对齐优化（与原函数相同）
        if objects:
            H_final = calculate_object_aligned_homography(H_base, objects, img_shape, weight=0.2)
            # 如果有物体对齐，提升质量评分
            if H_final is not H_base:
                quality = min(0.95, quality + 0.1 * len(objects))
        else:
            H_final = H_base
        
        return H_final, quality
    
    return combined_geometric_transform

if __name__ == "__main__":
    test_rotation_translation()