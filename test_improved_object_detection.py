#!/usr/bin/env python3
"""
测试改进的物体检测功能
验证是否能准确检测到有意义的特殊物体（车辆、行人、标志等）
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# 添加源代码路径
sys.path.append('src')
from dataset.utils import get_K_R

# 导入改进后的拼接函数
from warp_img_final_optimized import (
    detect_meaningful_objects,
    load_multiple_nuscenes_data,
    warp_img_final_optimized
)

def test_meaningful_object_detection():
    """
    测试改进的有意义物体检测功能
    """
    print("=== 测试改进的有意义物体检测功能 ===")
    print("新功能特点：")
    print("  ✓ 检测车辆、行人、交通标志等特殊物体")
    print("  ✓ 过滤掉随机噪声和无意义区域")
    print("  ✓ 基于物体大小、纹理复杂度进行智能筛选")
    print("  ✓ 物体类型分类（vehicle/person/sign/building）")
    
    real_data_path = "/mnt/vdb1/lyt/localdata/nuscenes/processed_data"
    
    if not os.path.exists(real_data_path):
        print("找不到数据路径")
        return
    
    # 加载2个场景的数据
    scenes_data = load_multiple_nuscenes_data(real_data_path, max_scenes=2)
    
    if not scenes_data:
        print("加载数据失败")
        return
    
    print(f"\n成功加载{len(scenes_data)}个场景")
    
    for scene_idx, (real_images, real_angles, scene_name) in enumerate(scenes_data):
        print(f"\n{'='*80}")
        print(f"测试场景{scene_idx+1}: {scene_name}")
        print(f"{'='*80}")
        
        # 测试相邻相机之间的物体检测
        camera_pairs = [
            (0, 1, "右前->前"),      # 右前 -> 前
            (1, 2, "前->左前"),      # 前 -> 左前
            (2, 3, "左前->左后"),    # 左前 -> 左后
            (3, 4, "左后->后"),      # 左后 -> 后
            (4, 5, "后->右后"),      # 后 -> 右后
        ]
        
        fig, axes = plt.subplots(len(camera_pairs), 3, figsize=(20, 6*len(camera_pairs)))
        fig.suptitle(f'场景{scene_idx+1}: 改进的有意义物体检测 ({scene_name})', fontsize=16)
        
        for pair_idx, (cam1_idx, cam2_idx, pair_name) in enumerate(camera_pairs):
            img1 = real_images[cam1_idx]
            img2 = real_images[cam2_idx]
            
            print(f"\n{pair_name}: {real_angles[cam1_idx]}° -> {real_angles[cam2_idx]}°")
            
            # 使用改进的物体检测
            objects = detect_meaningful_objects(img1, img2, min_match_count=6)
            
            # 创建标注图像
            img1_annotated = img1.copy()
            img2_annotated = img2.copy()
            
            # 不同类型物体使用不同颜色
            type_colors = {
                'vehicle': (255, 0, 0),    # 红色 - 车辆
                'person': (0, 255, 0),     # 绿色 - 行人
                'sign': (0, 0, 255),       # 蓝色 - 标志
                'building': (255, 255, 0), # 黄色 - 建筑
                'object': (255, 0, 255)    # 紫色 - 一般物体
            }
            
            print(f"  检测到 {len(objects)} 个有意义物体:")
            
            for i, obj in enumerate(objects):
                obj_type = obj['type']
                confidence = obj['confidence']
                complexity = obj['complexity']
                consistency = obj['size_consistency']
                
                print(f"    物体{i+1}: {obj_type}")
                print(f"      置信度: {confidence:.3f}")
                print(f"      纹理复杂度: {complexity:.1f}")
                print(f"      大小一致性: {consistency:.3f}")
                print(f"      中心偏移: {np.linalg.norm(obj['src_center'] - obj['dst_center']):.1f}像素")
                
                color = type_colors.get(obj_type, (128, 128, 128))
                
                # 在图1上标注
                bbox1 = obj['src_bbox']
                center1 = obj['src_center'].astype(int)
                min_pt1 = bbox1['min'].astype(int)
                max_pt1 = bbox1['max'].astype(int)
                
                cv2.rectangle(img1_annotated, tuple(min_pt1), tuple(max_pt1), color, 3)
                cv2.circle(img1_annotated, tuple(center1), 8, color, -1)
                
                # 标注文字
                label = f"{obj_type.upper()}{i+1}"
                cv2.putText(img1_annotated, label, tuple(center1-15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(img1_annotated, f"C:{confidence:.2f}", 
                           tuple(min_pt1 + [0, -10]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                # 在图2上标注
                bbox2 = obj['dst_bbox']
                center2 = obj['dst_center'].astype(int)
                min_pt2 = bbox2['min'].astype(int)
                max_pt2 = bbox2['max'].astype(int)
                
                cv2.rectangle(img2_annotated, tuple(min_pt2), tuple(max_pt2), color, 3)
                cv2.circle(img2_annotated, tuple(center2), 8, color, -1)
                
                cv2.putText(img2_annotated, label, tuple(center2-15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(img2_annotated, f"C:{confidence:.2f}", 
                           tuple(min_pt2 + [0, -10]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # 创建连线图
            if objects:
                h1, w1 = img1.shape[:2]
                h2, w2 = img2.shape[:2]
                h_max = max(h1, h2)
                
                img1_resized = cv2.resize(img1_annotated, (w1, h_max))
                img2_resized = cv2.resize(img2_annotated, (w2, h_max))
                combined_img = np.hstack([img1_resized, img2_resized])
                
                # 绘制连线和物体信息
                for i, obj in enumerate(objects):
                    color = type_colors.get(obj['type'], (128, 128, 128))
                    center1 = (obj['src_center'] * h_max / h1).astype(int)
                    center2 = (obj['dst_center'] * h_max / h2).astype(int)
                    center2[0] += w1
                    
                    # 绘制连线
                    cv2.line(combined_img, tuple(center1), tuple(center2), color, 3)
                    
                    # 在连线中点添加物体类型标签
                    mid_point = ((center1 + center2) // 2).astype(int)
                    cv2.putText(combined_img, obj['type'].upper(), 
                               tuple(mid_point), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            else:
                combined_img = np.hstack([img1, img2])
                h_max = combined_img.shape[0]
                # 添加"无物体"标识
                cv2.putText(combined_img, "NO MEANINGFUL OBJECTS DETECTED", 
                           (50, h_max//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # 显示结果
            axes[pair_idx, 0].imshow(img1_annotated)
            axes[pair_idx, 0].set_title(f'相机{cam1_idx} ({real_angles[cam1_idx]}°)\n{len(objects)}个有意义物体', fontsize=12)
            axes[pair_idx, 0].axis('off')
            
            axes[pair_idx, 1].imshow(img2_annotated)
            axes[pair_idx, 1].set_title(f'相机{cam2_idx} ({real_angles[cam2_idx]}°)\n物体类型标注', fontsize=12)
            axes[pair_idx, 1].axis('off')
            
            axes[pair_idx, 2].imshow(combined_img)
            types_found = set([obj['type'] for obj in objects]) if objects else set()
            axes[pair_idx, 2].set_title(f'{pair_name} 物体匹配\n类型: {", ".join(types_found) if types_found else "无"}', fontsize=12)
            axes[pair_idx, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'improved_object_detection_scene_{scene_idx+1}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\n场景{scene_idx+1}改进物体检测结果已保存: improved_object_detection_scene_{scene_idx+1}.png")
    
    # 生成物体类型统计
    create_object_type_statistics(scenes_data)
    
    print(f"\n{'='*80}")
    print("改进的有意义物体检测测试完成！")
    print("主要改进：")
    print("  ✓ 不再随机聚类，而是基于物体特征智能筛选")
    print("  ✓ 过滤掉纹理复杂度低的区域（如天空、道路）")
    print("  ✓ 检查物体大小合理性（2%-40%图像大小）")
    print("  ✓ 验证物体在两图中的一致性")
    print("  ✓ 简单的物体类型分类")
    print("\n生成的文件:")
    for i in range(len(scenes_data)):
        print(f"  - improved_object_detection_scene_{i+1}.png")
    print(f"  - object_type_statistics.png")
    print(f"{'='*80}")

def create_object_type_statistics(scenes_data):
    """
    创建物体类型统计图
    """
    type_counts = {'vehicle': 0, 'person': 0, 'sign': 0, 'building': 0, 'object': 0}
    confidence_by_type = {'vehicle': [], 'person': [], 'sign': [], 'building': [], 'object': []}
    
    print(f"\n=== 物体类型统计 ===")
    
    for scene_idx, (real_images, real_angles, scene_name) in enumerate(scenes_data):
        print(f"\n场景{scene_idx+1} ({scene_name}):")
        
        camera_pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
        
        for cam1_idx, cam2_idx in camera_pairs:
            img1 = real_images[cam1_idx]
            img2 = real_images[cam2_idx]
            
            objects = detect_meaningful_objects(img1, img2, min_match_count=6)
            
            for obj in objects:
                obj_type = obj['type']
                confidence = obj['confidence']
                type_counts[obj_type] += 1
                confidence_by_type[obj_type].append(confidence)
        
        # 统计该场景的物体类型
        scene_counts = {'vehicle': 0, 'person': 0, 'sign': 0, 'building': 0, 'object': 0}
        for cam1_idx, cam2_idx in camera_pairs:
            objects = detect_meaningful_objects(real_images[cam1_idx], real_images[cam2_idx], min_match_count=6)
            for obj in objects:
                scene_counts[obj['type']] += 1
        
        for obj_type, count in scene_counts.items():
            if count > 0:
                print(f"  {obj_type}: {count}个")
    
    # 创建统计图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 物体类型数量统计
    types = list(type_counts.keys())
    counts = list(type_counts.values())
    colors = ['red', 'green', 'blue', 'yellow', 'purple']
    
    bars = ax1.bar(types, counts, color=colors, alpha=0.7)
    ax1.set_title('检测到的物体类型数量统计', fontsize=14)
    ax1.set_ylabel('数量')
    ax1.set_xlabel('物体类型')
    
    # 在柱状图上添加数值标签
    for bar, count in zip(bars, counts):
        if count > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    str(count), ha='center', va='bottom', fontsize=12)
    
    # 各类型物体的平均置信度
    avg_confidences = []
    for obj_type in types:
        if confidence_by_type[obj_type]:
            avg_conf = np.mean(confidence_by_type[obj_type])
            avg_confidences.append(avg_conf)
        else:
            avg_confidences.append(0)
    
    bars2 = ax2.bar(types, avg_confidences, color=colors, alpha=0.7)
    ax2.set_title('各类型物体的平均置信度', fontsize=14)
    ax2.set_ylabel('平均置信度')
    ax2.set_xlabel('物体类型')
    ax2.set_ylim(0, 1)
    
    # 在柱状图上添加数值标签
    for bar, conf in zip(bars2, avg_confidences):
        if conf > 0:
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                    f'{conf:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('object_type_statistics.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n总体统计:")
    total_objects = sum(type_counts.values())
    print(f"  总检测物体数: {total_objects}")
    for obj_type, count in type_counts.items():
        if count > 0:
            percentage = count / total_objects * 100
            avg_conf = np.mean(confidence_by_type[obj_type]) if confidence_by_type[obj_type] else 0
            print(f"  {obj_type}: {count}个 ({percentage:.1f}%), 平均置信度: {avg_conf:.3f}")
    
    print(f"\n物体类型统计图已保存: object_type_statistics.png")

if __name__ == "__main__":
    test_meaningful_object_detection() 