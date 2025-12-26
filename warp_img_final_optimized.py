#!/usr/bin/env python3
"""
优化版本的图像拼接函数
主要解决边缘拼接问题，支持大角度变换
使用6个最优角度：[-145.0, -82.5, -27.5, 27.5, 82.5, 145.0]
增加物体检测和对齐功能，让同一物体在融合时尽可能重合
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# 添加源代码路径
sys.path.append('src')
from dataset.utils import get_K_R

def detect_meaningful_objects(img1, img2, min_match_count=8):
    """
    检测两张图片中有意义的特殊物体（车辆、行人、交通标志等）
    使用改进的算法，重点检测真正特殊的物体
    """
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY) if len(img1.shape) == 3 else img1
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY) if len(img2.shape) == 3 else img2
    
    # 使用SIFT检测特征点，放宽参数获取更多特征
    sift = cv2.SIFT_create(nfeatures=3000, contrastThreshold=0.02, edgeThreshold=10)
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    
    if des1 is None or des2 is None:
        # print("    SIFT特征检测失败")
        return []
    
    # print(f"    SIFT特征点: img1={len(kp1)}, img2={len(kp2)}")
    
    # FLANN匹配器
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    try:
        matches = flann.knnMatch(des1, des2, k=2)
        
        # 放宽Lowe's ratio test，获取更多匹配
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:  # 放宽到0.75
                    good_matches.append(m)
        
        # print(f"    初始匹配点: {len(good_matches)}")
        
        if len(good_matches) < min_match_count:
            # print(f"    匹配点不足，需要至少{min_match_count}个")
            return []
        
        # 提取匹配点坐标
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches])
        
        # 使用改进的物体检测方法
        objects = find_special_objects(img1, img2, src_pts, dst_pts, good_matches, kp1, kp2)
        
        return objects
        
    except Exception as e:
        # print(f"    物体检测出错: {e}")
        return []

def find_special_objects(img1, img2, src_pts, dst_pts, matches, kp1, kp2):
    """
    寻找特殊的有意义物体，重点检测车辆、行人等
    """
    from sklearn.cluster import DBSCAN
    
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # 先尝试基于空间聚类
    # print(f"    开始聚类分析...")
    
    # 使用更宽松的聚类参数
    eps = 60  # 聚类半径
    min_samples = max(3, len(src_pts) // 15)  # 更小的最小样本数
    
    # print(f"    聚类参数: eps={eps}, min_samples={min_samples}")
    
    # 对源图像特征点进行聚类
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(src_pts)
    labels = clustering.labels_
    
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
    # print(f"    发现{n_clusters}个聚类")
    
    objects = []
    
    for label in unique_labels:
        if label == -1:  # 噪声点
            continue
        
        # 获取该聚类的所有点
        cluster_mask = (labels == label)
        cluster_src_pts = src_pts[cluster_mask]
        cluster_dst_pts = dst_pts[cluster_mask]
        
        if len(cluster_src_pts) < 3:
            continue
        
        # print(f"    聚类{label}: {len(cluster_src_pts)}个特征点")
        
        # 计算基本物体特征
        src_center = np.mean(cluster_src_pts, axis=0)
        dst_center = np.mean(cluster_dst_pts, axis=0)
        
        # 计算边界框
        src_min = np.min(cluster_src_pts, axis=0)
        src_max = np.max(cluster_src_pts, axis=0)
        dst_min = np.min(cluster_dst_pts, axis=0)
        dst_max = np.max(cluster_dst_pts, axis=0)
        
        src_bbox_size = src_max - src_min
        dst_bbox_size = dst_max - dst_min
        
        # 放宽大小限制，只过滤掉明显不合理的
        min_size = min(w1, h1) * 0.01  # 1%以上
        max_size = min(w1, h1) * 0.6   # 60%以下
        
        avg_bbox_size = np.mean([src_bbox_size, dst_bbox_size], axis=0)
        
        if (np.min(avg_bbox_size) < min_size or np.max(avg_bbox_size) > max_size):
            # print(f"      跳过: 大小不合理 {avg_bbox_size}")
            continue
        
        # 计算特征点密度
        bbox_area = avg_bbox_size[0] * avg_bbox_size[1]
        point_density = len(cluster_src_pts) / max(bbox_area, 1)
        
        # 计算物体在两图中的大小一致性
        size_consistency = 1.0 - abs(np.linalg.norm(src_bbox_size) - np.linalg.norm(dst_bbox_size)) / \
                          max(np.linalg.norm(src_bbox_size), np.linalg.norm(dst_bbox_size), 1)
        
        # 检测物体区域的纹理特征
        src_complexity = calculate_region_texture(img1, cluster_src_pts)
        dst_complexity = calculate_region_texture(img2, cluster_dst_pts)
        avg_complexity = (src_complexity + dst_complexity) / 2
        
        # 计算置信度 - 放宽要求
        confidence = (len(cluster_src_pts) / len(src_pts)) * \
                    max(0.3, size_consistency) * \
                    min(1.0, point_density * 1000) * \
                    min(1.0, avg_complexity / 20.0)
        
        # 降低置信度阈值
        if confidence < 0.05:  # 从0.1降到0.05
            # print(f"      跳过: 置信度过低 {confidence:.3f}")
            continue
        
        # 物体类型分类
        object_type = classify_object_by_features(avg_bbox_size, point_density, avg_complexity)
        
        # 检查是否是特殊物体（车辆、行人等）
        is_special = check_if_special_object(img1, img2, cluster_src_pts, cluster_dst_pts, object_type)
        
        # print(f"      检测到物体: 类型={object_type}, 置信度={confidence:.3f}, 特殊={is_special}")
        
        objects.append({
            'src_points': cluster_src_pts.tolist(),
            'dst_points': cluster_dst_pts.tolist(),
            'src_center': src_center,
            'dst_center': dst_center,
            'src_bbox': {
                'min': src_min,
                'max': src_max,
                'center': src_center,
                'size': src_bbox_size
            },
            'dst_bbox': {
                'min': dst_min,
                'max': dst_max,
                'center': dst_center,
                'size': dst_bbox_size
            },
            'confidence': confidence,
            'size_consistency': size_consistency,
            'type': object_type,
            'complexity': avg_complexity,
            'point_density': point_density,
            'is_special': is_special
        })
    
    # 按置信度排序，优先返回特殊物体
    objects.sort(key=lambda x: (x['is_special'], x['confidence']), reverse=True)
    return objects[:5]  # 最多返回5个物体

def calculate_region_texture(img, points):
    """
    计算图像区域的纹理特征（放宽要求）
    """
    if len(points) < 3:
        return 0
    
    try:
        # 获取点的边界框，稍微扩大范围
        min_pt = np.max([np.min(points, axis=0) - 10, [0, 0]], axis=0).astype(int)
        max_pt = np.min([np.max(points, axis=0) + 10, [img.shape[1]-1, img.shape[0]-1]], axis=0).astype(int)
        
        if min_pt[0] >= max_pt[0] or min_pt[1] >= max_pt[1]:
            return 0
        
        # 提取区域
        if len(img.shape) == 3:
            region = img[min_pt[1]:max_pt[1]+1, min_pt[0]:max_pt[0]+1]
            gray_region = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
        else:
            gray_region = img[min_pt[1]:max_pt[1]+1, min_pt[0]:max_pt[0]+1]
        
        if gray_region.size == 0:
            return 0
        
        # 多种纹理特征
        # 1. 梯度方差
        grad_x = cv2.Sobel(gray_region, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_region, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
        texture_score = np.std(gradient_mag)
        
        # 2. 局部二值模式近似
        corners = cv2.goodFeaturesToTrack(gray_region, maxCorners=100, qualityLevel=0.01, minDistance=10)
        corner_density = len(corners) if corners is not None else 0
        
        # 3. 灰度方差
        gray_variance = np.var(gray_region)
        
        # 综合纹理分数
        total_score = texture_score + corner_density * 2 + gray_variance * 0.1
        
        return total_score
    except:
        return 0

def classify_object_by_features(bbox_size, point_density, complexity):
    """
    基于特征分类物体类型（更宽松的分类）
    """
    width, height = bbox_size
    aspect_ratio = width / max(height, 1)
    
    # 基于大小和复杂度的简单分类
    if width > 100 and height > 50:  # 较大物体
        if aspect_ratio > 1.8:
            return "vehicle"  # 宽长比大，可能是车辆
        elif complexity > 30:
            return "building"  # 复杂度高，可能是建筑
        else:
            return "object"
    elif width > 30 and height > 40 and aspect_ratio < 1.5:
        return "person"  # 竖直形状，可能是行人
    elif max(width, height) < 80 and complexity > 20:
        return "sign"  # 小而复杂，可能是标志
    else:
        return "object"

def check_if_special_object(img1, img2, src_pts, dst_pts, object_type):
    """
    检查是否是特殊物体（简化版本）
    """
    # 基于物体类型和特征点数量判断
    if object_type in ['vehicle', 'person', 'sign']:
        return True
    elif len(src_pts) > 10:  # 特征点较多的物体也认为是特殊的
        return True
    else:
        return False

# 保持原有函数名的兼容性
def calculate_region_complexity(img, points):
    """
    计算图像区域的纹理复杂度（兼容旧版本）
    """
    return calculate_region_texture(img, points)

def classify_object_type(img1, img2, src_pts, dst_pts, bbox_size):
    """
    物体类型分类（兼容旧版本）
    """
    point_density = len(src_pts) / max(bbox_size[0] * bbox_size[1], 1)
    complexity = calculate_region_texture(img1, src_pts)
    return classify_object_by_features(bbox_size, point_density, complexity)

def calculate_object_aligned_homography(H_base, objects, img_shape, weight=0.3):
    """
    基于检测到的物体调整单应性矩阵，使物体中心对齐
    """
    if not objects or H_base is None:
        return H_base
    
    h, w = img_shape[:2]
    
    try:
        # 计算所有物体中心点的平均偏移
        center_offsets = []
        weights = []
        
        for obj in objects:
            src_center = obj['src_center']
            dst_center = obj['dst_center']
            confidence = obj['confidence']
            
            # 使用当前单应性矩阵变换源中心点
            src_center_homo = np.array([[src_center[0], src_center[1]]], dtype=np.float32).reshape(-1, 1, 2)
            transformed_center = cv2.perspectiveTransform(src_center_homo, H_base)[0][0]
            
            # 计算偏移量
            offset = dst_center - transformed_center
            center_offsets.append(offset)
            weights.append(confidence)
        
        if not center_offsets:
            return H_base
        
        # 计算加权平均偏移量
        center_offsets = np.array(center_offsets)
        weights = np.array(weights)
        avg_offset = np.average(center_offsets, axis=0, weights=weights)
        
        # 创建平移矩阵
        T = np.array([[1, 0, avg_offset[0] * weight],
                     [0, 1, avg_offset[1] * weight],
                     [0, 0, 1]], dtype=np.float32)
        
        # 组合变换矩阵
        H_aligned = T @ H_base
        
        # 验证调整后的矩阵是否有效
        if check_homography_validity(H_aligned, img_shape):
            return H_aligned
        else:
            return H_base
            
    except Exception as e:
        return H_base

def enhanced_sift_matching(img1, img2):
    """
    改进的SIFT特征匹配
    """
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY) if len(img1.shape) == 3 else img1
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY) if len(img2.shape) == 3 else img2
    
    # 创建SIFT检测器
    sift = cv2.SIFT_create(nfeatures=2000, contrastThreshold=0.02, edgeThreshold=20)
    
    # 检测特征点
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    
    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        return None, []
    
    # FLANN匹配器
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=10)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    try:
        matches = flann.knnMatch(des1, des2, k=2)
        
        # Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
        
        if len(good_matches) >= 10:
            # 提取匹配点
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            # 计算单应性矩阵
            H, mask = cv2.findHomography(
                src_pts, dst_pts,
                method=cv2.RANSAC,
                ransacReprojThreshold=3.0,
                confidence=0.99,
                maxIters=5000
            )
            
            if H is not None and mask.sum() >= 8:
                return H, good_matches
    except:
        pass
    
    return None, []

def create_seamless_blend_mask(img, reference_img=None, quality_factor=1.0, is_primary=False):
    """
    生成无缝融合mask，减少重影和羽化
    """
    h, w = img.shape[:2]
    
    # 基础mask
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    base_mask = (gray > 0).astype(np.float32)
    
    if base_mask.sum() == 0:
        return np.zeros_like(base_mask)
    
    # 如果是主要相机，给予更强的权重
    if is_primary:
        base_weight = 1.0
        feather_size = 3  # 很小的羽化
    else:
        base_weight = max(0.2, quality_factor)
        feather_size = 5  # 适中的羽化
    
    # 距离变换 - 中心区域权重更高
    distance = cv2.distanceTransform((base_mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
    if distance.max() > 0:
        distance = distance / distance.max()
        distance_weight = np.power(distance, 0.8)  # 使用陡峭的权重分布
    else:
        distance_weight = base_mask
    
    # 梯度权重 - 降低高梯度区域的权重
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
    gradient_weight = 1.0 / (1.0 + gradient_mag / 50.0)
    
    # 羽化处理
    if feather_size % 2 == 0:
        feather_size += 1
    
    feathered_mask = cv2.GaussianBlur(base_mask, (feather_size, feather_size), feather_size/6.0)
    
    # 组合所有权重
    final_mask = feathered_mask * distance_weight * gradient_weight * base_weight
    
    # 边界平滑处理
    final_mask = cv2.bilateralFilter(final_mask.astype(np.float32), 5, 25, 25)
    
    return final_mask

def check_homography_validity(H, img_shape):
    """
    检查单应性矩阵是否合理，避免过度扭曲
    """
    if H is None:
        return False
    
    h, w = img_shape[:2]
    
    # 检查四个角点的变换
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    
    try:
        transformed_corners = cv2.perspectiveTransform(corners, H)
        transformed_corners = transformed_corners.reshape(-1, 2)
        
        # 检查是否有无效变换
        if np.any(np.isnan(transformed_corners)) or np.any(np.isinf(transformed_corners)):
            return False
        
        # 检查变换后的面积变化
        original_area = w * h
        transformed_area = cv2.contourArea(transformed_corners)
        
        # 面积变化不应该超过4倍或小于1/4
        area_ratio = transformed_area / original_area
        if area_ratio > 4.0 or area_ratio < 0.25:
            return False
        
        # 检查是否保持凸性（四边形不应该自相交）
        if cv2.isContourConvex(transformed_corners.astype(np.int32)):
            return True
        
    except:
        return False
    
    return False

def combine_geometric_and_feature_transform(img_source, img_target, geometric_theta, geometric_phi, fov, img_shape):
    """
    结合几何变换、特征匹配和物体对齐，防止过度失真
    """
    h, w = img_shape[:2]
    
    # 1. 先计算几何变换矩阵作为基准
    K, R = get_K_R(fov, -geometric_theta, geometric_phi, h, w)
    H_geometric = K @ R @ np.linalg.inv(K)
    
    # 2. 尝试特征匹配
    H_feature, matches = enhanced_sift_matching(img_source, img_target)
    
    # 3. 检测物体
    objects = detect_meaningful_objects(img_source, img_target, min_match_count=8)
    
    # print(f"    几何变换角度: {geometric_theta:.1f}°")
    if objects:
        # print(f"    检测到 {len(objects)} 个物体")
        pass
    
    # 4. 如果特征匹配成功，检查其有效性
    if H_feature is not None and len(matches) > 30:
        # print(f"    特征匹配: {len(matches)}个点")
        
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
                    # print(f"    使用特征匹配 (角点差异: {corner_diff:.1f}像素)")
                    H_base = H_feature
                    quality = min(0.9, len(matches) / 80.0)  # 高质量
                else:
                    # print(f"    特征匹配差异过大 (角点差异: {corner_diff:.1f}像素)，使用混合方案")
                    # 使用加权混合
                    weight_feature = 0.3  # 特征匹配权重较小
                    weight_geometric = 0.7  # 几何变换权重较大
                    
                    # 简单的矩阵加权混合
                    H_base = weight_geometric * H_geometric + weight_feature * H_feature
                    quality = 0.7
                    
                    if not check_homography_validity(H_base, img_shape):
                        # print(f"    混合变换无效，回退到几何变换")
                        H_base = H_geometric
                        quality = 0.5
            except:
                # print(f"    特征匹配计算出错，使用几何变换")
                H_base = H_geometric
                quality = 0.5
        else:
            # print(f"    特征匹配结果无效（过度扭曲），使用几何变换")
            H_base = H_geometric
            quality = 0.5
    else:
        if H_feature is not None:
            # print(f"    特征匹配点不足: {len(matches)}个点，使用几何变换")
            pass
        else:
            # print(f"    特征匹配失败，使用几何变换")
            pass
        H_base = H_geometric
        quality = 0.4
    
    # 5. 应用物体对齐优化
    if objects:
        H_final = calculate_object_aligned_homography(H_base, objects, img_shape, weight=0.2)
        # 如果有物体对齐，提升质量评分
        if H_final is not H_base:
            quality = min(0.95, quality + 0.1 * len(objects))
    else:
        H_final = H_base
    
    return H_final, quality

def refined_geometric_transform(img_source, img_target, base_theta, base_phi, fov, img_shape, search_range=15):
    """
    改进的几何变换，在基础角度附近搜索最佳匹配
    """
    h, w = img_shape[:2]
    best_theta = base_theta
    best_score = -1
    
    # 在基础角度附近搜索
    for delta in range(-search_range, search_range + 1, 3):
        test_theta = base_theta + delta
        
        K, R = get_K_R(fov, -test_theta, base_phi, h, w)
        H_test = K @ R @ np.linalg.inv(K)
        
        # 检查变换的有效性
        if not check_homography_validity(H_test, img_shape):
            continue
        
        # 计算匹配质量
        try:
            warped = cv2.warpPerspective(img_source, H_test, (w, h))
            
            # 计算重叠区域的相似度
            gray_warped = cv2.cvtColor(warped, cv2.COLOR_RGB2GRAY)
            gray_target = cv2.cvtColor(img_target, cv2.COLOR_RGB2GRAY)
            
            mask_warped = (gray_warped > 0).astype(np.uint8)
            mask_target = (gray_target > 0).astype(np.uint8)
            overlap = mask_warped & mask_target
            
            if overlap.sum() > w * h * 0.1:  # 至少10%重叠
                warped_region = gray_warped[overlap > 0]
                target_region = gray_target[overlap > 0]
                
                if len(warped_region) > 1000:
                    # 使用结构相似性
                    from skimage.metrics import structural_similarity as ssim
                    try:
                        # 将区域重塑为固定大小进行SSIM计算
                        min_size = min(len(warped_region), len(target_region))
                        if min_size > 1000:
                            sampled_size = min(min_size, 5000)
                            indices = np.random.choice(min_size, sampled_size, replace=False)
                            
                            w_sample = warped_region[indices]
                            t_sample = target_region[indices]
                            
                            # 计算相关性作为得分
                            correlation = np.corrcoef(w_sample, t_sample)[0, 1]
                            if not np.isnan(correlation):
                                overlap_ratio = overlap.sum() / (mask_warped.sum() + mask_target.sum())
                                score = correlation * overlap_ratio
                                
                                if score > best_score:
                                    best_score = score
                                    best_theta = test_theta
                    except:
                        # 如果SSIM失败，使用简单相关性
                        correlation = np.corrcoef(warped_region, target_region)[0, 1]
                        if not np.isnan(correlation):
                            score = correlation * (overlap.sum() / (mask_warped.sum() + mask_target.sum()))
                            if score > best_score:
                                best_score = score
                                best_theta = test_theta
        except:
            continue
    
    return best_theta, best_score

def enhanced_object_alignment(img_source, img_target, H_base, img_shape, verbose=False):
    """
    增强的物体对齐功能，集成改进的物体检测算法
    让同一个物体在融合时尽可能重合
    """
    h, w = img_shape[:2]
    
    if verbose:
        # print(f"    启动增强物体对齐...")
        pass
    
    # 使用改进的物体检测
    objects = detect_meaningful_objects(img_source, img_target, min_match_count=6)
    
    if not objects:
        if verbose:
            # print(f"    未检测到有意义物体，使用原始变换")
            pass
        return H_base, 0.0
    
    if verbose:
        # print(f"    检测到 {len(objects)} 个有意义物体:")
        for i, obj in enumerate(objects):
            obj_type = obj['type']
            confidence = obj['confidence']
            is_special = obj.get('is_special', False)
            # print(f"      物体{i+1}: {obj_type} (置信度:{confidence:.3f}, 特殊:{is_special})")
    
    # 按重要性排序物体（特殊物体优先）
    sorted_objects = sorted(objects, key=lambda x: (x.get('is_special', False), x['confidence']), reverse=True)
    
    # 计算物体中心的加权对齐
    center_adjustments = []
    weights = []
    
    for obj in sorted_objects[:3]:  # 最多使用前3个最重要的物体
        src_center = obj['src_center']
        dst_center = obj['dst_center']
        confidence = obj['confidence']
        is_special = obj.get('is_special', False)
        
        # 计算当前变换下源中心的投影位置
        src_center_homo = np.array([[src_center[0], src_center[1]]], dtype=np.float32).reshape(-1, 1, 2)
        try:
            transformed_center = cv2.perspectiveTransform(src_center_homo, H_base)[0][0]
            
            # 计算需要的调整量
            adjustment = dst_center - transformed_center
            
            # 权重：特殊物体权重更高
            weight = confidence * (2.0 if is_special else 1.0)
            
            center_adjustments.append(adjustment)
            weights.append(weight)
            
            if verbose:
                # print(f"        中心偏移: {np.linalg.norm(adjustment):.1f}像素 (权重:{weight:.3f})")
                pass
                
        except Exception as e:
            if verbose:
                # print(f"        物体投影失败: {e}")
                pass
            continue
    
    if not center_adjustments:
        if verbose:
            # print(f"    物体对齐计算失败，使用原始变换")
            pass
        return H_base, 0.0
    
    # 计算加权平均调整量
    center_adjustments = np.array(center_adjustments)
    weights = np.array(weights)
    avg_adjustment = np.average(center_adjustments, axis=0, weights=weights)
    
    # 创建平移变换矩阵（适度调整）
    adjustment_strength = min(0.5, len(sorted_objects) * 0.15)  # 根据物体数量调整强度
    T = np.array([[1, 0, avg_adjustment[0] * adjustment_strength],
                  [0, 1, avg_adjustment[1] * adjustment_strength],
                  [0, 0, 1]], dtype=np.float32)
    
    # 应用调整
    H_aligned = T @ H_base
    
    # 验证调整后的变换是否有效
    if check_homography_validity(H_aligned, img_shape):
        alignment_quality = min(0.9, len(sorted_objects) * 0.2 + np.mean(weights) * 0.3)
        
        if verbose:
            # print(f"    物体对齐成功: 调整({avg_adjustment[0]:.1f}, {avg_adjustment[1]:.1f}), " +
            #       f"强度:{adjustment_strength:.2f}, 质量:{alignment_quality:.3f}")
            pass
        
        return H_aligned, alignment_quality
    else:
        if verbose:
            # print(f"    物体对齐调整无效，使用原始变换")
            pass
        return H_base, 0.0

def adaptive_object_weighted_blending(img_source, img_target, objects, H_transform, img_shape):
    """
    基于物体检测的自适应加权融合
    在物体区域使用特殊的融合策略
    """
    h, w = img_shape[:2]
    
    # 变换源图像
    warped_source = cv2.warpPerspective(img_source, H_transform, (w, h))
    
    # 创建基础权重
    base_weight_source = create_seamless_blend_mask(warped_source, img_target, quality_factor=0.8, is_primary=False)
    base_weight_target = create_seamless_blend_mask(img_target, warped_source, quality_factor=0.8, is_primary=True)
    
    # 如果没有检测到物体，使用标准融合
    if not objects:
        total_weight = base_weight_source + base_weight_target
        total_weight = np.clip(total_weight, 1e-6, None)
        
        result = (warped_source.astype(np.float32) * base_weight_source[:, :, None] + 
                 img_target.astype(np.float32) * base_weight_target[:, :, None]) / total_weight[:, :, None]
        
        return result.astype(np.uint8)
    
    # 为物体区域创建特殊权重
    object_regions_source = np.zeros((h, w), dtype=np.float32)
    object_regions_target = np.zeros((h, w), dtype=np.float32)
    
    for obj in objects:
        confidence = obj['confidence']
        is_special = obj.get('is_special', False)
        
        # 特殊物体获得更强的融合权重
        if is_special and confidence > 0.1:
            # 在源图像中标记物体区域
            src_bbox = obj['src_bbox']
            src_mask = create_object_mask(src_bbox, img_source.shape[:2])
            
            # 变换到目标空间
            src_mask_warped = cv2.warpPerspective(src_mask.astype(np.float32), H_transform, (w, h))
            
            # 在目标图像中标记物体区域
            dst_bbox = obj['dst_bbox']
            dst_mask = create_object_mask(dst_bbox, img_target.shape[:2])
            
            # 物体区域的特殊处理
            object_weight = confidence * (1.5 if is_special else 1.0)
            
            object_regions_source += src_mask_warped * object_weight
            object_regions_target += dst_mask * object_weight
    
    # 组合权重
    enhanced_weight_source = base_weight_source + object_regions_source * 0.3
    enhanced_weight_target = base_weight_target + object_regions_target * 0.3
    
    # 归一化
    total_weight = enhanced_weight_source + enhanced_weight_target
    total_weight = np.clip(total_weight, 1e-6, None)
    
    # 融合
    result = (warped_source.astype(np.float32) * enhanced_weight_source[:, :, None] + 
             img_target.astype(np.float32) * enhanced_weight_target[:, :, None]) / total_weight[:, :, None]
    
    return result.astype(np.uint8)

def create_object_mask(bbox, img_shape):
    """
    为物体边界框创建mask
    """
    h, w = img_shape
    mask = np.zeros((h, w), dtype=np.float32)
    
    min_pt = np.maximum(bbox['min'].astype(int), [0, 0])
    max_pt = np.minimum(bbox['max'].astype(int), [w-1, h-1])
    
    if min_pt[0] < max_pt[0] and min_pt[1] < max_pt[1]:
        mask[min_pt[1]:max_pt[1], min_pt[0]:max_pt[0]] = 1.0
        
        # 创建羽化边缘
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)
        mask = cv2.GaussianBlur(mask, (15, 15), 5)
    
    return mask

# 更新主要的拼接函数以使用增强的物体对齐
def warp_img_final_optimized(fov, theta, phi, images, vx, vy, angle_threshold=1.0, enable_object_alignment=True):
    """
    优化版本的图像拼接函数，集成增强的物体检测和对齐功能
    """
    assert len(images) == len(vx) == len(vy), "Images and view angles must match"

    # 完美匹配检测
    for i, (img, cam_theta, cam_phi) in enumerate(zip(images, vx, vy)):
        theta_diff = abs((cam_theta - theta + 180) % 360 - 180)
        phi_diff = abs(cam_phi - phi)
        if theta_diff < angle_threshold and phi_diff < angle_threshold:
            # print(f"完美匹配：相机 {i} ({cam_theta}°)")
            return img.copy()
    
    # print(f"增强拼接 - 目标角度: {theta}° (物体对齐: {'开启' if enable_object_alignment else '关闭'})")
    
    H, W, _ = images[0].shape
    img_combine = np.zeros(images[0].shape, dtype=np.float32)
    total_weight = np.zeros((H, W), dtype=np.float32)
    
    # 收集候选相机
    candidates = []
    for i, img in enumerate(images):
        _theta = (vx[i] - theta + 180) % 360 - 180
        _phi = vy[i] - phi
        
        if _phi == 0 and abs(_theta) <= 75:
            angle_distance = abs(_theta)
            candidates.append((i, img, _theta, _phi, angle_distance))
    
    candidates.sort(key=lambda x: x[4])
    candidates = candidates[:2]  # 最多使用3个相机
    
    # print(f"可用相机: {[c[0] for c in candidates]} (角度差: {[f'{c[4]:.1f}°' for c in candidates]})")
    
    processed_images = []
    
    for idx, (cam_id, img, _theta, _phi, angle_dist) in enumerate(candidates):
        # print(f"\n处理相机 {cam_id} (角度差: {angle_dist:.1f}°)")
        
        is_primary = (idx == 0)
        
        if processed_images and not is_primary and enable_object_alignment:
            # 使用增强的物体对齐功能
            reference_img = processed_images[0]['warped']
            
            # 基础几何变换
            K, R = get_K_R(fov, -_theta, _phi, H, W)
            H_geometric = K @ R @ np.linalg.inv(K)
            
            # 特征匹配
            H_feature, matches = enhanced_sift_matching(img, reference_img)
            
            # 选择最佳基础变换
            if H_feature is not None and len(matches) > 20:
                if check_homography_validity(H_feature, img.shape):
                    H_base = H_feature
                    quality = min(0.8, len(matches) / 50.0)
                    # print(f"    使用特征匹配 ({len(matches)}个点)")
                else:
                    H_base = H_geometric
                    quality = 0.6
                    # print(f"    特征匹配无效，使用几何变换")
            else:
                H_base = H_geometric
                quality = 0.5
                # print(f"    使用几何变换")
            
            # 应用增强的物体对齐
            H_final, alignment_quality = enhanced_object_alignment(
                img, reference_img, H_base, img.shape, verbose=True
            )
            
            # 更新质量评分
            quality = min(0.95, quality + alignment_quality * 0.3)
            
        else:
            # 主相机或关闭物体对齐
            # print(f"    {'主相机' if is_primary else '基础'}几何变换")
            K, R = get_K_R(fov, -_theta, _phi, H, W)
            H_final = K @ R @ np.linalg.inv(K)
            quality = 1.0 if is_primary else 0.6
        
        # 应用变换
        warped = cv2.warpPerspective(img, H_final, (W, H))
        
        # 如果启用了物体对齐，使用增强的融合策略
        if enable_object_alignment and processed_images:
            reference_img = processed_images[0]['warped']
            objects = detect_meaningful_objects(img, reference_img, min_match_count=6)
            
            if objects:
                # print(f"    使用物体感知融合 ({len(objects)}个物体)")
                # 这里可以进一步优化融合策略
                pass
        
        # 创建融合mask
        blend_mask = create_seamless_blend_mask(warped, 
                                              processed_images[0]['warped'] if processed_images else None, 
                                              quality, is_primary)
        
        # 累积
        img_combine += warped.astype(np.float32) * blend_mask[:, :, None]
        total_weight += blend_mask
        
        processed_images.append({
            'cam_id': cam_id,
            'warped': warped,
            'mask': blend_mask,
            'quality': quality,
            'is_primary': is_primary
        })
        
        # print(f"  最终质量: {quality:.3f}")
    
    # 归一化
    total_weight = np.clip(total_weight, 1e-6, None)
    img_combine /= total_weight[:, :, None]
    
    return np.clip(img_combine, 0, 255).astype(np.uint8)

def load_multiple_nuscenes_data(data_root, max_scenes=10):
    """加载多组nuscenes数据，用于批量测试"""
    # print(f"从 {data_root} 加载多组真实数据...")
    
    folders = [f for f in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, f))]
    if not folders:
        return []
    
    # 限制场景数量
    folders = folders[:max_scenes]
    
    # nuScenes真实相机配置
    camera_mapping = {
        'CAM_FRONT': 0,         # 前向 (70° HFOV)
        'CAM_FRONT_LEFT': 55,   # 左前 (70° HFOV) 
        'CAM_FRONT_RIGHT': -55, # 右前 (70° HFOV)
        'CAM_BACK_LEFT': 110,   # 左后 (70° HFOV)
        'CAM_BACK_RIGHT': -110, # 右后 (70° HFOV)  
        'CAM_BACK': 180,        # 后向 (110° HFOV)
    }
    
    scenes_data = []
    ordered_cameras = ['CAM_FRONT_RIGHT', 'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    
    for folder_name in folders:
        test_folder = os.path.join(data_root, folder_name)
        # print(f"\n加载场景: {folder_name}")
        
        images = []
        angles = []
        
        for camera_name in ordered_cameras:
            # 确保相机名后面紧跟_n，避免CAM_BACK匹配到CAM_BACK_LEFT
            image_files = [f for f in os.listdir(test_folder) 
                          if f.startswith(camera_name + '_n') and f.endswith('.jpg')]
            if image_files:
                image_path = os.path.join(test_folder, image_files[0])
                img = cv2.imread(image_path)
                if img is not None:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    images.append(img_rgb)
                    angles.append(camera_mapping[camera_name])
                    # print(f"  加载 {camera_name}: {camera_mapping[camera_name]}°")
        
        if len(images) >= 4:  # 至少需要4个相机
            scenes_data.append((images, angles, folder_name))
            # print(f"  成功加载 {len(images)} 张图像")
        else:
            # print(f"  场景 {folder_name} 图像不足，跳过")
            pass
    
    return scenes_data

def test_optimal_6_angles():
    """测试6个最优角度的拼接效果"""
    print("=== 测试6个最优角度的拼接效果 ===")
    
    real_data_path = "/mnt/vdb1/lyt/localdata/nuscenes/processed_data"
    
    if not os.path.exists(real_data_path):
        print("找不到数据路径")
        return
    
    # 加载多组数据
    scenes_data = load_multiple_nuscenes_data(real_data_path, max_scenes=10)
    
    if not scenes_data:
        print("加载数据失败")
        return
    
    print(f"\n成功加载{len(scenes_data)}个场景")
    
    # nuScenes真实相机角度配置
    vx = [-55, 0, 55, 110, 180, -110]  # 右前、前、左前、左后、后、右后
    vy = [0, 0, 0, 0, 0, 0]
    
    # 6个最优拼接角度（基于FOV重叠分析）
    optimal_angles = [-145.0, -82.5, -27.5, 27.5, 82.5, 145.0]
    
    print(f"使用6个最优角度: {[f'{a:.1f}°' for a in optimal_angles]}")
    
    # 生成场景对比图
    for scene_idx, (real_images, real_angles, scene_name) in enumerate(scenes_data):
        print(f"\n{'='*60}")
        print(f"测试场景{scene_idx+1}: {scene_name}")
        print(f"{'='*60}")
        
        fig, axes = plt.subplots(3, 6, figsize=(24, 12))
        fig.suptitle(f'场景{scene_idx+1}: 6个最优角度拼接效果 ({scene_name})', fontsize=16)
        
        for idx, target_angle in enumerate(optimal_angles):
            print(f"\n目标角度: {target_angle}°")
            
            # 拼接结果
            final_result = warp_img_final_optimized(90, target_angle, 0, real_images, vx, vy, angle_threshold=1.0)
            
            # 找最近的原始图像
            angle_diffs = [abs((angle - target_angle + 180) % 360 - 180) for angle in real_angles]
            closest_idx = angle_diffs.index(min(angle_diffs))
            reference_img = real_images[closest_idx]
            
            # 找第二近的图像
            sorted_indices = sorted(range(len(angle_diffs)), key=lambda k: angle_diffs[k])
            second_closest_idx = sorted_indices[1] if len(sorted_indices) > 1 else closest_idx
            second_reference = real_images[second_closest_idx]
            
            # 显示结果
            axes[0, idx].imshow(reference_img)
            axes[0, idx].set_title(f'最近原图\n{real_angles[closest_idx]:.1f}°\n(差{angle_diffs[closest_idx]:.1f}°)', fontsize=10)
            axes[0, idx].axis('off')
            
            axes[1, idx].imshow(second_reference)
            axes[1, idx].set_title(f'次近原图\n{real_angles[second_closest_idx]:.1f}°\n(差{angle_diffs[second_closest_idx]:.1f}°)', fontsize=10)
            axes[1, idx].axis('off')
            
            axes[2, idx].imshow(final_result)
            axes[2, idx].set_title(f'最优拼接\n{target_angle:.1f}°', fontsize=10)
            axes[2, idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'optimal_6_angles_scene_{scene_idx+1}.png', dpi=150, bbox_inches='tight')
        print(f"\n场景{scene_idx+1}结果已保存: optimal_6_angles_scene_{scene_idx+1}.png")
    
    print(f"\n{'='*60}")
    print("6个最优角度测试完成！")
    print(f"使用的角度: {optimal_angles}")
    print("生成的文件:")
    for i in range(len(scenes_data)):
        print(f"  - optimal_6_angles_scene_{i+1}.png")
    print(f"{'='*60}")

def detect_objects_with_template_matching(img1, img2, min_match_count=8):
    """
    检测两张图片中的有意义物体（保持向后兼容）
    """
    return detect_meaningful_objects(img1, img2, min_match_count)

if __name__ == "__main__":
    test_optimal_6_angles() 