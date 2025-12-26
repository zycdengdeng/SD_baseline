# nuScenes图像拼接优化算法


## nuScenes相机配置

```
前方区域:                    后方区域:
    CAM_FRONT_LEFT(55°)          CAM_BACK_LEFT(110°)
         ↖                              ↖
CAM_FRONT(0°) ←→ 车辆 ←→ CAM_BACK(180°)
         ↘                              ↘  
    CAM_FRONT_RIGHT(-55°)        CAM_BACK_RIGHT(-110°)
```

**相机参数：**
- 前向相机：70° HFOV
- 后向相机：110° HFOV（更大视野）
- 总覆盖：360°全景

## 🎯 6个最优拼接角度

基于相机FOV重叠分析，算法选择了以下6个最优角度：

| 角度 | 位置描述 | 主要参与相机 | 优势 |
|------|----------|--------------|------|
| **-27.5°** | 右前方 | 右前相机 ↔ 前相机 | 重叠中心点 |
| **27.5°** | 左前方 | 前相机 ↔ 左前相机 | 重叠中心点 |
| **82.5°** | 左侧方 | 左前相机 ↔ 左后相机 | 重叠中心点 |
| **145.0°** | 左后方 | 左后相机 ↔ 后相机 | 重叠中心点 |
| **-82.5°** | 右侧方 | 右后相机 ↔ 右前相机 | 重叠中心点 |
| **-145.0°** | 右后方 | 后相机 ↔ 右后相机 | 重叠中心点 |


## 核心算法流程

### 1️⃣ 输入处理
```python
输入: 目标角度θ, 6张相机图像, 相机角度配置
↓
完美匹配检测: 如果目标角度与某相机角度差异<1°，直接返回该相机图像
```

### 2️⃣ 候选相机选择
```python
遍历所有相机:
  计算角度差异 = |相机角度 - 目标角度|
  如果角度差异 ≤ 75°:
    加入候选列表
↓
按角度差异排序，选择最近的3个相机
```

### 3️⃣ 主相机处理（第一个相机）
```python
主相机（角度差异最小）:
  使用几何变换: K,R = get_K_R(FOV, -角度差, 0, 高, 宽)
  变换矩阵: H = K @ R @ K^(-1)
  质量分数: 1.0（最高）
```

### 4️⃣ 辅助相机处理（其他相机）

#### 4️⃣a 混合变换策略
```python
对于每个辅助相机:
  1. 计算几何变换基准矩阵 H_geo
  2. 尝试SIFT特征匹配 → H_feat
  3. 检查特征匹配有效性:
     - 特征点数量 > 30
     - 变换后不过度扭曲
     - 角点差异 < 图像尺寸30%
  
  如果特征匹配有效:
    if 角点差异小:
      使用特征匹配结果 H_feat
    else:
      使用混合方案: H = 0.7×H_geo + 0.3×H_feat
  else:
    使用几何变换 H_geo
```

#### 4️⃣b 角度优化搜索
```python
如果混合变换质量不足（< 0.6）:
  在基础角度±15°范围内搜索:
    for 角度偏移 in [-15°, -12°, ..., +15°]:
      测试角度 = 基础角度 + 偏移
      计算变换质量（重叠相关性）
      记录最佳角度
  
  如果找到更好角度:
    使用优化后的几何变换
```

### 5️⃣ 安全检查
```python
对每个变换矩阵H检查:
  1. 四个角点变换后是否有效（无NaN/Inf）
  2. 变换后面积变化是否合理（0.25倍~4倍）
  3. 变换后形状是否保持凸性（无自相交）
  
如果检查失败:
  回退到基础几何变换
  质量分数降低到0.3
```

### 6️⃣ 图像变换与融合

#### 6️⃣a 图像变换
```python
对每个选中的相机:
  warped_image = cv2.warpPerspective(原图, H_best, (宽, 高))
```

#### 6️⃣b 融合权重计算
```python
创建融合mask:
  1. 基础mask = (灰度图 > 0)
  2. 距离权重 = distance_transform^0.8  # 中心区域权重高
  3. 梯度权重 = 1/(1 + 梯度强度/50)    # 降低高梯度区域权重
  4. 质量权重 = 变换质量分数
  5. 主相机权重 = 1.0, 辅助相机权重 = max(0.2, 质量分数)
  
最终mask = 高斯模糊(基础mask) × 距离权重 × 梯度权重 × 质量权重
```

#### 6️⃣c 加权融合
```python
结果图像 = Σ(变换图像 × 融合mask) / Σ(融合mask)
```

### 7️⃣ 输出结果
```python
输出: 无缝拼接的目标角度图像（0-255像素值）
```

## 关键函数说明

### `sift_match(img1, img2)`
**功能**：SIFT特征点匹配
```python
# 检测SIFT特征点 → FLANN匹配 → Lowe比值测试 → RANSAC单应性估计
返回: (单应性矩阵H, 匹配点列表) 或 (None, [])
```

### `check_valid_transform(H, shape)`
**功能**：检查变换矩阵有效性
```python
# 检查四角点变换 → 面积变化检查 → 凸性检查
返回: True(有效) 或 False(无效)
```

### `hybrid_transform(img_src, img_ref, theta, phi, fov, shape)`
**功能**：混合几何变换和特征匹配
```python
几何变换基准 + 特征匹配优化 + 智能选择策略
返回: (最佳变换矩阵, 质量分数)
```

### `search_best_angle(img_src, img_ref, base_theta, phi, fov, shape)`
**功能**：在±15°范围内搜索最佳几何角度
```python
# 角度扫描 → 重叠相关性评估 → 最佳角度选择
返回: (最佳角度, 质量分数)
```

### `create_blend_mask(img, ref_img, quality, is_primary)`
**功能**：创建多层权重融合mask
```python
距离变换 + 梯度权重 + 质量权重 + 主/辅相机权重
返回: 融合权重mask
```

### `warp_images(fov, theta, phi, images, vx, vy, angle_threshold)`
**功能**：主拼接函数
```python
完整的拼接流程：候选选择 → 变换计算 → 融合处理 → 结果输出
返回: 拼接后的图像
```

## 使用方法

### 基本使用
```python
from warp_img_final_optimized import warp_images

# nuScenes相机配置
vx = [-55, 0, 55, 110, 180, -110]  # 6个相机角度
vy = [0, 0, 0, 0, 0, 0]           # 水平放置

# 拼接到指定角度
result = warp_images(
    fov=90,                    # 输出视场角
    theta=27.5,               # 目标角度
    phi=0,                    # 俯仰角（通常为0）
    images=camera_images,     # 6张相机图像列表
    vx=vx,                    # 相机水平角度
    vy=vy,                    # 相机俯仰角度
    angle_threshold=1.0       # 完美匹配阈值
)
```

### 批量测试
```python
# 运行完整测试
python warp_img_final_optimized.py
```

## 输出结果

运行测试后将生成：
- `scene_1_results.png` - 场景1的6个角度拼接对比图
- `scene_2_results.png` - 场景2的6个角度拼接对比图
- ... (最多10个场景)

每张图包含3行×6列：
- **第1行**：最近原始相机图像
- **第2行**：次近原始相机图像  
- **第3行**：算法拼接结果

## 参数调节

### 关键参数
```python
# 特征匹配参数
nfeatures=2000              # SIFT特征点数量
contrastThreshold=0.02      # 对比度阈值
edgeThreshold=20           # 边缘阈值

# 变换有效性检查
area_ratio_range=[0.25, 4.0]  # 面积变化范围
max_corner_diff=0.3           # 最大角点差异（相对图像尺寸）

# 融合权重
primary_weight=1.0            # 主相机权重
min_secondary_weight=0.2      # 辅助相机最小权重
feather_size_primary=3        # 主相机羽化尺寸
feather_size_secondary=5      # 辅助相机羽化尺寸

# 角度搜索
search_range=15              # 搜索范围（±15°）
search_step=3               # 搜索步长（3°）
```

### 性能调节
```python
max_cameras=3               # 最大相机数量（影响质量和性能）
angle_limit=75              # 候选相机角度限制
min_overlap_ratio=0.1       # 最小重叠比例
min_feature_matches=30      # 最小特征匹配点数
```

#

### 数学模型
```python
# 几何变换矩阵
K, R = get_K_R(fov, -theta, phi, height, width)
H_geometric = K @ R @ K^(-1)

# 混合变换
H_hybrid = w_geo × H_geometric + w_feat × H_feature
where w_geo + w_feat = 1.0

# 融合权重
weight = feather_mask × distance_weight × gradient_weight × quality_factor
```
