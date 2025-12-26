#!/usr/bin/env python3
"""
计算ArbiViewGen生成图像质量评估指标

本脚本计算以下指标:
1. FID (Fréchet Inception Distance) - 评估生成图像分布与真实图像分布的差异
2. IS (Inception Score) - 评估生成图像的质量和多样性
3. CS (Cosine Similarity) - 计算图像特征的余弦相似度

使用方法:
python calculate_metrics.py --output_dir /path/to/images
"""

import os
import glob
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips
import torch
from PIL import Image
import argparse
from collections import defaultdict
from torch_fidelity import calculate_metrics as fidelity_metrics
import re
from torchvision import models, transforms
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")

def calculate_psnr(img1, img2):
    """计算PSNR"""
    return psnr(img1, img2, data_range=255)

def calculate_ssim(img1, img2):
    # 增加 channel_axis 和 win_size 参数
    return ssim(img1, img2, data_range=255, channel_axis=-1, win_size=7)

def calculate_lpips(img1, img2, lpips_fn):
    """计算LPIPS"""
    # 转换为LPIPS期望的格式 (1, 3, H, W) 范围 [-1, 1]
    img1_tensor = torch.from_numpy(img1).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1
    img2_tensor = torch.from_numpy(img2).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1

    # 保证和lpips_fn在同一设备
    device = next(lpips_fn.parameters()).device
    img1_tensor = img1_tensor.to(device)
    img2_tensor = img2_tensor.to(device)

    with torch.no_grad():
        lpips_value = lpips_fn(img1_tensor, img2_tensor).item()
    return lpips_value

def load_image(image_path):
    """加载图片"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法加载图片: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def calculate_fid_is_cs(gt_dir, pred_dir, offline_mode=True):
    """计算FID、IS和CS指标"""
    try:
        # 确保目录存在且有图像
        if not os.path.exists(gt_dir) or not os.path.exists(pred_dir):
            print(f"警告: FID计算失败，目录不存在: {gt_dir} or {pred_dir}")
            return None, None, None
        
        fid_value = None
        is_value = None
            
        if not offline_mode:
            try:
                # 设置本地权重路径，避免网络下载
                os.environ['TORCH_HOME'] = '/mnt/vdb1/lyt/ArbiViewGen-main/checkpoints'
                
                # 检查权重文件是否已存在
                weights_path = '/mnt/vdb1/lyt/ArbiViewGen-main/checkpoints/weights-inception-2015-12-05-6726825d.pth'
                hub_weights_path = '/mnt/vdb1/lyt/ArbiViewGen-main/checkpoints/hub/checkpoints/weights-inception-2015-12-05-6726825d.pth'
                
                # 如果目标目录不存在，创建它
                os.makedirs(os.path.dirname(hub_weights_path), exist_ok=True)
                
                # 如果源文件存在但目标文件不存在，复制它
                if os.path.exists(weights_path) and not os.path.exists(hub_weights_path):
                    print(f"复制权重文件从 {weights_path} 到 {hub_weights_path}")
                    import shutil
                    shutil.copy(weights_path, hub_weights_path)
                
                if os.path.exists(hub_weights_path):
                    print("使用本地权重文件计算FID和IS指标...")
                    # 使用torch-fidelity计算FID和IS
                    metrics_dict = fidelity_metrics(
                        input1=gt_dir,
                        input2=pred_dir,
                        cuda=False,  # 强制使用CPU
                        isc=True,
                        fid=True,
                        kid=False,  # KID计算较慢，先禁用
                        cache=True
                    )
                    
                    fid_value = metrics_dict.get('frechet_inception_distance')
                    is_value = metrics_dict.get('inception_score_mean')
                else:
                    print("权重文件不存在，跳过FID和IS计算")
            except Exception as e:
                print(f"FID和IS计算失败，跳过: {e}")
        else:
            print("离线模式：跳过FID和IS计算（需要网络下载权重）")
        
        # 计算CS (Cosine Similarity)
        print("使用CPU计算CS指标...")
        cs_value = calculate_cosine_similarity(gt_dir, pred_dir, force_cpu=True)
        
        return fid_value, is_value, cs_value
    except Exception as e:
        print(f"计算FID、IS和CS时出错: {e}")
        return None, None, None
        
def calculate_cosine_similarity(gt_dir, pred_dir, force_cpu=False):
    """计算图像特征的余弦相似度 (CS)"""
    try:
        # 设置权重路径，避免网络下载
        os.environ['TORCH_HOME'] = '/mnt/vdb1/lyt/ArbiViewGen-main/checkpoints'
        
        # 检查ResNet权重文件是否已存在
        resnet_weights_path = '/mnt/vdb1/lyt/ArbiViewGen-main/checkpoints/hub/checkpoints/resnet50-11ad3fa6.pth'
        
        # 如果目标目录不存在，创建它
        os.makedirs(os.path.dirname(resnet_weights_path), exist_ok=True)
        
        # 使用预训练的ResNet提取特征
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        model = torch.nn.Sequential(*list(model.children())[:-1])  # 移除最后的全连接层
        
        device = torch.device('cpu' if force_cpu else ('cuda' if torch.cuda.is_available() else 'cpu'))
        model = model.to(device)
        model.eval()
        
        # 图像预处理
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 获取所有图像文件
        gt_files = sorted(glob.glob(os.path.join(gt_dir, "*.png")))
        pred_files = sorted(glob.glob(os.path.join(pred_dir, "*.png")))
        
        if len(gt_files) == 0 or len(pred_files) == 0:
            print(f"警告: 目录中没有找到PNG图像: {gt_dir} or {pred_dir}")
            return None
            
        # 确保文件数量一致
        min_files = min(len(gt_files), len(pred_files))
        gt_files = gt_files[:min_files]
        pred_files = pred_files[:min_files]
        
        cs_values = []
        
        # 批量处理以提高效率
        batch_size = 10
        for i in range(0, min_files, batch_size):
            batch_gt_files = gt_files[i:i+batch_size]
            batch_pred_files = pred_files[i:i+batch_size]
            
            gt_features = []
            pred_features = []
            
            for gt_file, pred_file in zip(batch_gt_files, batch_pred_files):
                # 提取GT图像特征
                img = Image.open(gt_file).convert('RGB')
                img_tensor = preprocess(img).unsqueeze(0)
                img_tensor = img_tensor.to(device)
                with torch.no_grad():
                    gt_feature = model(img_tensor).squeeze()
                
                # 提取预测图像特征
                img = Image.open(pred_file).convert('RGB')
                img_tensor = preprocess(img).unsqueeze(0)
                img_tensor = img_tensor.to(device)
                with torch.no_grad():
                    pred_feature = model(img_tensor).squeeze()
                
                # 计算余弦相似度
                gt_feature = F.normalize(gt_feature, p=2, dim=0)
                pred_feature = F.normalize(pred_feature, p=2, dim=0)
                cs = torch.dot(gt_feature, pred_feature).item()
                cs_values.append(cs)
        
        # 返回平均余弦相似度
        return np.mean(cs_values)
        
    except Exception as e:
        print(f"计算CS时出错: {e}")
        return None

class FeatureExtractor:
    """特征提取器，用于计算多视角立体一致性"""
    def __init__(self, device, force_cpu=False):
        self.device = torch.device('cpu') if force_cpu else device
        
        # 设置权重路径，避免网络下载
        os.environ['TORCH_HOME'] = '/mnt/vdb1/lyt/ArbiViewGen-main/checkpoints'
        
        # 检查ResNet权重文件是否已存在
        resnet_weights_path = '/mnt/vdb1/lyt/ArbiViewGen-main/checkpoints/hub/checkpoints/resnet50-11ad3fa6.pth'
        
        # 如果目标目录不存在，创建它
        os.makedirs(os.path.dirname(resnet_weights_path), exist_ok=True)
        
        # 使用预训练的ResNet模型提取特征
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.model = torch.nn.Sequential(*list(self.model.children())[:-2])  # 移除最后的全连接层
        self.model.to(self.device).eval()
        
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def extract_features(self, image_path):
        """提取图像特征"""
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                features = self.model(image_tensor)
            
            # 归一化特征以计算余弦相似度
            features = F.normalize(features, p=2, dim=1)
            return features
        except Exception as e:
            print(f"提取特征失败 {image_path}: {e}")
            return None

def calculate_mvsc(left_img_path, middle_img_path, right_img_path, feature_extractor):
    """计算多视角立体一致性 (MVSC)
    
    使用预训练网络提取特征，然后计算:
    1. 中间视角与左侧视角的特征相似度
    2. 中间视角与右侧视角的特征相似度
    3. 计算这两个相似度的平均值作为MVSC指标
    """
    try:
        # 提取特征
        left_features = feature_extractor.extract_features(left_img_path)
        middle_features = feature_extractor.extract_features(middle_img_path)
        right_features = feature_extractor.extract_features(right_img_path)
        
        if left_features is None or middle_features is None or right_features is None:
            return None
            
        # 计算余弦相似度
        left_middle_sim = F.cosine_similarity(middle_features, left_features).mean().item()
        right_middle_sim = F.cosine_similarity(middle_features, right_features).mean().item()
        
        # MVSC是两个相似度的平均值
        mvsc = (left_middle_sim + right_middle_sim) / 2.0
        return mvsc
    except Exception as e:
        print(f"计算MVSC时出错: {e}")
        return None

def find_image_pairs(output_dir):
    """找到后缀为_0_gt.png和_0_pred.png以及部分_1_gt.png和_1_pred.png的图片对"""
    pairs = []
    
    # 找到视角0的图片对
    gt_files_0 = glob.glob(os.path.join(output_dir, "**/*_0_gt.png"), recursive=True)
    for gt_file in gt_files_0:
        pred_file = gt_file.replace("_0_gt.png", "_0_pred.png")
        if os.path.exists(pred_file):
            pairs.append((gt_file, pred_file))
    
    # 找到视角1的图片对，但只取20%
    gt_files_1 = glob.glob(os.path.join(output_dir, "**/*_1_gt.png"), recursive=True)
    import random
    random.seed(42)  # 固定随机种子，确保结果可重现
    selected_gt_files_1 = random.sample(gt_files_1, min(len(gt_files_1), len(gt_files_1) // 5))
    
    for gt_file in selected_gt_files_1:
        pred_file = gt_file.replace("_1_gt.png", "_1_pred.png")
        if os.path.exists(pred_file):
            pairs.append((gt_file, pred_file))
    
    return pairs

def find_multiview_triplets(output_dir):
    """查找多视角三元组：左侧视角、中间视角GT、中间视角预测、右侧视角"""
    # 修改正则表达式以匹配文件命名格式 0_0_0_gt.png
    pattern = re.compile(r'(\d+)_(\d+)_(\d+)_(gt|pred)\.png')
    all_files = glob.glob(os.path.join(output_dir, "**/*.png"), recursive=True)
    
    # 按批次和图像编号分组
    grouped_files = {}
    for file_path in all_files:
        basename = os.path.basename(file_path)
        match = pattern.match(basename)
        if match:
            batch_id, image_id, view_id, img_type = match.groups()
            key = f"{batch_id}_{image_id}"
            if key not in grouped_files:
                grouped_files[key] = {}
            view_key = f"{view_id}_{img_type}"
            grouped_files[key][view_key] = file_path
    
    # 查找完整的多视角三元组
    triplets = []
    for key, views in grouped_files.items():
        # 检查是否有完整的三元组：中间视角(1)的GT和预测，以及左(0)和右(2)视角
        # 注意：根据目录结构，视角编号可能是0(左)、1(中)、2(右)
        if '1_gt' in views and '1_pred' in views and '0_gt' in views and '2_gt' in views:
            triplet = {
                'left': views['0_gt'],         # 左侧视角
                'middle_gt': views['1_gt'],    # 中间视角 GT
                'middle_pred': views['1_pred'],# 中间视角 预测
                'right': views['2_gt']         # 右侧视角
            }
            triplets.append(triplet)
        # 或者检查是否有另一种排列：中间视角(0)的GT和预测，以及左(2)和右(1)视角
        elif '0_gt' in views and '0_pred' in views and '2_gt' in views and '1_gt' in views:
            triplet = {
                'left': views['2_gt'],         # 左侧视角
                'middle_gt': views['0_gt'],    # 中间视角 GT
                'middle_pred': views['0_pred'],# 中间视角 预测
                'right': views['1_gt']         # 右侧视角
            }
            triplets.append(triplet)
    
    print(f"找到多视角三元组的详细信息:")
    for i, triplet in enumerate(triplets[:5]):  # 只打印前5个作为示例
        print(f"  三元组 {i+1}:")
        print(f"    左侧: {os.path.basename(triplet['left'])}")
        print(f"    中间GT: {os.path.basename(triplet['middle_gt'])}")
        print(f"    中间预测: {os.path.basename(triplet['middle_pred'])}")
        print(f"    右侧: {os.path.basename(triplet['right'])}")
    
    return triplets

def main():
    parser = argparse.ArgumentParser(description="计算生成图片的评估指标")
    parser.add_argument("--output_dir", type=str, default="/mnt/vdc1/lyt/logs/tb_logs/test_pano_outpaint_dual_gpu_conservative/version_0/images/",
                       help="输出目录路径")
    parser.add_argument("--version", type=str, default=None,
                       help="指定版本目录，如果为None则处理所有版本")
    parser.add_argument("--compute_fid", action="store_true", default=True,
                       help="是否计算FID和IS指标")
    parser.add_argument("--compute_mvsc", action="store_true", 
                       help="是否计算多视角立体一致性指标")
    parser.add_argument("--cpu", action="store_true",
                       help="强制使用CPU进行计算，避免CUDA内存不足")
    parser.add_argument("--batch_size", type=int, default=10,
                       help="每批处理的图像数量，降低可减少内存使用")
    parser.add_argument("--reference_psnr", type=float, default=30.0,
                       help="PSNR参考值，用于计算比率")
    parser.add_argument("--reference_ssim", type=float, default=0.90,
                       help="SSIM参考值，用于计算比率")
    parser.add_argument("--reference_lpips", type=float, default=0.05,
                       help="LPIPS参考值，用于计算比率")
    parser.add_argument("--offline", action="store_true",
                       help="离线模式，跳过需要网络下载的指标（如FID/IS）")
    
    args = parser.parse_args()
    
    # 初始化设备
    device = torch.device('cpu' if args.cpu else ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"使用设备: {device}")
    
    # 只计算FID，不需要初始化LPIPS
    lpips_fn = None
    
    # 不需要计算多视角立体一致性
    feature_extractor = None
    
    # 确定要处理的目录
    if args.version:
        target_dirs = [os.path.join(args.output_dir, args.version)]
    else:
        # 直接使用指定的目录
        target_dirs = [args.output_dir]
    
    print(f"处理目录: {args.output_dir}")
    
    all_metrics = defaultdict(list)
    
    for target_dir in target_dirs:
        print(f"\n处理目录: {target_dir}")
        
        # 找到所有图片对
        image_pairs = find_image_pairs(target_dir)
        print(f"找到 {len(image_pairs)} 对图片")
        
        if len(image_pairs) == 0:
            print(f"警告: 在 {target_dir} 中没有找到图片对")
            continue
            
        # 只处理FID、IS和CS
        # 创建临时目录存放GT和预测图像
        gt_dir = os.path.join(target_dir, 'tmp_gt_for_metrics')
        pred_dir = os.path.join(target_dir, 'tmp_pred_for_metrics')
        
        os.makedirs(gt_dir, exist_ok=True)
        os.makedirs(pred_dir, exist_ok=True)
        
        # 复制图像到临时目录
        print("准备FID、IS和CS计算的图像...")
        for i, (gt_path, pred_path) in enumerate(image_pairs):
            try:
                img = Image.open(gt_path)
                img.save(os.path.join(gt_dir, f"{i:05d}.png"))
                
                img = Image.open(pred_path)
                img.save(os.path.join(pred_dir, f"{i:05d}.png"))
            except Exception as e:
                print(f"复制图像时出错: {e}")
                
        # 计算FID、IS和CS
        print("计算FID、IS和CS指标...")
        fid_value, is_value, cs_value = calculate_fid_is_cs(gt_dir, pred_dir, offline_mode=args.offline)
        
        if fid_value is not None:
            all_metrics['fid'].append(fid_value)
            print(f"FID={fid_value:.4f}")
            
        if is_value is not None:
            all_metrics['is'].append(is_value)
            print(f"IS={is_value:.4f}")
            
        if cs_value is not None:
            all_metrics['cs'].append(cs_value)
            print(f"CS={cs_value:.4f}")
        
        # 清理临时目录
        import shutil
        shutil.rmtree(gt_dir, ignore_errors=True)
        shutil.rmtree(pred_dir, ignore_errors=True)
    
    # 计算总体统计
    if all_metrics['fid']:
        print(f"\n=== 总体统计 ===")
        
        if all_metrics['fid']:
            avg_fid = np.mean(all_metrics['fid'])
            print(f"FID: \t {avg_fid:.4f}")
            
        if all_metrics['is']:
            avg_is = np.mean(all_metrics['is'])
            print(f"IS: \t {avg_is:.4f}")
            
        if all_metrics['cs']:
            avg_cs = np.mean(all_metrics['cs'])
            # CS是越高越好，范围在0-1之间，1表示完全相似
            reference_cs = 0.85  # 设定一个较高的CS参考值
            ratio_cs = avg_cs / reference_cs if reference_cs > 0 else 0
            print(f"CS: \t {avg_cs:.4f} \t {ratio_cs:.4f}")
        
        # 保存详细结果
        results_file = os.path.join(args.output_dir, "metrics_results.txt")
        with open(results_file, 'w') as f:
            f.write("=== 详细指标结果 ===\n")
            
            if all_metrics['fid']:
                f.write(f"FID: \t {avg_fid:.4f}\n")
            
            if all_metrics['is']:
                f.write(f"IS: \t {avg_is:.4f}\n")
                
            if all_metrics['cs']:
                f.write(f"CS: \t {avg_cs:.4f} \t {ratio_cs:.4f}\n\n")
        
        print(f"\n详细结果已保存到: {results_file}")
    else:
        print("没有找到有效的图片对")

if __name__ == "__main__":
    main() 