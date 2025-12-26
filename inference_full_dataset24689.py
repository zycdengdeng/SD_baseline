import os
import torch
import yaml
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm
import argparse

from src.lightning_depth import LidarDiffusionModule

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--data_roots', type=str, nargs='+', required=True, 
                        help='多个数据根目录，例如：002 004 006 008 009')
    parser.add_argument('--output_root', type=str, default='./inference_results')
    parser.add_argument('--num_steps', type=int, default=20, help='Number of diffusion steps')
    parser.add_argument('--save_original', action='store_true', help='是否保存原始proj和GT图像')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 加载模型
    print(f"加载模型: {args.checkpoint}")
    model = LidarDiffusionModule.load_from_checkpoint(args.checkpoint)
    model.eval()
    model.cuda()
    
    # 图像变换
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    
    # 视角列表
    view_names = ['FL.jpg', 'FN.jpg', 'FR.jpg', 'FW.jpg', 'RL.jpg', 'RN.jpg', 'RR.jpg']
    
    # 创建输出根目录
    os.makedirs(args.output_root, exist_ok=True)
    
    # 全局统计
    total_images = 0
    total_timestamps = 0
    failed_images = 0
    
    # 遍历所有数据根目录
    for data_root in args.data_roots:
        print(f"\n{'='*70}")
        print(f"处理数据目录: {data_root}")
        print(f"{'='*70}")
        
        if not os.path.exists(data_root):
            print(f"⚠️  目录不存在，跳过: {data_root}")
            continue
        
        # 获取数据根目录名称（例如 "002"）
        root_name = os.path.basename(data_root)
        
        # 获取所有时间戳目录
        timestamp_dirs = sorted([d for d in os.listdir(data_root) 
                               if os.path.isdir(os.path.join(data_root, d))])
        
        print(f"找到 {len(timestamp_dirs)} 个时间戳目录")
        
        # 遍历所有时间戳
        for timestamp_dir in tqdm(timestamp_dirs, desc=f"处理 {root_name}"):
            timestamp_path = os.path.join(data_root, timestamp_dir)
            proj_dir = os.path.join(timestamp_path, 'proj')
            gt_dir = os.path.join(timestamp_path, 'GT')
            
            # 检查proj和GT目录是否存在
            if not (os.path.exists(proj_dir) and os.path.exists(gt_dir)):
                continue
            
            total_timestamps += 1
            
            # 创建输出目录，按照 output_root/root_name/timestamp/ 组织
            output_timestamp_dir = os.path.join(args.output_root, root_name, timestamp_dir)
            output_generated_dir = os.path.join(output_timestamp_dir, 'generated')
            output_comparison_dir = os.path.join(output_timestamp_dir, 'comparison')
            
            os.makedirs(output_generated_dir, exist_ok=True)
            os.makedirs(output_comparison_dir, exist_ok=True)
            
            # 可选：保存原始图像
            if args.save_original:
                output_proj_dir = os.path.join(output_timestamp_dir, 'proj')
                output_gt_dir = os.path.join(output_timestamp_dir, 'GT')
                os.makedirs(output_proj_dir, exist_ok=True)
                os.makedirs(output_gt_dir, exist_ok=True)
            
            # 处理每个视角
            for view_name in view_names:
                proj_path = os.path.join(proj_dir, view_name)
                gt_path = os.path.join(gt_dir, view_name)
                
                if not (os.path.exists(proj_path) and os.path.exists(gt_path)):
                    continue
                
                try:
                    # 加载图像
                    proj_img = Image.open(proj_path).convert("RGB")
                    gt_img = Image.open(gt_path).convert("RGB")
                    
                    # 转换为tensor
                    proj_tensor = transform(proj_img).unsqueeze(0).cuda()
                    gt_tensor = transform(gt_img).unsqueeze(0).cuda()
                    
                    # 生成图像
                    with torch.no_grad():
                        generated = model.generate_from_lidar(proj_tensor, num_inference_steps=args.num_steps)
                        generated = torch.clamp(generated, 0, 1)
                    
                    # 保存生成的图像
                    output_path = os.path.join(output_generated_dir, view_name)
                    save_image(generated, output_path)
                    
                    # 可选：复制原始图像
                    if args.save_original:
                        proj_img.save(os.path.join(output_proj_dir, view_name))
                        gt_img.save(os.path.join(output_gt_dir, view_name))
                    
                    # 创建对比图（proj | generated | GT）
                    comparison = torch.cat([proj_tensor, generated, gt_tensor], dim=3)
                    comparison_path = os.path.join(output_comparison_dir, view_name)
                    save_image(comparison, comparison_path)
                    
                    total_images += 1
                    
                except Exception as e:
                    print(f"\n❌ 处理失败: {proj_path}")
                    print(f"   错误: {e}")
                    failed_images += 1
                    continue
    
    # 打印最终统计信息
    print(f"\n{'='*70}")
    print(f"推理完成！")
    print(f"{'='*70}")
    print(f"✅ 处理数据目录数: {len(args.data_roots)}")
    print(f"✅ 处理时间戳总数: {total_timestamps}")
    print(f"✅ 成功生成图像数: {total_images}")
    print(f"⚠️  失败图像数: {failed_images}")
    print(f"📁 输出目录: {args.output_root}")
    print(f"\n目录结构:")
    print(f"{args.output_root}/")
    print(f"  ├── 002/")
    print(f"  │   └── [时间戳]/")
    print(f"  │       ├── generated/   (生成的图像)")
    print(f"  │       ├── comparison/  (对比图: proj|generated|GT)")
    if args.save_original:
        print(f"  │       ├── proj/        (原始proj图像)")
        print(f"  │       └── GT/          (原始GT图像)")
    print(f"  ├── 004/")
    print(f"  ├── 006/")
    print(f"  ├── 008/")
    print(f"  └── 009/")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()