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
    parser.add_argument('--data_root', type=str, default='/mnt/zihanw/车路协同投影工作/batch_projection/乌鸡变版本/output_Result')
    parser.add_argument('--output_root', type=str, default='./inference_results')
    parser.add_argument('--num_steps', type=int, default=20, help='Number of diffusion steps')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for inference')
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
    
    # 统计
    total_images = 0
    processed_timestamps = []
    
    # 遍历所有时间戳
    for timestamp_dir in sorted(os.listdir(args.data_root)):
        timestamp_path = os.path.join(args.data_root, timestamp_dir)
        
        if not os.path.isdir(timestamp_path):
            continue
            
        proj_dir = os.path.join(timestamp_path, 'proj')
        gt_dir = os.path.join(timestamp_path, 'GT')
        
        if not (os.path.exists(proj_dir) and os.path.exists(gt_dir)):
            continue
        
        print(f"\n处理时间戳: {timestamp_dir}")
        processed_timestamps.append(timestamp_dir)
        
        # 创建输出目录，保持原始结构
        output_timestamp_dir = os.path.join(args.output_root, timestamp_dir)
        output_generated_dir = os.path.join(output_timestamp_dir, 'generated')
        output_comparison_dir = os.path.join(output_timestamp_dir, 'comparison')
        
        os.makedirs(output_generated_dir, exist_ok=True)
        os.makedirs(output_comparison_dir, exist_ok=True)
        
        # 复制原始proj和GT到输出目录（可选）
        output_proj_dir = os.path.join(output_timestamp_dir, 'proj')
        output_gt_dir = os.path.join(output_timestamp_dir, 'GT')
        os.makedirs(output_proj_dir, exist_ok=True)
        os.makedirs(output_gt_dir, exist_ok=True)
        
        # 处理每个视角
        for view_name in view_names:
            proj_path = os.path.join(proj_dir, view_name)
            gt_path = os.path.join(gt_dir, view_name)
            
            if not (os.path.exists(proj_path) and os.path.exists(gt_path)):
                print(f"  跳过 {view_name}: 文件不存在")
                continue
            
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
            
            # 复制原始图像
            proj_img.save(os.path.join(output_proj_dir, view_name))
            gt_img.save(os.path.join(output_gt_dir, view_name))
            
            # 创建对比图（proj | generated | GT）
            comparison = torch.cat([proj_tensor, generated, gt_tensor], dim=3)
            comparison_path = os.path.join(output_comparison_dir, view_name)
            save_image(comparison, comparison_path)
            
            total_images += 1
            print(f"  处理完成: {view_name}")
    
    # 打印统计信息
    print(f"\n========== 推理完成 ==========")
    print(f"处理时间戳数: {len(processed_timestamps)}")
    print(f"处理图像总数: {total_images}")
    print(f"输出目录: {args.output_root}")
    print(f"\n目录结构:")
    print(f"{args.output_root}/")
    print(f"  └── [时间戳]/")
    print(f"      ├── proj/        (原始proj图像)")
    print(f"      ├── GT/          (原始GT图像)")
    print(f"      ├── generated/   (生成的图像)")
    print(f"      └── comparison/  (对比图)")

if __name__ == "__main__":
    main()
