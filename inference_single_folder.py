import os
import torch
import argparse
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from src.lightning_depth import LidarDiffusionModule

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, required=True)
parser.add_argument('--input_dir', type=str, required=True, help='proj文件夹路径')
parser.add_argument('--output_dir', type=str, default='./single_inference')
parser.add_argument('--num_steps', type=int, default=20)
args = parser.parse_args()

# 加载模型
print(f"加载模型: {args.checkpoint}")
model = LidarDiffusionModule.load_from_checkpoint(args.checkpoint)
model.eval().cuda()

# 图像变换
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])

# 创建输出目录
os.makedirs(args.output_dir, exist_ok=True)
os.makedirs(f"{args.output_dir}/generated", exist_ok=True)
os.makedirs(f"{args.output_dir}/comparison", exist_ok=True)

# 处理每个图像（除了combined.jpg）
view_names = ['FL.jpg', 'FN.jpg', 'FR.jpg', 'FW.jpg', 'RL.jpg', 'RN.jpg', 'RR.jpg']

for view_name in view_names:
    input_path = os.path.join(args.input_dir, view_name)
    if not os.path.exists(input_path):
        print(f"跳过 {view_name}: 文件不存在")
        continue
    
    # 加载并处理图像
    proj_img = Image.open(input_path).convert("RGB")
    proj_tensor = transform(proj_img).unsqueeze(0).cuda()
    
    # 生成
    print(f"处理 {view_name}...")
    with torch.no_grad():
        generated = model.generate_from_lidar(proj_tensor, num_inference_steps=args.num_steps)
        generated = torch.clamp(generated, 0, 1)
    
    # 保存结果
    output_path = os.path.join(args.output_dir, "generated", view_name)
    save_image(generated, output_path)
    
    # 创建对比图
    comparison = torch.cat([proj_tensor, generated], dim=3)
    comparison_path = os.path.join(args.output_dir, "comparison", view_name)
    save_image(comparison, comparison_path)
    
    print(f"  ✓ 保存到: {output_path}")

print(f"\n完成！结果保存在: {args.output_dir}")
