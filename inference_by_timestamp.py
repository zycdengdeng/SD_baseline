"""
按时间戳推理脚本
- 每个时间戳生成一个文件夹
- 每个文件夹包含 7 个视角的对比图
- 可选择推理数量
"""
import os
import torch
import argparse
from collections import defaultdict
from PIL import Image
import torchvision.transforms as T
from torchvision.utils import save_image
from src.lightning_depth import LidarDiffusionModule

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True, help='Checkpoint path')
    parser.add_argument('--data_root', type=str, required=True, help='Data folder path')
    parser.add_argument('--output_dir', type=str, default='./inference_output', help='Output directory')
    parser.add_argument('--num_timestamps', type=int, default=5, help='Number of timestamps to process (use -1 for all)')
    parser.add_argument('--num_steps', type=int, default=50, help='Number of inference steps')
    parser.add_argument('--from_middle', action='store_true', help='Start from middle timestamps')
    return parser.parse_args()

def load_image(path, image_size=(512, 512)):
    transform = T.Compose([
        T.Resize(image_size),
        T.ToTensor(),
    ])
    img = Image.open(path).convert("RGB")
    return transform(img)

def main():
    args = parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载模型
    print(f"Loading checkpoint: {args.ckpt}")
    model = LidarDiffusionModule.load_from_checkpoint(args.ckpt)
    model.eval()
    model.cuda()
    print("Model loaded!")

    # 收集所有时间戳和对应的图像
    image_names = ['FL.jpg', 'FN.jpg', 'FR.jpg', 'FW.jpg', 'RL.jpg', 'RN.jpg', 'RR.jpg']
    timestamps_data = defaultdict(dict)

    print(f"Scanning data folder: {args.data_root}")
    for timestamp_dir in sorted(os.listdir(args.data_root)):
        timestamp_path = os.path.join(args.data_root, timestamp_dir)
        if not os.path.isdir(timestamp_path):
            continue

        # 支持 gt/GT
        gt_dir = os.path.join(timestamp_path, 'gt')
        if not os.path.exists(gt_dir):
            gt_dir = os.path.join(timestamp_path, 'GT')
        proj_dir = os.path.join(timestamp_path, 'proj')

        if os.path.exists(gt_dir) and os.path.exists(proj_dir):
            for img_name in image_names:
                gt_path = os.path.join(gt_dir, img_name)
                proj_path = os.path.join(proj_dir, img_name)
                if os.path.exists(gt_path) and os.path.exists(proj_path):
                    view = img_name.split('.')[0]
                    timestamps_data[timestamp_dir][view] = {
                        'gt_path': gt_path,
                        'proj_path': proj_path
                    }

    # 获取时间戳列表
    all_timestamps = sorted(timestamps_data.keys())
    total_timestamps = len(all_timestamps)
    print(f"Found {total_timestamps} timestamps")

    # 选择要处理的时间戳
    if args.num_timestamps == -1:
        selected_timestamps = all_timestamps
    else:
        if args.from_middle:
            # 从中间开始选择
            mid = total_timestamps // 2
            half = args.num_timestamps // 2
            start = max(0, mid - half)
            end = min(total_timestamps, start + args.num_timestamps)
            selected_timestamps = all_timestamps[start:end]
        else:
            selected_timestamps = all_timestamps[:args.num_timestamps]

    print(f"Processing {len(selected_timestamps)} timestamps")
    print(f"Timestamps: {selected_timestamps}")

    # 推理
    for i, timestamp in enumerate(selected_timestamps):
        print(f"\n[{i+1}/{len(selected_timestamps)}] Processing timestamp: {timestamp}")

        timestamp_output_dir = os.path.join(args.output_dir, timestamp)
        os.makedirs(timestamp_output_dir, exist_ok=True)

        views = timestamps_data[timestamp]
        for view, paths in views.items():
            gt_tensor = load_image(paths['gt_path']).unsqueeze(0).cuda()
            proj_tensor = load_image(paths['proj_path']).unsqueeze(0).cuda()

            with torch.no_grad():
                generated = model.generate_from_lidar(proj_tensor, num_inference_steps=args.num_steps)
                generated = torch.clamp(generated, 0, 1)

            # 保存对比图: proj | generated | gt
            comparison = torch.cat([proj_tensor.cpu(), generated.cpu(), gt_tensor.cpu()], dim=3)
            output_path = os.path.join(timestamp_output_dir, f"{view}_comparison.jpg")
            save_image(comparison, output_path)

            # 也单独保存生成的图像
            gen_path = os.path.join(timestamp_output_dir, f"{view}_generated.jpg")
            save_image(generated.cpu(), gen_path)

            print(f"  Saved {view}")

    print(f"\nDone! Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
