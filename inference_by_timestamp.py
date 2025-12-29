"""
交互式推理脚本
- 每个时间戳生成一个文件夹
- 每个文件夹包含 7 个视角的对比图
- 交互式选择参数，按 Enter 使用默认值
"""
import os
import torch
from collections import defaultdict
from PIL import Image
import torchvision.transforms as T
from torchvision.utils import save_image
from src.lightning_depth import LidarDiffusionModule


def prompt_input(prompt, default):
    """交互式输入，直接按 Enter 使用默认值"""
    user_input = input(f"{prompt} [{default}]: ").strip()
    return user_input if user_input else default


def load_image(path, image_size=(512, 512)):
    transform = T.Compose([
        T.Resize(image_size),
        T.ToTensor(),
    ])
    img = Image.open(path).convert("RGB")
    return transform(img)


def main():
    print("=" * 50)
    print("  交互式推理脚本")
    print("  按 Enter 使用默认值")
    print("=" * 50)
    print()

    # 默认值
    default_ckpt = "./experiments/p2g_20251226_155015/checkpoints/last.ckpt"
    default_data_root = "/mnt/zihanw/proj_utils_pro/blur投影/044/044"
    default_output_dir = "./inference_output"
    default_num_timestamps = "5"
    default_num_steps = "50"
    default_from_middle = "y"

    # 交互式输入
    ckpt_path = prompt_input("Checkpoint 路径", default_ckpt)
    data_root = prompt_input("数据文件夹路径", default_data_root)
    output_dir = prompt_input("输出目录", default_output_dir)
    num_timestamps_str = prompt_input("处理时间戳数量 (-1 表示全部)", default_num_timestamps)
    num_steps_str = prompt_input("推理步数", default_num_steps)
    from_middle_str = prompt_input("从中间开始选择 (y/n)", default_from_middle)

    # 转换参数
    num_timestamps = int(num_timestamps_str)
    num_steps = int(num_steps_str)
    from_middle = from_middle_str.lower() in ['y', 'yes', '1', 'true']

    print()
    print("=" * 50)
    print("  配置确认")
    print("=" * 50)
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  数据路径: {data_root}")
    print(f"  输出目录: {output_dir}")
    print(f"  时间戳数量: {num_timestamps}")
    print(f"  推理步数: {num_steps}")
    print(f"  从中间开始: {from_middle}")
    print("=" * 50)
    print()

    confirm = input("确认开始推理? (Enter 继续 / n 取消): ").strip().lower()
    if confirm == 'n':
        print("已取消")
        return

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 加载模型
    print(f"\n正在加载模型: {ckpt_path}")
    model = LidarDiffusionModule.load_from_checkpoint(ckpt_path)
    model.eval()
    model.cuda()
    print("模型加载完成!")

    # 收集所有时间戳和对应的图像
    image_names = ['FL.jpg', 'FN.jpg', 'FR.jpg', 'FW.jpg', 'RL.jpg', 'RN.jpg', 'RR.jpg']
    timestamps_data = defaultdict(dict)

    print(f"\n正在扫描数据: {data_root}")
    for timestamp_dir in sorted(os.listdir(data_root)):
        timestamp_path = os.path.join(data_root, timestamp_dir)
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
    print(f"找到 {total_timestamps} 个时间戳")

    # 选择要处理的时间戳
    if num_timestamps == -1:
        selected_timestamps = all_timestamps
    else:
        if from_middle:
            mid = total_timestamps // 2
            half = num_timestamps // 2
            start = max(0, mid - half)
            end = min(total_timestamps, start + num_timestamps)
            selected_timestamps = all_timestamps[start:end]
        else:
            selected_timestamps = all_timestamps[:num_timestamps]

    print(f"将处理 {len(selected_timestamps)} 个时间戳")
    print(f"时间戳列表: {selected_timestamps[:3]}...{selected_timestamps[-1] if len(selected_timestamps) > 3 else ''}")

    # 推理
    for i, timestamp in enumerate(selected_timestamps):
        print(f"\n[{i+1}/{len(selected_timestamps)}] 正在处理: {timestamp}")

        timestamp_output_dir = os.path.join(output_dir, timestamp)
        os.makedirs(timestamp_output_dir, exist_ok=True)

        views = timestamps_data[timestamp]
        for view, paths in views.items():
            gt_tensor = load_image(paths['gt_path']).unsqueeze(0).cuda()
            proj_tensor = load_image(paths['proj_path']).unsqueeze(0).cuda()

            with torch.no_grad():
                generated = model.generate_from_lidar(proj_tensor, num_inference_steps=num_steps)
                generated = torch.clamp(generated, 0, 1)

            # 保存对比图: proj | generated | gt
            comparison = torch.cat([proj_tensor.cpu(), generated.cpu(), gt_tensor.cpu()], dim=3)
            output_path = os.path.join(timestamp_output_dir, f"{view}_comparison.jpg")
            save_image(comparison, output_path)

            # 单独保存生成的图像
            gen_path = os.path.join(timestamp_output_dir, f"{view}_generated.jpg")
            save_image(generated.cpu(), gen_path)

            print(f"  {view} ✓")

    print(f"\n推理完成! 结果保存在: {output_dir}")
    print(f"共处理 {len(selected_timestamps)} 个时间戳, {len(selected_timestamps) * 7} 张图像")


if __name__ == "__main__":
    main()
