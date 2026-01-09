"""
将评测结果的图片按视角拼接成视频
"""

import os
import cv2
import numpy as np
from collections import defaultdict


def prompt_input(prompt, default):
    """交互式输入"""
    user_input = input(f"{prompt} [{default}]: ").strip()
    return user_input if user_input else default


def main():
    print("=" * 60)
    print("  图片转视频工具")
    print("  将评测结果按视角拼接成视频")
    print("=" * 60)
    print()

    # 默认值
    default_images_dir = "/mnt/zihanw/SD_baseline/evaRes/eval_20260109_133803/images"
    default_output_dir = "/mnt/zihanw/SD_baseline/evaRes/eval_20260109_133803/videos"
    default_fps = "10"
    default_view = "all"

    # 交互式输入
    images_dir = prompt_input("图片目录", default_images_dir)
    output_dir = prompt_input("视频输出目录", default_output_dir)
    fps = int(prompt_input("视频帧率 (fps)", default_fps))
    view_choice = prompt_input("选择视角 (FL/FN/FR/FW/RL/RN/RR/all)", default_view)

    # 扫描图片
    print(f"\n正在扫描: {images_dir}")

    if not os.path.exists(images_dir):
        print(f"错误: 目录不存在 {images_dir}")
        return

    # 按视角分组图片
    views = ['FL', 'FN', 'FR', 'FW', 'RL', 'RN', 'RR']
    view_images = defaultdict(list)

    for filename in sorted(os.listdir(images_dir)):
        if not filename.endswith('.jpg') and not filename.endswith('.png'):
            continue

        # 解析文件名: timestamp_VIEW.jpg
        for view in views:
            if f"_{view}." in filename:
                filepath = os.path.join(images_dir, filename)
                view_images[view].append(filepath)
                break

    # 统计
    print(f"\n找到图片:")
    for view in views:
        count = len(view_images[view])
        if count > 0:
            print(f"  {view}: {count} 张")

    total_images = sum(len(v) for v in view_images.values())
    if total_images == 0:
        print("错误: 没有找到图片")
        return

    # 确定要处理的视角
    if view_choice.lower() == 'all':
        views_to_process = [v for v in views if len(view_images[v]) > 0]
    else:
        views_to_process = [view_choice.upper()]

    print(f"\n将生成 {len(views_to_process)} 个视频: {views_to_process}")

    confirm = input("\n确认开始? (Enter 继续 / n 取消): ").strip().lower()
    if confirm == 'n':
        print("已取消")
        return

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 生成视频
    for view in views_to_process:
        images = view_images[view]
        if len(images) == 0:
            print(f"跳过 {view}: 没有图片")
            continue

        print(f"\n生成 {view} 视频 ({len(images)} 帧)...")

        # 读取第一张图片获取尺寸
        first_img = cv2.imread(images[0])
        if first_img is None:
            print(f"  错误: 无法读取 {images[0]}")
            continue

        height, width = first_img.shape[:2]
        print(f"  尺寸: {width}x{height}")

        # 创建视频写入器
        video_path = os.path.join(output_dir, f"video_{view}.mp4")

        # 尝试不同编码器
        codecs = [
            ('mp4v', '.mp4'),
            ('XVID', '.avi'),
            ('MJPG', '.avi'),
        ]

        success = False
        for codec, ext in codecs:
            try:
                test_path = os.path.join(output_dir, f"video_{view}{ext}")
                fourcc = cv2.VideoWriter_fourcc(*codec)
                out = cv2.VideoWriter(test_path, fourcc, fps, (width, height))

                if not out.isOpened():
                    continue

                for img_path in images:
                    frame = cv2.imread(img_path)
                    if frame is not None:
                        # 确保尺寸匹配
                        if frame.shape[:2] != (height, width):
                            frame = cv2.resize(frame, (width, height))
                        out.write(frame)

                out.release()

                # 验证文件
                if os.path.exists(test_path) and os.path.getsize(test_path) > 0:
                    print(f"  保存: {test_path}")
                    success = True
                    break
            except Exception as e:
                print(f"  编码器 {codec} 失败: {e}")

        if not success:
            print(f"  错误: 无法生成 {view} 视频")

    print(f"\n完成! 视频保存在: {output_dir}")


if __name__ == "__main__":
    main()
