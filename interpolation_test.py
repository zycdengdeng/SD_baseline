#!/usr/bin/env python3
"""
Camera interpolation test script
Generate 18 interpolated images per scene, save to individual folders
Images are named by angle
"""
import cv2
import numpy as np
import os
import sys
import random
import argparse
import json
import time
import gc
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial
from tqdm import tqdm

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("警告: psutil 未安装，系统监控功能将受限。建议运行: pip install psutil")

sys.path.append('src')
from dataset.utils import get_K_R

# 导入优化的拼接函数
from warp_img_final_optimized import warp_img_final_optimized, load_multiple_nuscenes_data

def save_progress(output_root, completed_scenes, total_scenes, current_batch=0):
    """保存进度信息到JSON文件"""
    progress_file = os.path.join(output_root, "progress.json")
    progress_data = {
        "completed_scenes": completed_scenes,
        "total_scenes": total_scenes,
        "current_batch": current_batch,
        "timestamp": time.time(),
        "completion_rate": len(completed_scenes) / total_scenes if total_scenes > 0 else 0
    }
    
    with open(progress_file, 'w', encoding='utf-8') as f:
        json.dump(progress_data, f, indent=2, ensure_ascii=False)

def load_progress(output_root):
    progress_file = os.path.join(output_root, "progress.json")
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"警告: 无法加载进度文件: {e}")
    return None

def load_analysis_result(output_root):
    """加载文件夹分析结果"""
    analysis_file = os.path.join(output_root, "folder_analysis_result.json")
    if os.path.exists(analysis_file):
        try:
            with open(analysis_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"警告: 无法加载分析结果文件: {e}")
    return None

def extract_timestamp_from_folder_name(folder_name):
    """从文件夹名称中提取时间戳"""
    # 格式：interpolated_XXXX_timestamp_1234567890123456
    import re
    match = re.search(r'timestamp_(\d+)', folder_name)
    if match:
        return match.group(1)
    return None

def find_missing_scene_data(missing_numbers, insufficient_folders, data_root, scenes_data):
    """找到缺失编号和不完整文件夹对应的原始场景数据"""
    repair_tasks = []
    
    # 创建时间戳到场景数据的映射
    timestamp_to_scene = {}
    for scene_data in scenes_data:
        image_paths, real_angles, scene_name = scene_data
        timestamp = extract_timestamp_from_folder_name(f"dummy_{scene_name}")
        if timestamp:
            timestamp_to_scene[timestamp] = scene_data
    
    print(f"创建了 {len(timestamp_to_scene)} 个时间戳映射")
    
    # 处理数量不足的文件夹
    print("\n=== 处理数量不足的文件夹 ===")
    for folder_info in insufficient_folders:
        folder_name = folder_info['folder_name']
        photo_count = folder_info['photo_count']
        
        # 提取编号和时间戳
        import re
        match = re.match(r'interpolated_(\d+)_timestamp_(\d+)', folder_name)
        if match:
            scene_idx = int(match.group(1)) - 1  # 转换为0索引
            timestamp = match.group(2)
            
            # 查找对应的原始场景数据
            if timestamp in timestamp_to_scene:
                scene_data = timestamp_to_scene[timestamp]
                repair_tasks.append({
                    'type': 'insufficient',
                    'scene_idx': scene_idx,
                    'scene_data': scene_data,
                    'folder_name': folder_name,
                    'current_count': photo_count,
                    'expected_count': 12
                })
                print(f"  找到修复任务: {folder_name} (当前{photo_count}张，需要12张)")
            else:
                print(f"  警告: 无法找到时间戳 {timestamp} 对应的原始数据")
    
    # 处理缺失编号 - 这些需要重新创建
    print("\n=== 处理缺失编号 ===")
    print(f"缺失编号数量: {len(missing_numbers)}")
    print(f"前10个缺失编号: {missing_numbers[:10]}")
    
    # 对于缺失编号，我们需要从原始数据中找到对应的场景
    # 但由于编号和原始数据的对应关系可能已经丢失，我们先跳过这部分
    # 专注于修复数量不足的文件夹
    
    return repair_tasks

def check_scene_completed(output_root, scene_idx, scene_name, num_interpolations):
    """检查场景是否已完成"""
    # 根据现有目录格式：interpolated_0001_timestamp_1543866038447444
    # scene_name 是类似 'timestamp_1543866038447444' 的格式
    scene_output_dir_name = f"interpolated_{scene_idx+1:04d}_{scene_name}"
    scene_output_dir = os.path.join(output_root, scene_output_dir_name)
    
    exists = os.path.exists(scene_output_dir)
    
    # 调试信息
    if scene_idx < 5:  # 只为前几个场景显示详细信息
        print(f"         检查目录: {scene_output_dir_name}")
        print(f"         目录存在: {exists}")
        if exists:
            print(f"  ✓ 场景 {scene_idx+1} 已完成（目录存在）")
        else:
            print(f"  ✗ 场景 {scene_idx+1} 未完成（目录不存在）")
    
    return exists

def load_scene_images(image_paths):
    """懒加载场景图像，统一尺寸为1920x1280"""
    images = []
    target_width, target_height = 1920, 1280
    
    for image_path in image_paths:
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 检查并统一尺寸
        current_height, current_width = img_rgb.shape[:2]
        if current_width != target_width or current_height != target_height:
            # 调整尺寸到1920x1280
            img_rgb = cv2.resize(img_rgb, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        
        images.append(img_rgb)
    return images

def process_single_scene(scene_data, scene_idx, output_root, interpolation_angles, vx, vy):
    """处理单个场景的插值图像生成"""
    image_paths, real_angles, scene_name = scene_data
    
    # 检查是否已完成
    if check_scene_completed(output_root, scene_idx, scene_name, len(interpolation_angles)):
        return f"Scene {scene_idx+1} already completed"
    
    # 懒加载图像
    real_images = load_scene_images(image_paths)
    
    # 创建时间戳场景文件夹
    scene_output_dir = os.path.join(output_root, f"interpolated_{scene_idx+1:04d}_{scene_name}")
    os.makedirs(scene_output_dir, exist_ok=True)
    
    results = []
    # 生成每个插值角度的图像
    for idx, target_angle in enumerate(interpolation_angles):
        # 生成插值图像
        result_img = warp_img_final_optimized(
            90, target_angle, 0, real_images, vx, vy, 
            angle_threshold=1.0, enable_object_alignment=True
        )
        
        # 格式化角度为文件名
        if target_angle >= 0:
            angle_str = f"pos_{target_angle:06.2f}"
        else:
            angle_str = f"neg_{abs(target_angle):06.2f}"
        
        # 保存图像
        filename = f"{angle_str}.jpg"
        filepath = os.path.join(scene_output_dir, filename)
        
        # 转换为BGR用于保存
        result_bgr = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filepath, result_bgr)
        
        results.append(f"Generated {filename}")
    
    # 不在这里清理内存，让Python的垃圾回收自动处理
    return f"Scene {scene_idx+1} completed with {len(results)} images"

def calculate_interpolation_angles():
    """
    根据新的相机配置计算插值角度（HFOV=25.2°）
    只在中间区域插值，左右两侧相机不向外插值
    """
    # 新的相机配置（按圆周顺序排列）
    camera_angles = [-90, -45, 0, 45, 90]  # 右侧、右前、前、左前、左侧
    camera_names = ['CAM_SIDE_RIGHT', 'CAM_FRONT_RIGHT', 'CAM_FRONT', 
                   'CAM_FRONT_LEFT', 'CAM_SIDE_LEFT']
    
    interpolation_angles = []
    interpolation_info = []
    
    # 只在中间4对相机之间插值，总共12个插值点
    # 跳过最外侧的相机对（-90°到90°），因为不需要向外插值
    for i in range(0, 4):  # 处理4对相机：-90°->-45°, -45°->0°, 0°->45°, 45°->90°
        current_angle = camera_angles[i]
        next_angle = camera_angles[i + 1]
        
        current_name = camera_names[i]
        next_name = camera_names[i + 1]
        
        # 计算3个插值点
        step = (next_angle - current_angle) / 4.0  # 分成4段，取中间3个点
        
        for j in range(1, 4):  # 1, 2, 3
            interp_angle = current_angle + step * j
            
            interpolation_angles.append(interp_angle)
            interpolation_info.append({
                'angle': interp_angle,
                'between': f"{current_name} ({current_angle}°) -> {next_name} ({next_angle}°)",
                'step': j,
                'total_steps': 4,
                'camera_pair': (current_name, next_name)
            })
    
    return interpolation_angles, interpolation_info

def load_random_timestamp_data(data_root, num_scenes=1000, resume_from_progress=None):
    """加载随机选择的时间戳数据，支持断点续传"""
    folders = [f for f in os.listdir(data_root) 
               if os.path.isdir(os.path.join(data_root, f)) and f.startswith('timestamp_')]
    
    if len(folders) < num_scenes:
        print(f"警告: 只找到{len(folders)}个时间戳场景，少于请求的{num_scenes}个")
        num_scenes = len(folders)
    
    # 如果有进度信息，使用相同的随机种子
    if resume_from_progress:
        random.seed(42)  # 固定种子确保相同的随机选择
        print("使用固定随机种子以支持断点续传")
    
    # 随机选择场景
    selected_folders = random.sample(folders, num_scenes)
    
    # 新的文件名映射到角度（基于文件命名：neg_XX.jpg, pos_XX.jpg）
    file_angle_mapping = {
        'neg_90.jpg': -90,    # 右侧 (25.2° HFOV)
        'neg_45.jpg': -45,    # 右前 (25.2° HFOV)
        'pos_0.jpg': 0,       # 前向 (25.2° HFOV)
        'pos_45.jpg': 45,     # 左前 (25.2° HFOV)
        'pos_90.jpg': 90,     # 左侧 (25.2° HFOV)
    }
    
    scenes_metadata = []
    # 按角度顺序排列文件名
    ordered_files = ['neg_90.jpg', 'neg_45.jpg', 'pos_0.jpg', 'pos_45.jpg', 'pos_90.jpg']
    
    # 懒加载：只返回路径信息，实际图像在需要时加载
    # 使用tqdm显示场景扫描进度
    for folder_name in tqdm(selected_folders, desc="扫描时间戳场景"):
        test_folder = os.path.join(data_root, folder_name)
        
        # 检查文件是否存在
        valid_files = []
        valid_angles = []
        
        for filename in ordered_files:
            image_path = os.path.join(test_folder, filename)
            if os.path.exists(image_path):
                valid_files.append(image_path)
                valid_angles.append(file_angle_mapping[filename])
        
        if len(valid_files) >= 4:  # 至少需要4个相机
            scenes_metadata.append((valid_files, valid_angles, folder_name))
    
    return scenes_metadata

def generate_interpolation_images(num_scenes=1000, batch_size=None, num_workers=None, resume=True):
    """
    生成插值图像的主函数
    
    Args:
        num_scenes: 要处理的场景数量，默认1000
        batch_size: 批量处理大小，如果为None则一次性处理所有场景
        num_workers: 并行工作进程数，如果为None则自动检测
        resume: 是否启用断点续传
    """
    print("=== 相机间插值图像生成===")
    print("功能: 每个时间戳场景生成12张插值图像，保存到各自文件夹")
    
    # 使用传入的路径参数，如果没有则使用默认值
    real_data_path = "/mnt/raid0/lyt/extracted_images_by_timestamp"
    output_root = "/mnt/raid0/lyt/interpolation_results_timestamp"
    
    if not os.path.exists(real_data_path):
        print(f"错误: 找不到数据路径 {real_data_path}")
        return
    
    # 创建输出根目录
    os.makedirs(output_root, exist_ok=True)
    
    # 检查断点续传
    progress_data = None
    if resume:
        progress_data = load_progress(output_root)
        if progress_data:
            print(f"发现断点续传数据: 已完成 {len(progress_data['completed_scenes'])}/{progress_data['total_scenes']} 个场景")
            print(f"完成率: {progress_data['completion_rate']:.2%}")
    
    # 加载随机时间戳场景的数据
    print(f"正在扫描随机{num_scenes}个时间戳场景...")
    scenes_data = load_random_timestamp_data(real_data_path, num_scenes=num_scenes, resume_from_progress=progress_data)
    
    if not scenes_data:
        print("错误: 加载数据失败")
        return
    
    print(f"成功扫描{len(scenes_data)}个场景")
    
    # 、CPU核心数设置
    if num_workers is None:
        cpu_count = mp.cpu_count()
        # 使用更合理的核心数分配
        if cpu_count >= 64:  # 大型服务器
            num_workers = min(16, cpu_count // 8)  # 使用1/8的核心数，最多16个
        elif cpu_count >= 32:  # 中型服务器
            num_workers = min(12, cpu_count // 4)  # 使用1/4的核心数，最多12个
        elif cpu_count >= 16:  # 小型服务器
            num_workers = min(8, cpu_count // 2)   # 使用1/2的核心数，最多8个
        else:  # 普通机器
            num_workers = max(2, cpu_count - 2)    # 保留2个核心给系统
    
    # 合理的最大工作进程数限制
    num_workers = min(num_workers, 16)  # 硬性限制最多16个进程
    
    print(f"使用 {num_workers} 个工作进程进行并行处理 (CPU核心数: {mp.cpu_count()})")
    
    # 新的相机角度配置（HFOV=25.2°）
    vx = [-90, -45, 0, 45, 90]  # 右侧、右前、前、左前、左侧
    vy = [0, 0, 0, 0, 0]
    
    # 计算插值角度
    interpolation_angles, interpolation_info = calculate_interpolation_angles()
    
    print(f"每个时间戳场景将生成{len(interpolation_angles)}张插值图像（{len(interpolation_angles)}个插值点）")
    
    # 创建输出根目录
    os.makedirs(output_root, exist_ok=True)
    
    # 批量处理逻辑
    if batch_size is None:
        batch_size = len(scenes_data)
    
    print(f"批量处理设置: 每批处理{batch_size}个场景")
    
    # 为每个场景生成插值图像
    total_images = len(scenes_data) * len(interpolation_angles)
    
    with tqdm(total=total_images, desc="生成插值图像") as pbar:
        for batch_start in range(0, len(scenes_data), batch_size):
            batch_end = min(batch_start + batch_size, len(scenes_data))
            batch_scenes = scenes_data[batch_start:batch_end]
            
            print(f"\n处理批次 {batch_start//batch_size + 1}/{(len(scenes_data)-1)//batch_size + 1}")
            print(f"场景范围: {batch_start+1}-{batch_end}")
            
            for scene_idx, (image_paths, real_angles, scene_name) in enumerate(batch_scenes, start=batch_start):
                # 懒加载图像并统一尺寸
                real_images = load_scene_images(image_paths)
                
                # 创建时间戳场景文件夹
                scene_output_dir = os.path.join(output_root, f"interpolated_{scene_idx+1:04d}_{scene_name}")
                os.makedirs(scene_output_dir, exist_ok=True)
                
                # 生成每个插值角度的图像
                for idx, target_angle in enumerate(interpolation_angles):
                    # 生成插值图像（使用智能物体对齐）
                    result_img = warp_img_final_optimized(
                        90, target_angle, 0, real_images, vx, vy, 
                        angle_threshold=1.0, enable_object_alignment=True
                    )
                    
                    # 格式化角度为文件名
                    if target_angle >= 0:
                        angle_str = f"pos_{target_angle:06.2f}"
                    else:
                        angle_str = f"neg_{abs(target_angle):06.2f}"
                    
                    # 保存图像
                    filename = f"{angle_str}.jpg"
                    filepath = os.path.join(scene_output_dir, filename)
                    
                    # 转换为BGR用于保存
                    result_bgr = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(filepath, result_bgr)
                    
                    # 更新进度条
                    pbar.set_postfix({
                        'Batch': f"{batch_start//batch_size + 1}/{(len(scenes_data)-1)//batch_size + 1}",
                        'Scene': f"{scene_idx+1}/{len(scenes_data)}",
                        'Image': f"{idx+1}/{len(interpolation_angles)}",
                        'Angle': f"{target_angle:.2f}°"
                    })
                    pbar.update(1)
    
    print(f"\n完成！")
    print(f"生成了{len(scenes_data)}个时间戳场景，每个场景{len(interpolation_angles)}张图像")
    print(f"总计{total_images}张图像")
    print(f"输出目录: {output_root}")
    
    # 输出时间戳场景信息
    print(f"\n时间戳场景列表:")
    for i, (_, _, scene_name) in enumerate(scenes_data):
        print(f"  interpolated_{i+1:04d}_{scene_name}")

def main():
    """主函数，解析命令行参数并执行插值图像生成或修复"""
    parser = argparse.ArgumentParser(description='时间戳场景插值图像生成工具（支持断点续传、并行处理和修复功能）')
    parser.add_argument('--mode', type=str, choices=['generate', 'repair'], default='generate',
                       help='运行模式: generate=生成插值图像, repair=修复缺失/不完整的图像 (默认: generate)')
    parser.add_argument('--num_scenes', type=int, default=1000, 
                       help='要处理的场景数量 (默认: 1000)')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='批量处理大小，如果不指定则自动设置')
    parser.add_argument('--num_workers', type=int, default=None,
                       help='并行工作进程数 (默认: CPU核心数-1)')
    parser.add_argument('--resume', action='store_true', default=True,
                       help='启用断点续传 (默认: True)')
    parser.add_argument('--no_resume', action='store_true',
                       help='禁用断点续传，重新开始')
    parser.add_argument('--data_path', type=str, 
                       default="/mnt/raid0/lyt/extracted_images_by_timestamp",
                       help='输入数据路径 (默认: /mnt/raid0/lyt/extracted_images_by_timestamp)')
    parser.add_argument('--output_path', type=str,
                       default="/mnt/raid0/lyt/interpolation_results_timestamp", 
                       help='输出路径 (默认: /mnt/raid0/lyt/interpolation_results_timestamp)')
    
    args = parser.parse_args()
    
    print(f"运行模式: {args.mode}")
    print(f"配置参数:")
    print(f"  - 工作进程数: {args.num_workers if args.num_workers else '自动检测'}")
    print(f"  - 输入路径: {args.data_path}")
    print(f"  - 输出路径: {args.output_path}")
    print()
    
    if args.mode == 'repair':
        # 修复模式
        repair_missing_interpolations(
            output_root=args.output_path,
            data_root=args.data_path,
            num_workers=args.num_workers
        )
    else:
        # 生成模式
        resume = args.resume and not args.no_resume
        print(f"  - 场景数量: {args.num_scenes}")
        print(f"  - 批量大小: {args.batch_size if args.batch_size else '自动设置'}")
        print(f"  - 断点续传: {'启用' if resume else '禁用'}")
        print()
        
        generate_interpolation_images_with_params(
            num_scenes=args.num_scenes,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            resume=resume,
            data_path=args.data_path,
            output_path=args.output_path
        )

def repair_missing_interpolations(output_root=None, data_root=None, num_workers=None, dry_run=False):
    """修复缺失和不完整的插值图像"""
    print("=== 修复缺失和不完整的插值图像 ===")
    
    if output_root is None:
        output_root = "/mnt/raid0/lyt/interpolation_results_timestamp"
    if data_root is None:
        data_root = "/mnt/raid0/lyt/extracted_images_by_timestamp"
    
    # 检查路径
    if not os.path.exists(output_root):
        print(f"错误: 输出目录不存在 {output_root}")
        return
    if not os.path.exists(data_root):
        print(f"错误: 数据目录不存在 {data_root}")
        return
    
    # 加载分析结果
    analysis_result = load_analysis_result(output_root)
    if not analysis_result:
        print("错误: 无法加载分析结果文件")
        return
    
    missing_numbers = analysis_result.get('missing_numbers', [])
    insufficient_folders = analysis_result.get('insufficient_photos', [])
    
    print(f"发现 {len(missing_numbers)} 个缺失编号")
    print(f"发现 {len(insufficient_folders)} 个数量不足的文件夹")
    
    if len(insufficient_folders) == 0:
        print("没有需要修复的文件夹")
        return
    
    # 加载所有可用的时间戳场景数据
    print("\n正在加载时间戳场景数据...")
    all_scenes_data = load_random_timestamp_data(data_root, num_scenes=50000)  # 加载更多数据用于匹配
    
    if not all_scenes_data:
        print("错误: 无法加载场景数据")
        return
    
    print(f"加载了 {len(all_scenes_data)} 个场景数据")
    
    # 找到需要修复的任务
    repair_tasks = find_missing_scene_data(missing_numbers, insufficient_folders, data_root, all_scenes_data)
    
    if not repair_tasks:
        print("没有找到可修复的任务")
        return
    
    print(f"\n找到 {len(repair_tasks)} 个可修复的任务")
    
    # 设置工作进程数
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() // 4)  # 使用较少的进程避免系统过载
    
    print(f"使用 {num_workers} 个工作进程进行修复")
    
    # 插值参数
    vx = [-90, -45, 0, 45, 90]
    vy = [0, 0, 0, 0, 0]
    interpolation_angles, _ = calculate_interpolation_angles()
    
    print(f"每个场景需要生成 {len(interpolation_angles)} 张插值图像")
    
    # 开始修复
    repaired_count = 0
    
    for task in tqdm(repair_tasks, desc="修复插值图像"):
        try:
            if task['type'] == 'insufficient':
                scene_idx = task['scene_idx']
                scene_data = task['scene_data']
                folder_name = task['folder_name']
                current_count = task['current_count']
                
                # 获取现有的图像文件
                scene_output_dir = os.path.join(output_root, folder_name)
                existing_files = set()
                if os.path.exists(scene_output_dir):
                    existing_files = set(os.listdir(scene_output_dir))
                
                # 加载场景图像
                image_paths, real_angles, scene_name = scene_data
                real_images = load_scene_images(image_paths)
                
                # 确保输出目录存在
                os.makedirs(scene_output_dir, exist_ok=True)
                
                # 生成缺失的插值图像
                generated_count = 0
                for target_angle in interpolation_angles:
                    # 生成文件名
                    if target_angle >= 0:
                        angle_str = f"pos_{target_angle:06.2f}"
                    else:
                        angle_str = f"neg_{abs(target_angle):06.2f}"
                    
                    filename = f"{angle_str}.jpg"
                    
                    # 检查文件是否已存在且完整
                    if filename not in existing_files:
                        # 生成插值图像
                        result_img = warp_img_final_optimized(
                            90, target_angle, 0, real_images, vx, vy,
                            angle_threshold=1.0, enable_object_alignment=True
                        )
                        
                        # 保存图像
                        filepath = os.path.join(scene_output_dir, filename)
                        result_bgr = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(filepath, result_bgr)
                        generated_count += 1
                
                if generated_count > 0:
                    repaired_count += 1
                    print(f"  修复完成: {folder_name} (生成了 {generated_count} 张图像)")
                
        except Exception as e:
            print(f"  修复失败: {task.get('folder_name', 'Unknown')} - {e}")
            continue
    
    print(f"\n修复完成！")
    print(f"成功修复了 {repaired_count} 个文件夹")
    print(f"建议重新运行分析脚本验证修复结果")

def generate_interpolation_images_with_params(num_scenes=1000, batch_size=None, 
                                            num_workers=None, resume=True,
                                            data_path=None, output_path=None):
    """
    带参数的插值图像生成函数（兼容性包装器）
    
    Args:
        num_scenes: 要处理的场景数量
        batch_size: 批量处理大小
        num_workers: 并行工作进程数
        resume: 是否启用断点续传
        data_path: 输入数据路径
        output_path: 输出路径
    """
    if data_path is None:
        data_path = "/mnt/raid0/lyt/extracted_images_by_timestamp"
    if output_path is None:
        output_path = "/mnt/raid0/lyt/interpolation_results_timestamp"
    
    # 修改generate_interpolation_images函数中的路径，然后调用它
    import types
    
    # 创建一个修改版本的函数
    def modified_generate_interpolation_images(num_scenes, batch_size, num_workers, resume):
        print("=== 相机间插值图像生成（优化版）===")
        print("功能: 每个时间戳场景生成12张插值图像，支持断点续传和并行处理")
        
        real_data_path = data_path
        output_root = output_path
        
        if not os.path.exists(real_data_path):
            print(f"错误: 找不到数据路径 {real_data_path}")
            return
        
        # 创建输出根目录
        os.makedirs(output_root, exist_ok=True)
        
        # 检查断点续传
        progress_data = None
        if resume:
            progress_data = load_progress(output_root)
            if progress_data:
                print(f"发现断点续传数据: 已完成 {len(progress_data['completed_scenes'])}/{progress_data['total_scenes']} 个场景")
                print(f"完成率: {progress_data['completion_rate']:.2%}")
        
        # 加载随机时间戳场景的数据
        print(f"正在扫描随机{num_scenes}个时间戳场景...")
        scenes_data = load_random_timestamp_data(real_data_path, num_scenes=num_scenes, resume_from_progress=progress_data)
        
        if not scenes_data:
            print("错误: 加载数据失败")
            return
        
        print(f"成功扫描{len(scenes_data)}个场景")
        
        # 自动检测CPU核心数
        if num_workers is None:
            num_workers = max(1, mp.cpu_count() - 1)  # 保留一个核心给系统
        
        print(f"使用 {num_workers} 个工作进程进行并行处理")
        
        # 新的相机角度配置（HFOV=25.2°）
        vx = [-90, -45, 0, 45, 90]  # 右侧、右前、前、左前、左侧
        vy = [0, 0, 0, 0, 0]
        
        # 计算插值角度
        interpolation_angles, interpolation_info = calculate_interpolation_angles()
        
        print(f"每个时间戳场景将生成{len(interpolation_angles)}张插值图像（{len(interpolation_angles)}个插值点）")
        
        # 过滤已完成的场景（断点续传）
        completed_scenes = set()
        if progress_data:
            completed_scenes = set(progress_data.get('completed_scenes', []))
        
        # 先扫描输出目录，提取已完成的时间戳
        print(f"正在扫描输出目录中的已完成场景...")
        completed_timestamps = set()
        if os.path.exists(output_root):
            import re
            for dir_name in os.listdir(output_root):
                if dir_name.startswith('interpolated_') and os.path.isdir(os.path.join(output_root, dir_name)):
                    # 提取时间戳，格式：interpolated_xxxx_timestamp_1234567890123456
                    match = re.search(r'timestamp_(\d+)', dir_name)
                    if match:
                        timestamp = match.group(1)
                        completed_timestamps.add(timestamp)
        
        print(f"发现 {len(completed_timestamps)} 个已完成的时间戳")
        if len(completed_timestamps) > 0:
            sample_timestamps = list(completed_timestamps)[:5]
            print(f"示例已完成时间戳: {sample_timestamps}")
        
        # 筛选未完成的场景（基于时间戳）
        print(f"正在筛选未完成的场景...")
        remaining_scenes = []
        completed_scene_indices = []
        
        for idx, scene_data in enumerate(scenes_data):
            _, _, scene_name = scene_data
            # 从scene_name中提取时间戳：timestamp_1234567890123456
            timestamp_match = re.search(r'timestamp_(\d+)', scene_name)
            if timestamp_match:
                timestamp = timestamp_match.group(1)
                
                # 只显示前几个场景的匹配情况
                if idx < 5:
                    status = "已完成" if timestamp in completed_timestamps else "待处理"
                    print(f"  场景 {idx+1}: {scene_name} -> {status}")
                
                if timestamp in completed_timestamps:
                    completed_scene_indices.append(idx + 1)
                else:
                    remaining_scenes.append((idx, scene_data))
            else:
                # 无法提取时间戳，认为未完成
                remaining_scenes.append((idx, scene_data))
                if idx < 5:
                    print(f"  场景 {idx+1}: {scene_name} -> 无法提取时间戳（待处理）")
        
        print(f"已完成场景数: {len(completed_scene_indices)}")
        print(f"剩余需要处理: {len(remaining_scenes)} 个场景")
        
        # 重新编号：为剩余场景分配连续的新编号
        if remaining_scenes:
            # 基于已完成场景的数量，从下一个连续编号开始
            num_completed = len(completed_scene_indices)
            start_new_idx = num_completed + 1  # 从已完成数量+1开始
            
            print(f"\n重新编号策略:")
            print(f"  已完成场景数量: {num_completed}")
            print(f"  新场景编号从 {start_new_idx} 开始连续编号")
            
            # 为每个剩余场景分配新的连续编号
            renumbered_scenes = []
            for i, (original_idx, scene_data) in enumerate(remaining_scenes):
                new_idx = start_new_idx + i - 1  # 新的连续编号（从0开始的索引）
                renumbered_scenes.append((new_idx, scene_data))
            
            remaining_scenes = renumbered_scenes
            
            print(f"\n接下来要处理的前5个场景（重新编号后）:")
            for i, (new_idx, scene_data) in enumerate(remaining_scenes[:5]):
                _, _, scene_name = scene_data
                print(f"  新编号 {new_idx+1}: {scene_name}")
        else:
            print(f"\n没有剩余场景需要处理")
        
        # 开始实际处理
        print(f"\n=== 开始处理剩余场景 ===")
        print(f"总场景数: {len(scenes_data)}")
        print(f"已完成时间戳数: {len(completed_timestamps)}")
        print(f"已完成场景数: {len(completed_scene_indices)}")
        print(f"剩余需要处理场景数: {len(remaining_scenes)}")
        
        if not remaining_scenes:
            print(f"\n所有场景都已完成！")
            return
        
        
        # 批量处理逻辑 - 合理的批处理大小
        if batch_size is None:
            # 每个工作进程处理3-4个场景，提高效率
            batch_size = min(num_workers * 4, len(remaining_scenes), 32)  # 每个工作进程处理4个场景，最多32个
        
        # 适中的批处理大小限制
        batch_size = min(batch_size, 24)  # 硬性限制最多24个场景一批
        
        print(f"批量处理设置: 每批处理{batch_size}个场景")
        
        # 创建部分函数以固定参数
        process_func = partial(process_single_scene, 
                              output_root=output_root,
                              interpolation_angles=interpolation_angles,
                              vx=vx, vy=vy)
        
        # 为剩余场景生成插值图像
        total_scenes = len(remaining_scenes)
        completed_in_session = 0
        
        # 添加CPU使用率监控和自适应调节
        def check_system_load():
            """检查系统负载，如果过高则建议降低并行度"""
            if not PSUTIL_AVAILABLE:
                return 0, 0
            try:
                cpu_percent = psutil.cpu_percent(interval=0.5)  # 减少检查时间
                load_avg = os.getloadavg()[0] if hasattr(os, 'getloadavg') else 0
                return cpu_percent, load_avg
            except Exception as e:
                print(f"警告: 无法获取系统负载信息: {e}")
                return 0, 0
        
        # 检查初始系统负载
        initial_cpu, initial_load = check_system_load()
        print(f"初始系统状态: CPU使用率 {initial_cpu:.1f}%, 系统负载 {initial_load:.2f}")
        
        # 只在系统负载非常高时才减少工作进程数
        if initial_load > 50 or initial_cpu > 90:
            num_workers = max(2, num_workers // 2)
            print(f"⚠️  系统负载很高，自动降低工作进程数到 {num_workers}")
        
        # 使用进程池并行处理（添加系统负载监控）
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
                with tqdm(total=total_scenes, desc="生成插值图像") as pbar:
                    for batch_start in range(0, len(remaining_scenes), batch_size):
                        batch_end = min(batch_start + batch_size, len(remaining_scenes))
                        batch_scenes = remaining_scenes[batch_start:batch_end]
                        
                        # 获取这批场景的实际编号范围
                        if batch_scenes:
                            first_scene_idx = batch_scenes[0][0]
                            last_scene_idx = batch_scenes[-1][0]
                            print(f"\n处理批次 {batch_start//batch_size + 1}/{(len(remaining_scenes)-1)//batch_size + 1}")
                            print(f"场景编号范围: {first_scene_idx+1}-{last_scene_idx+1}")
                        
                        # 提交批次任务
                        futures = []
                        for scene_idx, scene_data in batch_scenes:
                            future = executor.submit(process_func, scene_data, scene_idx)
                            futures.append((future, scene_idx, scene_data))
                        
                        # 等待批次完成 - 添加进程间延迟以降低CPU竞争
                        for i, (future, scene_idx, scene_data) in enumerate(futures):
                            result = future.result()  # 移除超时和异常处理
                            _, _, scene_name = scene_data
                            scene_id = f"interpolated_{scene_idx+1:04d}_{scene_name}"
                            completed_scenes.add(scene_id)
                            completed_in_session += 1
                            
                            # 更新进度条
                            pbar.set_postfix({
                                'Batch': f"{batch_start//batch_size + 1}/{(len(remaining_scenes)-1)//batch_size + 1}",
                                'Completed': f"{completed_in_session}/{total_scenes}",
                                'Total': f"{len(completed_scenes)}/{len(scenes_data)}"
                            })
                            pbar.update(1)
                            
                            # 适度的任务间延迟
                            if i < len(futures) - 1:  # 不在最后一个任务后延迟
                                time.sleep(0.05)  # 50ms延迟
                        
                        # 保存进度
                        save_progress(output_root, list(completed_scenes), len(scenes_data), 
                                    batch_start//batch_size + 1)
                        
                        # 批量内存清理 - 每完成一个批次进行内存管理
                        print(f"  批次完成，进行内存清理...")
                        
                        # 强制垃圾回收，释放内存
                        if PSUTIL_AVAILABLE:
                            try:
                                # 获取当前进程的内存使用情况
                                process = psutil.Process(os.getpid())
                                memory_before = process.memory_info().rss / 1024 / 1024  # MB
                                
                                # 执行垃圾回收
                                gc.collect()
                                
                                # 检查内存使用情况
                                memory_after = process.memory_info().rss / 1024 / 1024  # MB
                                print(f"  内存使用: {memory_before:.1f}MB -> {memory_after:.1f}MB")
                            except Exception as e:
                                print(f"  内存监控失败: {e}")
                                gc.collect()
                        else:
                            # 没有psutil时仍然执行垃圾回收
                            gc.collect()
                            print(f"  执行垃圾回收（无内存监控）")
                        
                        # 批次间适度延迟
                        if batch_start + batch_size < len(remaining_scenes):  # 不是最后一个批次
                            print(f"  等待1秒让系统恢复...")
                            time.sleep(1)
                            
                            # 每个批次后检查系统负载
                            cpu_percent, load_avg = check_system_load()
                            print(f"  当前系统状态: CPU使用率 {cpu_percent:.1f}%, 系统负载 {load_avg:.2f}")
                            
                            # 只在负载非常高时增加延迟
                            if load_avg > 100 or cpu_percent > 95:
                                extra_delay = min(5, load_avg / 20)  # 最多额外等待5秒
                                print(f"  ⚠️  系统负载过高，额外等待 {extra_delay:.1f} 秒...")
                                time.sleep(extra_delay)
        
        # 最终保存进度
        save_progress(output_root, list(completed_scenes), len(scenes_data))
        
        print(f"\n完成！")
        print(f"本次会话完成了{completed_in_session}个场景")
        print(f"总共完成了{len(completed_scenes)}/{len(scenes_data)}个时间戳场景")
        print(f"每个场景{len(interpolation_angles)}张图像")
        print(f"输出目录: {output_root}")
        
        if len(completed_scenes) < len(scenes_data):
            remaining = len(scenes_data) - len(completed_scenes)
            print(f"\n还有 {remaining} 个场景未完成，可以使用 --resume 选项继续")
        
        # 输出完成的时间戳场景信息
        print(f"\n已完成的时间戳场景数量: {len(completed_scenes)}")
    
    # 调用修改版本的函数
    modified_generate_interpolation_images(num_scenes, batch_size, num_workers, resume)

if __name__ == "__main__":
    # 设置多进程启动方法
    if hasattr(mp, 'set_start_method'):
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass  
    
    main() 
