#!/usr/bin/env python3
"""
多进程优化版本的相机间插值测试脚本
使用多进程并行处理场景，充分利用CPU资源，同时控制内存使用
"""
import cv2
import numpy as np
import os
import sys
import random
import argparse
import multiprocessing as mp
import psutil
import time
from tqdm import tqdm
from functools import partial
import gc
import threading
import queue

# 添加源代码路径
sys.path.append('src')
from dataset.utils import get_K_R

# 导入优化的拼接函数
from warp_img_final_optimized import warp_img_final_optimized, load_multiple_nuscenes_data

class MemoryMonitor:
    """内存监控类"""
    def __init__(self, max_memory_percent=85):
        self.max_memory_percent = max_memory_percent
        self.process = psutil.Process()
        
    def get_memory_usage(self):
        """获取当前内存使用率"""
        memory_info = psutil.virtual_memory()
        return memory_info.percent
    
    def get_process_memory(self):
        """获取当前进程内存使用"""
        return self.process.memory_info().rss / 1024 / 1024  # MB
    
    def should_wait_for_memory(self):
        """检查是否需要等待内存释放"""
        return self.get_memory_usage() > self.max_memory_percent

def calculate_optimal_process_count():
    """
    根据CPU核心数和内存大小计算最优进程数
    """
    cpu_count = mp.cpu_count()
    memory_gb = psutil.virtual_memory().total / (1024**3)
    
    # 基于CPU核心数
    cpu_processes = max(1, cpu_count - 1)  # 留一个核心给系统
    
    # 基于内存（每个进程大约需要2-4GB内存）
    memory_processes = max(1, int(memory_gb / 3))
    
    # 取较小值，确保不会内存溢出
    optimal_processes = min(cpu_processes, memory_processes)
    
    print(f"系统信息: CPU核心数={cpu_count}, 内存={memory_gb:.1f}GB")
    print(f"建议进程数: CPU建议={cpu_processes}, 内存建议={memory_processes}")
    print(f"最终选择进程数: {optimal_processes}")
    
    return optimal_processes

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
    for i in range(len(camera_angles) - 1):  # 处理4对相机
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

def load_scene_data(scene_path_info):
    """
    加载单个场景的数据（用于多进程）
    """
    scene_path, scene_name = scene_path_info
    
    # 新的文件名映射到角度
    file_angle_mapping = {
        'neg_90.jpg': -90,    # 右侧 (25.2° HFOV)
        'neg_45.jpg': -45,    # 右前 (25.2° HFOV)
        'pos_0.jpg': 0,       # 前向 (25.2° HFOV)
        'pos_45.jpg': 45,     # 左前 (25.2° HFOV)
        'pos_90.jpg': 90,     # 左侧 (25.2° HFOV)
    }
    
    # 按角度顺序排列文件名
    ordered_files = ['neg_90.jpg', 'neg_45.jpg', 'pos_0.jpg', 'pos_45.jpg', 'pos_90.jpg']
    
    images = []
    angles = []
    
    for filename in ordered_files:
        image_path = os.path.join(scene_path, filename)
        if os.path.exists(image_path):
            img = cv2.imread(image_path)
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img_rgb)
                angles.append(file_angle_mapping[filename])
    
    if len(images) >= 4:  # 至少需要4个相机
        return (images, angles, scene_name)
    else:
        return None

def process_single_scene_streaming(args):
    """
    流式处理单个场景的所有插值图像（多进程工作函数）
    即时加载和处理，不预加载数据
    """
    scene_path_info, interpolation_angles, output_root, vx, vy = args
    scene_path, scene_name, scene_idx = scene_path_info
    
    try:
        # 即时加载场景数据
        scene_data = load_scene_data((scene_path, scene_name))
        
        if scene_data is None:
            return {
                'scene_idx': scene_idx,
                'scene_name': scene_name,
                'success_count': 0,
                'total_count': len(interpolation_angles),
                'error': '场景数据加载失败'
            }
        
        real_images, real_angles, scene_name = scene_data
        
        # 创建场景输出目录
        scene_output_dir = os.path.join(output_root, f"interpolated_{scene_idx+1:04d}_{scene_name}")
        os.makedirs(scene_output_dir, exist_ok=True)
        
        results = []
        
        # 为当前场景生成所有插值图像
        for idx, target_angle in enumerate(interpolation_angles):
            try:
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
                
                results.append(f"场景{scene_idx+1} - 角度{target_angle:.2f}° - 成功")
                
                # 清理内存
                del result_img, result_bgr
                gc.collect()
                
            except Exception as e:
                error_msg = f"场景{scene_idx+1} - 角度{target_angle:.2f}° - 错误: {str(e)}"
                results.append(error_msg)
                continue
        
        # 清理内存
        del real_images, scene_data
        gc.collect()
        
        return {
            'scene_idx': scene_idx,
            'scene_name': scene_name,
            'success_count': len([r for r in results if "成功" in r]),
            'total_count': len(interpolation_angles),
            'results': results
        }
        
    except Exception as e:
        return {
            'scene_idx': scene_idx,
            'scene_name': scene_name if 'scene_name' in locals() else 'unknown',
            'success_count': 0,
            'total_count': len(interpolation_angles),
            'error': str(e)
        }

def get_scene_list(data_root, num_scenes=1000):
    """
    获取场景列表，不预加载数据
    """
    folders = [f for f in os.listdir(data_root) 
               if os.path.isdir(os.path.join(data_root, f)) and f.startswith('timestamp_')]
    
    if len(folders) < num_scenes:
        print(f"警告: 只找到{len(folders)}个时间戳场景，少于请求的{num_scenes}个")
        num_scenes = len(folders)
    
    # 随机选择场景
    selected_folders = random.sample(folders, num_scenes)
    
    # 返回场景路径信息，不预加载数据
    scene_path_infos = [(os.path.join(data_root, folder_name), folder_name, idx) 
                        for idx, folder_name in enumerate(selected_folders)]
    
    print(f"找到{len(scene_path_infos)}个场景，将使用流式处理")
    
    return scene_path_infos

def generate_interpolation_images_multiprocessing(num_scenes=1000, 
                                                 num_processes=None,
                                                 data_path=None, 
                                                 output_path=None,
                                                 memory_limit_percent=85):
    """
    多进程版本的插值图像生成函数
    """
    print("=== 多进程相机间插值图像生成 ===")
    print("功能: 使用多进程并行生成插值图像，最大化CPU利用率")
    
    # 默认路径
    if data_path is None:
        data_path = "/mnt/raid0/lyt/extracted_images_by_timestamp"
    if output_path is None:
        output_path = "/mnt/raid0/lyt/interpolation_results_timestamp_mp"
    
    # 检查输入路径
    if not os.path.exists(data_path):
        print(f"错误: 找不到数据路径 {data_path}")
        return
    
    # 计算最优进程数
    if num_processes is None:
        num_processes = calculate_optimal_process_count()
    
    # 初始化内存监控
    memory_monitor = MemoryMonitor(memory_limit_percent)
    
    # 获取场景列表（不预加载数据）
    print(f"正在扫描{num_scenes}个时间戳场景...")
    scene_path_infos = get_scene_list(data_path, num_scenes=num_scenes)
    
    if not scene_path_infos:
        print("错误: 找不到有效场景")
        return
    
    print(f"找到{len(scene_path_infos)}个场景，使用流式处理")
    
    # 相机配置
    vx = [-90, -45, 0, 45, 90]  # 右侧、右前、前、左前、左侧
    vy = [0, 0, 0, 0, 0]
    
    # 计算插值角度
    interpolation_angles, interpolation_info = calculate_interpolation_angles()
    print(f"每个场景将生成{len(interpolation_angles)}张插值图像")
    
    # 创建输出目录
    os.makedirs(output_path, exist_ok=True)
    
    # 准备多进程参数（流式处理）
    process_args = []
    for scene_path_info in scene_path_infos:
        args = (scene_path_info, interpolation_angles, output_path, vx, vy)
        process_args.append(args)
    
    print(f"开始多进程流式处理，使用{num_processes}个进程...")
    print(f"总计需要生成{len(scene_path_infos) * len(interpolation_angles)}张图像")
    print(f"内存使用策略: 流式加载，每个进程即时加载和释放数据")
    
    # 使用多进程处理场景
    start_time = time.time()
    successful_scenes = 0
    total_images = 0
    
    # 使用进程池处理
    with mp.Pool(num_processes) as pool:
        # 使用imap_unordered获得更好的负载均衡
        results = list(tqdm(
            pool.imap_unordered(process_single_scene_streaming, process_args),
            total=len(process_args),
            desc="生成插值图像",
            unit="场景"
        ))
    
    # 统计结果
    for result in results:
        if 'error' not in result:
            successful_scenes += 1
            total_images += result['success_count']
            if result['success_count'] < result['total_count']:
                print(f"  场景 {result['scene_idx']+1} ({result['scene_name']}): "
                      f"{result['success_count']}/{result['total_count']} 张图像成功")
        else:
            print(f"  场景 {result['scene_idx']+1} ({result['scene_name']}) 失败: {result['error']}")
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    print(f"\n=== 处理完成 ===")
    print(f"处理时间: {processing_time:.2f} 秒")
    print(f"成功处理场景: {successful_scenes}/{len(scene_path_infos)}")
    print(f"总计生成图像: {total_images}/{len(scene_path_infos) * len(interpolation_angles)}")
    print(f"平均每场景处理时间: {processing_time/len(scene_path_infos):.2f} 秒")
    print(f"图像生成速度: {total_images/processing_time:.2f} 张/秒")
    print(f"输出目录: {output_path}")
    
    # 显示内存使用情况
    final_memory = memory_monitor.get_memory_usage()
    process_memory = memory_monitor.get_process_memory()
    print(f"最终内存使用: 系统{final_memory:.1f}%, 进程{process_memory:.1f}MB")

def main():
    """主函数，解析命令行参数并执行多进程插值图像生成"""
    parser = argparse.ArgumentParser(description='多进程时间戳场景插值图像生成工具')
    parser.add_argument('--num_scenes', type=int, default=100000, 
                       help='要处理的场景数量 (默认: 100000)')
    parser.add_argument('--num_processes', type=int, default=None,
                       help='并行进程数，如果不指定则自动计算最优值')
    parser.add_argument('--data_path', type=str, 
                       default="/mnt/raid0/lyt/extracted_images_by_timestamp",
                       help='输入数据路径')
    parser.add_argument('--output_path', type=str,
                       default="/mnt/raid0/lyt/interpolation_results_timestamp_mp", 
                       help='输出路径')
    parser.add_argument('--memory_limit', type=int, default=85,
                       help='内存使用限制百分比 (默认: 85)')
    
    args = parser.parse_args()
    
    print(f"配置参数:")
    print(f"  - 场景数量: {args.num_scenes}")
    print(f"  - 进程数: {args.num_processes if args.num_processes else '自动计算'}")
    print(f"  - 输入路径: {args.data_path}")
    print(f"  - 输出路径: {args.output_path}")
    print(f"  - 内存限制: {args.memory_limit}%")
    print()
    
    # 设置多进程启动方法
    if sys.platform != 'win32':
        mp.set_start_method('spawn', force=True)
    
    # 调用多进程生成函数
    generate_interpolation_images_multiprocessing(
        num_scenes=args.num_scenes,
        num_processes=args.num_processes,
        data_path=args.data_path,
        output_path=args.output_path,
        memory_limit_percent=args.memory_limit
    )

if __name__ == "__main__":
    main()
