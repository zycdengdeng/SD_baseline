#!/usr/bin/env python3
"""
批量预加载版本的相机间6个中间角度插值脚本
使用批量预加载 + 多进程 + 物体对齐
"""
import cv2
import numpy as np
import os
import sys
import multiprocessing as mp
from functools import partial
from tqdm import tqdm
import time
import gc

# 添加源代码路径
sys.path.append('src')
from dataset.utils import get_K_R

# 导入优化的拼接函数
from warp_img_final_optimized import warp_img_final_optimized

def calculate_middle_angles():
    """计算6对相邻相机之间的中间角度"""
    camera_angles = [-55, 0, 55, 110, 180, -110]
    camera_names = ['CAM_FRONT_RIGHT', 'CAM_FRONT', 'CAM_FRONT_LEFT', 
                   'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    
    middle_angles = []
    middle_info = []
    
    for i in range(6):
        current_angle = camera_angles[i]
        next_angle = camera_angles[(i + 1) % len(camera_angles)]
        
        current_name = camera_names[i]
        next_name = camera_names[(i + 1) % len(camera_names)]
        
        # 处理角度跳跃
        if current_angle == 180 and next_angle == -110:
            next_angle_normalized = 250
            middle_angle = (current_angle + next_angle_normalized) / 2.0
            if middle_angle > 180:
                middle_angle = middle_angle - 360
        else:
            middle_angle = (current_angle + next_angle) / 2.0
        
        middle_angles.append(middle_angle)
        middle_info.append({
            'angle': middle_angle,
            'between': f"{current_name} ({current_angle}°) -> {next_name} ({next_angle}°)",
            'camera_pair': (current_name, next_name)
        })
    
    return middle_angles, middle_info

def calculate_24_interpolation_angles():
    """计算6对相邻相机之间的24个插值角度（每对4个）"""
    camera_angles = [-55, 0, 55, 110, 180, -110]
    camera_names = ['CAM_FRONT_RIGHT', 'CAM_FRONT', 'CAM_FRONT_LEFT', 
                   'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    
    interpolation_angles = []
    interpolation_info = []
    
    for i in range(6):  # 6对相机，每对4个插值点 = 24张
        current_angle = camera_angles[i]
        next_angle = camera_angles[(i + 1) % len(camera_angles)]
        
        current_name = camera_names[i]
        next_name = camera_names[(i + 1) % len(camera_names)]
        
        # 处理角度跳跃（180度到-110度的情况）
        if current_angle == 180 and next_angle == -110:
            # 从180度顺时针到250度（-110度）
            next_angle_normalized = 250
            angle_diff = next_angle_normalized - current_angle
        else:
            angle_diff = next_angle - current_angle
        
        # 计算4个插值点
        step = angle_diff / 5.0  # 分成5段，取中间4个点
        
        for j in range(1, 5):  # 1, 2, 3, 4
            if current_angle == 180 and next_angle == -110:
                # 特殊处理180到-110的情况
                interp_angle = 180 + step * j
                if interp_angle > 180:
                    interp_angle = interp_angle - 360  # 转换为负角度表示
            else:
                interp_angle = current_angle + step * j
            
            interpolation_angles.append(interp_angle)
            interpolation_info.append({
                'angle': interp_angle,
                'between': f"{current_name} ({current_angle}°) -> {next_name} ({next_angle}°)",
                'step': j,
                'total_steps': 5,
                'camera_pair': (current_name, next_name)
            })
    
    return interpolation_angles, interpolation_info

def load_batch_scenes(data_root, scene_batch):
    """批量加载一组场景的图像到内存"""
    # 相机配置
    camera_mapping = {
        'CAM_FRONT': 0,
        'CAM_FRONT_LEFT': 55,
        'CAM_FRONT_RIGHT': -55,
        'CAM_BACK_LEFT': 110,
        'CAM_BACK_RIGHT': -110,
        'CAM_BACK': 180,
    }
    
    ordered_cameras = ['CAM_FRONT_RIGHT', 'CAM_FRONT', 'CAM_FRONT_LEFT', 
                      'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    
    batch_data = {}
    
    for scene_name in tqdm(scene_batch, desc="批量加载场景", leave=False):
        folder_name = scene_name.split('_')[-1]
        test_folder = os.path.join(data_root, folder_name)
        
        if not os.path.exists(test_folder):
            continue
        
        images = []
        angles = []
        
        for camera_name in ordered_cameras:
            image_files = [f for f in os.listdir(test_folder) 
                          if f.startswith(camera_name + '_n') and f.endswith('.jpg')]
            if image_files:
                image_path = os.path.join(test_folder, image_files[0])
                img = cv2.imread(image_path)
                if img is not None:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    images.append(img_rgb)
                    angles.append(camera_mapping[camera_name])
        
        if len(images) >= 4:
            batch_data[scene_name] = (images, angles, scene_name)
    
    return batch_data

def process_scene_from_memory(args):
    """从内存中的数据处理单个场景"""
    scene_name, scene_data, output_root, middle_angles = args
    
    if scene_data is None:
        return 0
    
    real_images, real_angles, _ = scene_data
    
    # 创建输出目录
    scene_output_dir = os.path.join(output_root, scene_name)
    os.makedirs(scene_output_dir, exist_ok=True)
    
    # nuScenes相机角度配置
    vx = [-55, 0, 55, 110, 180, -110]
    vy = [0, 0, 0, 0, 0, 0]
    
    generated_count = 0
    
    # 生成每个中间角度的图像
    for target_angle in middle_angles:
        try:
            # 启用物体对齐的高质量拼接
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
            
            # 转换为BGR并保存
            result_bgr = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(filepath, result_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            generated_count += 1
            
        except Exception as e:
            print(f"场景 {scene_name} 角度 {target_angle}° 生成失败: {e}")
            continue
    
    return generated_count

def get_scene_list_from_txt(txt_path):
    """从txt文件中读取场景列表"""
    with open(txt_path, 'r') as f:
        scene_list = [line.strip() for line in f.readlines()]
    return scene_list

def simple_half_stitch(img1, img2):
    """
    简单的图像拼接：取img1的右半部分 + img2的左半部分
    """
    h, w = img1.shape[:2]
    half_w = w // 2
    
    # 取img1的右半部分和img2的左半部分
    right_half = img1[:, half_w:]
    left_half = img2[:, :half_w]
    
    # 拼接：img1的右半 + img2的左半
    stitched = np.concatenate([right_half, left_half], axis=1)
    
    return stitched

def simple_fraction_stitch(img1, img2, fraction1, fraction2):
    """
    简单的图像拼接：取img1的fraction1部分 + img2的fraction2部分
    fraction1: img1的右半部分比例 (0.0-1.0)
    fraction2: img2的左半部分比例 (0.0-1.0)
    """
    h, w = img1.shape[:2]
    
    # 计算分割点
    split1 = int(w * (1 - fraction1))  # img1的右半部分起始点
    split2 = int(w * fraction2)        # img2的左半部分结束点
    
    # 取img1的右半部分和img2的左半部分
    right_part = img1[:, split1:]
    left_part = img2[:, :split2]
    
    # 拼接：img1的右半 + img2的左半
    stitched = np.concatenate([right_part, left_part], axis=1)
    
    return stitched

def simple_fraction_stitch_reversed(img1, img2, fraction1, fraction2):
    """
    反向的简单图像拼接：取img2的fraction1部分 + img1的fraction2部分
    fraction1: img2的右半部分比例 (0.0-1.0)
    fraction2: img1的左半部分比例 (0.0-1.0)
    """
    h, w = img1.shape[:2]
    
    # 计算分割点
    split1 = int(w * (1 - fraction1))  # img2的右半部分起始点
    split2 = int(w * fraction2)        # img1的左半部分结束点
    
    # 取img2的右半部分和img1的左半部分
    right_part = img2[:, split1:]
    left_part = img1[:, :split2]
    
    # 拼接：img2的右半 + img1的左半
    stitched = np.concatenate([right_part, left_part], axis=1)
    
    return stitched

def simple_fraction_stitch_corrected(img1, img2, fraction1, fraction2):
    """
    修正的简单图像拼接：取img1的fraction1部分 + img2的fraction2部分
    fraction1: img1的右半部分比例 (0.0-1.0)
    fraction2: img2的左半部分比例 (0.0-1.0)
    注意：img1是左图，img2是右图
    """
    h, w = img1.shape[:2]
    
    # 计算分割点
    split1 = int(w * (1 - fraction1))  # img1的右半部分起始点
    split2 = int(w * fraction2)        # img2的左半部分结束点
    
    # 取img1的右半部分和img2的左半部分
    right_part = img1[:, split1:]
    left_part = img2[:, :split2]
    
    # 拼接：img1的右半 + img2的左半
    stitched = np.concatenate([right_part, left_part], axis=1)
    
    return stitched


def process_single_timestamp_simple_stitch(args):
    """处理单个时间戳的所有角度图像生成 - 使用简单拼接方法"""
    timestamp, real_data_path, output_root, target_angles = args
    
    # 场景路径
    scene_path = os.path.join(real_data_path, timestamp)
    
    if not os.path.exists(scene_path):
        return 0, f"场景路径不存在: {scene_path}"
    
    # 相机配置
    camera_mapping = {
        'CAM_FRONT': 0,
        'CAM_FRONT_LEFT': 55,
        'CAM_FRONT_RIGHT': -55,
        'CAM_BACK_LEFT': 110,
        'CAM_BACK_RIGHT': -110,
        'CAM_BACK': 180,
    }
    
    ordered_cameras = ['CAM_FRONT_RIGHT', 'CAM_FRONT', 'CAM_FRONT_LEFT', 
                      'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    
    # 加载场景数据
    images_dict = {}
    
    for camera_name in ordered_cameras:
        image_files = [f for f in os.listdir(scene_path) 
                      if f.startswith(camera_name + '_n') and f.endswith('.jpg')]
        if image_files:
            image_path = os.path.join(scene_path, image_files[0])
            img = cv2.imread(image_path)
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                angle = camera_mapping[camera_name]
                images_dict[angle] = img_rgb
    
    if len(images_dict) < 4:
        return 0, f"图像不足（只有{len(images_dict)}张）"
    
    # 创建时间戳输出目录
    timestamp_dir = os.path.join(output_root, f"timestamp_{timestamp}")
    os.makedirs(timestamp_dir, exist_ok=True)
    
    generated_count = 0
    
    # 定义拼接规则：目标角度 -> (左图角度, 右图角度)
    # simple_half_stitch(left_img, right_img) 会产生 left_img的右半 + right_img的左半
    # 由于-角度是顺时针，需要调整左右顺序
    stitch_rules = {
        -27.5: (0, -55),    # -27.5度 = 0度右半 + (-55)度左半
        27.5: (55, 0),      # 27.5度 = 55度右半 + 0度左半  
        82.5: (110, 55),    # 82.5度 = 110度右半 + 55度左半
        145.0: (180, 110),  # 145.0度 = 180度右半 + 110度左半
        -82.5: (-55, -110), # -82.5度 = (-55)度右半 + (-110)度左半
        -145.0: (-110, 180) # -145.0度 = (-110)度右半 + 180度左半
    }
    
    # 调试：打印可用的相机角度（可选）
    # if len(images_dict) > 0:
    #     print(f"    时间戳 {timestamp} 可用相机角度: {sorted(images_dict.keys())}")
    
    # 生成每个目标角度的图像
    for target_angle in target_angles:
        try:
            if target_angle in stitch_rules:
                left_angle, right_angle = stitch_rules[target_angle]
                
                # 检查是否有对应的图像
                if left_angle in images_dict and right_angle in images_dict:
                    left_img = images_dict[left_angle]
                    right_img = images_dict[right_angle]
                    
                    # 使用简单拼接方法
                    result_img = simple_half_stitch(left_img, right_img)
                    
                    # 格式化角度为文件名
                    if target_angle >= 0:
                        angle_str = f"{target_angle:06.2f}"
                    else:
                        angle_str = f"{target_angle:07.2f}"
                    
                    # 按照用户指定的命名格式
                    filename = f"timestamp_{timestamp}_angle_{angle_str}.jpg"
                    filepath = os.path.join(timestamp_dir, filename)
                    
                    # 转换为BGR并保存
                    result_bgr = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(filepath, result_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    
                    generated_count += 1
                else:
                    # 某些时间戳可能缺少特定相机角度的图像
                    pass
                    
        except Exception as e:
            continue
    
    return generated_count, f"成功生成{generated_count}张图像"

def process_single_timestamp(args):
    """处理单个时间戳的所有角度图像生成"""
    timestamp, real_data_path, output_root, target_angles = args
    
    # 场景路径
    scene_path = os.path.join(real_data_path, timestamp)
    
    if not os.path.exists(scene_path):
        return 0, f"场景路径不存在: {scene_path}"
    
    # 相机配置
    camera_mapping = {
        'CAM_FRONT': 0,
        'CAM_FRONT_LEFT': 55,
        'CAM_FRONT_RIGHT': -55,
        'CAM_BACK_LEFT': 110,
        'CAM_BACK_RIGHT': -110,
        'CAM_BACK': 180,
    }
    
    ordered_cameras = ['CAM_FRONT_RIGHT', 'CAM_FRONT', 'CAM_FRONT_LEFT', 
                      'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    
    # 加载场景数据
    images = []
    angles = []
    
    for camera_name in ordered_cameras:
        image_files = [f for f in os.listdir(scene_path) 
                      if f.startswith(camera_name + '_n') and f.endswith('.jpg')]
        if image_files:
            image_path = os.path.join(scene_path, image_files[0])
            img = cv2.imread(image_path)
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img_rgb)
                angles.append(camera_mapping[camera_name])
    
    if len(images) < 4:
        return 0, f"图像不足（只有{len(images)}张）"
    
    # nuScenes相机角度配置
    vx = [-55, 0, 55, 110, 180, -110]  # 右前、前、左前、左后、后、右后
    vy = [0, 0, 0, 0, 0, 0]
    
    # 创建时间戳输出目录
    timestamp_dir = os.path.join(output_root, f"timestamp_{timestamp}")
    os.makedirs(timestamp_dir, exist_ok=True)
    
    generated_count = 0
    
    # 生成每个目标角度的图像
    for target_angle in target_angles:
        try:
            # 使用纯几何变换（关闭物体对齐）
            result_img = warp_img_final_optimized(
                90, target_angle, 0, images, vx, vy, 
                angle_threshold=1.0, enable_object_alignment=False
            )
            
            # 格式化角度为文件名 (按照用户要求的格式)
            if target_angle >= 0:
                angle_str = f"{target_angle:06.2f}"
            else:
                angle_str = f"{target_angle:07.2f}"
            
            # 按照用户指定的命名格式
            filename = f"timestamp_{timestamp}_angle_{angle_str}.jpg"
            filepath = os.path.join(timestamp_dir, filename)
            
            # 转换为BGR并保存
            result_bgr = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(filepath, result_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            generated_count += 1
            
        except Exception as e:
            continue
    
    return generated_count, f"成功生成{generated_count}张图像"


def generate_167_timestamps_images_simple_stitch():
    """
    为167个时间戳生成简单拼接图像
    """
    print("=== 生成167个时间戳的简单拼接图像 ===")
    
    real_data_path = "/mnt/vdb1/lyt/localdata/nuscenes/processed_data"
    output_root = "timestamp_images_simple_stitch"
    
    # 167个时间戳列表
    timestamps = [
        "1537297676650253", "1542800689447619", "1542800309448236", "1538033755447568", "1535657545049969",
        "1538448758547578", "1535729331397165", "1531885992048261", "1538986259296943", "1535489941196216",
        "1535488129298125", "1542799639197823", "1537286922401853", "1533115035397015", "1538985953297524",
        "1538985492197746", "1537290784048665", "1537297574398457", "1533202432548729", "1537845616798505",
        "1537298030699722", "1533112863197144", "1537845290697996", "1542194135398242", "1538448739697143",
        "1537292745197772", "1542798966448066", "1533151621047706", "1533201470948018", "1533151490447282",
        "1537297673148878", "1535478732548287", "1535639661149817", "1533582606697759", "1534867103199100",
        "1537298175699186", "1533112948047251", "1535478536446553", "1537932125947573", "1535639743298733",
        "1535573303948645", "1537297687148194", "1538984451898628", "1537287050451585", "1542193253447056",
        "1537286916449787", "1533280926199994", "1531713383948093", "1535659406797724", "1535487687296439",
        "1535478867949633", "1537293225300092", "1537845628646841", "1542799160647687", "1537289857548079",
        "1533201593949092", "1535489323946970", "1533280289449672", "1531886330948237", "1537295921399240",
        "1535658901048600", "1538986164797677", "1535487845397220", "1535385098400887", "1537292612297038",
        "1535488471296946", "1534968042448006", "1533202614150287", "1537298222048295", "1535657466149105",
        "1535488348696255", "1533112803297900", "1542800871447242", "1537295947398884", "1533583053697111",
        "1533112619197293", "1526915265447261", "1532402173448300", "1538033894797973", "1538986174197439",
        "1533107791696948", "1542193186447638", "1538985962146551", "1532662170449093", "1533151722898031",
        "1532621750148578", "1538448851147954", "1538985931797335", "1538984244946513", "1535478747946733",
        "1542193792148167", "1542193794698419", "1537845370197632", "1537298076699860", "1537296521549233",
        "1537297342949177", "1542194374697894", "1533153556898236", "1533280293399480", "1537296933899521",
        "1532402194895946", "1533583260696175", "1535573831449092", "1538033689697296", "1535658849048223",
        "1533202663198952", "1537852965647912", "1535573090448420", "1542800635698179", "1533201747949093",
        "1542801023898417", "1537289866648173", "1533115040798333", "1531886324549363", "1537297200449110",
        "1542801655797174", "1537290015198738", "1535573093950361", "1535488190198490", "1535729026448358",
        "1533281529699213", "1542800518448439", "1535729287046411", "1533113384446771", "1535729331897050",
        "1531281645698628", "1535658891148969", "1537298058398581", "1533280314200205", "1533151524948156",
        "1542799750047559", "1537290838300124", "1533281741301095", "1537297439949479", "1535574121797012",
        "1538985033897081", "1531713369298701", "1533115161197184", "1533583322146665", "1533202656799543",
        "1532402739196326", "1542193264048407", "1542798728448386", "1538985939197607", "1537297835300101",
        "1535729561947330", "1537297981198689", "1535728963947813", "1526915782698607", "1542801021047908",
        "1533112940447890", "1538033866897263", "1538984237446737", "1535656875798715", "1542801727897683",
        "1532402584946961", "1537299143399713", "1542193851697006", "1535729033399075", "1532400351198130",
        "1532402046448260", "1535729911896630", "1532400190698090", "1535487479197014", "1537845521447214",
        "1537290172449593", "1533582778796768"
    ]
    
    # 6个最优角度（基于FOV重叠分析）
    target_angles = [-145.0, -82.5, -27.5, 27.5, 82.5, 145.0]
    
    print(f"时间戳数量: {len(timestamps)}")
    print(f"每个时间戳生成角度: {target_angles}")
    print(f"总共将生成: {len(timestamps) * len(target_angles)} 张图像")
    print(f"输出目录: {output_root}")
    print(f"使用方法: 简单拼接（左图右半 + 右图左半）")
    
    # 检查路径
    if not os.path.exists(real_data_path):
        print(f"错误: 找不到数据路径 {real_data_path}")
        return
    
    # 创建输出目录
    os.makedirs(output_root, exist_ok=True)
    
    # 多进程设置
    max_processes = mp.cpu_count()
    conservative_processes = min(max_processes // 2, 16)  # 保守一些，避免内存问题
    print(f"使用 {conservative_processes} 个进程并行处理")
    
    # 准备多进程参数
    process_args = [(timestamp, real_data_path, output_root, target_angles) 
                   for timestamp in timestamps]
    
    # 多进程处理
    start_time = time.time()
    
    print(f"\n开始生成图像...")
    
    with mp.Pool(processes=conservative_processes) as pool:
        results = list(tqdm(
            pool.imap_unordered(process_single_timestamp_simple_stitch, process_args),
            total=len(process_args),
            desc="生成简单拼接图像"
        ))
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # 统计结果
    total_images = sum(result[0] for result in results)
    successful_timestamps = sum(1 for result in results if result[0] > 0)
    
    print(f"\n=== 生成完成！ ===")
    print(f"总处理时间: {total_time/60:.1f} 分钟")
    print(f"成功处理时间戳: {successful_timestamps}/{len(timestamps)}")
    print(f"总生成图像数: {total_images}")
    print(f"平均处理速度: {total_images/total_time:.1f} 张/秒")
    print(f"输出目录: {output_root}")
    
    # 显示失败的时间戳
    failed_timestamps = [timestamps[i] for i, result in enumerate(results) if result[0] == 0]
    if failed_timestamps:
        print(f"\n处理失败的时间戳 ({len(failed_timestamps)}个):")
        for i, timestamp in enumerate(failed_timestamps[:10]):  # 只显示前10个
            print(f"  {i+1}. {timestamp}")
        if len(failed_timestamps) > 10:
            print(f"  ... 还有 {len(failed_timestamps)-10} 个")


def generate_167_timestamps_images():
    """
    为167个时间戳生成纯几何变换图像
    """
    print("=== 生成167个时间戳的纯几何变换图像 ===")
    
    real_data_path = "/mnt/vdb1/lyt/localdata/nuscenes/processed_data"
    output_root = "timestamp_images_geometric"
    
    # 167个时间戳列表
    timestamps = [
        "1537297676650253", "1542800689447619", "1542800309448236", "1538033755447568", "1535657545049969",
        "1538448758547578", "1535729331397165", "1531885992048261", "1538986259296943", "1535489941196216",
        "1535488129298125", "1542799639197823", "1537286922401853", "1533115035397015", "1538985953297524",
        "1538985492197746", "1537290784048665", "1537297574398457", "1533202432548729", "1537845616798505",
        "1537298030699722", "1533112863197144", "1537845290697996", "1542194135398242", "1538448739697143",
        "1537292745197772", "1542798966448066", "1533151621047706", "1533201470948018", "1533151490447282",
        "1537297673148878", "1535478732548287", "1535639661149817", "1533582606697759", "1534867103199100",
        "1537298175699186", "1533112948047251", "1535478536446553", "1537932125947573", "1535639743298733",
        "1535573303948645", "1537297687148194", "1538984451898628", "1537287050451585", "1542193253447056",
        "1537286916449787", "1533280926199994", "1531713383948093", "1535659406797724", "1535487687296439",
        "1535478867949633", "1537293225300092", "1537845628646841", "1542799160647687", "1537289857548079",
        "1533201593949092", "1535489323946970", "1533280289449672", "1531886330948237", "1537295921399240",
        "1535658901048600", "1538986164797677", "1535487845397220", "1535385098400887", "1537292612297038",
        "1535488471296946", "1534968042448006", "1533202614150287", "1537298222048295", "1535657466149105",
        "1535488348696255", "1533112803297900", "1542800871447242", "1537295947398884", "1533583053697111",
        "1533112619197293", "1526915265447261", "1532402173448300", "1538033894797973", "1538986174197439",
        "1533107791696948", "1542193186447638", "1538985962146551", "1532662170449093", "1533151722898031",
        "1532621750148578", "1538448851147954", "1538985931797335", "1538984244946513", "1535478747946733",
        "1542193792148167", "1542193794698419", "1537845370197632", "1537298076699860", "1537296521549233",
        "1537297342949177", "1542194374697894", "1533153556898236", "1533280293399480", "1537296933899521",
        "1532402194895946", "1533583260696175", "1535573831449092", "1538033689697296", "1535658849048223",
        "1533202663198952", "1537852965647912", "1535573090448420", "1542800635698179", "1533201747949093",
        "1542801023898417", "1537289866648173", "1533115040798333", "1531886324549363", "1537297200449110",
        "1542801655797174", "1537290015198738", "1535573093950361", "1535488190198490", "1535729026448358",
        "1533281529699213", "1542800518448439", "1535729287046411", "1533113384446771", "1535729331897050",
        "1531281645698628", "1535658891148969", "1537298058398581", "1533280314200205", "1533151524948156",
        "1542799750047559", "1537290838300124", "1533281741301095", "1537297439949479", "1535574121797012",
        "1538985033897081", "1531713369298701", "1533115161197184", "1533583322146665", "1533202656799543",
        "1532402739196326", "1542193264048407", "1542798728448386", "1538985939197607", "1537297835300101",
        "1535729561947330", "1537297981198689", "1535728963947813", "1526915782698607", "1542801021047908",
        "1533112940447890", "1538033866897263", "1538984237446737", "1535656875798715", "1542801727897683",
        "1532402584946961", "1537299143399713", "1542193851697006", "1535729033399075", "1532400351198130",
        "1532402046448260", "1535729911896630", "1532400190698090", "1535487479197014", "1537845521447214",
        "1537290172449593", "1533582778796768"
    ]
    
    # 6个最优角度（基于FOV重叠分析）
    target_angles = [-145.0, -82.5, -27.5, 27.5, 82.5, 145.0]
    
    print(f"时间戳数量: {len(timestamps)}")
    print(f"每个时间戳生成角度: {target_angles}")
    print(f"总共将生成: {len(timestamps) * len(target_angles)} 张图像")
    print(f"输出目录: {output_root}")
    
    # 检查路径
    if not os.path.exists(real_data_path):
        print(f"错误: 找不到数据路径 {real_data_path}")
        return
    
    # 创建输出目录
    os.makedirs(output_root, exist_ok=True)
    
    # 多进程设置
    max_processes = mp.cpu_count()
    conservative_processes = min(max_processes // 2, 16)  # 保守一些，避免内存问题
    print(f"使用 {conservative_processes} 个进程并行处理")
    
    # 准备多进程参数
    process_args = [(timestamp, real_data_path, output_root, target_angles) 
                   for timestamp in timestamps]
    
    # 多进程处理
    start_time = time.time()
    
    print(f"\n开始生成图像...")
    
    with mp.Pool(processes=conservative_processes) as pool:
        results = list(tqdm(
            pool.imap_unordered(process_single_timestamp, process_args),
            total=len(process_args),
            desc="生成时间戳图像"
        ))
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # 统计结果
    total_images = sum(result[0] for result in results)
    successful_timestamps = sum(1 for result in results if result[0] > 0)
    
    print(f"\n=== 生成完成！ ===")
    print(f"总处理时间: {total_time/60:.1f} 分钟")
    print(f"成功处理时间戳: {successful_timestamps}/{len(timestamps)}")
    print(f"总生成图像数: {total_images}")
    print(f"平均处理速度: {total_images/total_time:.1f} 张/秒")
    print(f"输出目录: {output_root}")
    
    # 显示失败的时间戳
    failed_timestamps = [timestamps[i] for i, result in enumerate(results) if result[0] == 0]
    if failed_timestamps:
        print(f"\n处理失败的时间戳 ({len(failed_timestamps)}个):")
        for i, timestamp in enumerate(failed_timestamps[:10]):  # 只显示前10个
            print(f"  {i+1}. {timestamp}")
        if len(failed_timestamps) > 10:
            print(f"  ... 还有 {len(failed_timestamps)-10} 个")


def generate_single_scene_single_angle():
    """
    生成单个场景的单个角度图像 - 只用几何变换
    """
    print("=== 生成单个场景单个角度图像（纯几何变换） ===")
    
    real_data_path = "/mnt/vdb1/lyt/localdata/nuscenes/processed_data"
    target_scene = "scene_1535573303948645"  # 目标场景
    target_angle = 145.0  # 目标角度
    output_root = "output"
    
    # 检查路径
    if not os.path.exists(real_data_path):
        print(f"错误: 找不到数据路径 {real_data_path}")
        return
    
    # 找到场景文件夹（去掉scene_前缀）
    scene_folder = target_scene.replace("scene_", "")
    scene_path = os.path.join(real_data_path, scene_folder)
    
    if not os.path.exists(scene_path):
        print(f"错误: 找不到场景路径 {scene_path}")
        return
    
    print(f"目标场景: {target_scene}")
    print(f"目标角度: {target_angle}°")
    print(f"使用纯几何变换（无特征匹配和物体对齐）")
    
    # 相机配置
    camera_mapping = {
        'CAM_FRONT': 0,
        'CAM_FRONT_LEFT': 55,
        'CAM_FRONT_RIGHT': -55,
        'CAM_BACK_LEFT': 110,
        'CAM_BACK_RIGHT': -110,
        'CAM_BACK': 180,
    }
    
    ordered_cameras = ['CAM_FRONT_RIGHT', 'CAM_FRONT', 'CAM_FRONT_LEFT', 
                      'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    
    # 加载场景数据
    print(f"正在加载场景数据...")
    images = []
    angles = []
    
    for camera_name in ordered_cameras:
        image_files = [f for f in os.listdir(scene_path) 
                      if f.startswith(camera_name + '_n') and f.endswith('.jpg')]
        if image_files:
            image_path = os.path.join(scene_path, image_files[0])
            img = cv2.imread(image_path)
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img_rgb)
                angles.append(camera_mapping[camera_name])
                print(f"  加载 {camera_name}: {camera_mapping[camera_name]}°")
    
    if len(images) < 4:
        print(f"错误: 场景 {target_scene} 图像不足（只有{len(images)}张）")
        return
    
    print(f"成功加载 {len(images)} 张图像")
    
    # nuScenes相机角度配置
    vx = [-55, 0, 55, 110, 180, -110]  # 右前、前、左前、左后、后、右后
    vy = [0, 0, 0, 0, 0, 0]
    
    # 创建输出目录
    os.makedirs(output_root, exist_ok=True)
    
    print(f"\n开始生成目标角度图像...")
    start_time = time.time()
    
    try:
        # 使用纯几何变换（关闭物体对齐）
        result_img = warp_img_final_optimized(
            90, target_angle, 0, images, vx, vy, 
            angle_threshold=1.0, enable_object_alignment=False
        )
        
        # 格式化角度为文件名
        if target_angle >= 0:
            angle_str = f"pos_{target_angle:06.2f}"
        else:
            angle_str = f"neg_{abs(target_angle):06.2f}"
        
        # 保存图像
        filename = f"{angle_str}.jpg"
        filepath = os.path.join(output_root, filename)
        
        # 转换为BGR并保存
        result_bgr = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filepath, result_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"\n=== 生成完成！ ===")
        print(f"处理时间: {processing_time:.2f} 秒")
        print(f"输出文件: {filepath}")
        print(f"场景: {target_scene}")
        print(f"角度: {target_angle}°")
        print(f"使用方法: 纯几何变换")
        
        return True
        
    except Exception as e:
        print(f"生成失败: {e}")
        return False


def generate_middle_angle_images_batch():
    """
    批量预加载生成6个中间角度插值图像
    """
    print("=== 批量预加载训练集场景6个中间角度插值图像生成 ===")
    print("优化: 批量预加载 + 多进程 + 物体对齐")
    
    real_data_path = "/mnt/vdb1/lyt/localdata/nuscenes/processed_data"
    scenes_txt_path = "train_scenes_70percent.txt"
    output_root = "/mnt/vdc1/lyt/interpolation_6_angles"
    batch_size = 800  # 每批处理800个场景（约24GB内存）
    
    # 检查路径
    if not os.path.exists(real_data_path):
        print(f"错误: 找不到数据路径 {real_data_path}")
        return
    
    if not os.path.exists(scenes_txt_path):
        print(f"错误: 找不到场景列表文件 {scenes_txt_path}")
        return
    
    # 读取场景列表
    print("正在读取70%训练集场景列表...")
    train_scene_list = get_scene_list_from_txt(scenes_txt_path)
    print(f"找到 {len(train_scene_list)} 个训练集场景（70%）")
    
    # 计算中间角度
    middle_angles, middle_info = calculate_middle_angles()
    print(f"每个场景生成 {len(middle_angles)} 张图像")
    
    print("中间角度列表:")
    for i, info in enumerate(middle_info):
        print(f"  {i+1}. {info['angle']:7.1f}° ({info['between']})")
    
    # 创建输出目录
    os.makedirs(output_root, exist_ok=True)
    
    # 多进程设置
    max_processes = mp.cpu_count()
    conservative_processes = min(max_processes // 3, 20)
    print(f"系统有 {max_processes} 个CPU核心")
    print(f"使用 {conservative_processes} 个进程并行处理（保守模式）")
    print(f"批处理大小: {batch_size} 个场景/批")
    
    # 分批处理
    total_batches = (len(train_scene_list) + batch_size - 1) // batch_size
    total_generated = 0
    total_successful = 0
    start_time = time.time()
    
    for batch_idx in range(total_batches):
        batch_start = batch_idx * batch_size
        batch_end = min((batch_idx + 1) * batch_size, len(train_scene_list))
        scene_batch = train_scene_list[batch_start:batch_end]
        
        print(f"\n=== 处理批次 {batch_idx + 1}/{total_batches} ===")
        print(f"场景范围: {batch_start + 1}-{batch_end} ({len(scene_batch)} 个场景)")
        
        # 批量加载当前批次的场景
        print(f"正在批量加载 {len(scene_batch)} 个场景到内存...")
        batch_data = load_batch_scenes(real_data_path, scene_batch)
        print(f"成功加载 {len(batch_data)} 个场景")
        
        if not batch_data:
            print("批次数据为空，跳过...")
            continue
        
        # 准备多进程参数
        process_args = [(scene_name, batch_data.get(scene_name), output_root, middle_angles) 
                       for scene_name in scene_batch]
        
        # 多进程处理当前批次
        batch_start_time = time.time()
        
        with mp.Pool(processes=conservative_processes) as pool:
            results = list(tqdm(
                pool.imap_unordered(process_scene_from_memory, process_args),
                total=len(process_args),
                desc=f"批次{batch_idx + 1}生成插值"
            ))
        
        batch_end_time = time.time()
        
        # 统计当前批次结果
        batch_generated = sum(results)
        batch_successful = sum(1 for r in results if r > 0)
        batch_time = batch_end_time - batch_start_time
        
        total_generated += batch_generated
        total_successful += batch_successful
        
        print(f"批次 {batch_idx + 1} 完成:")
        print(f"  处理时间: {batch_time:.1f} 秒")
        print(f"  成功场景: {batch_successful}/{len(scene_batch)}")
        print(f"  生成图像: {batch_generated} 张")
        print(f"  处理速度: {batch_generated/batch_time:.1f} 张/秒")
        
        # 清理内存
        del batch_data
        gc.collect()
        
        # 显示总体进度
        elapsed_time = time.time() - start_time
        processed_scenes = batch_end
        remaining_scenes = len(train_scene_list) - processed_scenes
        if processed_scenes > 0:
            eta = elapsed_time * remaining_scenes / processed_scenes
            print(f"总体进度: {processed_scenes}/{len(train_scene_list)} ({processed_scenes/len(train_scene_list)*100:.1f}%)")
            print(f"预计剩余时间: {eta/3600:.1f} 小时")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\n=== 全部完成！ ===")
    print(f"总处理时间: {total_time/3600:.1f} 小时")
    print(f"成功处理: {total_successful}/{len(train_scene_list)} 个场景")
    print(f"生成图像: {total_generated} 张")
    print(f"平均速度: {total_generated/total_time:.1f} 张/秒")
    print(f"输出目录: {output_root}")
    
    # 输出角度信息
    print(f"\n生成的6个中间角度:")
    for i, info in enumerate(middle_info):
        angle = info['angle']
        between = info['between']
        print(f"  {i+1}. {angle:7.1f}° (在 {between} 之间)")

def test_single_simple_stitch():
    """
    测试单个时间戳的简单拼接功能
    """
    print("=== 测试单个时间戳的简单拼接功能 ===")
    
    real_data_path = "/mnt/vdb1/lyt/localdata/nuscenes/processed_data"
    output_root = "test_simple_stitch_corrected"
    target_timestamp = "1535573303948645"
    target_angles = [-145.0, -82.5, -27.5, 27.5, 82.5, 145.0]  # 测试所有角度
    
    # 创建测试参数
    args = (target_timestamp, real_data_path, output_root, target_angles)
    
    # 调用处理函数
    result_count, result_msg = process_single_timestamp_simple_stitch(args)
    
    print(f"测试结果: {result_msg}")
    if result_count > 0:
        print(f"成功生成 {result_count} 张图像")
        print(f"输出目录: {output_root}/timestamp_{target_timestamp}/")
    else:
        print("测试失败")

def generate_single_timestamp_24_angles_geometric():
    """
    为指定时间戳生成24个插值角度的纯几何拼接图像
    """
    print("=== 生成指定时间戳的24个插值角度图像（纯几何拼接） ===")
    
    real_data_path = "/mnt/vdb1/lyt/localdata/nuscenes/processed_data"
    output_root = "/mnt/vdb1/lyt/interpolation_24_angles_geometric"
    target_timestamp = "1531883992949188"
    
    # 检查路径
    if not os.path.exists(real_data_path):
        print(f"错误: 找不到数据路径 {real_data_path}")
        return
    
    # 场景路径
    scene_path = os.path.join(real_data_path, target_timestamp)
    
    if not os.path.exists(scene_path):
        print(f"错误: 找不到场景路径 {scene_path}")
        return
    
    print(f"目标时间戳: {target_timestamp}")
    print(f"使用纯几何拼接（无特征匹配和物体对齐）")
    
    # 相机配置
    camera_mapping = {
        'CAM_FRONT': 0,
        'CAM_FRONT_LEFT': 55,
        'CAM_FRONT_RIGHT': -55,
        'CAM_BACK_LEFT': 110,
        'CAM_BACK_RIGHT': -110,
        'CAM_BACK': 180,
    }
    
    ordered_cameras = ['CAM_FRONT_RIGHT', 'CAM_FRONT', 'CAM_FRONT_LEFT', 
                      'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    
    # 加载场景数据
    print(f"正在加载场景数据...")
    images = []
    angles = []
    
    for camera_name in ordered_cameras:
        image_files = [f for f in os.listdir(scene_path) 
                      if f.startswith(camera_name + '_n') and f.endswith('.jpg')]
        if image_files:
            image_path = os.path.join(scene_path, image_files[0])
            img = cv2.imread(image_path)
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img_rgb)
                angles.append(camera_mapping[camera_name])
                print(f"  加载 {camera_name}: {camera_mapping[camera_name]}°")
    
    if len(images) < 4:
        print(f"错误: 场景 {target_timestamp} 图像不足（只有{len(images)}张）")
        return
    
    print(f"成功加载 {len(images)} 张图像")
    
    # 计算24个插值角度
    interpolation_angles, interpolation_info = calculate_24_interpolation_angles()
    print(f"将生成 {len(interpolation_angles)} 张插值图像")
    
    # 显示插值角度信息
    print(f"\n插值角度列表:")
    for i, info in enumerate(interpolation_info):
        angle = info['angle']
        between = info['between']
        step = info['step']
        print(f"  {i+1:2d}. {angle:7.1f}° (步骤{step}/4, {between})")
    
    # nuScenes相机角度配置
    vx = [-55, 0, 55, 110, 180, -110]  # 右前、前、左前、左后、后、右后
    vy = [0, 0, 0, 0, 0, 0]
    
    # 创建输出目录
    timestamp_dir = os.path.join(output_root, f"timestamp_{target_timestamp}")
    os.makedirs(timestamp_dir, exist_ok=True)
    
    print(f"\n开始生成插值图像...")
    start_time = time.time()
    
    generated_count = 0
    
    # 生成每个插值角度的图像
    for idx, target_angle in enumerate(interpolation_angles):
        try:
            # 使用纯几何变换（关闭物体对齐）
            result_img = warp_img_final_optimized(
                90, target_angle, 0, images, vx, vy, 
                angle_threshold=1.0, enable_object_alignment=False
            )
            
            # 格式化角度为文件名
            if target_angle >= 0:
                angle_str = f"pos_{target_angle:06.2f}"
            else:
                angle_str = f"neg_{abs(target_angle):06.2f}"
            
            # 保存图像
            filename = f"{angle_str}.jpg"
            filepath = os.path.join(timestamp_dir, filename)
            
            # 转换为BGR并保存
            result_bgr = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(filepath, result_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            generated_count += 1
            
            # 显示进度
            print(f"  进度: {idx+1}/{len(interpolation_angles)} - 角度 {target_angle:7.1f}° -> {filename}")
            
        except Exception as e:
            print(f"  错误: 角度 {target_angle}° 生成失败: {e}")
            continue
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    print(f"\n=== 生成完成！ ===")
    print(f"处理时间: {processing_time:.2f} 秒")
    print(f"成功生成: {generated_count}/{len(interpolation_angles)} 张图像")
    print(f"输出目录: {timestamp_dir}")
    print(f"时间戳: {target_timestamp}")
    print(f"使用方法: 纯几何拼接")
    
    if generated_count == len(interpolation_angles):
        print(f"✓ 所有 {len(interpolation_angles)} 张图像生成成功！")
    else:
        print(f"⚠ 有 {len(interpolation_angles) - generated_count} 张图像生成失败")
    
    return generated_count

def generate_single_timestamp_24_angles_simple_stitch():
    """
    为指定时间戳生成24个插值角度的简单拼接图像
    """
    print("=== 生成指定时间戳的24个插值角度图像（简单拼接） ===")
    
    real_data_path = "/mnt/vdb1/lyt/localdata/nuscenes/processed_data"
    output_root = "/mnt/vdb1/lyt/interpolation_24_angles_simple_stitch"
    target_timestamp = "1531883992949188"
    
    # 检查路径
    if not os.path.exists(real_data_path):
        print(f"错误: 找不到数据路径 {real_data_path}")
        return
    
    # 场景路径
    scene_path = os.path.join(real_data_path, target_timestamp)
    
    if not os.path.exists(scene_path):
        print(f"错误: 找不到场景路径 {scene_path}")
        return
    
    print(f"目标时间戳: {target_timestamp}")
    print(f"使用简单拼接（1/5 + 4/5 或 4/5 + 1/5）")
    
    # 相机配置
    camera_mapping = {
        'CAM_FRONT': 0,
        'CAM_FRONT_LEFT': 55,
        'CAM_FRONT_RIGHT': -55,
        'CAM_BACK_LEFT': 110,
        'CAM_BACK_RIGHT': -110,
        'CAM_BACK': 180,
    }
    
    ordered_cameras = ['CAM_FRONT_RIGHT', 'CAM_FRONT', 'CAM_FRONT_LEFT', 
                      'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    
    # 加载场景数据
    print(f"正在加载场景数据...")
    images_dict = {}
    
    for camera_name in ordered_cameras:
        image_files = [f for f in os.listdir(scene_path) 
                      if f.startswith(camera_name + '_n') and f.endswith('.jpg')]
        if image_files:
            image_path = os.path.join(scene_path, image_files[0])
            img = cv2.imread(image_path)
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                angle = camera_mapping[camera_name]
                images_dict[angle] = img_rgb
                print(f"  加载 {camera_name}: {camera_mapping[camera_name]}°")
    
    if len(images_dict) < 4:
        print(f"错误: 场景 {target_timestamp} 图像不足（只有{len(images_dict)}张）")
        return
    
    print(f"成功加载 {len(images_dict)} 张图像")
    
    # 计算24个插值角度
    interpolation_angles, interpolation_info = calculate_24_interpolation_angles()
    print(f"将生成 {len(interpolation_angles)} 张插值图像")
    
    # 显示插值角度信息
    print(f"\n插值角度列表:")
    for i, info in enumerate(interpolation_info):
        angle = info['angle']
        between = info['between']
        step = info['step']
        print(f"  {i+1:2d}. {angle:7.1f}° (步骤{step}/4, {between})")
    
    # 创建输出目录
    timestamp_dir = os.path.join(output_root, f"timestamp_{target_timestamp}")
    os.makedirs(timestamp_dir, exist_ok=True)
    
    print(f"\n开始生成插值图像...")
    start_time = time.time()
    
    generated_count = 0
    
    # 定义拼接规则：目标角度 -> (左图角度, 右图角度, 左图比例, 右图比例)
    # 每对相机4个插值点，使用不同的拼接比例
    # 注意：左图是角度较小的，右图是角度较大的
    stitch_rules = {
        # CAM_FRONT_RIGHT (-55°) -> CAM_FRONT (0°)
        -44.0: (0, -55, 0.2, 0.8),    # 步骤1: 0°左1/5 + (-55°)右4/5
        -33.0: (0, -55, 0.4, 0.6),    # 步骤2: 0°左2/5 + (-55°)右3/5
        -22.0: (0, -55, 0.6, 0.4),    # 步骤3: 0°左3/5 + (-55°)右2/5
        -11.0: (0, -55, 0.8, 0.2),    # 步骤4: 0°左4/5 + (-55°)右1/5
        
        # CAM_FRONT (0°) -> CAM_FRONT_LEFT (55°)
        11.0: (55, 0, 0.2, 0.8),      # 步骤1: 55°左1/5 + 0°右4/5
        22.0: (55, 0, 0.4, 0.6),      # 步骤2: 左图2/5 + 右图3/5
        33.0: (55, 0, 0.6, 0.4),      # 步骤3: 左图3/5 + 右图2/5
        44.0: (55, 0, 0.8, 0.2),      # 步骤4: 55°右4/5 + 0°左1/5
        
        # CAM_FRONT_LEFT (55°) -> CAM_BACK_LEFT (110°)
        66.0: (110, 55, 0.2, 0.8),    # 步骤1: 左图1/5 + 右图4/5
        77.0: (110, 55, 0.4, 0.6),    # 步骤2: 左图2/5 + 右图3/5
        88.0: (110, 55, 0.6, 0.4),    # 步骤3: 左图3/5 + 右图2/5
        99.0: (110, 55, 0.8, 0.2),    # 步骤4: 110°右4/5 + 55°左1/5
        
        # CAM_BACK_LEFT (110°) -> CAM_BACK (180°)
        124.0: (180, 110, 0.2, 0.8),    # 步骤1: 0°左1/5 + 110°右4/5
        138.0: (180, 110, 0.4, 0.6),  # 步骤2: 左图2/5 + 右图3/5
        152.0: (180, 110, 0.6, 0.4),  # 步骤3: 左图3/5 + 右图2/5
        166.0: (180, 110, 0.8, 0.2),  # 步骤4: 180°右4/5 + 110°左1/5
        
        # CAM_BACK (180°) -> CAM_BACK_RIGHT (-110°)
        -166.0: (-110, 180, 0.2, 0.8), # 步骤1: (-110°)左1/5 + 180°右4/5
        -152.0: (-110, 180, 0.4, 0.6), # 步骤2: (-110°)左2/5 + 180°右3/5
        -138.0: (-110, 180, 0.6, 0.4), # 步骤3: (-110°)左3/5 + 180°右2/5
        -124.0: (-110, 180, 0.8, 0.2), # 步骤4: (-110°)左4/5 + 180°右1/5
        
        # CAM_BACK_RIGHT (-110°) -> CAM_FRONT_RIGHT (-55°)
        -66.0: (-55, -110, 0.8, 0.2), # 步骤1: (-55°)右4/5 + (-110°)左1/5
        -77.0: (-55, -110, 0.6, 0.4), # 步骤2: (-55°)右3/5 + (-110°)左2/5
        -88.0: (-55, -110, 0.4, 0.6), # 步骤3: (-55°)右2/5 + (-110°)左3/5
        -99.0: (-55, -110, 0.2, 0.8), # 步骤4: (-55°)左1/5 + (-110°)右4/5
    }
    
    # 生成每个目标角度的图像
    for idx, target_angle in enumerate(interpolation_angles):
        try:
            if target_angle in stitch_rules:
                left_angle, right_angle, left_fraction, right_fraction = stitch_rules[target_angle]
                
                # 检查是否有对应的图像
                if left_angle in images_dict and right_angle in images_dict:
                    left_img = images_dict[left_angle]
                    right_img = images_dict[right_angle]
                    
                    # 使用新的拼接方法
                    if target_angle in [-66.0, -77.0, -88.0, -99.0]:
                        # 对于负角度区域，使用反向拼接
                        # 我们想要：(-55°)右4/5 + (-110°)左1/5
                        # 所以：img2=(-55°)右4/5 + img1=(-110°)左1/5
                        result_img = simple_fraction_stitch_reversed(right_img, left_img, left_fraction, right_fraction)
                    elif target_angle in [-11.0, -22.0, -33.0, -44.0]:
                        # 对于负角度区域，使用正常拼接
                        # 我们想要：0°右4/5 + (-55°)左1/5
                        # 所以：img1=0°右4/5 + img2=(-55°)左1/5
                        result_img = simple_fraction_stitch(left_img, right_img, left_fraction, right_fraction)
                    else:
                        # 对于其他角度，使用正常拼接
                        result_img = simple_fraction_stitch(left_img, right_img, left_fraction, right_fraction)
                    
                    # 格式化角度为文件名
                    if target_angle >= 0:
                        angle_str = f"pos_{target_angle:06.2f}"
                    else:
                        angle_str = f"neg_{abs(target_angle):06.2f}"
                    
                    # 保存图像
                    filename = f"{angle_str}.jpg"
                    filepath = os.path.join(timestamp_dir, filename)
                    
                    # 转换为BGR并保存
                    result_bgr = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(filepath, result_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    
                    generated_count += 1
                    
                    # 显示进度
                    print(f"  进度: {idx+1}/{len(interpolation_angles)} - 角度 {target_angle:7.1f}° -> {filename} (拼接: {left_fraction:.1f}+{right_fraction:.1f})")
                else:
                    print(f"  错误: 缺少角度 {left_angle}° 或 {right_angle}° 的图像")
            else:
                print(f"  错误: 角度 {target_angle}° 没有对应的拼接规则")
                
        except Exception as e:
            print(f"  错误: 角度 {target_angle}° 生成失败: {e}")
            continue
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    print(f"\n=== 生成完成！ ===")
    print(f"处理时间: {processing_time:.2f} 秒")
    print(f"成功生成: {generated_count}/{len(interpolation_angles)} 张图像")
    print(f"输出目录: {timestamp_dir}")
    print(f"时间戳: {target_timestamp}")
    print(f"使用方法: 简单拼接（1/5 + 4/5 或 4/5 + 1/5）")
    
    if generated_count == len(interpolation_angles):
        print(f"✓ 所有 {len(interpolation_angles)} 张图像生成成功！")
    else:
        print(f"⚠ 有 {len(interpolation_angles) - generated_count} 张图像生成失败")
    
    return generated_count

if __name__ == "__main__":
    # 设置多进程启动方法
    mp.set_start_method('spawn', force=True)
    
    # 生成指定时间戳的24个插值角度图像（简单拼接）
    generate_single_timestamp_24_angles_simple_stitch() 