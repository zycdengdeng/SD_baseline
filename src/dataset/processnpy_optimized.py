import os
import cv2
import numpy as np
from tqdm import tqdm
from itertools import product
from multiprocessing import Pool, cpu_count, Manager
from functools import partial
import pickle
import hashlib
import logging
import traceback
import gc

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('processnpy.log')
    ]
)

def get_K_R(FOV, THETA, PHI, height, width):
    f = 0.5 * width * 1 / np.tan(0.5 * FOV / 180.0 * np.pi)
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    K = np.array([
        [f, 0, cx],
        [0, f, cy],
        [0, 0,  1],
    ], np.float32)

    y_axis = np.array([0.0, 1.0, 0.0], np.float32)
    x_axis = np.array([1.0, 0.0, 0.0], np.float32)
    R1, _ = cv2.Rodrigues(y_axis * np.radians(THETA))
    R2, _ = cv2.Rodrigues(np.dot(R1, x_axis) * np.radians(PHI))
    R = R2 @ R1
    return K, R

def parse_angle_from_filename(filename):
    name = os.path.basename(filename).replace('.jpg', '')
    if name.startswith("pos_"):
        return float(name[4:])
    elif name.startswith("neg_"):
        return -float(name[4:])
    else:
        raise ValueError(f"Unexpected filename format: {filename}")

def load_image_cached(img_path, cache_dict):
    if img_path in cache_dict:
        return cache_dict[img_path]
    
    try:
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"无法读取图像: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cache_dict[img_path] = img
        return img
    except Exception as e:
        logging.error(f"读取图像失败 {img_path}: {str(e)}")
        raise

def process_single_scene_optimized(args):
    scene_dir, real_angles, fov, num_left, num_right, use_cache = args
    
    scene_name = os.path.basename(scene_dir)
    # 只记录到日志文件，不输出到控制台
    logging.info(f"开始处理场景: {scene_name}")
    
    try:
        scene_results = []
        all_files = [f for f in os.listdir(scene_dir) if f.endswith(".jpg")]
        
        if not all_files:
            logging.warning(f"场景 {scene_name} 中没有找到jpg文件")
            return []
            
        angle_to_file = {}
        failed_files = []
        
        img_cache = {}
        if use_cache:
            logging.info(f"场景 {scene_name}: 预加载 {len(all_files)} 张图像")
            for f in all_files:
                try:
                    angle = parse_angle_from_filename(f)
                    angle_to_file[angle] = f
                    img_path = os.path.join(scene_dir, f)
                    load_image_cached(img_path, img_cache)
                except Exception as e:
                    failed_files.append(f)
                    logging.error(f"场景 {scene_name} 文件 {f} 处理失败: {str(e)}")
                    continue
        else:
            for f in all_files:
                try:
                    angle = parse_angle_from_filename(f)
                    angle_to_file[angle] = f
                except Exception as e:
                    failed_files.append(f)
                    logging.error(f"场景 {scene_name} 文件 {f} 解析失败: {str(e)}")
                    continue

        if failed_files:
            logging.warning(f"场景 {scene_name} 有 {len(failed_files)} 个文件处理失败: {failed_files}")

        all_angles = sorted(angle_to_file.keys())
        if not all_angles:
            logging.error(f"场景 {scene_name} 没有有效的角度文件")
            return []

        logging.info(f"场景 {scene_name}: 找到 {len(all_angles)} 个有效角度")

        for real_angle in real_angles:
            if real_angle not in angle_to_file:
                logging.warning(f"场景 {scene_name} 缺少角度 {real_angle}")
                continue

            real_file = angle_to_file[real_angle]
            real_path = os.path.join(scene_dir, real_file)
            
            try:
                if use_cache:
                    real_img = img_cache[real_path]
                else:
                    real_img = cv2.imread(real_path)
                    if real_img is None:
                        raise ValueError(f"无法读取图像: {real_path}")
                    real_img = cv2.cvtColor(real_img, cv2.COLOR_BGR2RGB)
                    
                h, w, _ = real_img.shape
                
                def angle_diff(a, b):
                    return ((b - a + 180) % 360) - 180
                    
                left_angles = sorted([a for a in all_angles if angle_diff(real_angle, a) < 0],
                         key=lambda x: abs(angle_diff(real_angle, x)))[:num_left]
                right_angles = sorted([a for a in all_angles if angle_diff(real_angle, a) > 0],
                          key=lambda x: abs(angle_diff(real_angle, x)))[:num_right]

                if len(left_angles) < num_left or len(right_angles) < num_right:
                    logging.warning(f"场景 {scene_name} 角度 {real_angle} 的左右角度不足: 左{len(left_angles)}/{num_left}, 右{len(right_angles)}/{num_right}")
                    continue

                left_Ks, left_Rs = [], []
                right_Ks, right_Rs = [], []

                for angle in left_angles:
                    K, R = get_K_R(fov, angle - real_angle, 0, h, w)
                    left_Ks.append(K)
                    left_Rs.append(R)

                for angle in right_angles:
                    K, R = get_K_R(fov, angle - real_angle, 0, h, w)
                    right_Ks.append(K)
                    right_Rs.append(R)

                for i, j in product(range(num_left), range(num_right)):
                    scene_results.append({
                        'scene': scene_name,
                        'real_img_path': real_path,
                        'real_angle': real_angle,
                        'cond_img_paths': [
                            os.path.join(scene_dir, angle_to_file[left_angles[i]]),
                            os.path.join(scene_dir, angle_to_file[right_angles[j]])
                        ],
                        'Ks': [left_Ks[i], right_Ks[j]],
                        'Rs': [left_Rs[i], right_Rs[j]],
                        'cond_angles': [left_angles[i], right_angles[j]]
                    })
                    
            except Exception as e:
                logging.error(f"场景 {scene_name} 处理角度 {real_angle} 时出错: {str(e)}")
                continue
        
        logging.info(f"场景 {scene_name} 处理完成，生成 {len(scene_results)} 个样本")
        
        # 清理内存
        if use_cache:
            del img_cache
            gc.collect()
            
        return scene_results
        
    except Exception as e:
        logging.error(f"场景 {scene_name} 处理失败: {str(e)}")
        logging.error(traceback.format_exc())
        return []

def generate_condition_pairs_all_scenes_optimized(
    parent_dir,
    save_path,
    real_angles=[0, 55, 110, 180, -110, -55],
    fov=90,
    num_left=3,
    num_right=3,
    n_processes=None,
    use_cache=True
    ):
    
    if n_processes is None:
        n_processes = min(int(cpu_count() * 0.8), 80)
    
    scene_dirs = sorted([os.path.join(parent_dir, d) for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d))])
    
    print(f"开始处理数据集")
    print(f"使用 {n_processes} 个进程处理 {len(scene_dirs)} 个场景")
    print(f"图像缓存: {'启用' if use_cache else '禁用'}")
    print(f"CPU核心数: {cpu_count()}, 内存: 456GB")
    print()
    
    args_list = [(scene_dir, real_angles, fov, num_left, num_right, use_cache) 
                 for scene_dir in scene_dirs]
    
    all_results = []
    failed_scenes = []
    
    with Pool(processes=n_processes) as pool:
        results = list(tqdm(
            pool.imap(process_single_scene_optimized, args_list),
            total=len(scene_dirs),
            desc="Processing scenes"
        ))
        
        for i, scene_results in enumerate(results):
            if scene_results:
                all_results.extend(scene_results)
            else:
                failed_scenes.append(os.path.basename(scene_dirs[i]))
    
    if failed_scenes:
        print(f"⚠️  有 {len(failed_scenes)} 个场景处理失败，详情请查看日志文件")
        logging.warning(f"有 {len(failed_scenes)} 个场景处理失败: {failed_scenes}")
    
    if not all_results:
        print("❌ 没有生成任何有效结果！")
        logging.error("没有生成任何有效结果！")
        return
    
    np.random.shuffle(all_results)
    split_idx = int(0.7 * len(all_results))
    train_results = all_results[:split_idx]
    test_results = all_results[split_idx:]

    base_path = os.path.splitext(save_path)[0]
    os.makedirs(os.path.dirname(base_path), exist_ok=True)
    np.save(base_path + "_train.npy", train_results)
    np.save(base_path + "_test.npy", test_results)

    print(f"✅ 保存 {len(train_results)} 个训练样本到 {base_path}_train.npy")
    print(f"✅ 保存 {len(test_results)} 个测试样本到 {base_path}_test.npy")
    print(f"总计处理 {len(all_results)} 个样本")
    
    logging.info(f"✅ 保存 {len(train_results)} 个训练样本到 {base_path}_train.npy")
    logging.info(f"✅ 保存 {len(test_results)} 个测试样本到 {base_path}_test.npy")
    logging.info(f"总计处理 {len(all_results)} 个样本")

if __name__ == "__main__":
    generate_condition_pairs_all_scenes_optimized(
        parent_dir="/mnt/vdc1/lyt/interpolation_results",
        save_path="/mnt/vdb1/lyt/condition_pairs.npy",
        real_angles=[0, 55, 110, 180, -110, -55],
        fov=90,
        n_processes=80,
        use_cache=True
    ) 