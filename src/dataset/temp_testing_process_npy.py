import os
import cv2
import numpy as np
from itertools import combinations
from tqdm import tqdm

def normalize_angle(angle):
    angle = angle % 360
    if angle >= 180:
        angle -= 360
    return angle

def calculate_angle_midpoint(angle1, angle2):
    angle1 = normalize_angle(angle1)
    angle2 = normalize_angle(angle2)
    
    #计算两个角度之间的差值
    diff = normalize_angle(angle2 - angle1)
    
    #中点就是第一个角度加上差值的一半
    midpoint = normalize_angle(angle1 + diff / 2)
    
    return midpoint

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
    
def generate_pairs_with_midpoint_axis(
    parent_dir,
    save_path,
    real_angles=[0, 55, 110, 180, -110, -55],
    fov=90
):
    results = []
    max_pairs = 1000000  # 生成全部数据 

    scene_dirs = sorted([d for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d))])
    cnt=0
    for scene_dir in tqdm(scene_dirs, desc="Processing scenes"):
        cnt+=1
        if cnt<=20:
            continue
        scene_path = os.path.join(parent_dir, scene_dir)
        files = [f for f in os.listdir(scene_path) if f.endswith(".jpg")]
        angle_to_file = {}
        for f in files:
            try:
                angle = parse_angle_from_filename(f)
                angle_to_file[angle] = f
            except:
                continue
        
        sorted_angles = sorted(real_angles, key=lambda x: (x + 360) % 360)
        available_angles = set(angle_to_file.keys())
        
        pairs = []
        for i in range(len(sorted_angles)):
            left_angle = sorted_angles[i]
            right_angle = sorted_angles[(i + 1) % len(sorted_angles)]
            pairs.append((left_angle, right_angle))

        for left_angle, right_angle in pairs:
            if len(results) >= max_pairs:
                break
            if left_angle not in available_angles or right_angle not in available_angles:
                continue
            middle_angle = calculate_angle_midpoint(left_angle, right_angle)
            
            relative_left_angle = normalize_angle(left_angle - middle_angle)
            relative_right_angle = normalize_angle(right_angle - middle_angle)

            left_path = os.path.join(scene_path, angle_to_file[left_angle])
            right_path = os.path.join(scene_path, angle_to_file[right_angle])

            target_img = cv2.imread(left_path)
            target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
            h, w, _ = target_img.shape

            K_left, R_left = get_K_R(fov, relative_left_angle, 0, h, w)
            K_right, R_right = get_K_R(fov, relative_right_angle, 0, h, w)

            results.append({
                'scene': scene_dir,
                'real_angle': middle_angle,
                'cond_img_paths': [left_path, right_path],
                'cond_angles': [relative_left_angle, relative_right_angle],
                'Ks': [K_left, K_right],
                'Rs': [R_left, R_right]
            })
            
        if len(results) >= max_pairs:
            break  

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, results)
    print(f"✅ Saved {len(results)} test pairs with midpoint axes to {save_path}")

generate_pairs_with_midpoint_axis(
    parent_dir="/mnt/vdc1/lyt/interpolation_results",
    save_path="/mnt/vdb1/lyt/midpoint_condition_pairs_test.npy",
    real_angles=[0, 55, 110, 180, -110, -55],
    fov=90
)
