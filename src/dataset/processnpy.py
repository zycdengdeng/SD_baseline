import os
import cv2
import numpy as np
from tqdm import tqdm
from itertools import product
import numpy as np
import cv2

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

def get_depth_path(img_path, depth_suffix="_depth",
                   try_exts=(".png", ".npy", ".jpg", ".exr", ".pfm")):
    """
    根据图像路径自动推断对应深度文件路径：
    e.g. /a/pos_55.jpg -> /a/pos_55_depth.png (若存在)
    依次尝试 try_exts 中的扩展名；若找不到返回 None
    """
    dirn, base = os.path.split(img_path)
    stem, _ = os.path.splitext(base)
    cand_list = [os.path.join(dirn, stem + depth_suffix + ext) for ext in try_exts]
    for p in cand_list:
        if os.path.isfile(p):
            return p
    return None

def generate_condition_pairs_all_scenes(
    parent_dir,
    save_path,
    real_angles=[0, 55, 110, 180, -110, -55],
    fov=90,
    num_left=3,
    num_right=3
    ):
    all_results = []

    scene_dirs = sorted([os.path.join(parent_dir, d) for d in os.listdir(parent_dir)
                         if os.path.isdir(os.path.join(parent_dir, d))])

    skipped_no_depth = 0
    for scene_dir in tqdm(scene_dirs, desc="Processing all scenes"):
        scene_name = os.path.basename(scene_dir)
        all_files = [f for f in os.listdir(scene_dir) if f.endswith(".jpg")]
        angle_to_file = {}

        for f in all_files:
            try:
                angle = parse_angle_from_filename(f)
                angle_to_file[angle] = f
            except:
                continue

        all_angles = sorted(angle_to_file.keys())

        for real_angle in real_angles:
            if real_angle not in angle_to_file:
                continue

            real_file = angle_to_file[real_angle]
            real_path = os.path.join(scene_dir, real_file)
            real_depth_path = get_depth_path(real_path)
            if real_depth_path is None:
                skipped_no_depth += 1
                continue

            real_img = cv2.imread(real_path)
            real_img = cv2.cvtColor(real_img, cv2.COLOR_BGR2RGB)
            h, w, _ = real_img.shape

            def angle_diff(a, b):
                return ((b - a + 180) % 360) - 180

            left_angles = sorted([a for a in all_angles if angle_diff(real_angle, a) < 0],
                                 key=lambda x: abs(angle_diff(real_angle, x)))[:num_left]
            right_angles = sorted([a for a in all_angles if angle_diff(real_angle, x) > 0],
                                  key=lambda x: abs(angle_diff(real_angle, x)))[:num_right]

            if len(left_angles) < num_left or len(right_angles) < num_right:
                continue

            left_imgs, left_Ks, left_Rs = [], [], []
            right_imgs, right_Ks, right_Rs = [], [], []

            for angle in left_angles:
                img_p = os.path.join(scene_dir, angle_to_file[angle])
                depth_p = get_depth_path(img_p)
                if depth_p is None:
                    depth_missing = True
                    break
                img = cv2.imread(img_p)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                K, R = get_K_R(fov, angle - real_angle, 0, h, w)
                left_imgs.append(img); left_Ks.append(K); left_Rs.append(R)
            else:
                depth_missing = False

            if depth_missing:
                skipped_no_depth += 1
                continue

            for angle in right_angles:
                img_p = os.path.join(scene_dir, angle_to_file[angle])
                depth_p = get_depth_path(img_p)
                if depth_p is None:
                    depth_missing = True
                    break
                img = cv2.imread(img_p)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                K, R = get_K_R(fov, angle - real_angle, 0, h, w)
                right_imgs.append(img); right_Ks.append(K); right_Rs.append(R)
            if depth_missing:
                skipped_no_depth += 1
                continue

            for i, j in product(range(num_left), range(num_right)):
                left_img_path = os.path.join(scene_dir, angle_to_file[left_angles[i]])
                right_img_path = os.path.join(scene_dir, angle_to_file[right_angles[j]])
                left_depth_path = get_depth_path(left_img_path)
                right_depth_path = get_depth_path(right_img_path)

                if (left_depth_path is None) or (right_depth_path is None):
                    skipped_no_depth += 1
                    continue

                all_results.append({
                    'scene': scene_name,
                    'real_img_path': real_path,
                    'real_depth_path': real_depth_path,              
                    'real_angle': real_angle,
                    'cond_img_paths': [left_img_path, right_img_path],
                    'cond_depth_paths_depth': [left_depth_path, right_depth_path],  
                    'Ks': [left_Ks[i], right_Ks[j]],
                    'Rs': [left_Rs[i], right_Rs[j]],
                    'cond_angles': [left_angles[i], right_angles[j]]
                })

    np.random.shuffle(all_results)
    split_idx = int(0.7 * len(all_results))
    train_results = all_results[:split_idx]
    test_results = all_results[split_idx:]

    
    base_path = os.path.splitext(save_path)[0]
    os.makedirs(os.path.dirname(base_path), exist_ok=True)
    np.save(base_path + "_train.npy", train_results)
    np.save(base_path + "_test.npy", test_results)

    print(f"✅ Saved {len(train_results)} training samples to {base_path}_train.npy")
    print(f"✅ Saved {len(test_results)} testing samples to {base_path}_test.npy")
    print(f"⚠️ Skipped samples due to missing depth: {skipped_no_depth}")


generate_condition_pairs_all_scenes(
    parent_dir="/mnt/vdc1/lyt/interpolation_results",
    save_path="/mnt/vdb1/lyt/condition_pairs.npy",
    real_angles=[0, 55, 110, 180, -110, -55],
    fov=90
)
