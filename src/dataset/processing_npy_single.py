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

def generate_condition_pairs_all_scenes(
    parent_dir,
    save_path,
    real_angles=[0, 55, 110, 180, -110, -55],
    fov=90,
    num_left=3,
    num_right=3
    ):
    
    all_results = []

    scene_dirs = sorted([os.path.join(parent_dir, d) for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d))])
    
    for scene_dir in tqdm(scene_dirs, desc="Processing all scenes"):
        scene_name = "/mnt/vdc1/lyt/interpolation_results/scene_01_1532402421648955"
        
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
            real_img = cv2.imread(real_path)
            real_img = cv2.cvtColor(real_img, cv2.COLOR_BGR2RGB)
            h, w, _ = real_img.shape
            
            def angle_diff(a, b):
                """Return signed shortest distance from angle a to b."""
                return ((b - a + 180) % 360) - 180
            left_angles = sorted([a for a in all_angles if angle_diff(real_angle, a) < 0],
                     key=lambda x: abs(angle_diff(real_angle, x)))[:num_left]
            right_angles = sorted([a for a in all_angles if angle_diff(real_angle, a) > 0],
                      key=lambda x: abs(angle_diff(real_angle, x)))[:num_right]

            if len(left_angles) < num_left or len(right_angles) < num_right:
                continue

            left_imgs, left_Ks, left_Rs = [], [], []
            right_imgs, right_Ks, right_Rs = [], [], []

            for angle in left_angles:
                img = cv2.imread(os.path.join(scene_dir, angle_to_file[angle]))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                K, R = get_K_R(fov, angle - real_angle, 0, h, w)
                left_imgs.append(img)
                left_Ks.append(K)
                left_Rs.append(R)

            for angle in right_angles:
                img = cv2.imread(os.path.join(scene_dir, angle_to_file[angle]))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                K, R = get_K_R(fov, angle - real_angle, 0, h, w)
                right_imgs.append(img)
                right_Ks.append(K)
                right_Rs.append(R)

            for i, j in product(range(num_left), range(num_right)):
                all_results.append({
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
        break
    
    test_results = all_results
    # === Save separately ===
    base_path = os.path.splitext(save_path)[0]  
    os.makedirs(os.path.dirname(base_path), exist_ok=True)
    np.save
    np.save(base_path + "_test.npy", test_results)
    print(f"✅ Saved {len(test_results)} testing samples to {base_path}_test.npy")


generate_condition_pairs_all_scenes(
    parent_dir="/mnt/vdc1/lyt/interpolation_results",
    save_path="/mnt/vdb1/lyt/condition_pairs_single.npy",
    real_angles=[0, 55, 110, 180, -110, -55],
    fov=90
)