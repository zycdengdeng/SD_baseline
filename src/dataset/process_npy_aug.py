import os
import json
import cv2
import numpy as np
from tqdm import tqdm
from itertools import product

def get_K_R_abs(FOV, yaw_deg, pitch_deg, height, width):
    """
    Intrinsics K and absolute rotation R (world->cam) from yaw (around y) and pitch (around x).
    Angles in degrees. World axes: x-right, y-up, z-forward.
    """
    f = 0.5 * width / np.tan(0.5 * FOV / 180.0 * np.pi)
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    K = np.array([[f, 0, cx],
                  [0, f, cy],
                  [0, 0,  1]], np.float32)

    # Rodrigues expects axis-angle (axis * radians)
    y_axis = np.array([0.0, 1.0, 0.0], np.float32)
    x_axis = np.array([1.0, 0.0, 0.0], np.float32)

    R_yaw,  _ = cv2.Rodrigues(y_axis * np.radians(yaw_deg))
    R_pitch,_ = cv2.Rodrigues(x_axis * np.radians(pitch_deg))

    # yaw then pitch
    R = R_pitch @ R_yaw  # world->cam
    return K, R

def relative_RT(R_src, C_src, R_tgt, C_tgt):
    """
    Compute relative transform (cond/source -> real/target):
    X_tgt = R_rel * X_src + t_rel
    with R_rel = R_tgt * R_src^T, t_rel = R_tgt * (C_src - C_tgt)
    """
    R_rel = R_tgt @ R_src.T
    t_rel = R_tgt @ (C_src - C_tgt)
    return R_rel.astype(np.float32), t_rel.astype(np.float32)

def euler_from_R_yxz(R):
    """
    Extract yaw(y), pitch(x), roll(z) in degrees from a rotation matrix
    for a 'yaw (y) then pitch (x) then roll (z)' convention.
    Axes: x-right, y-up, z-forward.
    """
    # yaw = atan2(R[0,2], R[2,2])
    yaw = np.degrees(np.arctan2(R[0,2], R[2,2]))
    # pitch = -asin(R[1,2])
    pitch = np.degrees(np.arcsin(np.clip(-R[1,2], -1.0, 1.0)))
    # roll = atan2(R[1,0], R[1,1])
    roll = np.degrees(np.arctan2(R[1,0], R[1,1]))
    return float(yaw), float(pitch), float(roll)

def baseline_angles(R_tgt, C_src, C_tgt):
    """
    Baseline direction of source center as seen in the target camera frame.
    Returns (azimuth_deg, elevation_deg).
    """
    b = (C_src - C_tgt).astype(np.float32)     # world
    b_cam = (R_tgt @ b).astype(np.float32)     # target camera coords
    bx, by, bz = float(b_cam[0]), float(b_cam[1]), float(b_cam[2])
    azimuth = np.degrees(np.arctan2(bx, bz))   # left/right around target
    elevation = np.degrees(np.arctan2(by, np.hypot(bx, bz)))
    return float(azimuth), float(elevation)

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
    dirn, base = os.path.split(img_path)
    stem, _ = os.path.splitext(base)
    for ext in try_exts:
        p = os.path.join(dirn, stem + depth_suffix + ext)
        if os.path.isfile(p):
            return p
    return None

def load_scene_positions(scene_dir, default_height=1.6):
    """
    Load optional per-image camera centers (world coords) from poses.json in the scene folder.
    Format:
    {
      "pos_0.jpg":  {"x": 0.0, "y": 1.6, "z": 0.0},
      "pos_55.jpg": {"x": 0.8, "y": 1.6, "z": 0.2},
      ...
    }
    If missing, returns a function that supplies (0, default_height, 0).
    """
    poses_path = os.path.join(scene_dir, "poses.json")
    if os.path.isfile(poses_path):
        with open(poses_path, "r") as f:
            poses = json.load(f)
        def getter(filename):
            ent = poses.get(filename, None)
            if ent is None:
                return np.array([0.0, default_height, 0.0], dtype=np.float32)
            return np.array([float(ent.get("x", 0.0)),
                             float(ent.get("y", default_height)),
                             float(ent.get("z", 0.0))], dtype=np.float32)
        return getter, True
    else:
        def getter(_filename):
            return np.array([0.0, default_height, 0.0], dtype=np.float32)
        return getter, False

def generate_condition_pairs_all_scenes(
    parent_dir,
    save_path,
    real_angles=(0, 55, 110, 180, -110, -55),
    fov=90,
    num_left=3,
    num_right=3,
    camera_height=1.6,   # meters; used if poses.json missing or y not provided
    pitch_deg=0.0        # per-image pitch; customize if you have it per file
    ):
    all_results = []

    scene_dirs = sorted([os.path.join(parent_dir, d) for d in os.listdir(parent_dir)
                         if os.path.isdir(os.path.join(parent_dir, d))])

    skipped_no_depth = 0
    missing_pose_files = 0

    for scene_dir in tqdm(scene_dirs, desc="Processing all scenes"):
        scene_name = os.path.basename(scene_dir)
        all_files = [f for f in os.listdir(scene_dir) if f.endswith(".jpg")]
        angle_to_file = {}

        for f in all_files:
            try:
                ang = parse_angle_from_filename(f)
                angle_to_file[ang] = f
            except:
                continue

        all_angles = sorted(angle_to_file.keys())
        if not all_angles:
            continue

        get_center, has_poses = load_scene_positions(scene_dir, default_height=camera_height)
        if not has_poses:
            missing_pose_files += 1

        # Pre-load absolute extrinsics for every angle/file (after we know h,w)
        abs_K = {}
        abs_R = {}
        abs_C = {}  # camera centers in world coords

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
            if real_img is None:
                continue
            real_img = cv2.cvtColor(real_img, cv2.COLOR_BGR2RGB)
            h, w, _ = real_img.shape

            # Compute absolute R, K, C for all angles once we know h,w
            if not abs_K:
                for a in all_angles:
                    fname = angle_to_file[a]
                    K_a, R_a = get_K_R_abs(fov, yaw_deg=a, pitch_deg=pitch_deg, height=h, width=w)
                    C_a = get_center(fname).astype(np.float32)
                    abs_K[a] = K_a
                    abs_R[a] = R_a.astype(np.float32)
                    abs_C[a] = C_a

            # ---------- Placement-aware split by baseline azimuth ----------
            R_tgt = abs_R[real_angle]
            C_tgt = abs_C[real_angle]

            # Compute azimuth for every candidate angle relative to target
            az_by_angle = {}
            dist_by_angle = {}

            for a in all_angles:
                if a == real_angle:
                    continue
                az, el = baseline_angles(R_tgt, abs_C[a], C_tgt)
                az_by_angle[a] = (az, el)
                dist_by_angle[a] = float(np.linalg.norm(abs_C[a] - C_tgt))

            # Left = negative az, Right = positive az, each sorted by |az|
            left_angles = [a for a in all_angles if a in az_by_angle and az_by_angle[a][0] < 0]
            right_angles = [a for a in all_angles if a in az_by_angle and az_by_angle[a][0] > 0]

            left_angles  = sorted(left_angles,  key=lambda a: dist_by_angle[a])[:num_left]
            right_angles = sorted(right_angles, key=lambda a: dist_by_angle[a])[:num_right]


            if len(left_angles) < num_left or len(right_angles) < num_right:
                # Not enough neighbors on both sides — skip this real
                continue

            # Prepare lists
            left_imgs, left_Ks, left_Rrels, left_Trels = [], [], [], []
            right_imgs, right_Ks, right_Rrels, right_Trels = [], [], [], []
            # Also record angles for debugging/analysis
            left_baseline_az_el, right_baseline_az_el = [], []
            left_heading_ypr, right_heading_ypr = [], []

            # Collect left side
            depth_missing = False
            for a in left_angles:
                img_p = os.path.join(scene_dir, angle_to_file[a])
                depth_p = get_depth_path(img_p)
                if depth_p is None:
                    depth_missing = True
                    break
                img = cv2.imread(img_p)
                if img is None:
                    depth_missing = True
                    break
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                R_src = abs_R[a]
                C_src = abs_C[a]
                R_rel, t_rel = relative_RT(R_src, C_src, R_tgt, C_tgt)
                yaw_rel, pitch_rel, roll_rel = euler_from_R_yxz(R_rel)
                az, el = az_by_angle[a]

                left_imgs.append(img)
                left_Ks.append(abs_K[a])
                left_Rrels.append(R_rel)
                left_Trels.append(t_rel)
                left_baseline_az_el.append((az, el))
                left_heading_ypr.append((yaw_rel, pitch_rel, roll_rel))
            if depth_missing:
                skipped_no_depth += 1
                continue

            # Collect right side
            for a in right_angles:
                img_p = os.path.join(scene_dir, angle_to_file[a])
                depth_p = get_depth_path(img_p)
                if depth_p is None:
                    depth_missing = True
                    break
                img = cv2.imread(img_p)
                if img is None:
                    depth_missing = True
                    break
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                R_src = abs_R[a]
                C_src = abs_C[a]
                R_rel, t_rel = relative_RT(R_src, C_src, R_tgt, C_tgt)
                yaw_rel, pitch_rel, roll_rel = euler_from_R_yxz(R_rel)
                az, el = az_by_angle[a]

                right_imgs.append(img)
                right_Ks.append(abs_K[a])
                right_Rrels.append(R_rel)
                right_Trels.append(t_rel)
                right_baseline_az_el.append((az, el))
                right_heading_ypr.append((yaw_rel, pitch_rel, roll_rel))
            if depth_missing:
                skipped_no_depth += 1
                continue

            # Pair up left/right combos
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
                    'real_angle': float(real_angle),

                    'cond_img_paths': [left_img_path, right_img_path],
                    'cond_depth_paths_depth': [left_depth_path, right_depth_path],

                    # intrinsics for each cond view (could be same across all)
                    'Ks': [left_Ks[i], right_Ks[j]],

                    # relative rotations (cond -> real)
                    'Rs': [left_Rrels[i], right_Rrels[j]],

                    # relative translations (cond -> real), shape (3,)
                    'Ts': [left_Trels[i], right_Trels[j]],

                    # placement-aware baseline angles (azimuth,elevation) in target cam frame
                    'cond_baseline_az_el': [left_baseline_az_el[i], right_baseline_az_el[j]],

                    # orientation deltas (yaw,pitch,roll) from R_rel (optional but handy)
                    'cond_heading_ypr': [left_heading_ypr[i], right_heading_ypr[j]],

                    # keep for reference
                    'cond_angles': [float(left_angles[i]), float(right_angles[j])],
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
    if missing_pose_files > 0:
        print(f"ℹ️ {missing_pose_files} scene(s) had no poses.json; used default centers (0, {camera_height}, 0).")
