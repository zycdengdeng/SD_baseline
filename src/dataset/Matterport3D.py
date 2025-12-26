import torch
import os
import numpy as np
import cv2
import random
from .utils import get_K_R
import copy

class MP3Ddataset(torch.utils.data.Dataset):
    def __init__(self, config, mode='train'):
        self.mode = mode
        self.image_root_dir = config['image_root_dir']
        self.fov = config['fov']
        self.resolution = config['resolution']
        self.crop_size = config['crop_size']

        train_path = os.path.join(self.image_root_dir, "condition_pairs_train.npy")
        test_path  = os.path.join(self.image_root_dir, "condition_pairs_test.npy")

        if mode == 'train':
            self.data = np.load(train_path, allow_pickle=True)
        else:
            self.data = np.load(test_path, allow_pickle=True)

    def __len__(self):
        return len(self.data)
    
    def crop_img(self, img, K):
        margin = (self.resolution - self.crop_size) // 2
        img_crop = img[margin:-margin, margin:-margin]
        K = copy.deepcopy(K)
        K[0, 2] -= margin
        K[1, 2] -= margin
        return img_crop, K

    def load_depth(self, path):
        if path.endswith(".npy"):
            depth = np.load(path).astype(np.float32)
        else:
            depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if depth is None:
                raise FileNotFoundError(f"[ERROR] Failed to load depth at: {path}")
            depth = depth.astype(np.float32)

        depth = cv2.resize(depth, (self.resolution, self.resolution), interpolation=cv2.INTER_NEAREST)
        if depth.max() > 0:
            depth_norm = depth / (depth.max() + 1e-8)
        else:
            depth_norm = depth
        return depth_norm

    def __getitem__(self, idx):
        item = self.data[idx].item() if isinstance(self.data[idx], np.ndarray) else self.data[idx]

        # Paths
        real_path        = item['real_img_path']
        cond_paths       = item['cond_img_paths']
        real_depth_path  = item['real_depth_path']
        cond_depth_paths = item['cond_depth_paths_depth']

        all_paths        = [real_path] + cond_paths
        all_depth_paths  = [real_depth_path] + cond_depth_paths

        # Load images
        real_img = cv2.imread(real_path)
        if real_img is None:
            raise FileNotFoundError(f"[ERROR] Failed to load real image at: {real_path}")

        cond_imgs = []
        for p in cond_paths:
            img = cv2.imread(p)
            if img is None:
                raise FileNotFoundError(f"[ERROR] Failed to load conditional image at: {p}")
            cond_imgs.append(img)

        all_imgs = [real_img] + cond_imgs
        all_imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in all_imgs]
        all_imgs = [cv2.resize(img, (self.resolution, self.resolution)) for img in all_imgs]
        images   = (np.stack(all_imgs).astype(np.float32) / 127.5) - 1.0   # [m, H, W, 3]

        # Depths
        all_depths = [self.load_depth(p) for p in all_depth_paths]
        depths     = np.stack(all_depths).astype(np.float32)               # [m, H, W]

        # Intrinsics/extrinsics (stack real first, then conds)
        K_target, R_target = get_K_R(self.fov, 0, 0, self.resolution, self.resolution)
        Ks = [K_target] + item['Ks']                                       # [m, 3, 3]
        Rs = [R_target] + item['Rs']                                       # [m, 3, 3]
        Ks = np.stack(Ks).astype(np.float32)
        Rs = np.stack(Rs).astype(np.float32)

        # --------- NEW: Translations T (cond -> real) ----------
        # Expect in pairs file: item['Ts'] = [t_left (3,), t_right (3,)]
        # Real/target view's T is zero.
        if 'Ts' in item:
            Ts = [np.zeros(3, dtype=np.float32)] + [np.array(t, dtype=np.float32) for t in item['Ts']]
        else:
            # Fallback if Ts not present
            Ts = [np.zeros(3, dtype=np.float32)] + [np.zeros(3, dtype=np.float32) for _ in cond_paths]
        Ts = np.stack(Ts).astype(np.float32)                             


        m = images.shape[0]
        prompts = [""] * m

        return {
            'image_paths': all_paths,          # list length m
            'images': images,                  # [m, H, W, 3], float32 in [-1, 1]
            'depth_inv_norm': depths,          # [m, H, W], float32 in [0,1]
            'prompt': prompts,                 # list length m
            'R': Rs,                           # [m, 3, 3]
            'K': Ks,                           # [m, 3, 3]
            'T': Ts                            # [m, 3]  <-- added
        }
