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
        test_path = os.path.join(self.image_root_dir, "midpoint_condition_pairs_test_fixed_mvdiffusion.npy")

        if mode == 'train':
            self.data = np.load(train_path, allow_pickle=True)
        else:
            self.data = np.load(test_path, allow_pickle=True)

    def __len__(self):
        return len(self.data)
    
    def crop_img(self, img, K):
        margin = (self.resolution-self.crop_size)//2
        img_crop = img[margin:-margin, margin:-margin]
        K=copy.deepcopy(K)
        K[0, 2] -= margin
        K[1, 2] -= margin

        return img_crop, K

    def __getitem__(self, idx):
        item = self.data[idx].item() if isinstance(self.data[idx], np.ndarray) else self.data[idx]
        real_path = item['real_img_path']
        cond_paths = item['cond_img_paths']
        all_paths = [real_path] + cond_paths
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
        K_target, R_target = get_K_R(90, 0, 0,self.resolution, self.resolution)
        Ks = [K_target] + item['Ks']
        Rs = [R_target] + item['Rs']
        Ks = np.stack(Ks).astype(np.float32)
        Rs = np.stack(Rs).astype(np.float32)
        images = (np.stack(all_imgs).astype(np.float32) / 127.5) - 1.0

        return {
            'image_paths': all_paths,
            'images': images,
            'prompt': [""] * 3,
            'R': Rs,
            'K': Ks
        }
   