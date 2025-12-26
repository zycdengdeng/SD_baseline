import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torchvision.transforms as T
import pytorch_lightning as pl


class RGBLiDARDataset(Dataset):
    def __init__(self, data_root, transform=None, image_size=(512, 512)):
        self.data_root = data_root
        self.image_pairs = []
        
        # 遍历所有matched_*目录
        for subdir in os.listdir(data_root):
            if subdir.startswith('matched_'):
                subdir_path = os.path.join(data_root, subdir)
                rgb_dir = os.path.join(subdir_path, 'rgb')
                lidar_dir = os.path.join(subdir_path, 'lidar')
                
                if os.path.exists(rgb_dir) and os.path.exists(lidar_dir):
                    rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.endswith(('.png', '.jpg'))])
                    lidar_files = sorted([f for f in os.listdir(lidar_dir) if f.endswith(('.png', '.jpg'))])
                    
                    # 匹配相同名称的文件（除了扩展名）
                    for rgb_file in rgb_files:
                        base_name = os.path.splitext(rgb_file)[0]
                        # 查找对应的lidar文件
                        lidar_file = None
                        for lf in lidar_files:
                            if os.path.splitext(lf)[0] == base_name:
                                lidar_file = lf
                                break
                        
                        if lidar_file:
                            self.image_pairs.append({
                                'rgb_path': os.path.join(rgb_dir, rgb_file),
                                'lidar_path': os.path.join(lidar_dir, lidar_file)
                            })

        print(f"找到 {len(self.image_pairs)} 对RGB-LiDAR图片")

        self.rgb_transform = transform or T.Compose([
            T.Resize(image_size),
            T.ToTensor(),  # (3,H,W)
        ])

        self.lidar_transform = T.Compose([
            T.Resize(image_size),
            T.ToTensor(),  # (3,H,W) 
        ])

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        pair = self.image_pairs[idx]
        
        # --- RGB ---
        rgb_img = Image.open(pair['rgb_path']).convert("RGB")
        rgb_tensor = self.rgb_transform(rgb_img)

        # --- LiDAR (3-channel) ---
        lidar_img = Image.open(pair['lidar_path']).convert("RGB")  # keep 3 channels
        lidar_tensor = self.lidar_transform(lidar_img)

        return {
            "rgb": lidar_tensor.float(),            # (3,H,W)
            "lidar_sparse": rgb_tensor.float()      # (3,H,W)
        }



class LidarDataModule(pl.LightningDataModule):
    def __init__(self, config, batch_size=4, num_workers=4):
        super().__init__()
        self.config = config
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        data_cfg = self.config['dataset']
        data_root = data_cfg['train_rgb_dir']  # 现在都指向同一个根目录
        image_size = tuple(data_cfg.get('image_size', (768, 512)))
        
        # 创建完整数据集
        full_dataset = RGBLiDARDataset(
            data_root,
            image_size=image_size
        )
        
        # 划分训练集和验证集 (80%训练，20%验证)
        total_len = len(full_dataset)
        train_len = int(0.8 * total_len)
        val_len = total_len - train_len
        
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            full_dataset, [train_len, val_len],
            generator=torch.Generator().manual_seed(42)  # 固定随机种子
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers,
                          pin_memory=True,
                          drop_last=True,
                          prefetch_factor=2,  # 预取更多批次
                          persistent_workers=True)  # 保持workers活跃

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=1,
                          shuffle=False,
                          num_workers=self.num_workers,
                          pin_memory=True,
                          drop_last=False)
