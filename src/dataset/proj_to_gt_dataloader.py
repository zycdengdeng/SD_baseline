import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torchvision.transforms as T
import pytorch_lightning as pl


class ProjToGTDataset(Dataset):
    def __init__(self, data_root, transform=None, image_size=(512, 512)):
        """
        数据集加载器：从proj生成GT
        
        数据结构：
        data_root/
        ├── 1743646155857/
        │   ├── GT/
        │   │   ├── FL.jpg
        │   │   ├── FN.jpg
        │   │   └── ...
        │   └── proj/
        │       ├── FL.jpg
        │       ├── FN.jpg
        │       └── ...
        └── 1743646155986/
            └── ...
        """
        self.data_root = data_root
        self.image_pairs = []
        
        # 定义需要的图像名称（排除combined.jpg）
        image_names = ['FL.jpg', 'FN.jpg', 'FR.jpg', 'FW.jpg', 'RL.jpg', 'RN.jpg', 'RR.jpg']
        
        # 遍历所有时间戳目录
        for timestamp_dir in sorted(os.listdir(data_root)):
            timestamp_path = os.path.join(data_root, timestamp_dir)
            
            # 检查是否是目录且包含GT和proj子目录
            if os.path.isdir(timestamp_path):
                gt_dir = os.path.join(timestamp_path, 'GT')
                proj_dir = os.path.join(timestamp_path, 'proj')
                
                if os.path.exists(gt_dir) and os.path.exists(proj_dir):
                    # 对每个图像名称创建配对
                    for img_name in image_names:
                        gt_path = os.path.join(gt_dir, img_name)
                        proj_path = os.path.join(proj_dir, img_name)
                        
                        # 确保两个文件都存在
                        if os.path.exists(gt_path) and os.path.exists(proj_path):
                            self.image_pairs.append({
                                'gt_path': gt_path,
                                'proj_path': proj_path,
                                'timestamp': timestamp_dir,
                                'view': img_name.split('.')[0]  # FL, FN, FR等
                            })

        print(f"找到 {len(self.image_pairs)} 对 proj-GT 图片")
        print(f"时间戳数量: {len(set([p['timestamp'] for p in self.image_pairs]))}")
        print(f"每个时间戳的视角数: {len(image_names)}")

        # 图像变换
        self.transform = transform or T.Compose([
            T.Resize(image_size),
            T.ToTensor(),  # 转换为 [0, 1] 范围
        ])

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        pair = self.image_pairs[idx]
        
        # 加载GT图像（目标）
        gt_img = Image.open(pair['gt_path']).convert("RGB")
        gt_tensor = self.transform(gt_img)
        
        # 加载proj图像（条件/引导）
        proj_img = Image.open(pair['proj_path']).convert("RGB")
        proj_tensor = self.transform(proj_img)
        
        return {
            "rgb": gt_tensor.float(),           # GT作为目标（在扩散模型中要生成的）
            "lidar_sparse": proj_tensor.float(), # proj作为条件（引导生成过程）
            "metadata": {
                "timestamp": pair['timestamp'],
                "view": pair['view'],
                "gt_path": pair['gt_path'],
                "proj_path": pair['proj_path']
            }
        }


class ProjToGTDataModule(pl.LightningDataModule):
    def __init__(self, config, batch_size=4, num_workers=4, val_split=0.2):
        super().__init__()
        self.config = config
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split

    def setup(self, stage=None):
        data_cfg = self.config['dataset']
        data_root = data_cfg['data_root']  # 新的配置项
        image_size = tuple(data_cfg.get('image_size', (512, 512)))
        
        # 创建完整数据集
        full_dataset = ProjToGTDataset(
            data_root,
            image_size=image_size
        )
        
        # 划分训练集和验证集 (默认80%训练，20%验证)
        total_len = len(full_dataset)
        train_len = int((1 - self.val_split) * total_len)
        val_len = total_len - train_len
        
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            full_dataset, [train_len, val_len],
            generator=torch.Generator().manual_seed(42)  # 固定随机种子以保证可重复性
        )
        
        print(f"训练集大小: {len(self.train_dataset)}")
        print(f"验证集大小: {len(self.val_dataset)}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            prefetch_factor=2,
            persistent_workers=True if self.num_workers > 0 else False
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=1,  # 验证时使用batch_size=1以便更好地观察单个样本
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=True if self.num_workers > 0 else False
        )