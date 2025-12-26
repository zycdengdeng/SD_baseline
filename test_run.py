import os
# 必须在导入PyTorch之前设置环境变量
os.environ["CUDA_VISIBLE_DEVICES"] = "5,6,7"

import yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint, LearningRateMonitor, Callback
from pytorch_lightning.loggers import TensorBoardLogger
from torchvision.utils import save_image
from datetime import datetime
import argparse

from src.lightning_depth import LidarDiffusionModule
from src.dataset.current_data_loader import LidarDataModule

def parse_args():
    parser = argparse.ArgumentParser(description='LiDAR Diffusion Training')
    parser.add_argument('--ckpt_path', type=str, default=None, help='Path to checkpoint for resuming training')
    parser.add_argument('--exp_name', type=str, default=None, help='Experiment name (if not provided, uses timestamp)')
    return parser.parse_args()

args = parse_args()

# 检查GPU设置
print(f"可用GPU数量: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# 创建统一的保存路径
if args.ckpt_path and args.exp_name is None:
    # 如果是断点续训且没有指定新实验名，尝试使用原实验目录
    try:
        # 从检查点路径推断原实验目录
        ckpt_parts = args.ckpt_path.split('/')
        exp_idx = ckpt_parts.index('experiments') + 1
        original_exp_name = ckpt_parts[exp_idx]
        save_dir = f"./experiments/{original_exp_name}"
        print(f"Resuming training in original experiment directory: {save_dir}")
    except (ValueError, IndexError):
        # 如果无法推断，创建新目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = f"./experiments/lidar_diffusion_resume_{timestamp}"
        print(f"Could not infer original experiment directory, creating new one: {save_dir}")
elif args.exp_name:
    save_dir = f"./experiments/{args.exp_name}"
    print(f"Using specified experiment directory: {save_dir}")
else:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"./experiments/lidar_diffusion_{timestamp}"
    print(f"Creating new experiment directory: {save_dir}")

os.makedirs(save_dir, exist_ok=True)
os.makedirs(f"{save_dir}/checkpoints", exist_ok=True)
os.makedirs(f"{save_dir}/images", exist_ok=True)
os.makedirs(f"{save_dir}/logs", exist_ok=True)

if args.ckpt_path:
    print(f"Resuming from checkpoint: {args.ckpt_path}")


datamodule = LidarDataModule(
    config=config,
    batch_size=config["train"]["batch_size"],  # 每张卡的batch size，总batch=20*3=60
    num_workers=12  # 数据加载workers，适配3张卡
)


model = LidarDiffusionModule(
    lr=float(config.get("lr", 1e-4)),
    t_max=int(config["model"].get("t_max", 1000)),
    weight_decay=float(config.get("weight_decay", 0.0)),
    model_id=config["model"].get("model_id", "runwayml/stable-diffusion-v1-5"),
    lidar_channels=config["model"].get("lidar_channels", 3),
)

# 设置回调函数
checkpoint_callback = ModelCheckpoint(
    dirpath=f"{save_dir}/checkpoints",
    filename="epoch-{epoch:02d}-val_loss-{val/loss:.3f}",
    monitor="val/loss",
    mode="min",
    save_top_k=10,  # 保存最好的10个模型
    save_last=True,  # 保存最后一个epoch的模型
    save_on_train_epoch_end=False,  # 在验证结束后保存
    verbose=True
)

lr_monitor = LearningRateMonitor(logging_interval="epoch")

# 自定义回调：每个epoch后生成图片
class ImageGenerationCallback(Callback):
    def __init__(self, save_dir):
        self.save_dir = save_dir
        
    def on_validation_epoch_end(self, trainer, pl_module):
        # 只在rank 0上生成图片，避免多卡重复生成
        if trainer.current_epoch % 1 == 0 and trainer.global_rank == 0:  # 每个epoch都生成
            pl_module.eval()
            
            # 为当前epoch创建专门的文件夹
            epoch_dir = f"{self.save_dir}/images/epoch_{trainer.current_epoch:02d}"
            os.makedirs(epoch_dir, exist_ok=True)
            
            # 获取验证批次
            datamodule = trainer.datamodule
            val_dataloader = datamodule.val_dataloader()
            val_iterator = iter(val_dataloader)
            
            # 保存10个不同的样本
            num_samples = 10
            for sample_idx in range(num_samples):
                try:
                    val_batch = next(val_iterator)
                    # 从批次中选择不同的样本，而不是总是选择第一个
                    batch_size = val_batch["lidar_sparse"].shape[0]
                    sample_in_batch = sample_idx % batch_size
                    
                    lidar_sparse = val_batch["lidar_sparse"][sample_in_batch:sample_in_batch+1].to(pl_module.device)
                    rgb_original = val_batch["rgb"][sample_in_batch:sample_in_batch+1]  # 原始RGB
                    lidar_original = val_batch["lidar_sparse"][sample_in_batch:sample_in_batch+1]  # 原始LiDAR
                    
                    # 尝试获取数据源信息
                    try:
                        # 获取数据集中的文件路径信息
                        dataset = datamodule.val_dataset.dataset if hasattr(datamodule.val_dataset, 'dataset') else datamodule.val_dataset
                        
                        # 计算实际的数据索引
                        if hasattr(val_batch, 'get') and 'idx' in val_batch:
                            # 如果批次包含索引信息
                            actual_idx = val_batch['idx'][sample_in_batch].item()
                        else:
                            # 否则根据批次和样本位置估算
                            actual_idx = sample_idx
                        
                        if hasattr(dataset, 'image_pairs') and len(dataset.image_pairs) > actual_idx:
                            # 获取当前样本的源文件信息
                            sample_info = dataset.image_pairs[actual_idx]
                            rgb_path = sample_info['rgb_path']
                            # 提取匹配目录名和文件名
                            matched_dir = rgb_path.split('/')[-3]  # matched_XXXXXXXXX
                            file_name = rgb_path.split('/')[-1].split('.')[0]  # pos_011.25
                            source_info = f"{matched_dir}_{file_name}"
                        else:
                            source_info = f"sample_{sample_idx:03d}"
                    except:
                        source_info = f"sample_{sample_idx:03d}"
                    
                    with torch.no_grad():
                        # 生成图片
                        out_rgb = pl_module.generate_from_lidar(lidar_sparse, num_inference_steps=20)
                        
                        # 确保所有张量都在正确的范围内
                        out_rgb = torch.clamp(out_rgb, 0, 1)
                        rgb_original = torch.clamp(rgb_original, 0, 1)
                        lidar_original = torch.clamp(lidar_original, 0, 1)
                        
                        # 保存文件名前缀
                        file_prefix = f"{epoch_dir}/{source_info}"
                        
                        # 保存生成的图片
                        save_image(out_rgb, f"{file_prefix}_generated.jpg")
                        
                        # 保存原始RGB图片
                        save_image(rgb_original, f"{file_prefix}_original_rgb.jpg")
                        
                        # 保存原始LiDAR图片
                        save_image(lidar_original, f"{file_prefix}_original_lidar.jpg")
                        
                except Exception as e:
                    print(f"Sample {sample_idx+1} failed: {e}")
                    break

image_callback = ImageGenerationCallback(save_dir)

# 设置日志记录器
logger = TensorBoardLogger(
    save_dir=f"{save_dir}/logs",
    name="lidar_diffusion",
    version=None
)

# 环境变量已在文件开头设置

trainer = pl.Trainer(
    max_epochs=config["train"].get("max_epochs", 50),  # 改为50个epoch
    accelerator="gpu",
    devices=3,  # 使用3张卡
    strategy="ddp",  # 分布式数据并行
    log_every_n_steps=10,
    callbacks=[TQDMProgressBar(), checkpoint_callback, lr_monitor, image_callback],
    # precision="16-mixed",  
    logger=logger,
    enable_checkpointing=True,
    enable_progress_bar=True,
    enable_model_summary=True,
    gradient_clip_val=1.0,  
    accumulate_grad_batches=1,  
    max_steps=-1,  
    sync_batchnorm=True,  # 多卡训练时启用同步BN
    check_val_every_n_epoch=1,  
    val_check_interval=1.0,  
)

# 开始训练（如果提供了检查点路径，则从检查点恢复训练）
trainer.fit(model, datamodule=datamodule, ckpt_path=args.ckpt_path)

# 只在主进程中执行推理
if trainer.global_rank == 0:
    model = model.to("cuda:0")  # 使用第一张卡进行推理（现在是GPU 5）
    model.eval()
    datamodule.setup()
    batch = next(iter(datamodule.val_dataloader()))
    rgb = batch["rgb"].to(model.device)
    lidar_sparse = batch["lidar_sparse"].to(model.device)

    with torch.no_grad():
        out_rgb = model.generate_from_lidar(lidar_sparse, num_inference_steps=30)

    # 归一化到 [0,1] 范围
    out_rgb = (out_rgb - out_rgb.min()) / (out_rgb.max() - out_rgb.min())

    # 保存生成的图片到实验目录
    output_path = f"{save_dir}/images/generated_output.jpg"
    save_image(out_rgb, output_path)

    print(f"Generated: {output_path}")
    print(f"Best model: {checkpoint_callback.best_model_path}")
    print(f"Output shape: {out_rgb.shape}")  