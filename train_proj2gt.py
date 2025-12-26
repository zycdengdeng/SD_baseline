import os
# 必须在导入PyTorch之前设置环境变量
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

'''
python train_proj2gt.py --config config_proj2gt.yaml
python /mnt/zihanw/ArbiViewGen-main/train_proj2gt.py --config config_proj2gt.yaml --ckpt_path path/to/checkpoint.ckpt

sudo -u lyt bash -c "source /opt/anaconda3/etc/profile.d/conda.sh && conda activate MVDiffusion && cd /mnt/zihanw/ArbiViewGen-main && python train_proj2gt.py --config config_proj2gt.yaml"
'''

import yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint, LearningRateMonitor, Callback
from pytorch_lightning.loggers import TensorBoardLogger
from torchvision.utils import save_image
from datetime import datetime
import argparse

# 导入修改后的模块
from src.lightning_depth import LidarDiffusionModule
from proj_to_gt_dataloader import ProjToGTDataModule  # 使用新的数据加载器

def parse_args():
    parser = argparse.ArgumentParser(description='Proj to GT Diffusion Training')
    parser.add_argument('--config', type=str, default='config_proj2gt.yaml', help='Path to config file')
    parser.add_argument('--ckpt_path', type=str, default=None, help='Path to checkpoint for resuming training')
    parser.add_argument('--exp_name', type=str, default=None, help='Experiment name (if not provided, uses timestamp)')
    return parser.parse_args()

args = parse_args()

# 检查GPU设置
print(f"可用GPU数量: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

# 加载配置文件
with open(args.config, "r") as f:
    config = yaml.safe_load(f)

print(f"数据根目录: {config['dataset']['data_root']}")
print(f"图像尺寸: {config['dataset']['image_size']}")
print(f"批次大小: {config['train']['batch_size']}")

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
        save_dir = f"./experiments/proj2gt_resume_{timestamp}"
        print(f"Could not infer original experiment directory, creating new one: {save_dir}")
elif args.exp_name:
    save_dir = f"./experiments/{args.exp_name}"
    print(f"Using specified experiment directory: {save_dir}")
else:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"./experiments/proj2gt_{timestamp}"
    print(f"Creating new experiment directory: {save_dir}")

os.makedirs(save_dir, exist_ok=True)
os.makedirs(f"{save_dir}/checkpoints", exist_ok=True)
os.makedirs(f"{save_dir}/images", exist_ok=True)
os.makedirs(f"{save_dir}/logs", exist_ok=True)

if args.ckpt_path:
    print(f"Resuming from checkpoint: {args.ckpt_path}")

# 使用新的数据模块
datamodule = ProjToGTDataModule(
    config=config,
    batch_size=config["train"]["batch_size"],  # 每张卡的batch size，总batch=20*3=60
    num_workers=12,  # 数据加载workers，适配3张卡
    val_split=0.2  # 20%作为验证集
)

# 创建模型
model = LidarDiffusionModule(
    lr=float(config["train"].get("lr", 1e-4)),
    t_max=int(config["model"].get("t_max", 1000)),
    weight_decay=float(config["train"].get("weight_decay", 0.0)),
    model_id=config["model"].get("model_id", "runwayml/stable-diffusion-v1-5"),
    lidar_channels=config["model"].get("lidar_channels", 3),  # proj是RGB图像，所以是3通道
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
            num_samples = min(10, len(val_dataloader))
            for sample_idx in range(num_samples):
                try:
                    val_batch = next(val_iterator)
                    
                    # 获取数据
                    gt_image = val_batch["rgb"][0:1].to(pl_module.device)  # GT图像
                    proj_image = val_batch["lidar_sparse"][0:1].to(pl_module.device)  # proj图像作为条件
                    
                    # 获取元数据
                    if "metadata" in val_batch:
                        metadata = val_batch["metadata"]
                        timestamp = metadata["timestamp"][0]
                        view = metadata["view"][0]
                        source_info = f"{timestamp}_{view}"
                    else:
                        source_info = f"sample_{sample_idx:03d}"
                    
                    with torch.no_grad():
                        # 从proj图像生成GT风格的图像
                        generated = pl_module.generate_from_lidar(proj_image, num_inference_steps=20)
                        
                        # 确保所有张量都在正确的范围内
                        generated = torch.clamp(generated, 0, 1)
                        gt_image = torch.clamp(gt_image, 0, 1)
                        proj_image = torch.clamp(proj_image, 0, 1)
                        
                        # 保存文件名前缀
                        file_prefix = f"{epoch_dir}/{source_info}"
                        
                        # 保存生成的图片
                        save_image(generated, f"{file_prefix}_generated.jpg")
                        
                        # 保存原始GT图片
                        save_image(gt_image, f"{file_prefix}_gt.jpg")
                        
                        # 保存原始proj图片
                        save_image(proj_image, f"{file_prefix}_proj.jpg")
                        
                        # 创建对比图（三张图片并排）
                        comparison = torch.cat([proj_image, generated, gt_image], dim=3)  # 水平拼接
                        save_image(comparison, f"{file_prefix}_comparison.jpg")
                        
                except StopIteration:
                    break
                except Exception as e:
                    print(f"Sample {sample_idx+1} failed: {e}")
                    continue

image_callback = ImageGenerationCallback(save_dir)

# 设置日志记录器
logger = TensorBoardLogger(
    save_dir=f"{save_dir}/logs",
    name="proj2gt_diffusion",
    version=None
)

# 创建训练器
trainer = pl.Trainer(
    max_epochs=config["train"].get("max_epochs", 50),
    accelerator="gpu",
    devices=3,  # 使用3张卡
    strategy="ddp",  # 分布式数据并行
    log_every_n_steps=10,
    callbacks=[TQDMProgressBar(), checkpoint_callback, lr_monitor, image_callback],
    # precision="16-mixed",  # 可选：混合精度训练以节省显存
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
    print("\n=== 训练完成，生成最终测试样本 ===")
    model = model.to("cuda:0")  # 使用第一张卡进行推理（现在是GPU 5）
    model.eval()
    
    # 设置数据模块并获取一个批次
    datamodule.setup()
    batch = next(iter(datamodule.val_dataloader()))
    
    gt_image = batch["rgb"].to(model.device)
    proj_image = batch["lidar_sparse"].to(model.device)
    
    with torch.no_grad():
        # 生成图像
        generated = model.generate_from_lidar(proj_image, num_inference_steps=30)
    
    # 归一化到 [0,1] 范围
    generated = torch.clamp(generated, 0, 1)
    
    # 保存最终生成的图片到实验目录
    output_path = f"{save_dir}/images/final_generated.jpg"
    save_image(generated, output_path)
    
    # 保存对比图
    comparison = torch.cat([proj_image, generated, gt_image], dim=3)
    comparison_path = f"{save_dir}/images/final_comparison.jpg"
    save_image(comparison, comparison_path)
    
    print(f"Generated: {output_path}")
    print(f"Comparison: {comparison_path}")
    print(f"Best model: {checkpoint_callback.best_model_path}")
    print(f"Output shape: {generated.shape}")
