import argparse
import yaml
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from diffusers import DDPMScheduler, AutoencoderKL
from src.dataset.current_data_loader import LidarDataModule
from src.models.depth.MVDepthModel import LidarDiffusionModel



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("main_cfg_path", type=str, help="Path to YAML config")
    parser.add_argument("--exp_name", type=str, default="lidar_diffusion_exp")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser = pl.Trainer.add_argparse_args(parser)
    return parser.parse_args()


class LidarDiffusionModule(pl.LightningModule):
    def __init__(self, lr=1e-4, t_max=1000, weight_decay=0.0,
                 model_id="runwayml/stable-diffusion-v1-5", lidar_channels=1):
        super().__init__()
        self.save_hyperparameters()

        # pretrained VAE + scheduler
        self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad = False

        #self.model = LidarDiffusionModel(model_id=model_id, lidar_channels=lidar_channels)
        in_ch = 4 + lidar_channels  # 4 latent channels + LiDAR 条件
        #self.model = LidarDiffusionModel(in_ch=in_ch, out_ch=4, base=64, time_dim=128)
        self.model = LidarDiffusionModel(model_id=model_id, lidar_channels=lidar_channels)

        self.noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

        self.lr = lr
        self.weight_decay = weight_decay

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(),
                                 lr=self.lr, weight_decay=self.weight_decay)

    def encode_rgb(self, rgb):
        latents = self.vae.encode(rgb).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor
        return latents

    def decode_latents(self, latents):
        latents = latents / self.vae.config.scaling_factor
        imgs = self.vae.decode(latents).sample
        return imgs.clamp(0, 1)

    def downsample_lidar(self, lidar, target_shape):
        return F.interpolate(lidar, size=target_shape,
                             mode="bilinear", align_corners=False)
    
    def compute_smoothness_loss(self, pred):
        """计算预测结果的平滑性损失"""
        # 梯度损失
        grad_x = torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:])
        grad_y = torch.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :])
        
        return grad_x.mean() + grad_y.mean()

    def training_step(self, batch, batch_idx):
        rgb = batch["rgb"]                 
        lidar_sparse = batch["lidar_sparse"]  

        latents = self.encode_rgb(rgb)
        lidar_cond = self.downsample_lidar(lidar_sparse, latents.shape[-2:])

        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0, self.noise_scheduler.num_train_timesteps,
            (latents.size(0),), device=self.device, dtype=torch.long
        )

        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        noise_pred = self.model(noisy_latents, lidar_cond, timesteps)

        # 改进的mask策略
        valid_mask = (rgb.sum(1, keepdim=True) > 0).float()
        invalid_mask = 1.0 - valid_mask
        
        valid_mask_latent = F.interpolate(valid_mask, size=latents.shape[-2:], mode="nearest")
        invalid_mask_latent = F.interpolate(invalid_mask, size=latents.shape[-2:], mode="nearest")

        # 有效区域的正常监督
        valid_loss = ((noise - noise_pred) ** 2 * valid_mask_latent).sum() / (valid_mask_latent.sum() + 1e-6)
        
        # 无效区域鼓励生成零值
        target_noise_invalid = torch.zeros_like(noise)
        invalid_loss = ((target_noise_invalid - noise_pred) ** 2 * invalid_mask_latent).sum() / (invalid_mask_latent.sum() + 1e-6)
        
        # 平滑性正则化
        smoothness_loss = self.compute_smoothness_loss(noise_pred)
        
        # 组合损失
        total_loss = valid_loss + 0.1 * invalid_loss + 0.01 * smoothness_loss

        self.log("train/loss", total_loss, prog_bar=True)
        self.log("train/valid_loss", valid_loss)
        self.log("train/invalid_loss", invalid_loss)
        self.log("train/smoothness_loss", smoothness_loss)
        
        return total_loss

    def validation_step(self, batch, batch_idx):
        rgb = batch["rgb"]
        lidar_sparse = batch["lidar_sparse"]
        lidar_sparse = F.interpolate(batch["lidar_sparse"], size=(128, 128), mode="bilinear")

        latents = self.encode_rgb(rgb)
        lidar_cond = self.downsample_lidar(lidar_sparse, latents.shape[-2:])

        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps,
                                  (latents.size(0),), device=self.device, dtype=torch.long)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        noise_pred = self.model(noisy_latents, lidar_cond, timesteps)

        # 与训练步骤保持一致的mask策略
        valid_mask = (rgb.sum(1, keepdim=True) > 0).float()
        invalid_mask = 1.0 - valid_mask
        
        valid_mask_latent = F.interpolate(valid_mask, size=latents.shape[-2:], mode="nearest")
        invalid_mask_latent = F.interpolate(invalid_mask, size=latents.shape[-2:], mode="nearest")

        # 有效区域的损失
        valid_loss = ((noise - noise_pred) ** 2 * valid_mask_latent).sum() / (valid_mask_latent.sum() + 1e-6)
        
        # 无效区域的损失
        target_noise_invalid = torch.zeros_like(noise)
        invalid_loss = ((target_noise_invalid - noise_pred) ** 2 * invalid_mask_latent).sum() / (invalid_mask_latent.sum() + 1e-6)
        
        # 平滑性损失
        smoothness_loss = self.compute_smoothness_loss(noise_pred)
        
        # 组合损失
        total_loss = valid_loss + 0.1 * invalid_loss + 0.01 * smoothness_loss

        self.log("val/loss", total_loss, prog_bar=True)
        self.log("val/valid_loss", valid_loss)
        self.log("val/invalid_loss", invalid_loss) 
        self.log("val/smoothness_loss", smoothness_loss)
        
        return total_loss

    @torch.no_grad()
    def generate(self, rgb, lidar_sparse,
                 num_inference_steps=50, strength=1.0):
        self.model.eval()
        scheduler = self.noise_scheduler
        scheduler.set_timesteps(num_inference_steps)

        latents = self.encode_rgb(rgb)
        lidar_cond = self.downsample_lidar(lidar_sparse, latents.shape[-2:])

        init_index = int((num_inference_steps - 1) * strength)
        init_index = max(0, min(init_index, num_inference_steps - 1))
        init_t = scheduler.timesteps[init_index]

        noise = torch.randn_like(latents)
        sample = scheduler.add_noise(latents, noise, init_t)

        for i in range(init_index, num_inference_steps):
            t = scheduler.timesteps[i]
            t_batch = torch.full((latents.size(0),), t,
                                 device=self.device, dtype=torch.long)
            noise_pred = self.model(sample, lidar_cond, t_batch)
            sample = scheduler.step(noise_pred, t, sample).prev_sample

        return self.decode_latents(sample)
    
    @torch.no_grad()
    def generate_from_lidar(self, lidar_sparse, num_inference_steps=50):
        device = next(self.parameters()).device  
        self.model.eval()
        scheduler = self.noise_scheduler
        scheduler.set_timesteps(num_inference_steps)

        b, _, H, W = lidar_sparse.shape
        # 使用输入LiDAR的实际尺寸
        latent_shape = (b, self.vae.config.latent_channels, H // 8, W // 8)
        #latent_shape = (b, self.vae.config.latent_channels, H // 8, W // 8)
        sample = torch.randn(latent_shape, device=device)  # GPU

        lidar_cond = self.downsample_lidar(lidar_sparse.to(device), sample.shape[-2:])  # GPU

        for t in scheduler.timesteps:
            t_batch = torch.full((b,), t, device=device, dtype=torch.long)  # GPU
            noise_pred = self.model(sample, lidar_cond, t_batch)            # GPU
            sample = scheduler.step(noise_pred, t, sample).prev_sample      

        return self.decode_latents(sample.to(device)) 



if __name__ == "__main__":
    args = parse_args()
    torch.set_float32_matmul_precision("medium")

    with open(args.main_cfg_path, "r") as f:
        config = yaml.safe_load(f)

    datamodule = LidarDataModule(
        config=config,
        batch_size=config["train"]["batch_size"],
        num_workers=args.num_workers
    )

    model = LidarDiffusionModule(
        lr=float(config.get("lr", 1e-4)),
        t_max=int(config["model"].get("t_max", 1000)),
        weight_decay=float(config.get("weight_decay", 0.0)),
        model_id=config["model"].get("model_id", "runwayml/stable-diffusion-v1-5"),
        lidar_channels=config["model"].get("lidar_channels", 1),
    )

    checkpoint_cb = ModelCheckpoint(save_top_k=1, monitor="val/loss", mode="min")
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    logger = TensorBoardLogger(save_dir=config.get("log_dir", "./logs"), name=args.exp_name)

    trainer = pl.Trainer.from_argparse_args(
        args,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices="auto",
        max_epochs=config["train"].get("max_epochs", 50),
        callbacks=[checkpoint_cb, lr_monitor],
        logger=logger
    )

    trainer.fit(model, datamodule=datamodule)
