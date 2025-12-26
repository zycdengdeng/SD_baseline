import argparse
import yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from src.dataset import Scannetdataset
from src.dataset.Matterport3D import MP3Ddataset

from src.lightning_depth import LidarDiffusionModule


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('main_cfg_path', type=str, help='Path to YAML config')
    parser.add_argument('--exp_name', type=str, default='lidar_diffusion_exp')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size per device')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--ckpt_path', type=str, default=None, help='optional checkpoint to load')
    # Expose common PL Trainer flags (e.g., --max_epochs, --devices, --accelerator, etc.)
    parser = pl.Trainer.add_argparse_args(parser)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    torch.set_float32_matmul_precision('medium')

    with open(args.main_cfg_path, 'rb') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    config.setdefault('train', {})
    config.setdefault('dataset', {})
    config.setdefault('model', {})
    config['train']['max_epochs'] = getattr(args, 'max_epochs', config['train'].get('max_epochs', 50))
    config['train']['batch_size'] = args.batch_size

    config['model'].setdefault('base', 64)        # UNet base channels
    config['model'].setdefault('out_ch', 3)       # predict 3-ch (colored)
    config['model'].setdefault('t_max', 1000)     # diffusion train timesteps
    config['model'].setdefault('time_dim', 128)   # timestep embedding dim
    config.setdefault('lr', 1e-4)
    config.setdefault('weight_decay', 0.0)
    config.setdefault('log_dir', './logs/tb_logs')

    ds_name = config['dataset'].get('name', 'mp3d')
    if ds_name == 'mp3d':
        train_dataset = MP3Ddataset(config['dataset'], mode='train')
        val_dataset   = MP3Ddataset(config['dataset'], mode='val')
    elif ds_name == 'scannet':
        train_dataset = Scannetdataset(config['dataset'], mode='train')
        val_dataset   = Scannetdataset(config['dataset'], mode='val')
    else:
        raise ValueError(f"Unknown dataset name: {ds_name}")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['train']['batch_size'],
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    # -------- Model --------
    model = LidarDiffusionModule(
        lr=float(config['lr']),
        base=int(config['model']['base']),
        out_ch=int(config['model']['out_ch']),
        t_max=int(config['model']['t_max']),
        weight_decay=float(config['weight_decay']),
        time_dim=int(config['model']['time_dim'])
    )

    if args.ckpt_path:
        state = torch.load(args.ckpt_path, map_location='cpu')
        sd = state.get('state_dict', state) if isinstance(state, dict) else state
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[load_state_dict] missing={len(missing)} unexpected={len(unexpected)}")

    checkpoint_cb = ModelCheckpoint(
        save_top_k=1,
        monitor="val/loss",
        mode="min",
        save_last=True,
        filename='epoch={epoch}-val_loss={val/loss:.4f}'
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    logger = TensorBoardLogger(
        save_dir=config['log_dir'],
        name=args.exp_name,
        default_hp_metric=False
    )

    trainer = pl.Trainer.from_argparse_args(
        args,
        accelerator=('gpu' if torch.cuda.is_available() else 'cpu'),
        devices='auto',
        max_epochs=config['train']['max_epochs'],
        callbacks=[checkpoint_cb, lr_monitor],
        logger=logger
    )

    trainer.fit(model, train_loader, val_loader)
