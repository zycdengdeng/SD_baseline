import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_bn_relu(in_ch, out_ch, k=3, s=1, p=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            conv_bn_relu(in_ch, out_ch),
            conv_bn_relu(out_ch, out_ch),
        )
    def forward(self, x):
        return self.net(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x):
        return self.conv(self.pool(x))

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x, skip):
        x = self.up(x)
        dh = skip.size(2) - x.size(2)
        dw = skip.size(3) - x.size(3)
        if dh or dw:
            x = F.pad(x, (0, dw, 0, dh))
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)

# --------- timestep embedding (sinusoidal) ----------
def timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """
    timesteps: (B,) long
    returns: (B, dim) sinusoidal embedding
    """
    device = timesteps.device
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(0, half, device=device, dtype=torch.float32) / max(half - 1, 1))
    args = timesteps.float()[:, None] * freqs[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, 2*half)
    if dim % 2 == 1:
        emb = F.pad(emb, (0,1))
    return emb

class LidarDiffusionUNet(nn.Module):
    """
    Latent diffusion UNet:
    Input: concat([noisy_latents(4), lidar_cond(?channels)]) -> (B,in_ch,H,W)
    Extra: timestep embedding t
    Output: predicted noise for latents (B,4,H,W)
    """
    def __init__(self, in_ch=5, out_ch=4, base=64, time_dim=128):
        super().__init__()
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, base * 16),
            nn.SiLU(),
            nn.Linear(base * 16, base * 16),
        )

        self.inc = DoubleConv(in_ch, base)
        self.d1  = Down(base, base*2)
        self.d2  = Down(base*2, base*4)
        self.d3  = Down(base*4, base*8)
        self.bn  = DoubleConv(base*8, base*16)   # bottleneck (channels match time proj)

        self.u3  = Up(base*16 + base*8, base*8)
        self.u2  = Up(base*8  + base*4, base*4)
        self.u1  = Up(base*4  + base*2, base*2)
        self.u0  = Up(base*2  + base,   base)
        self.out = nn.Conv2d(base, out_ch, kernel_size=1)

    def forward(self, noisy_latents, lidar_cond, timesteps: torch.Tensor):
        # concat inputs
        x = torch.cat([noisy_latents, lidar_cond], dim=1)  # (B,in_ch,H,W)

        # encoder
        x0 = self.inc(x)
        x1 = self.d1(x0)
        x2 = self.d2(x1)
        x3 = self.d3(x2)

        # time embedding
        t_emb = timestep_embedding(timesteps, self.time_dim)  # (B, time_dim)
        t_emb = self.time_mlp(t_emb)                          # (B, 16b)
        t_emb = t_emb[:, :, None, None]                       # (B, 16b, 1, 1)

        xb = self.bn(x3) + t_emb

        # decoder
        x  = self.u3(xb, x3)
        x  = self.u2(x,  x2)
        x  = self.u1(x,  x1)
        x  = self.u0(x,  x0)

        return self.out(x)  # (B,4,H,W)
