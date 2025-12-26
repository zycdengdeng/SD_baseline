import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel

class LidarDiffusionModel(nn.Module):
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5", lidar_channels=1):
        super().__init__()
        self.unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")

        old_conv = self.unet.conv_in
        in_ch = old_conv.in_channels
        out_ch = old_conv.out_channels
        new_conv = nn.Conv2d(in_ch + lidar_channels, out_ch,
                             kernel_size=old_conv.kernel_size,
                             stride=old_conv.stride,
                             padding=old_conv.padding)

        with torch.no_grad():
            new_conv.weight[:, :in_ch] = old_conv.weight
            if lidar_channels > 0:
                nn.init.zeros_(new_conv.weight[:, in_ch:])
        new_conv.bias = old_conv.bias

        self.unet.conv_in = new_conv
        self.lidar_channels = lidar_channels

    def forward(self, noisy_latents, lidar_cond, timesteps, encoder_hidden_states=None):
        x = torch.cat([noisy_latents, lidar_cond], dim=1)
        if encoder_hidden_states is None:
            B = noisy_latents.size(0)
            encoder_hidden_states = torch.zeros(B, 1, 768, device=noisy_latents.device)
        return self.unet(x, timesteps, encoder_hidden_states).sample
