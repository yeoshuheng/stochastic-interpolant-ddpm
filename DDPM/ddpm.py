from .backward import UNetBlock
from .embedding import SinusoidalPositionEmbeddings
import torch
import torch.nn as nn

class DDPM(nn.Module):
    def __init__(self):
        super().__init__()
        image_channels = 3 #rgb
        down_sample_channels = [64, 128, 256, 512, 1024]
        up_sample_channels = [1024, 512, 256, 128, 64]
        time_emb_dim = 32
        # Generates the time embeddings.
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )
        # Initial convolution (3 -> 64)
        self.conv0 = nn.Conv2d(image_channels, down_sample_channels[0], 3, padding = 1)
        self.downsample = nn.ModuleList([
            UNetBlock(down_sample_channels[i], 
                      down_sample_channels[i + 1], 
                      time_emb_dim) for i in range(len(down_sample_channels) - 1)
        ])
        self.upsample = nn.ModuleList([
            UNetBlock(up_sample_channels[i],
                      up_sample_channels[i + 1],
                      time_emb_dim) for i in range(len(up_sample_channels) - 1)
        ])
        # Final convolution (64 -> 3)
        self.output = nn.Conv2d(up_sample_channels[-1], image_channels, 1)

    def forward(self, x, t):
        t = self.time_mlp(t)
        x = self.conv0(x)
        residuals = []
        for down_block in self.downsample:
            x = down_block(x, t)
            residuals.append(x)
        for up_block in self.upsample:
            res_x = residuals.pop()
            x = torch.cat((res_x, x), dim = 1) # Combine residual channels with current channel
            x = up_block(x, t)
        return self.output(x)