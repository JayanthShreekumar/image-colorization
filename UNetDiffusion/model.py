import torch
import torch.nn as nn
import torch.nn.functional as F


# Time Embedding to feed time step information to UNet

def sinusoidal_time_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(
        torch.arange(half, dtype=torch.float32, device=t.device) 
        * -(torch.log(torch.tensor(10000.0)) / (half - 1))
    )
    emb = t[:, None] * freqs[None]
    return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


class TimeEmbed(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, t):
        return self.layer(t)


# Residual Block of UNet (A variation of UNet called ResUNet) - matches original DDPM model by Ho

class ResBlock(nn.Module):
    def __init__(self, in_c, out_c, time_dim):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(in_c)
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1)

        self.time_emb = nn.Linear(time_dim, out_c)

        self.norm2 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1)

        self.skip = nn.Conv2d(in_c, out_c, 1) if in_c != out_c else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_emb(t_emb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


# UNet downsampling block

class Down(nn.Module):
    def __init__(self, in_c, out_c, time_dim):
        super().__init__()
        self.res1 = ResBlock(in_c, out_c, time_dim)
        self.res2 = ResBlock(out_c, out_c, time_dim)
        self.down = nn.Conv2d(out_c, out_c, 4, 2, 1)

    def forward(self, x, t_emb):
        x = self.res1(x, t_emb)
        x = self.res2(x, t_emb)
        skip = x
        x = self.down(x)
        return x, skip


# UNet upsampling block

class Up(nn.Module):
    def __init__(self, in_c, skip_c, out_c, time_dim):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, 4, 2, 1)
        self.res1 = ResBlock(out_c + skip_c, out_c, time_dim)
        self.res2 = ResBlock(out_c, out_c, time_dim)

    def forward(self, x, skip, t_emb):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)  # concat after upsampling
        x = self.res1(x, t_emb)
        x = self.res2(x, t_emb)
        return x


# Full model definition

class UNetDiffusion(nn.Module):
    def __init__(self, in_channels=1, out_channels=3, time_dim=256):
        super().__init__()

        # Time embedding
        self.time_mlp = TimeEmbed(time_dim)

        # Downsampling
        self.down1 = Down(in_channels, 64, time_dim)
        self.down2 = Down(64, 128, time_dim)
        self.down3 = Down(128, 256, time_dim)
        self.down4 = Down(256, 512, time_dim)

        # Bottleneck
        self.bot1 = ResBlock(512, 512, time_dim)
        self.bot2 = ResBlock(512, 512, time_dim)

        # Upsampling
        self.up1 = Up(in_c=512, skip_c=512, out_c=256, time_dim=time_dim)  # bottleneck -> down4 skip
        self.up2 = Up(in_c=256, skip_c=256, out_c=128, time_dim=time_dim)  # up1 -> down3 skip
        self.up3 = Up(in_c=128, skip_c=128, out_c=64, time_dim=time_dim)   # up2 -> down2 skip
        self.up4 = Up(in_c=64, skip_c=64, out_c=32, time_dim=time_dim)     # up3 -> down1 skip

        self.final = nn.Conv2d(32, out_channels, 1)

    def forward(self, x, t):
        # Time embedding
        t_emb = sinusoidal_time_embedding(t, self.time_mlp.layer[0].in_features)
        t_emb = self.time_mlp(t_emb)

        # Down
        x, s1 = self.down1(x, t_emb)
        x, s2 = self.down2(x, t_emb)
        x, s3 = self.down3(x, t_emb)
        x, s4 = self.down4(x, t_emb)

        # Bottleneck
        x = self.bot1(x, t_emb)
        x = self.bot2(x, t_emb)

        # Up
        x = self.up1(x, s4, t_emb)
        x = self.up2(x, s3, t_emb)
        x = self.up3(x, s2, t_emb)
        x = self.up4(x, s1, t_emb)

        return self.final(x)