import torch
import torch.nn as nn
import torch.nn.functional as F

class DownBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, 3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class UNetVAE(nn.Module):
    def __init__(self, in_channels=1, out_channels=3, latent_dim=256):
        super().__init__()

        self.down1 = DownBlock(1,   64)     # 256 → 128
        self.down2 = DownBlock(64,  128)    # 128 → 64
        self.down3 = DownBlock(128, 256)    # 64 → 32
        self.down4 = DownBlock(256, 512)    # 32 → 16
        self.down5 = DownBlock(512, 512)    # 16 → 8   (bottleneck)

        self.flatten_dim = 512 * 8 * 8
        self.fc_mu     = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, self.flatten_dim)

        self.up0  = UpBlock(512, 512)             # 8 → 16
        self.dec1 = ConvBlock(512 + 512, 512)

        self.up1  = UpBlock(512, 256)             # 16 → 32
        self.dec2 = ConvBlock(256 + 256, 256)

        self.up2  = UpBlock(256, 128)             # 32 → 64
        self.dec3 = ConvBlock(128 + 128, 128)

        self.up3  = UpBlock(128, 64)              # 64 → 128
        self.dec4 = ConvBlock(64 + 64, 64)

        self.up4  = UpBlock(64, 32)               # 128 → 256
        self.dec5 = ConvBlock(32, 32)

        self.out_conv = nn.Conv2d(32, out_channels, 3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):

        s1 = self.down1(x)     # 64×128×128
        s2 = self.down2(s1)    # 128×64×64
        s3 = self.down3(s2)    # 256×32×32
        s4 = self.down4(s3)    # 512×16×16
        bottleneck = self.down5(s4)  # 512×8×8

        flat = bottleneck.reshape(x.size(0), -1)
        mu = self.fc_mu(flat)
        logvar = self.fc_logvar(flat)
        z = self.reparameterize(mu, logvar)

        decoded = self.fc_decode(z).reshape(-1, 512, 8, 8)

        up = self.up0(decoded)                  # → 16×16
        up = self.dec1(torch.cat([up, s4], dim=1))

        up = self.up1(up)                       # → 32×32
        up = self.dec2(torch.cat([up, s3], dim=1))

        up = self.up2(up)                       # → 64×64
        up = self.dec3(torch.cat([up, s2], dim=1))

        up = self.up3(up)                       # → 128×128
        up = self.dec4(torch.cat([up, s1], dim=1))

        up = self.up4(up)                       # → 256×256
        up = self.dec5(up)

        out = self.out_conv(up)
        return self.sigmoid(out), mu, logvar