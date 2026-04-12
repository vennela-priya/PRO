"""segmentation/attention_unet.py — Attention U-Net (Oktay et al. 2018)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class AttentionGate(nn.Module):
    """Additive attention gate — gates skip connection using decoder signal."""

    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g,   F_int, 1, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l,   F_int, 1, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # g = gating signal (decoder), x = skip connection (encoder)
        g_up = F.interpolate(g, size=x.shape[2:], mode='bilinear', align_corners=True)
        psi  = self.relu(self.W_g(g_up) + self.W_x(x))
        psi  = self.psi(psi)          # (B,1,H,W) attention map
        return x * psi                # gated skip


class AttentionUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base_filters=16, bilinear=True):
        super().__init__()
        f = base_filters
        self.pool = nn.MaxPool2d(2)

        # Encoder
        self.enc1      = ConvBlock(in_channels, f)
        self.enc2      = ConvBlock(f,     f * 2)
        self.enc3      = ConvBlock(f * 2, f * 4)
        self.enc4      = ConvBlock(f * 4, f * 8)
        self.bottleneck = ConvBlock(f * 8, f * 16)

        # Decoder with attention gates
        self.up4  = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.att4 = AttentionGate(F_g=f * 16, F_l=f * 8, F_int=f * 4)
        self.dec4 = ConvBlock(f * 16 + f * 8, f * 8)

        self.up3  = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.att3 = AttentionGate(F_g=f * 8,  F_l=f * 4, F_int=f * 2)
        self.dec3 = ConvBlock(f * 8  + f * 4, f * 4)

        self.up2  = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.att2 = AttentionGate(F_g=f * 4,  F_l=f * 2, F_int=f)
        self.dec2 = ConvBlock(f * 4  + f * 2, f * 2)

        self.up1  = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.att1 = AttentionGate(F_g=f * 2,  F_l=f,     F_int=max(f // 2, 1))
        self.dec1 = ConvBlock(f * 2  + f,      f)

        self.outc = nn.Conv2d(f, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bottleneck(self.pool(e4))

        g4 = self.up4(b)
        d4 = self.dec4(torch.cat([self.att4(g4, e4), g4], dim=1))

        g3 = self.up3(d4)
        d3 = self.dec3(torch.cat([self.att3(g3, e3), g3], dim=1))

        g2 = self.up2(d3)
        d2 = self.dec2(torch.cat([self.att2(g2, e2), g2], dim=1))

        g1 = self.up1(d2)
        d1 = self.dec1(torch.cat([self.att1(g1, e1), g1], dim=1))

        return self.outc(d1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
