"""
segmentation/unet.py
====================
U-Net implementation in PyTorch — Ronneberger et al. (2015) with
modern improvements (BatchNorm, bilinear upsampling option).

Architecture
------------
  Encoder : 4 DoubleConv + MaxPool blocks (stride-2 downsampling)
  Bridge  : DoubleConv bottleneck
  Decoder : 4 Up blocks (bilinear upsample + skip concat + DoubleConv)
  Head    : 1×1 Conv → out_channels logits

Input  : (B, C_in, H, W)   — C_in = 1 (grayscale) or 3 (RGB)
Output : (B, 1,   H, W)   — raw logits; apply sigmoid for probability map

Usage
-----
    model = UNet(in_channels=1, base_filters=64)
    logits = model(x)                              # (B,1,H,W)
    prob, binary = model.predict_mask(x, threshold=0.5)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Building blocks ───────────────────────────────────────────────────────

class DoubleConv(nn.Module):
    """Two consecutive Conv2d → BatchNorm → ReLU layers."""

    def __init__(self, in_ch: int, out_ch: int, mid_ch: int | None = None):
        super().__init__()
        mid_ch = mid_ch or out_ch
        self.block = nn.Sequential(
            nn.Conv2d(in_ch,  mid_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    """MaxPool2d(2) → DoubleConv  (encoder step, halves spatial dims)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool_conv(x)


class Up(nn.Module):
    """
    Upsample → concatenate skip connection → DoubleConv  (decoder step).

    Parameters
    ----------
    in_ch    : total channels after concatenation  (skip + upsampled)
    out_ch   : output channels
    bilinear : True → bilinear upsample (memory-efficient),
               False → transposed convolution (learnable upsampling)
    """

    def __init__(self, in_ch: int, out_ch: int, bilinear: bool = True):
        super().__init__()
        if bilinear:
            self.up   = nn.Upsample(scale_factor=2, mode="bilinear",
                                    align_corners=True)
            self.conv = DoubleConv(in_ch, out_ch, in_ch // 2)
        else:
            self.up   = nn.ConvTranspose2d(in_ch, in_ch // 2,
                                           kernel_size=2, stride=2)
            self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        x1 : upsampled tensor from deeper level
        x2 : skip-connection tensor from encoder (same resolution before pooling)
        """
        x1 = self.up(x1)

        # Pad x1 to exactly match x2 spatial size (handles odd dimensions)
        dH = x2.size(2) - x1.size(2)
        dW = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [dW // 2, dW - dW // 2,
                        dH // 2, dH - dH // 2])

        return self.conv(torch.cat([x2, x1], dim=1))


class OutConv(nn.Module):
    """1×1 convolution to map to desired number of output channels."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# ── U-Net ─────────────────────────────────────────────────────────────────

class UNet(nn.Module):
    """
    Standard U-Net with 4 encoder + 4 decoder levels.

    Parameters
    ----------
    in_channels  : 1 (grayscale) or 3 (RGB)
    out_channels : 1 for binary segmentation, N for multi-class
    base_filters : filters at first level (doubles each level); default 64
    bilinear     : bilinear upsample (True) vs transposed conv (False)

    Approximate parameter counts
    ─────────────────────────────
    base_filters=32 →  ~7.8 M params   (lightweight, fits 4 GB VRAM)
    base_filters=64 → ~31.0 M params   (standard, fits 8 GB VRAM)
    """

    def __init__(
        self,
        in_channels:  int  = 1,
        out_channels: int  = 1,
        base_filters: int  = 64,
        bilinear:     bool = True,
    ):
        super().__init__()
        f      = base_filters
        factor = 2 if bilinear else 1

        # ── Encoder ───────────────────────────────────────────────────
        self.inc   = DoubleConv(in_channels, f)
        self.down1 = Down(f,            f * 2)
        self.down2 = Down(f * 2,        f * 4)
        self.down3 = Down(f * 4,        f * 8)
        self.down4 = Down(f * 8,        f * 16 // factor)   # bottleneck

        # ── Decoder ───────────────────────────────────────────────────
        self.up1  = Up(f * 16,          f * 8  // factor, bilinear)
        self.up2  = Up(f * 8,           f * 4  // factor, bilinear)
        self.up3  = Up(f * 4,           f * 2  // factor, bilinear)
        self.up4  = Up(f * 2,           f,                bilinear)
        self.outc = OutConv(f, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)   # bottleneck

        # Decoder (skip connections)
        x  = self.up1(x5, x4)
        x  = self.up2(x,  x3)
        x  = self.up3(x,  x2)
        x  = self.up4(x,  x1)

        return self.outc(x)    # raw logits  [B, out_ch, H, W]

    @torch.no_grad()
    def predict_mask(
        self,
        x:         torch.Tensor,
        threshold: float = 0.5,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convenience method for inference.

        Returns
        -------
        binary : (B, 1, H, W) float32 {0.0, 1.0}
        prob   : (B, 1, H, W) float32 [0, 1] probability map
        """
        prob   = torch.sigmoid(self.forward(x))
        binary = (prob >= threshold).float()
        return binary, prob

    def count_parameters(self) -> int:
        """Return number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
