"""
resnet_cbam.py
==============
ResNet-50 with CBAM (Convolutional Block Attention Module).
Secondary backbone in the ensemble.

CBAM Reference: Woo et al., ECCV 2018.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Optional


# ---------------------------------------------------------------------------
# CBAM Components
# ---------------------------------------------------------------------------

class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation style channel attention."""

    def __init__(self, in_channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        mid = max(in_channels // reduction, 1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, in_channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """Spatial attention map from avg + max pooled channel features."""

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(concat))


class CBAM(nn.Module):
    """Full CBAM: Channel Attention + Spatial Attention."""

    def __init__(self, in_channels: int, reduction: int = 16, spatial_kernel: int = 7) -> None:
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


# ---------------------------------------------------------------------------
# ResNet-50 + CBAM Classifier
# ---------------------------------------------------------------------------

class ResNetCBAMClassifier(nn.Module):
    """
    ResNet-50 backbone with CBAM attention injected after each stage,
    followed by custom classification head.

    Parameters
    ----------
    num_classes   : output classes
    pretrained    : load ImageNet weights
    cbam_reduction: channel reduction ratio for CBAM
    dropout       : dropout probability
    hidden_dim    : FC hidden dimension
    freeze_backbone : freeze backbone parameters
    """

    def __init__(
        self,
        num_classes: int = 3,
        pretrained: bool = True,
        cbam_reduction: int = 16,
        dropout: float = 0.3,
        hidden_dim: int = 512,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()

        # Load backbone without head
        self.backbone = timm.create_model(
            "resnet50",
            pretrained=pretrained,
            num_classes=0,
            global_pool="",          # disable pooling, we'll handle it
        )

        # Inject CBAM after layer3 (1024-ch) and layer4 (2048-ch)
        self.cbam3 = CBAM(1024, reduction=cbam_reduction)
        self.cbam4 = CBAM(2048, reduction=cbam_reduction)

        self.pool = nn.AdaptiveAvgPool2d(1)

        in_features = 2048
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout / 2),
            nn.Linear(hidden_dim, num_classes),
        )

        self._init_classifier()
        if freeze_backbone:
            self.freeze_backbone()

    def _init_classifier(self) -> None:
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Manual forward through ResNet stages
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.act1(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.cbam3(x)           # CBAM after layer3
        x = self.backbone.layer4(x)
        x = self.cbam4(x)           # CBAM after layer4

        x = self.pool(x).flatten(1)
        return self.classifier(x)

    @property
    def target_layer(self) -> nn.Module:
        """GradCAM target layer."""
        return self.backbone.layer4[-1]
