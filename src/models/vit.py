"""
vit.py
======
Vision Transformer ViT-B/16 fine-tuned for brain tumor classification.
Tertiary backbone in the ensemble.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import timm
from typing import Optional


class ViTClassifier(nn.Module):
    """
    ViT-B/16 with custom MLP classification head.

    Accepts 224x224 images (16x16 patches → 196 tokens).

    Parameters
    ----------
    num_classes     : number of output classes
    pretrained      : load ImageNet-21k pretrained weights
    dropout         : dropout applied before classifier head
    freeze_backbone : freeze all transformer parameters (Phase 1)
    """

    def __init__(
        self,
        num_classes: int = 3,
        pretrained: bool = True,
        dropout: float = 0.1,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()

        self.backbone = timm.create_model(
            "vit_base_patch16_224",
            pretrained=pretrained,
            num_classes=0,          # strip head
            global_pool="token",   # use CLS token
        )
        in_features = self.backbone.embed_dim   # 768 for ViT-B

        self.head_drop = nn.Dropout(p=dropout)
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )

        self._init_head()
        if freeze_backbone:
            self.freeze_backbone()

    def _init_head(self) -> None:
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def freeze_backbone(self) -> None:
        """Freeze transformer blocks (keep patch embed trainable for speed)."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self, unfreeze_last_n_blocks: Optional[int] = None) -> None:
        """Unfreeze backbone. Optionally unfreeze only the last N transformer blocks."""
        if unfreeze_last_n_blocks is None:
            for param in self.backbone.parameters():
                param.requires_grad = True
        else:
            # Freeze all first, then selectively unfreeze last N blocks
            for param in self.backbone.parameters():
                param.requires_grad = False
            total_blocks = len(self.backbone.blocks)
            for i in range(total_blocks - unfreeze_last_n_blocks, total_blocks):
                for param in self.backbone.blocks[i].parameters():
                    param.requires_grad = True
            for param in self.backbone.norm.parameters():
                param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)          # [B, 768]
        features = self.head_drop(features)
        return self.classifier(features)

    @property
    def target_layer(self) -> nn.Module:
        """GradCAM target — last attention block norm."""
        return self.backbone.blocks[-1].norm1
