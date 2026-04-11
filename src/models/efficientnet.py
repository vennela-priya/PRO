"""
efficientnet.py
===============
EfficientNetV2-M classifier for brain tumor classification.
Primary backbone in the ensemble.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import timm
from typing import Optional


class EfficientNetV2Classifier(nn.Module):
    """
    EfficientNetV2-M with custom classification head.

    Architecture
    ------------
    EfficientNetV2-M backbone (ImageNet pretrained)
      → Global Average Pooling
      → Dropout(p)
      → Linear(hidden)
      → GELU
      → Dropout(p/2)
      → Linear(num_classes)

    Parameters
    ----------
    num_classes : number of output classes
    pretrained  : load ImageNet weights
    dropout     : dropout probability
    hidden_dim  : intermediate FC dimension
    freeze_backbone : freeze all backbone parameters
    """

    def __init__(
        self,
        num_classes: int = 3,
        pretrained: bool = True,
        dropout: float = 0.3,
        hidden_dim: int = 512,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()

        self.backbone = timm.create_model(
            "efficientnetv2_m",
            pretrained=pretrained,
            num_classes=0,          # remove default head
            global_pool="avg",
        )
        in_features = self.backbone.num_features

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
        """Freeze all backbone weights (Phase 1 training)."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze all backbone weights (Phase 2 training)."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)

    def get_feature_extractor(self) -> nn.Module:
        """Return backbone only (for XAI / feature extraction)."""
        return self.backbone

    @property
    def target_layer(self) -> nn.Module:
        """GradCAM target layer — last conv block."""
        return self.backbone.blocks[-1]
