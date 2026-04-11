"""
unet.py
=======
U-Net segmentation model with EfficientNet-B4 encoder.
Outputs binary tumor mask + bounding box coordinates.
Loss: Dice + BCE (λ=0.5 each).
Target DSC ≥ 0.91.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

logger = logging.getLogger(__name__)


class BrainTumorUNet(nn.Module):
    """
    EfficientNet-B4 encoder + U-Net decoder for binary tumor segmentation.

    Parameters
    ----------
    encoder_name   : segmentation_models_pytorch encoder name
    encoder_weights: pretrained weights ('imagenet' | None)
    in_channels    : input channels (3 for RGB)
    classes        : output classes (1 for binary mask)
    """

    def __init__(
        self,
        encoder_name: str = "efficientnet-b4",
        encoder_weights: str = "imagenet",
        in_channels: int = 3,
        classes: int = 1,
    ) -> None:
        super().__init__()

        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=None,          # raw logits; sigmoid applied in loss/inference
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits [B, 1, H, W]."""
        return self.unet(x)

    def predict_mask(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Return binary mask [B, 1, H, W]."""
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
            return (probs > threshold).float()

    def predict_bbox(self, x: torch.Tensor, threshold: float = 0.5) -> list[Optional[Tuple]]:
        """
        Predict bounding boxes from binary masks.

        Returns list of (x1, y1, x2, y2) or None if no tumor found.
        """
        masks = self.predict_mask(x, threshold)  # [B, 1, H, W]
        bboxes = []
        for mask in masks[:, 0]:  # iterate batch
            ys, xs = torch.where(mask > 0)
            if len(ys) == 0:
                bboxes.append(None)
            else:
                bboxes.append((xs.min().item(), ys.min().item(),
                               xs.max().item(), ys.max().item()))
        return bboxes

    def freeze_encoder(self) -> None:
        for param in self.unet.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self) -> None:
        for param in self.unet.encoder.parameters():
            param.requires_grad = True


# ---------------------------------------------------------------------------
# Combined Dice + BCE Loss
# ---------------------------------------------------------------------------

class DiceBCELoss(nn.Module):
    """
    Combined Dice + Binary Cross-Entropy loss for segmentation.

        L = λ_dice * L_dice + λ_bce * L_bce

    Parameters
    ----------
    dice_weight : weight for Dice component (default 0.5)
    bce_weight  : weight for BCE component (default 0.5)
    smooth      : smoothing constant for Dice
    """

    def __init__(
        self,
        dice_weight: float = 0.5,
        bce_weight: float = 0.5,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def dice_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = self.bce(logits, targets)
        dice = self.dice_loss(logits, targets)
        return self.dice_weight * dice + self.bce_weight * bce


# ---------------------------------------------------------------------------
# DSC metric helper
# ---------------------------------------------------------------------------

def dice_score(
    pred_mask: torch.Tensor,
    true_mask: torch.Tensor,
    smooth: float = 1.0,
) -> float:
    """Compute batch-averaged Dice Similarity Coefficient."""
    pred = (pred_mask > 0.5).float()
    intersection = (pred * true_mask).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + true_mask.sum(dim=(1, 2, 3))
    dsc = (2 * intersection + smooth) / (union + smooth)
    return dsc.mean().item()
