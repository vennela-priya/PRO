"""
segmentation/losses.py
======================
Loss functions for binary tumour segmentation.

Available
---------
dice_loss       — pure Dice loss (1 - Dice coefficient)
BCEDiceLoss     — weighted sum of BCE + Dice (most stable in practice)

Both accept raw logits (no sigmoid needed externally).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    smooth: float = 1.0,
) -> torch.Tensor:
    """
    Dice loss computed from raw logits.

    Parameters
    ----------
    logits  : (B, 1, H, W)  — raw model output (no sigmoid)
    targets : (B, 1, H, W)  — binary ground-truth {0, 1}
    smooth  : Laplace smoothing term (avoids divide-by-zero)

    Returns
    -------
    Scalar tensor — Dice loss value in [0, 1].
    """
    probs = torch.sigmoid(logits)

    # Flatten spatial dims
    probs   = probs.view(probs.size(0),   -1)   # (B, H*W)
    targets = targets.view(targets.size(0), -1) # (B, H*W)

    intersection = (probs * targets).sum(dim=1)             # (B,)
    dice = (2.0 * intersection + smooth) / (
        probs.sum(dim=1) + targets.sum(dim=1) + smooth
    )                                                        # (B,)
    return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    """
    Combined Binary Cross-Entropy + Dice loss.

        L = alpha * BCE(logits, targets) + (1 - alpha) * Dice(logits, targets)

    Parameters
    ----------
    alpha  : weight for BCE term (0.5 → equal weighting, default)
    smooth : smoothing constant for Dice denominator

    Notes
    -----
    * BCE stabilises gradients early in training (avoids flat Dice landscape).
    * Dice aligns the objective with the evaluation metric.
    * alpha=0.5 is a strong default; tune towards 0.3 if masks are large,
      towards 0.7 if masks are small relative to image.
    """

    def __init__(self, alpha: float = 0.5, smooth: float = 1.0):
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0,1], got {alpha}")
        self.alpha  = alpha
        self.smooth = smooth

    def forward(
        self,
        logits:  torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Parameters
        ----------
        logits  : (B, 1, H, W)  raw model output
        targets : (B, 1, H, W)  binary float {0.0, 1.0}

        Returns
        -------
        total_loss : scalar tensor (backpropagatable)
        components : dict with keys 'bce', 'dice', 'total' (float values
                     for logging — already detached)
        """
        bce  = F.binary_cross_entropy_with_logits(logits, targets)
        dice = dice_loss(logits, targets, self.smooth)

        total = self.alpha * bce + (1.0 - self.alpha) * dice

        components = {
            "bce":   bce.item(),
            "dice":  dice.item(),
            "total": total.item(),
        }
        return total, components
