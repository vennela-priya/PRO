"""
losses.py
=========
Classification losses:
- WeightedLabelSmoothingCrossEntropy
- FocalLoss
- CombinedLoss (CrossEntropy + Focal)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedLabelSmoothingCE(nn.Module):
    """
    Cross-entropy with label smoothing and optional class weights.

    Parameters
    ----------
    num_classes     : number of output classes
    smoothing       : label smoothing epsilon (0.1 recommended)
    class_weights   : 1-D tensor of class weights (for imbalance)
    """

    def __init__(
        self,
        num_classes: int = 3,
        smoothing: float = 0.1,
        class_weights: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.register_buffer("class_weights", class_weights)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits  : [B, C] raw logits
        targets : [B]   integer class labels
        """
        log_probs = F.log_softmax(logits, dim=-1)

        # Smooth targets
        with torch.no_grad():
            smooth_targets = torch.full_like(log_probs, self.smoothing / (self.num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        loss = -(smooth_targets * log_probs).sum(dim=-1)  # [B]

        if self.class_weights is not None:
            w = self.class_weights[targets]
            loss = loss * w

        return loss.mean()


class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al., 2017) for handling class imbalance.

        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Parameters
    ----------
    gamma         : focusing parameter (default 2.0)
    class_weights : alpha weights per class
    reduction     : 'mean' | 'sum' | 'none'
    """

    def __init__(
        self,
        gamma: float = 2.0,
        class_weights: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.register_buffer("class_weights", class_weights)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, weight=self.class_weights, reduction="none")
        pt = torch.exp(-ce_loss)
        focal = (1.0 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal.mean()
        elif self.reduction == "sum":
            return focal.sum()
        return focal


class CombinedClassificationLoss(nn.Module):
    """
    Weighted sum of label-smoothed CE + Focal loss.

    L = α * CE_smooth + (1-α) * Focal
    """

    def __init__(
        self,
        num_classes: int = 3,
        smoothing: float = 0.1,
        focal_gamma: float = 2.0,
        alpha: float = 0.7,
        class_weights: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.ce = WeightedLabelSmoothingCE(num_classes, smoothing, class_weights)
        self.focal = FocalLoss(focal_gamma, class_weights)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.alpha * self.ce(logits, targets) + (1 - self.alpha) * self.focal(logits, targets)
