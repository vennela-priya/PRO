"""segmentation/losses.py — Loss functions for binary tumour segmentation."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TverskyLoss(nn.Module):
    """Tversky loss — penalises FP more than FN when alpha > beta."""

    def __init__(self, alpha=0.7, beta=0.3, smooth=1.0):
        super().__init__()
        self.alpha  = alpha
        self.beta   = beta
        self.smooth = smooth

    def forward(self, logits, targets):
        probs     = torch.sigmoid(logits)
        probs_f   = probs.view(probs.size(0),     -1)
        targets_f = targets.view(targets.size(0), -1)
        TP = (probs_f * targets_f).sum(1)
        FP = (probs_f * (1 - targets_f)).sum(1)
        FN = ((1 - probs_f) * targets_f).sum(1)
        tversky = (TP + self.smooth) / (
            TP + self.alpha * FP + self.beta * FN + self.smooth)
        return (1 - tversky).mean()


class DiceLoss(nn.Module):
    """Soft Dice loss."""

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs     = torch.sigmoid(logits)
        probs_f   = probs.view(probs.size(0),     -1)
        targets_f = targets.view(targets.size(0), -1)
        inter = (probs_f * targets_f).sum(1)
        dice  = (2 * inter + self.smooth) / (
            probs_f.sum(1) + targets_f.sum(1) + self.smooth)
        return (1 - dice).mean()


class BoundaryLoss(nn.Module):
    """BCE weighted 2x at tumour boundary pixels."""

    def __init__(self, boundary_weight=2.0, kernel_size=5):
        super().__init__()
        self.boundary_weight = boundary_weight
        self.kernel_size     = kernel_size

    def forward(self, logits, targets):
        bce_map = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none')
        with torch.no_grad():
            pad      = self.kernel_size // 2
            dil      = F.max_pool2d(targets,  self.kernel_size, stride=1, padding=pad)
            ero      = -F.max_pool2d(-targets, self.kernel_size, stride=1, padding=pad)
            boundary = (dil - ero).clamp(0, 1)
            weight   = 1.0 + self.boundary_weight * boundary
        return (bce_map * weight).mean()


class CombinedSegLoss(nn.Module):
    """
    0.5 * Tversky + 0.3 * Dice + 0.2 * Boundary.
    forward() returns a SCALAR tensor (not a tuple).
    """

    def __init__(
        self,
        tversky_alpha=0.7, tversky_beta=0.3,
        boundary_weight=2.0,
        w_tversky=0.5, w_dice=0.3, w_boundary=0.2,
        smooth=1.0,
    ):
        super().__init__()
        self.tversky  = TverskyLoss(tversky_alpha, tversky_beta, smooth)
        self.dice     = DiceLoss(smooth)
        self.boundary = BoundaryLoss(boundary_weight)
        self.w_t = w_tversky
        self.w_d = w_dice
        self.w_b = w_boundary

    def forward(self, logits, targets):
        return (
            self.w_t * self.tversky(logits, targets) +
            self.w_d * self.dice(logits, targets)    +
            self.w_b * self.boundary(logits, targets)
        )
