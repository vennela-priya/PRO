"""
segmentation/metrics.py
=======================
Evaluation metrics for binary segmentation masks.

All functions accept either:
  • torch.Tensor  (B, 1, H, W) float32
  • numpy.ndarray (H, W) or (B, H, W) float32/uint8

and return a plain Python float (mean over batch).
"""
from __future__ import annotations

import numpy as np
import torch


# ── Internal helpers ──────────────────────────────────────────────────────

def _to_binary_tensors(
    pred:   "torch.Tensor | np.ndarray",
    target: "torch.Tensor | np.ndarray",
    threshold: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Ensure both inputs are binary float tensors of shape (B, H*W)."""
    def _cast(x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x.astype(np.float32))
        return x.float()

    pred   = _cast(pred)
    target = _cast(target)

    # Threshold probability maps
    if pred.max() <= 1.0 and pred.dtype == torch.float32:
        pred = (pred >= threshold).float()

    # Flatten to (B, N)
    pred   = pred.view(-1,   pred.numel() //   pred.shape[0])
    target = target.view(-1, target.numel() // target.shape[0])

    return pred, target


# ── Public metrics ────────────────────────────────────────────────────────

def dice_score(
    pred:      "torch.Tensor | np.ndarray",
    target:    "torch.Tensor | np.ndarray",
    threshold: float = 0.5,
    smooth:    float = 1.0,
) -> float:
    """
    Dice Similarity Coefficient  =  2·|P ∩ T| / (|P| + |T|)

    Returns
    -------
    Mean Dice over the batch, in [0, 1].
    1.0 = perfect overlap, 0.0 = no overlap.
    """
    pred, target = _to_binary_tensors(pred, target, threshold)

    intersection = (pred * target).sum(dim=1)
    dsc = (2.0 * intersection + smooth) / (
        pred.sum(dim=1) + target.sum(dim=1) + smooth
    )
    return float(dsc.mean().item())


def iou_score(
    pred:      "torch.Tensor | np.ndarray",
    target:    "torch.Tensor | np.ndarray",
    threshold: float = 0.5,
    smooth:    float = 1.0,
) -> float:
    """
    Intersection over Union (Jaccard Index)  =  |P ∩ T| / |P ∪ T|

    Returns
    -------
    Mean IoU over the batch, in [0, 1].
    """
    pred, target = _to_binary_tensors(pred, target, threshold)

    intersection = (pred * target).sum(dim=1)
    union        = pred.sum(dim=1) + target.sum(dim=1) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return float(iou.mean().item())


def pixel_accuracy(
    pred:      "torch.Tensor | np.ndarray",
    target:    "torch.Tensor | np.ndarray",
    threshold: float = 0.5,
) -> float:
    """
    Pixel-wise accuracy  =  correct pixels / total pixels.

    Returns
    -------
    Mean pixel accuracy over the batch, in [0, 1].
    """
    pred, target = _to_binary_tensors(pred, target, threshold)
    correct = (pred == target).float().sum(dim=1)
    total   = pred.size(1)
    return float((correct / total).mean().item())


def compute_all_metrics(
    pred:      "torch.Tensor | np.ndarray",
    target:    "torch.Tensor | np.ndarray",
    threshold: float = 0.5,
) -> dict[str, float]:
    """
    Compute Dice, IoU, and pixel accuracy in one call.

    Returns
    -------
    {'dice': float, 'iou': float, 'pixel_acc': float}
    """
    return {
        "dice":      dice_score(pred, target, threshold),
        "iou":       iou_score(pred,  target, threshold),
        "pixel_acc": pixel_accuracy(pred, target, threshold),
    }
