"""
augmentation.py
===============
Albumentations-based augmentation pipelines + CutMix / MixUp implementations.
All transforms are composable and seed-safe.
"""

from __future__ import annotations

import random
from typing import Callable, Optional, Tuple

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2


# ---------------------------------------------------------------------------
# Albumentations pipelines
# ---------------------------------------------------------------------------

def build_train_transforms(
    image_size: int = 224,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225),
    rotation_degrees: int = 15,
    hflip_p: float = 0.5,
    vflip_p: float = 0.1,
    elastic_p: float = 0.3,
    noise_p: float = 0.2,
    erasing_p: float = 0.1,
) -> A.Compose:
    """Return heavy training augmentation pipeline."""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=hflip_p),
        A.VerticalFlip(p=vflip_p),
        A.Rotate(limit=rotation_degrees, p=0.7, border_mode=0),
        A.OneOf([
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0),
            A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=1.0),
        ], p=elastic_p),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05, p=0.6),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MedianBlur(blur_limit=5, p=1.0),
        ], p=0.2),
        A.GaussNoise(var_limit=(10.0, 50.0), p=noise_p),
        A.CoarseDropout(
            max_holes=8, max_height=image_size // 10, max_width=image_size // 10,
            fill_value=0, p=erasing_p,
        ),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=0, p=0.4),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


def build_val_transforms(
    image_size: int = 224,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225),
) -> A.Compose:
    """Return minimal val/test augmentation pipeline (resize + normalize only)."""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


def build_tta_transforms(
    image_size: int = 224,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225),
) -> list[A.Compose]:
    """Return list of TTA transforms (5 variants)."""
    base = [
        A.Resize(image_size, image_size),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ]
    variants = [
        A.Compose(base),
        A.Compose([A.Resize(image_size, image_size), A.HorizontalFlip(p=1.0),
                   A.Normalize(mean=mean, std=std), ToTensorV2()]),
        A.Compose([A.Resize(image_size, image_size), A.Rotate(limit=10, p=1.0),
                   A.Normalize(mean=mean, std=std), ToTensorV2()]),
        A.Compose([A.Resize(image_size, image_size), A.Rotate(limit=-10, p=1.0),
                   A.Normalize(mean=mean, std=std), ToTensorV2()]),
        A.Compose([A.Resize(image_size, image_size),
                   A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=5, p=1.0),
                   A.Normalize(mean=mean, std=std), ToTensorV2()]),
    ]
    return variants


# ---------------------------------------------------------------------------
# CutMix
# ---------------------------------------------------------------------------

def cutmix_data(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 0.4,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    CutMix augmentation (Yun et al., 2019).

    Returns
    -------
    mixed_x, y_a, y_b, lam
    """
    if alpha <= 0:
        return x, y, y, 1.0

    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    _, _, H, W = x.size()
    cut_ratio = np.sqrt(1.0 - lam)
    cut_h = int(H * cut_ratio)
    cut_w = int(W * cut_ratio)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    x1 = np.clip(cx - cut_w // 2, 0, W)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    y2 = np.clip(cy + cut_h // 2, 0, H)

    mixed_x = x.clone()
    mixed_x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
    lam = 1.0 - (x2 - x1) * (y2 - y1) / (W * H)

    return mixed_x, y, y[index], lam


# ---------------------------------------------------------------------------
# MixUp
# ---------------------------------------------------------------------------

def mixup_data(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 0.2,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    MixUp augmentation (Zhang et al., 2018).

    Returns
    -------
    mixed_x, y_a, y_b, lam
    """
    if alpha <= 0:
        return x, y, y, 1.0

    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1.0 - lam) * x[index]
    return mixed_x, y, y[index], lam


def mixup_criterion(
    criterion: Callable,
    pred: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """Compute mixed loss for MixUp or CutMix."""
    return lam * criterion(pred, y_a) + (1.0 - lam) * criterion(pred, y_b)
