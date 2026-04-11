"""
segmentation/dataset.py
=======================
PyTorch Dataset for brain-tumour segmentation.

Supports two modes
------------------
1. Paired mode  — directory of images + matching binary masks (PNG/JPG/NPY)
2. Image-only   — directory of images; masks are loaded from a pre-generated
                  pseudo-label cache (created by pseudo_masks.py)

Directory layout (paired)
--------------------------
    data_root/
    ├── images/   *.jpg  *.png  (MRI scans)
    └── masks/    *.png  (binary, 0 or 255)

Augmentation
------------
train: random horizontal + vertical flip, random rotation (±20°), random
       brightness/contrast jitter, random Gaussian noise, normalise
val  : centre-crop resize + normalise only

The same spatial transforms are applied identically to image and mask using
a shared random seed (no dependency on albumentations or torchvision ≥0.9).
"""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


# ── ImageNet normalisation constants ─────────────────────────────────────
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# ── Augmentation helpers ──────────────────────────────────────────────────

class SegAugment:
    """
    Paired image + mask augmentation.  All spatial ops use the same seed so
    image and mask are always transformed identically.
    """

    def __init__(
        self,
        image_size: int   = 224,
        train:      bool  = True,
        flip_p:     float = 0.5,
        rot_deg:    float = 20.0,
        brightness: float = 0.2,
        noise_std:  float = 0.02,
    ):
        self.image_size = image_size
        self.train      = train
        self.flip_p     = flip_p
        self.rot_deg    = rot_deg
        self.brightness = brightness
        self.noise_std  = noise_std

    def __call__(
        self,
        image: np.ndarray,   # uint8 RGB (H, W, 3)
        mask:  np.ndarray,   # uint8  (H, W)  values 0 or 255
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (image_tensor [3,H,W], mask_tensor [1,H,W])."""
        h, w = self.image_size, self.image_size

        # ── Resize (always) ───────────────────────────────────────────
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
        mask  = cv2.resize(mask,  (w, h), interpolation=cv2.INTER_NEAREST)

        if self.train:
            # ── Horizontal flip ───────────────────────────────────────
            if random.random() < self.flip_p:
                image = cv2.flip(image, 1)
                mask  = cv2.flip(mask,  1)

            # ── Vertical flip ─────────────────────────────────────────
            if random.random() < self.flip_p:
                image = cv2.flip(image, 0)
                mask  = cv2.flip(mask,  0)

            # ── Random rotation ───────────────────────────────────────
            if random.random() < 0.7:
                angle  = random.uniform(-self.rot_deg, self.rot_deg)
                M      = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
                image  = cv2.warpAffine(image, M, (w, h),
                                        flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_REFLECT)
                mask   = cv2.warpAffine(mask,  M, (w, h),
                                        flags=cv2.INTER_NEAREST,
                                        borderMode=cv2.BORDER_REFLECT)

            # ── Brightness / contrast jitter (image only) ─────────────
            if random.random() < 0.6:
                alpha = 1.0 + random.uniform(-self.brightness, self.brightness)
                beta  = random.uniform(-10, 10)
                image = np.clip(image.astype(np.float32) * alpha + beta,
                                0, 255).astype(np.uint8)

            # ── Gaussian noise (image only) ───────────────────────────
            if random.random() < 0.3:
                noise = np.random.normal(0, self.noise_std * 255,
                                        image.shape).astype(np.float32)
                image = np.clip(image.astype(np.float32) + noise,
                                0, 255).astype(np.uint8)

        # ── To float tensors ──────────────────────────────────────────
        img_f  = image.astype(np.float32) / 255.0
        img_f  = (img_f - _MEAN) / _STD                   # normalise
        img_t  = torch.from_numpy(img_f.transpose(2, 0, 1))  # [3,H,W]

        msk_f  = (mask > 127).astype(np.float32)           # binarise
        msk_t  = torch.from_numpy(msk_f).unsqueeze(0)      # [1,H,W]

        return img_t, msk_t


# ── Dataset ───────────────────────────────────────────────────────────────

class BrainSegDataset(Dataset):
    """
    Brain MRI segmentation dataset.

    Parameters
    ----------
    image_dir     : folder containing MRI images
    mask_dir      : folder containing matching binary masks (same filename stem)
                    Pass None if masks come from pseudo_cache_dir.
    pseudo_cache  : folder of pre-computed pseudo masks (*.npy float32 [H,W])
                    Used when mask_dir is None.
    augment       : SegAugment instance (or any callable(img, mask)->tensors)
    image_size    : square resize target (default 224)
    train         : whether this split uses training augmentation

    Notes
    -----
    * Image and mask filenames must share the same stem, e.g.
      images/img_001.jpg  ←→  masks/img_001.png
    * Mask pixel values: 0 = background, 255 (or any >0) = tumour.
    """

    def __init__(
        self,
        image_dir:    str | Path,
        mask_dir:     Optional[str | Path] = None,
        pseudo_cache: Optional[str | Path] = None,
        image_size:   int  = 224,
        train:        bool = True,
    ):
        self.image_dir    = Path(image_dir)
        self.mask_dir     = Path(mask_dir)     if mask_dir     else None
        self.pseudo_cache = Path(pseudo_cache) if pseudo_cache else None
        self.augment      = SegAugment(image_size=image_size, train=train)

        if self.mask_dir is None and self.pseudo_cache is None:
            raise ValueError(
                "Provide either mask_dir (real masks) or "
                "pseudo_cache (pseudo-label directory)."
            )

        # Collect image paths (recursive — handles class subfolders)
        self.image_paths: list[Path] = sorted(
            p for p in self.image_dir.rglob('*')
            if p.suffix.lower() in _IMG_EXTS
        )
        if not self.image_paths:
            raise FileNotFoundError(
                f"No images found in {self.image_dir}. "
                f"Expected extensions: {_IMG_EXTS}"
            )

    def _load_image(self, path: Path) -> np.ndarray:
        """Load image as uint8 RGB numpy array."""
        img = cv2.imread(str(path))
        if img is None:
            raise IOError(f"Could not read image: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _load_mask(self, image_path: Path) -> np.ndarray:
        """
        Load binary mask for a given image.
        Falls back to pseudo cache if mask_dir is None.
        Returns uint8 array {0, 255} shape (H, W).
        """
        stem = image_path.stem

        if self.mask_dir is not None:
            # Try multiple extensions
            for ext in [".png", ".jpg", ".jpeg", ".bmp"]:
                mp = self.mask_dir / (stem + ext)
                if mp.exists():
                    m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
                    if m is not None:
                        return (m > 0).astype(np.uint8) * 255

        if self.pseudo_cache is not None:
            npy_path = self.pseudo_cache / (stem + ".npy")
            if npy_path.exists():
                arr = np.load(str(npy_path)).astype(np.float32)
                return (arr > 0.5).astype(np.uint8) * 255

        # Return empty mask as last resort (skip sample in training via
        # custom collate_fn or handle upstream)
        h, w = 224, 224
        return np.zeros((h, w), dtype=np.uint8)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        img_path  = self.image_paths[idx]
        image     = self._load_image(img_path)
        mask      = self._load_mask(img_path)

        img_t, msk_t = self.augment(image, mask)

        return {
            "image":    img_t,    # [3, H, W]
            "mask":     msk_t,    # [1, H, W]
            "filename": img_path.name,
        }


# ── Factory helpers ───────────────────────────────────────────────────────

def make_dataloaders(
    image_dir:    str,
    mask_dir:     Optional[str] = None,
    pseudo_cache: Optional[str] = None,
    image_size:   int   = 224,
    val_split:    float = 0.15,
    batch_size:   int   = 8,
    num_workers:  int   = 0,
    seed:         int   = 42,
) -> tuple:
    """
    Create train / val DataLoaders from a single image directory.

    Parameters
    ----------
    val_split : fraction of data held out for validation (default 15 %)

    Returns
    -------
    (train_loader, val_loader, n_train, n_val)
    """
    from torch.utils.data import DataLoader, random_split, Subset

    full_ds = BrainSegDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        pseudo_cache=pseudo_cache,
        image_size=image_size,
        train=True,
    )

    n_total = len(full_ds)
    n_val   = max(1, int(n_total * val_split))
    n_train = n_total - n_val

    gen     = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=gen)

    # Val split: disable augmentation
    val_ds.dataset = BrainSegDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        pseudo_cache=pseudo_cache,
        image_size=image_size,
        train=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, n_train, n_val
