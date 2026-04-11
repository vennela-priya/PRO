"""
dataset.py
==========
PyTorch Dataset implementations:
- BrainTumorDataset  : classification (Figshare + Br35H)
- BrainTumorSegDataset : segmentation (with masks)
- Weighted sampler factory to handle class imbalance
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Classification Dataset
# ---------------------------------------------------------------------------

class BrainTumorDataset(Dataset):
    """
    Brain tumor MRI classification dataset.

    Parameters
    ----------
    df : DataFrame with columns [path, label, patient_id]
    transform : albumentations Compose transform
    cache : whether to cache images in RAM (fast but memory intensive)
    """

    CLASS_NAMES = ["meningioma", "glioma", "pituitary"]

    def __init__(
        self,
        df: pd.DataFrame,
        transform: Optional[Callable] = None,
        cache: bool = False,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.cache = cache
        self._cache: Dict[int, np.ndarray] = {}

        if cache:
            logger.info("Pre-caching dataset into RAM…")
            for idx in range(len(self.df)):
                self._cache[idx] = self._load_image(idx)

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, idx: int) -> np.ndarray:
        path = self.df.iloc[idx]["path"]
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"Image not found: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img = self._cache[idx] if self.cache and idx in self._cache else self._load_image(idx)
        label = int(self.df.iloc[idx]["label"])

        if self.transform:
            augmented = self.transform(image=img)
            img = augmented["image"]  # -> torch.Tensor [C, H, W]
        else:
            img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0

        return {"image": img, "label": torch.tensor(label, dtype=torch.long)}

    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency class weights for weighted loss."""
        counts = self.df["label"].value_counts().sort_index()
        weights = 1.0 / counts.values.astype(np.float32)
        weights /= weights.sum()
        return torch.from_numpy(weights)

    def get_sample_weights(self) -> List[float]:
        """Per-sample weights for WeightedRandomSampler."""
        class_counts = self.df["label"].value_counts().to_dict()
        n_total = len(self.df)
        sample_weights = [
            n_total / class_counts[int(row["label"])]
            for _, row in self.df.iterrows()
        ]
        return sample_weights


# ---------------------------------------------------------------------------
# Segmentation Dataset
# ---------------------------------------------------------------------------

class BrainTumorSegDataset(Dataset):
    """
    Brain tumor MRI segmentation dataset.
    Returns (image, mask) pairs.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        transform: Optional[Callable] = None,
        mask_col: str = "mask_path",
        image_size: int = 256,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.mask_col = mask_col
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        img = cv2.cvtColor(cv2.imread(str(row["path"])), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(row[self.mask_col]), cv2.IMREAD_GRAYSCALE)

        if mask is None:
            mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]
        else:
            img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
            mask = torch.from_numpy(mask).float() / 255.0

        mask = (mask > 0.5).float().unsqueeze(0)  # [1, H, W]
        return {"image": img, "mask": mask, "label": torch.tensor(int(row["label"]))}


# ---------------------------------------------------------------------------
# DataLoader factories
# ---------------------------------------------------------------------------

def build_classification_loaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_transform: Callable,
    val_transform: Callable,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    oversample: bool = True,
    cache: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train / val / test DataLoaders for classification.

    Applies WeightedRandomSampler to train set when ``oversample=True``.
    """
    train_ds = BrainTumorDataset(train_df, transform=train_transform, cache=cache)
    val_ds = BrainTumorDataset(val_df, transform=val_transform, cache=cache)
    test_ds = BrainTumorDataset(test_df, transform=val_transform, cache=False)

    train_sampler = None
    shuffle = True
    if oversample:
        sample_weights = train_ds.get_sample_weights()
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        shuffle = False

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=shuffle if train_sampler is None else False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    logger.info(
        f"DataLoaders — train: {len(train_loader)} batches, "
        f"val: {len(val_loader)}, test: {len(test_loader)}"
    )
    return train_loader, val_loader, test_loader


def build_seg_loaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_transform: Callable,
    val_transform: Callable,
    batch_size: int = 16,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build train / val / test DataLoaders for segmentation."""
    train_ds = BrainTumorSegDataset(train_df, transform=train_transform)
    val_ds = BrainTumorSegDataset(val_df, transform=val_transform)
    test_ds = BrainTumorSegDataset(test_df, transform=val_transform)

    make_loader = lambda ds, shuffle: DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    return make_loader(train_ds, True), make_loader(val_ds, False), make_loader(test_ds, False)
