"""
tests/test_data.py — Unit tests for data pipeline.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.augmentation import (
    build_train_transforms,
    build_val_transforms,
    cutmix_data,
    mixup_data,
)
from src.data.preprocessing import apply_clahe, patient_stratified_split, verify_no_leakage


# ---------------------------------------------------------------------------
# Augmentation tests
# ---------------------------------------------------------------------------

class TestAugmentations:
    def test_train_transform_output_shape(self):
        tfm = build_train_transforms(image_size=224)
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        out = tfm(image=img)["image"]
        assert out.shape == (3, 224, 224), f"Expected (3,224,224), got {out.shape}"

    def test_val_transform_output_shape(self):
        tfm = build_val_transforms(image_size=224)
        img = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        out = tfm(image=img)["image"]
        assert out.shape == (3, 224, 224)

    def test_train_transform_dtype(self):
        tfm = build_train_transforms()
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        out = tfm(image=img)["image"]
        assert out.dtype == torch.float32


class TestCutMix:
    def test_cutmix_output_shape(self):
        x = torch.randn(4, 3, 224, 224)
        y = torch.tensor([0, 1, 2, 0])
        mixed_x, y_a, y_b, lam = cutmix_data(x, y, alpha=0.4)
        assert mixed_x.shape == x.shape
        assert 0.0 <= lam <= 1.0

    def test_mixup_output_shape(self):
        x = torch.randn(4, 3, 224, 224)
        y = torch.tensor([0, 1, 2, 0])
        mixed_x, y_a, y_b, lam = mixup_data(x, y, alpha=0.2)
        assert mixed_x.shape == x.shape
        assert 0.0 <= lam <= 1.0


# ---------------------------------------------------------------------------
# CLAHE tests
# ---------------------------------------------------------------------------

class TestPreprocessing:
    def test_clahe_output_shape(self):
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        out = apply_clahe(img)
        assert out.shape == img.shape
        assert out.dtype == np.uint8

    def test_clahe_grayscale(self):
        img = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
        out = apply_clahe(img)
        assert out.shape == (224, 224)


# ---------------------------------------------------------------------------
# Split tests
# ---------------------------------------------------------------------------

class TestSplits:
    def _make_df(self):
        records = []
        for i in range(200):
            records.append({
                "path": f"/data/{i}.png",
                "label": i % 3,
                "patient_id": f"patient_{i // 3}",
            })
        return pd.DataFrame(records)

    def test_split_ratios(self):
        df = self._make_df()
        train, val, test = patient_stratified_split(df, 0.70, 0.15, 0.15, seed=42)
        total = len(train) + len(val) + len(test)
        assert total == len(df), "Split sizes must sum to total"

    def test_no_leakage(self):
        df = self._make_df()
        train, val, test = patient_stratified_split(df, seed=42)
        assert verify_no_leakage(train, val, test), "Data leakage detected!"

    def test_all_classes_in_train(self):
        df = self._make_df()
        train, val, test = patient_stratified_split(df, seed=42)
        assert set(train["label"].unique()) == {0, 1, 2}, "All classes must be in train split"
