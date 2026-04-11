"""
preprocessing.py
================
MRI image preprocessing pipeline:
- DICOM / PNG / JPEG / .mat loading
- Skull stripping (optional intensity-based approximation)
- Histogram normalization (CLAHE)
- Resize, centre-crop
- Patient-level train/val/test splitting with stratification
"""

from __future__ import annotations

import os
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from scipy.io import loadmat
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLAHE normalisation
# ---------------------------------------------------------------------------

def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_size: int = 8) -> np.ndarray:
    """Apply Contrast Limited Adaptive Histogram Equalisation to a grayscale image."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    if image.ndim == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return clahe.apply(image)


# ---------------------------------------------------------------------------
# Skull stripping (intensity-based fast approximation)
# ---------------------------------------------------------------------------

def skull_strip(image: np.ndarray) -> np.ndarray:
    """
    Fast intensity-based skull approximation.
    Returns image with background zeroed out.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image.copy()
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        brain_mask = np.zeros_like(mask)
        cv2.drawContours(brain_mask, [largest], -1, 255, -1)
        if image.ndim == 3:
            return cv2.bitwise_and(image, image, mask=brain_mask)
        return cv2.bitwise_and(image, image, mask=brain_mask)
    return image


# ---------------------------------------------------------------------------
# Figshare .mat loader
# ---------------------------------------------------------------------------

def load_mat_sample(mat_path: str) -> Tuple[np.ndarray, int, np.ndarray]:
    """
    Load a single Figshare brain tumor .mat file.

    Returns
    -------
    image : H x W uint8
    label : int  (1=meningioma, 2=glioma, 3=pituitary)
    mask  : H x W binary mask (tumor region)
    """
    data = loadmat(mat_path)
    image = data["cjdata"]["image"][0, 0].astype(np.float32)
    label = int(data["cjdata"]["label"][0, 0][0, 0])
    mask = data["cjdata"]["tumorMask"][0, 0].astype(np.uint8)

    # Normalise to [0, 255]
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return image, label - 1, mask  # zero-indexed label


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

class FigsharePreprocessor:
    """
    Preprocesses the Figshare Brain Tumor Dataset.

    Parameters
    ----------
    raw_dir : path to directory containing .mat files
    out_dir  : path where processed PNG files will be saved
    image_size : output spatial resolution
    skull_strip : whether to apply skull stripping
    clahe : whether to apply CLAHE
    """

    CLASS_MAP = {0: "meningioma", 1: "glioma", 2: "pituitary"}

    def __init__(
        self,
        raw_dir: str,
        out_dir: str,
        image_size: int = 224,
        apply_skull_strip: bool = False,
        apply_clahe: bool = True,
    ) -> None:
        self.raw_dir = Path(raw_dir)
        self.out_dir = Path(out_dir)
        self.image_size = image_size
        self.apply_skull_strip = apply_skull_strip
        self.apply_clahe = apply_clahe
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def process_all(self) -> pd.DataFrame:
        """Process every .mat file; return metadata DataFrame."""
        records: List[Dict] = []
        mat_files = sorted(self.raw_dir.glob("**/*.mat"))
        logger.info(f"Found {len(mat_files)} .mat files in {self.raw_dir}")

        for mat_file in mat_files:
            try:
                img, label, mask = load_mat_sample(str(mat_file))
                img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

                if self.apply_skull_strip:
                    img_rgb = skull_strip(img_rgb)

                if self.apply_clahe:
                    img_rgb = apply_clahe(img_rgb)

                img_rgb = cv2.resize(img_rgb, (self.image_size, self.image_size))

                # Derive patient ID from filename (e.g. 1.mat → patient_1)
                stem = mat_file.stem
                patient_id = f"patient_{stem}"

                class_dir = self.out_dir / self.CLASS_MAP[label]
                class_dir.mkdir(parents=True, exist_ok=True)

                out_path = class_dir / f"{stem}.png"
                Image.fromarray(img_rgb).save(out_path)

                # Save mask
                mask_dir = self.out_dir / "masks" / self.CLASS_MAP[label]
                mask_dir.mkdir(parents=True, exist_ok=True)
                mask_resized = cv2.resize(mask * 255, (self.image_size, self.image_size))
                Image.fromarray(mask_resized).save(mask_dir / f"{stem}_mask.png")

                records.append({
                    "path": str(out_path),
                    "mask_path": str(mask_dir / f"{stem}_mask.png"),
                    "label": label,
                    "class_name": self.CLASS_MAP[label],
                    "patient_id": patient_id,
                    "source": "figshare",
                })

            except Exception as exc:
                logger.warning(f"Failed to process {mat_file}: {exc}")

        df = pd.DataFrame(records)
        df.to_csv(self.out_dir / "metadata.csv", index=False)
        logger.info(f"Processed {len(df)} samples → {self.out_dir}")
        return df


# ---------------------------------------------------------------------------
# Patient-level stratified split
# ---------------------------------------------------------------------------

def patient_stratified_split(
    df: pd.DataFrame,
    train_ratio: float = 0.80,
    val_ratio: float = 0.20,
    test_ratio: float = 0.0,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset at *patient* level to prevent data leakage.

    Parameters
    ----------
    df           : DataFrame with columns [path, label, patient_id]
    train_ratio  : fraction for training
    val_ratio    : fraction for validation
    test_ratio   : fraction for test (0.0 = no internal test split;
                   use final_test.py with the locked holdout instead)

    Returns train_df, val_df, test_df  (test_df is empty when test_ratio=0)
    """
    total = train_ratio + val_ratio + test_ratio
    assert abs(total - 1.0) < 1e-6 or abs(total - (train_ratio + val_ratio)) < 1e-6, \
        f"Ratios must sum to 1.0 (got {total:.3f})"

    # Aggregate per patient — take majority label
    patient_df = (
        df.groupby("patient_id")["label"]
        .agg(lambda x: x.mode()[0])
        .reset_index()
        .rename(columns={"label": "patient_label"})
    )

    if test_ratio > 0:
        train_patients, temp_patients = train_test_split(
            patient_df,
            test_size=(val_ratio + test_ratio),
            stratify=patient_df["patient_label"],
            random_state=seed,
        )
        val_frac = val_ratio / (val_ratio + test_ratio)
        val_patients, test_patients = train_test_split(
            temp_patients,
            test_size=(1.0 - val_frac),
            stratify=temp_patients["patient_label"],
            random_state=seed,
        )
        test_ids = set(test_patients["patient_id"])
    else:
        # No internal test split — split only into train and val
        train_patients, val_patients = train_test_split(
            patient_df,
            test_size=val_ratio,
            stratify=patient_df["patient_label"],
            random_state=seed,
        )
        test_ids = set()

    train_ids = set(train_patients["patient_id"])
    val_ids   = set(val_patients["patient_id"])

    # Verify no leakage
    assert train_ids.isdisjoint(val_ids),  "Data leakage: train ∩ val ≠ ∅"
    assert train_ids.isdisjoint(test_ids), "Data leakage: train ∩ test ≠ ∅"
    assert val_ids.isdisjoint(test_ids),   "Data leakage: val ∩ test ≠ ∅"

    train_df = df[df["patient_id"].isin(train_ids)].copy()
    val_df   = df[df["patient_id"].isin(val_ids)].copy()
    test_df  = df[df["patient_id"].isin(test_ids)].copy() if test_ids else pd.DataFrame()

    logger.info(
        f"Split sizes → train: {len(train_df)}, val: {len(val_df)}, "
        f"test: {len(test_df) if not test_df.empty else 'N/A (locked holdout)'}"
    )
    return train_df, val_df, test_df


def verify_no_leakage(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> bool:
    """Explicitly verify there is no patient-level overlap between splits."""
    train_ids = set(train_df["patient_id"].unique())
    val_ids = set(val_df["patient_id"].unique())
    test_ids = set(test_df["patient_id"].unique())

    ok = (
        len(train_ids & val_ids) == 0
        and len(train_ids & test_ids) == 0
        and len(val_ids & test_ids) == 0
    )
    if ok:
        logger.info("Patient-level split verification PASSED — no data leakage detected.")
    else:
        logger.error("LEAKAGE DETECTED in patient splits!")
    return ok
