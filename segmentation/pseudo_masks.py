"""
segmentation/pseudo_masks.py
============================
Generate pseudo segmentation masks from Grad-CAM activations.

Why this exists
---------------
The classification dataset (glioma / meningioma / notumor / pituitary) has
no pixel-level annotations. This module converts the Grad-CAM heatmaps into
coarse binary masks that can bootstrap U-Net training without any manual
labelling. Quality is lower than hand-drawn masks but often sufficient for
learning a strong initialisation, especially when combined with real masks
from BraTS during fine-tuning.

Pipeline
--------
  MRI image
     │
     ▼
  EnsembleGradCAM.generate()  →  cam [H,W] float32 [0,1]
     │
     ▼
  gradcam_to_mask()           →  binary mask {0,255}  (threshold + morphology)
     │
     ▼
  saved as  <output_dir>/<stem>.png

Usage
-----
  # Quick single-image call
  from segmentation.pseudo_masks import gradcam_to_mask
  mask = gradcam_to_mask(cam_array, threshold=0.4)

  # Batch over a dataset folder
  from segmentation.pseudo_masks import build_pseudo_dataset
  n = build_pseudo_dataset(
      images_dir   = "data/Testing",
      output_dir   = "data/pseudo_masks",
      gradcam_fn   = my_gradcam_callable,   # img_np -> cam_np
      skip_notumor = True,
  )
  print(f"Generated {n} masks")
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# ── Core converter ────────────────────────────────────────────────────────

def gradcam_to_mask(
    cam:         np.ndarray,
    threshold:   float = 0.40,
    open_ksize:  int   = 5,
    close_ksize: int   = 11,
    min_area:    int   = 100,
) -> np.ndarray:
    """
    Convert a normalised Grad-CAM heatmap to a cleaned binary mask.

    Parameters
    ----------
    cam         : float32 array [0, 1], any spatial size
    threshold   : activation level above which pixels are considered tumour.
                  0.40 works well for most brain MRI datasets.
    open_ksize  : morphological opening kernel size (removes salt noise).
                  Set 0 to skip.
    close_ksize : morphological closing kernel size (fills interior holes).
                  Set 0 to skip.
    min_area    : connected components smaller than this pixel count are
                  removed (helps discard tiny spurious activations).

    Returns
    -------
    uint8 array {0, 255} — same spatial size as `cam`.
    """
    # Ensure float in [0, 1]
    c = cam.copy().astype(np.float32)
    if c.max() > c.min():
        c = (c - c.min()) / (c.max() - c.min() + 1e-8)

    binary = (c >= threshold).astype(np.uint8) * 255

    # Morphological opening — break thin connections, remove noise
    if open_ksize > 0:
        k      = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  k)

    # Morphological closing — fill small holes inside the tumour blob
    if close_ksize > 0:
        k      = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)

    # Remove tiny connected components
    if min_area > 0 and binary.any():
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8)
        for lbl in range(1, n_labels):
            if stats[lbl, cv2.CC_STAT_AREA] < min_area:
                binary[labels == lbl] = 0

    return binary   # uint8 {0, 255}


def mask_to_overlay(
    img_rgb: np.ndarray,
    mask:    np.ndarray,
    color:   tuple[int, int, int] = (255, 60, 60),
    alpha:   float = 0.40,
) -> np.ndarray:
    """
    Blend a binary mask over an RGB image.

    Parameters
    ----------
    img_rgb : uint8 (H, W, 3)
    mask    : uint8 (H, W) — {0, 255}
    color   : RGB colour for the tumour region
    alpha   : opacity of the colour overlay

    Returns
    -------
    uint8 (H, W, 3)
    """
    overlay   = img_rgb.copy()
    color_layer = np.zeros_like(img_rgb)
    color_layer[mask > 0] = color
    overlay = cv2.addWeighted(overlay, 1.0 - alpha, color_layer, alpha, 0)
    return overlay


# ── Batch builder ─────────────────────────────────────────────────────────

def build_pseudo_dataset(
    images_dir:    str,
    output_dir:    str,
    gradcam_fn:    Callable[[np.ndarray], np.ndarray],
    threshold:     float  = 0.40,
    skip_notumor:  bool   = True,
    notumor_label: str    = "notumor",
    save_npy:      bool   = True,
    save_png:      bool   = True,
    verbose:       bool   = True,
) -> int:
    """
    Run Grad-CAM over every image in `images_dir` and save binary masks.

    Handles both flat directories and class-subfolder layouts:
        images_dir/glioma/img001.jpg
        images_dir/meningioma/img002.jpg
        images_dir/notumor/img003.jpg   ← written as all-zero mask

    Parameters
    ----------
    images_dir   : root directory to scan recursively for images
    output_dir   : where to save .png and/or .npy masks (flat structure)
    gradcam_fn   : callable that accepts a uint8 RGB numpy array (H,W,3)
                   and returns a float32 Grad-CAM array normalised to [0,1]
    threshold    : binarisation threshold for gradcam_to_mask()
    skip_notumor : if True, write an all-zero mask for notumor images
                   instead of running Grad-CAM (saves time + avoids noise)
    notumor_label: substring used to identify notumor images by path
    save_npy     : save float32 prob map as .npy (useful for soft labels)
    save_png     : save uint8 mask as .png (useful for visualisation)
    verbose      : print a progress line every 50 images

    Returns
    -------
    Number of masks written.
    """
    img_root   = Path(images_dir)
    mask_root  = Path(output_dir)
    mask_root.mkdir(parents=True, exist_ok=True)

    img_paths = sorted(
        p for p in img_root.rglob("*")
        if p.suffix.lower() in _IMG_EXTS
    )
    if not img_paths:
        logger.warning(f"[PseudoMasks] No images found under {images_dir}")
        return 0

    count = 0
    for i, p in enumerate(img_paths, 1):
        try:
            img = cv2.imread(str(p))
            if img is None:
                logger.warning(f"[PseudoMasks] Skipping unreadable: {p.name}")
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w    = img_rgb.shape[:2]

            is_notumor = notumor_label in str(p).lower()

            if is_notumor and skip_notumor:
                prob_map = np.zeros((h, w), dtype=np.float32)
                mask_png = np.zeros((h, w), dtype=np.uint8)
            else:
                cam      = gradcam_fn(img_rgb)                          # [0,1]
                prob_map = cv2.resize(
                    cam.astype(np.float32), (w, h),
                    interpolation=cv2.INTER_LINEAR,
                )
                mask_raw = gradcam_to_mask(prob_map, threshold=threshold)
                mask_png = mask_raw                                     # {0,255}

            # Build flat output filename (class/img001 → class_img001)
            try:
                rel = p.relative_to(img_root)
            except ValueError:
                rel = Path(p.name)
            flat_stem = str(rel).replace(os.sep, "_").replace("/", "_")
            flat_stem = Path(flat_stem).stem

            if save_npy:
                np.save(str(mask_root / (flat_stem + ".npy")), prob_map)
            if save_png:
                cv2.imwrite(str(mask_root / (flat_stem + ".png")), mask_png)

            count += 1
            if verbose and count % 50 == 0:
                logger.info(
                    f"[PseudoMasks] {count}/{len(img_paths)} done ...")

        except Exception as e:
            logger.warning(f"[PseudoMasks] Error on {p.name}: {e}")
            continue

    logger.info(
        f"[PseudoMasks] Done — {count} masks written to {output_dir}")
    return count
