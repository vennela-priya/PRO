"""
segmentation/postprocess.py
============================
Standalone morphological post-processing pipeline for binary tumour masks.

Applies the following steps in order:
  1. Threshold probability map → binary mask
  2. Morphological OPEN  (remove salt noise, radius ≈ 1 % of image)
  3. Morphological CLOSE (fill small holes, radius ≈ 2 % of image)
  4. Remove connected components smaller than min_area_pct of image
  5. Optional ERODE  (tighten boundary by n_erode pixels)

Usage
-----
    from segmentation.postprocess import refine_mask

    binary = refine_mask(prob_map, threshold=0.45, erode_px=2)
"""
from __future__ import annotations

import cv2
import numpy as np


def refine_mask(
    prob_map:     np.ndarray,
    threshold:    float = 0.45,
    min_area_pct: float = 0.3,
    erode_px:     int   = 2,
) -> np.ndarray:
    """
    Convert a probability map to a clean, tightly-bounded binary mask.

    Parameters
    ----------
    prob_map     : (H, W) float32 in [0, 1]  — raw sigmoid output
    threshold    : probability cut-off for binarisation (default 0.45)
    min_area_pct : connected components smaller than this fraction of
                   total image pixels (in percent) are removed (default 0.3)
    erode_px     : additional boundary erosion in pixels after cleanup;
                   set to 0 to skip erosion (reduces over-segmentation)

    Returns
    -------
    clean_mask : (H, W) uint8 {0, 1}
    """
    H, W   = prob_map.shape
    binary = (prob_map >= threshold).astype(np.uint8)

    # ── 1. Morphological OPEN — remove isolated noise blobs ──────────
    r_open  = max(3, int(min(H, W) * 0.012))
    k_open  = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (r_open * 2 + 1, r_open * 2 + 1))
    binary  = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k_open)

    # ── 2. Morphological CLOSE — fill internal holes ──────────────────
    r_close = max(5, int(min(H, W) * 0.025))
    k_close = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (r_close * 2 + 1, r_close * 2 + 1))
    binary  = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_close)

    # ── 3. Remove small components ────────────────────────────────────
    min_px  = max(1, int(H * W * min_area_pct / 100))
    n_comp, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8)
    clean = np.zeros_like(binary)
    for lbl in range(1, n_comp):
        if stats[lbl, cv2.CC_STAT_AREA] >= min_px:
            clean[labels == lbl] = 1

    # ── 4. Boundary erosion — tighten contour ─────────────────────────
    if erode_px > 0:
        k_erode = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (erode_px * 2 + 1, erode_px * 2 + 1))
        clean = cv2.erode(clean, k_erode, iterations=1)

    return clean


def overlay_masks(
    img_rgb:     np.ndarray,
    pred_mask:   np.ndarray,
    gt_mask:     np.ndarray | None = None,
    pred_color:  tuple = (220, 50, 50),   # red — predicted
    gt_color:    tuple = (50, 200, 50),   # green — ground truth
    alpha:       float = 0.38,
) -> np.ndarray:
    """
    Produce an RGB overlay image showing prediction and optional GT mask.

    Parameters
    ----------
    img_rgb    : (H, W, 3) uint8 background image
    pred_mask  : (H, W) uint8 {0, 1}
    gt_mask    : (H, W) uint8 {0, 1} — optional ground truth
    pred_color : RGB colour for predicted mask (default red)
    gt_color   : RGB colour for GT mask (default green)
    alpha      : blending alpha for colour layers (default 0.38)

    Returns
    -------
    (H, W, 3) uint8 overlay
    """
    H, W = img_rgb.shape[:2]
    out  = img_rgb.copy().astype(np.float32)

    if gt_mask is not None:
        gt_layer = np.zeros((H, W, 3), dtype=np.float32)
        gt_layer[gt_mask == 1] = gt_color
        out = out * (1 - alpha) + gt_layer * alpha

    pred_layer = np.zeros((H, W, 3), dtype=np.float32)
    pred_layer[pred_mask == 1] = pred_color
    out = out * (1 - alpha) + pred_layer * alpha

    return np.clip(out, 0, 255).astype(np.uint8)


def draw_contours(
    img_rgb:    np.ndarray,
    pred_mask:  np.ndarray,
    gt_mask:    np.ndarray | None = None,
    pred_color: tuple = (220, 50, 50),   # red
    gt_color:   tuple = (50, 200, 50),   # green
    thickness:  int   = 2,
) -> np.ndarray:
    """
    Draw contours of prediction (and optional GT) on the image.

    Parameters
    ----------
    img_rgb    : (H, W, 3) uint8 background
    pred_mask  : (H, W) uint8 {0, 1}
    gt_mask    : (H, W) uint8 {0, 1} optional
    thickness  : contour line thickness in pixels

    Returns
    -------
    (H, W, 3) uint8 annotated image
    """
    out = img_rgb.copy()

    def _draw(mask, color):
        cnts, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, color, thickness)

    if gt_mask is not None:
        _draw(gt_mask,   gt_color)
    _draw(pred_mask, pred_color)

    return out
