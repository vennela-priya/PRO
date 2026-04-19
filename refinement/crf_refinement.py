"""
refinement/crf_refinement.py
=============================
Dense CRF post-processing to sharpen mask boundaries.

Primary: pydensecrf (DenseCRF2D) — snaps mask edges to image gradients.
Fallback: bilateral filter + Canny edge snapping (pure OpenCV, no extra deps).

Both approaches use the RGB image as a guide to align mask boundaries
with the true image edges, improving segmentation quality.
"""
from __future__ import annotations

import logging
import warnings

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ── Primary: DenseCRF ─────────────────────────────────────────────────────────

def _try_import_pydensecrf() -> bool:
    """
    Attempt to import pydensecrf; return True if available.

    Returns
    -------
    bool
        True if pydensecrf.densecrf is importable, False otherwise.
    """
    try:
        import pydensecrf.densecrf as dcrf   # noqa: F401
        return True
    except ImportError:
        return False


def _apply_densecrf(
    image_rgb:  np.ndarray,
    mask_prob:  np.ndarray,
    n_iter:     int = 5,
) -> np.ndarray:
    """
    Apply DenseCRF2D to refine mask boundaries using image colour information.

    Parameters
    ----------
    image_rgb : np.ndarray  uint8 RGB image, shape (H, W, 3).
    mask_prob : np.ndarray  Float probability map in [0, 1], shape (H, W).
    n_iter    : int         Number of CRF inference iterations (default 5).

    Returns
    -------
    np.ndarray
        Refined binary mask, dtype=uint8, values {0, 1}, shape (H, W).
    """
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import (
        unary_from_softmax,
        create_pairwise_bilateral,
        create_pairwise_gaussian,
    )

    H, W = image_rgb.shape[:2]
    n_labels = 2

    d = dcrf.DenseCRF2D(W, H, n_labels)

    # Unary potential from mask probability
    fg = mask_prob.astype(np.float32)
    bg = 1.0 - fg
    probs = np.stack([bg, fg], axis=0)   # (2, H, W)
    probs = np.clip(probs, 1e-6, 1.0)
    probs = probs / probs.sum(axis=0, keepdims=True)

    U = unary_from_softmax(probs)
    d.setUnaryEnergy(U)

    # Pairwise Gaussian (spatial smoothness)
    d.addPairwiseGaussian(sxy=3, compat=3)

    # Pairwise bilateral (appearance / colour)
    feats = create_pairwise_bilateral(
        sdims=(80, 80), schan=(13, 13, 13),
        img=image_rgb, chdim=2
    )
    d.addPairwiseEnergy(feats, compat=10)

    # Inference
    Q = d.inference(n_iter)
    refined_prob = np.array(Q).reshape(n_labels, H, W)[1]   # foreground prob

    return (refined_prob >= 0.5).astype(np.uint8)


# ── Fallback: Bilateral + Canny ───────────────────────────────────────────────

def _crf_fallback(
    image_rgb: np.ndarray,
    mask_prob: np.ndarray,
) -> np.ndarray:
    """
    Bilateral filter + Canny edge snapping fallback when pydensecrf is unavailable.

    Steps:
      1. Bilateral filter the image to smooth regions while preserving edges.
      2. Detect edges with Canny.
      3. Threshold the probability mask.
      4. Snap mask boundary towards detected edges using dilation + AND.
      5. Remove very small components.

    Parameters
    ----------
    image_rgb : np.ndarray  uint8 RGB image, shape (H, W, 3).
    mask_prob : np.ndarray  Float probability map in [0, 1], shape (H, W).

    Returns
    -------
    np.ndarray
        Refined binary mask, dtype=uint8, values {0, 1}, shape (H, W).
    """
    H, W = image_rgb.shape[:2]

    # 1. Bilateral filter on greyscale — preserves edges
    gray      = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    bilateral = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

    # 2. Canny edge map
    edges = cv2.Canny(bilateral, threshold1=30, threshold2=90)
    # Dilate edges to create a snapping band
    k_edge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    edge_band = cv2.dilate(edges, k_edge, iterations=1)

    # 3. Threshold probability mask
    mask_prob_f = mask_prob.astype(np.float32)
    if mask_prob_f.shape != (H, W):
        mask_prob_f = cv2.resize(mask_prob_f, (W, H), interpolation=cv2.INTER_LINEAR)

    binary = (mask_prob_f >= 0.45).astype(np.uint8)

    # 4. Slightly dilate mask then AND with ~edge_band to pull boundary inward at edges
    k_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.dilate(binary, k_dil, iterations=2)

    # Where edges exist OUTSIDE the mask, contract the dilation back
    non_edge = (edge_band == 0).astype(np.uint8)
    snapped   = (dilated & non_edge) | binary   # keep interior, snap exterior

    # 5. Clean up: open (remove noise) + close (fill small holes)
    k_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    snapped = cv2.morphologyEx(snapped, cv2.MORPH_OPEN,  k_clean)
    snapped = cv2.morphologyEx(snapped, cv2.MORPH_CLOSE, k_clean)

    # Remove very small components (< 0.5% of image)
    min_px = max(1, int(H * W * 0.005))
    n_c, labels, stats, _ = cv2.connectedComponentsWithStats(snapped, 8)
    result = np.zeros_like(snapped)
    for lbl in range(1, n_c):
        if stats[lbl, cv2.CC_STAT_AREA] >= min_px:
            result[labels == lbl] = 1

    return result.astype(np.uint8)


# ── Public API ────────────────────────────────────────────────────────────────

def apply_crf(
    image_rgb: np.ndarray,
    mask_prob: np.ndarray,
    n_iter:    int = 5,
) -> np.ndarray:
    """
    Apply DenseCRF to sharpen mask boundaries.

    Falls back to bilateral filter + Canny edge snapping if pydensecrf is
    unavailable.

    Parameters
    ----------
    image_rgb : np.ndarray
        uint8 RGB image, shape (H, W, 3).
    mask_prob : np.ndarray
        Float probability map in [0, 1], shape (H, W).
        Will be resized to match image_rgb if needed.
    n_iter : int
        Number of CRF inference iterations (default 5; ignored by fallback).

    Returns
    -------
    np.ndarray
        Refined binary mask, dtype=uint8, values {0, 1}, shape (H, W).
    """
    # Ensure uint8 RGB
    if image_rgb.dtype != np.uint8:
        image_rgb = np.clip(image_rgb, 0, 255).astype(np.uint8)
    if image_rgb.ndim == 2:
        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_GRAY2RGB)

    H, W = image_rgb.shape[:2]
    # Resize mask_prob to image size if needed
    mp = mask_prob.astype(np.float32)
    if mp.shape != (H, W):
        mp = cv2.resize(mp, (W, H), interpolation=cv2.INTER_LINEAR)

    if _try_import_pydensecrf():
        try:
            logger.debug("[CRF] Using pydensecrf DenseCRF2D")
            return _apply_densecrf(image_rgb, mp, n_iter=n_iter)
        except Exception as e:
            logger.warning(f"[CRF] DenseCRF failed ({e}), falling back to bilateral+Canny")

    logger.debug("[CRF] Using bilateral+Canny fallback")
    return _crf_fallback(image_rgb, mp)


# ── Test / Demo ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import glob
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root))
    output_dir = project_root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s %(levelname)s %(message)s")

    from refinement.mask_refinement import _make_gaussian_blob_mask, clean_mask

    # Find 3 images
    patterns = [
        str(project_root / "Project" / "Testing" / "**" / "*.jpg"),
        str(project_root / "data"    / "**"            / "*.jpg"),
    ]
    image_paths: list[str] = []
    for pat in patterns:
        image_paths.extend(glob.glob(pat, recursive=True))
        if len(image_paths) >= 3:
            break
    image_paths = image_paths[:3]

    panels = []
    for img_path in image_paths:
        bgr = cv2.imread(img_path)
        if bgr is None:
            continue
        if bgr.ndim == 2:
            bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
        h, w  = bgr.shape[:2]
        rgb   = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        raw_mask  = _make_gaussian_blob_mask(h, w, 0.30)
        clean     = clean_mask(raw_mask).astype(np.float32)

        refined   = apply_crf(rgb, clean, n_iter=5)

        before_pct = float(clean.mean()   * 100)
        after_pct  = float(refined.mean() * 100)
        print(
            f"  {Path(img_path).name:40s}  "
            f"before={before_pct:.1f}%  after={after_pct:.1f}%"
        )

        orig_rs   = cv2.resize(rgb, (256, 256))
        mask_rs   = cv2.resize((clean * 255).astype(np.uint8), (256, 256))
        mask_rgb  = np.stack([mask_rs] * 3, axis=-1)
        ref_rs    = cv2.resize((refined * 255).astype(np.uint8), (256, 256))
        ref_rgb   = np.stack([ref_rs] * 3, axis=-1)
        row       = np.concatenate([orig_rs, mask_rgb, ref_rgb], axis=1)
        panels.append(row)

    if panels:
        grid     = np.concatenate(panels, axis=0)
        out_path = str(output_dir / "stage5_crf.png")
        cv2.imwrite(out_path, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
        print(f"\nSaved -> {out_path}")
