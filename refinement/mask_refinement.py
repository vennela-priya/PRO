"""
refinement/mask_refinement.py
==============================
Morphological pseudo-mask cleaning pipeline.

Pipeline:
    raw float mask
        -> Gaussian blur (reduce noise)
        -> Otsu threshold (adaptive binarisation)
        -> Morphological open  (remove small noise)
        -> Morphological close (fill small holes)
        -> Largest connected component (keep dominant region)
        -> Hole fill (scipy.ndimage)
        -> binary uint8 output
"""
from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
from scipy import ndimage


# ── Public API ────────────────────────────────────────────────────────────────

def clean_mask(mask: np.ndarray) -> np.ndarray:
    """
    Full cleaning pipeline: blur -> Otsu -> open -> close -> largest CC -> hole fill.

    Parameters
    ----------
    mask : np.ndarray
        Input mask as float32 in [0, 1] or uint8 in {0, 255} or {0, 1}.
        Shape (H, W) or (H, W, 1).

    Returns
    -------
    np.ndarray
        Cleaned binary mask, dtype=uint8, values in {0, 1}, shape (H, W).
    """
    # Ensure 2-D
    if mask.ndim == 3:
        mask = mask[:, :, 0]

    # Normalise to [0, 255] uint8 for OpenCV
    m = mask.astype(np.float32)
    if m.max() <= 1.0:
        m = (m * 255.0).astype(np.uint8)
    else:
        m = m.astype(np.uint8)

    # 1. Gaussian blur — suppress high-frequency noise before thresholding
    blurred = cv2.GaussianBlur(m, (7, 7), sigmaX=2.0)

    # 2. Otsu threshold — adaptive global threshold
    binary = _otsu_threshold(blurred)

    # 3 & 4. Morphological open + close
    binary = _morphological_cleanup(binary)

    # 5. Keep only largest connected component
    binary = _largest_connected_component(binary)

    # 6. Fill any remaining interior holes
    binary = _fill_holes(binary)

    return binary.astype(np.uint8)


# ── Private helpers ───────────────────────────────────────────────────────────

def _otsu_threshold(mask: np.ndarray) -> np.ndarray:
    """
    Apply Otsu's automatic threshold to a grayscale uint8 image.

    Parameters
    ----------
    mask : np.ndarray
        Grayscale uint8 image, shape (H, W).

    Returns
    -------
    np.ndarray
        Binary image with values {0, 1}, shape (H, W), dtype=uint8.
    """
    _, binary = cv2.threshold(
        mask.astype(np.uint8), 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return (binary > 0).astype(np.uint8)


def _morphological_cleanup(binary: np.ndarray) -> np.ndarray:
    """
    Apply morphological opening (noise removal) then closing (hole fill).

    Parameters
    ----------
    binary : np.ndarray
        Binary image with values {0, 1}, shape (H, W).

    Returns
    -------
    np.ndarray
        Cleaned binary image, same shape, dtype=uint8.
    """
    h, w = binary.shape[:2]
    # Kernel radius scales with image size (roughly 2% of short side, min 3)
    r_open  = max(3, int(min(h, w) * 0.020))
    r_close = max(5, int(min(h, w) * 0.030))

    k_open  = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (r_open * 2 + 1,  r_open * 2 + 1))
    k_close = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (r_close * 2 + 1, r_close * 2 + 1))

    b = binary.astype(np.uint8)
    b = cv2.morphologyEx(b, cv2.MORPH_OPEN,  k_open)
    b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, k_close)
    return b


def _largest_connected_component(binary: np.ndarray) -> np.ndarray:
    """
    Keep only the largest connected component in a binary mask.

    If the mask is entirely zero, return it unchanged.

    Parameters
    ----------
    binary : np.ndarray
        Binary image with values {0, 1}, shape (H, W), dtype=uint8.

    Returns
    -------
    np.ndarray
        Binary image retaining only the largest component, dtype=uint8.
    """
    b = binary.astype(np.uint8)
    n_comp, labels, stats, _ = cv2.connectedComponentsWithStats(b, connectivity=8)

    if n_comp <= 1:          # nothing or all background
        return b

    # Component 0 is background — ignore it
    areas = stats[1:, cv2.CC_STAT_AREA]   # (n_comp-1,)
    largest_label = int(np.argmax(areas)) + 1   # +1 for background offset

    result = np.zeros_like(b)
    result[labels == largest_label] = 1
    return result


def _fill_holes(binary: np.ndarray) -> np.ndarray:
    """
    Fill interior holes in a binary mask using scipy.ndimage binary_fill_holes.

    Parameters
    ----------
    binary : np.ndarray
        Binary image with values {0, 1}, shape (H, W).

    Returns
    -------
    np.ndarray
        Binary image with holes filled, dtype=uint8.
    """
    filled = ndimage.binary_fill_holes(binary.astype(bool))
    return filled.astype(np.uint8)


# ── Test / Demo ───────────────────────────────────────────────────────────────

def _make_gaussian_blob_mask(h: int, w: int, radius_frac: float = 0.30) -> np.ndarray:
    """
    Generate a synthetic Gaussian blob mask centred in the image.

    Parameters
    ----------
    h, w        : image height and width in pixels
    radius_frac : blob radius as a fraction of min(h, w)

    Returns
    -------
    np.ndarray
        Float32 mask in [0, 1], shape (H, W).
    """
    cx, cy = w // 2, h // 2
    radius = radius_frac * min(h, w)
    Y, X = np.ogrid[:h, :w]
    dist_sq = (X - cx) ** 2 + (Y - cy) ** 2
    blob = np.exp(-dist_sq / (2 * radius ** 2)).astype(np.float32)
    # Add noise to simulate imperfect pseudo masks
    rng  = np.random.RandomState(42)
    blob += rng.normal(0, 0.08, blob.shape).astype(np.float32)
    blob = np.clip(blob, 0.0, 1.0)
    return blob


if __name__ == "__main__":
    import sys
    import glob

    project_root = Path(__file__).resolve().parent.parent
    output_dir   = project_root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Find 3 real images ---
    patterns = [
        str(project_root / "Project" / "Testing" / "**" / "*.jpg"),
        str(project_root / "data"    / "**"            / "*.jpg"),
    ]
    image_paths: list[str] = []
    for pat in patterns:
        found = glob.glob(pat, recursive=True)
        image_paths.extend(found)
        if len(image_paths) >= 3:
            break
    image_paths = image_paths[:3]

    if not image_paths:
        print("No real images found — creating synthetic images for demo")
        # create tiny synthetic images
        for i in range(3):
            synth = (np.random.randint(40, 200, (256, 256, 3), dtype=np.uint8))
            tmp   = str(output_dir / f"_synth_{i}.jpg")
            cv2.imwrite(tmp, synth)
            image_paths.append(tmp)

    print(f"Using {len(image_paths)} images")

    panels = []
    for img_path in image_paths:
        bgr = cv2.imread(img_path)
        if bgr is None:
            print(f"  Could not read {img_path}")
            continue
        if bgr.ndim == 2:
            bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
        h, w = bgr.shape[:2]

        raw_mask  = _make_gaussian_blob_mask(h, w, radius_frac=0.30)
        clean     = clean_mask(raw_mask)

        before_pct = float(raw_mask.mean() * 100)
        after_pct  = float(clean.mean()    * 100)
        print(f"  {Path(img_path).name:40s}  before={before_pct:5.1f}%  after={after_pct:5.1f}%")

        # Build comparison panel (original | raw mask overlay | clean mask overlay)
        rgb       = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        raw_vis   = (raw_mask * 255).astype(np.uint8)
        raw_rgb   = cv2.applyColorMap(raw_vis, cv2.COLORMAP_JET)
        raw_rgb   = cv2.cvtColor(raw_rgb, cv2.COLOR_BGR2RGB)
        raw_rgb   = cv2.resize(raw_rgb, (256, 256))

        clean_vis = (clean * 255).astype(np.uint8)
        clean_rgb = cv2.applyColorMap(clean_vis, cv2.COLORMAP_JET)
        clean_rgb = cv2.cvtColor(clean_rgb, cv2.COLOR_BGR2RGB)
        clean_rgb = cv2.resize(clean_rgb, (256, 256))

        orig_rs   = cv2.resize(rgb, (256, 256))
        row       = np.concatenate([orig_rs, raw_rgb, clean_rgb], axis=1)
        panels.append(row)

    if panels:
        grid = np.concatenate(panels, axis=0)
        out_path = str(output_dir / "stage2_comparison.png")
        cv2.imwrite(out_path, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
        print(f"\nSaved comparison grid -> {out_path}")
    else:
        print("No panels to save.")
