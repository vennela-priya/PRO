"""
refinement/boundary_extraction.py
====================================
Tumour boundary extraction and visualisation.

Finds contours in the binary mask, filters by minimum area,
smooths each contour with approxPolyDP (and convex hull as fallback for
jagged results), then draws them on the image with the specified colour
and thickness.
"""
from __future__ import annotations

import cv2
import numpy as np


# ── Public API ────────────────────────────────────────────────────────────────

def extract_boundary(
    image:     np.ndarray,
    mask:      np.ndarray,
    color:     tuple = (0, 255, 0),
    thickness: int   = 2,
    min_area:  int   = 100,
) -> tuple[np.ndarray, list]:
    """
    Find, filter, smooth contours and draw them on the image.

    Parameters
    ----------
    image     : np.ndarray
        Input image (H, W, 3) in RGB or BGR format; not modified in-place.
    mask      : np.ndarray
        Binary mask, shape (H, W).  Values in {0, 1} or {0, 255}.
    color     : tuple
        BGR contour colour (default green = (0, 255, 0)).
    thickness : int
        Contour line thickness in pixels (default 2).
    min_area  : int
        Minimum contour area in pixels to keep (default 100).

    Returns
    -------
    tuple[np.ndarray, list]
        (annotated_image, contour_list) where:
        - annotated_image is a copy of `image` with contours drawn.
        - contour_list is a list of smoothed contour arrays (each Nx1x2 int32).
    """
    # Ensure mask is uint8 with values {0, 1}
    m = mask.astype(np.uint8)
    if m.max() > 1:
        m = (m > 127).astype(np.uint8)

    # Ensure 2-D
    if m.ndim == 3:
        m = m[:, :, 0]

    # Find contours
    contours_raw, _ = cv2.findContours(
        m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Filter by area and smooth
    contours_filtered: list[np.ndarray] = []
    for cnt in contours_raw:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        smoothed = _smooth_contour(cnt)
        contours_filtered.append(smoothed)

    # Draw on copy of image
    annotated = image.copy()
    cv2.drawContours(annotated, contours_filtered, -1, color, thickness)

    return annotated, contours_filtered


def _smooth_contour(contour: np.ndarray) -> np.ndarray:
    """
    Smooth a contour with approxPolyDP; fall back to convex hull if still jagged.

    A contour is considered "jagged" if the number of vertices is greater than
    40% of the raw vertex count after approximation.

    Parameters
    ----------
    contour : np.ndarray
        Raw contour array of shape (N, 1, 2), dtype int32.

    Returns
    -------
    np.ndarray
        Smoothed contour, same dtype, shape (M, 1, 2) with M <= N.
    """
    if len(contour) < 4:
        return contour

    arc_len  = cv2.arcLength(contour, closed=True)
    epsilon  = 0.01 * arc_len                        # 1% of perimeter
    approx   = cv2.approxPolyDP(contour, epsilon, closed=True)

    # Jagged check: too many vertices remaining relative to original
    raw_n    = len(contour)
    approx_n = len(approx)
    if approx_n > max(4, int(raw_n * 0.40)):
        # Fall back to convex hull to guarantee a smooth polygon
        hull = cv2.convexHull(contour)
        return hull

    return approx


# ── Test / Demo ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import glob
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root))
    output_dir = project_root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

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
        h, w = bgr.shape[:2]
        rgb  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        raw_mask = _make_gaussian_blob_mask(h, w, 0.30)
        clean    = clean_mask(raw_mask)

        annotated, contours = extract_boundary(
            rgb, clean, color=(0, 255, 0), thickness=2, min_area=100
        )

        print(
            f"  {Path(img_path).name:40s}  "
            f"contours={len(contours)}  "
            f"mask={clean.mean()*100:.1f}%"
        )

        orig_rs  = cv2.resize(rgb,       (256, 256))
        ann_rs   = cv2.resize(annotated, (256, 256))
        mask_rs  = cv2.resize((clean * 255).astype(np.uint8), (256, 256))
        mask_rgb = np.stack([mask_rs] * 3, axis=-1)
        row      = np.concatenate([orig_rs, mask_rgb, ann_rs], axis=1)
        panels.append(row)

    if panels:
        grid     = np.concatenate(panels, axis=0)
        out_path = str(output_dir / "stage6_boundary.png")
        cv2.imwrite(out_path, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
        print(f"\nSaved -> {out_path}")
