"""
refinement/classical_mask_generator.py
=======================================
Pure classical image processing pipeline for brain tumor mask generation.

Root cause of GradCAM failure
------------------------------
GradCAM computes gradients w.r.t. classifier activations — it finds the
region that CHANGES THE CLASSIFICATION SCORE most, not the actual tumor.
When classifier weights are imperfect, GradCAM can confidently point to
the wrong hemisphere.  This pipeline replaces the entire model-based chain
with a physically motivated rule:

    Brain tumors on contrast-enhanced MRI are HYPERINTENSE (bright).
    They are spatially compact, inside the brain, and NOT the skull.

Pipeline per image
------------------
 1.  Grayscale conversion
 2.  Brain extraction   — Otsu + close + fill_holes + largest-CC + erode-skull
 3.  CLAHE              — local contrast enhancement inside the brain
 4.  Multi-percentile   — try class-specific brightness levels (high → low)
 5.  Fill holes         — handles ring-enhancing glioma (bright ring, dark core)
 6.  Morphological close → merge nearby bright pixels into solid blob
 7.  Border prune       — remove components touching the image edge (skull)
 8.  Score each blob    — brightness × compactness  (within class size bounds)
 9.  Best blob          — pick highest-scoring valid component
10.  Adaptive smooth    — close+open with kernel ∝ blob radius → no jagged edges
11.  Final fill_holes   — ensure perfectly solid interior
12.  cv2.fitEllipse     — draw perfectly smooth boundary (blue, BGR)

No-tumor handling
-----------------
If tumor_class == "notumor": return zero mask immediately (no processing).
During live inference (unknown class): if no bright compact blob survives
scoring, return zero mask.

Usage
-----
    # Validation demo (4 USE-Me Test images):
    python refinement/classical_mask_generator.py

    # Single image:
    python refinement/classical_mask_generator.py --image path.jpg --cls glioma

    # Process entire dataset and save masks:
    python refinement/classical_mask_generator.py --dataset
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from scipy import ndimage

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

logger = logging.getLogger(__name__)

# ── Class metadata ─────────────────────────────────────────────────────────────

CLASS_NAMES  = ["glioma", "meningioma", "notumor", "pituitary"]
TUMOR_CLASSES = {"glioma", "meningioma", "pituitary"}

# Brightness percentile sweep per class.
# Each value P means: "keep the top (100-P)% brightest brain pixels".
# We try multiple levels; the best-scored blob across all levels wins.
#   glioma     : moderately bright ring        → try p80..p92
#   meningioma : very bright lobulated mass    → try p72..p84
#   pituitary  : tiny hyperintense central spot → try p86..p96
CLASS_PCT_SWEEP: dict[str, list[int]] = {
    "glioma":     [80, 84, 88, 92],
    "meningioma": [72, 76, 80, 84],
    "pituitary":  [86, 90, 93, 96],
}

# Morphological close kernel (pixels) in the bright binary map.
CLASS_CLOSE_KERNEL: dict[str, int] = {
    "glioma":     7,
    "meningioma": 9,
    "pituitary":  5,
}

# Valid tumor area as % of TOTAL IMAGE pixels (min, max)
CLASS_SIZE_BOUNDS: dict[str, tuple[float, float]] = {
    "glioma":     (0.30, 28.0),
    "meningioma": (0.50, 50.0),
    # pituitary min = 0.30% so tiny noise blobs (< 200px in 512x512) are ignored
    "pituitary":  (0.30, 12.0),
}

# Border margin fraction.
# meningioma = 0.0: surface tumors legitimately touch the brain/image edge;
# the skull-erode step already removes pure skull pixels so no false positives.
CLASS_BORDER_MARGIN: dict[str, float] = {
    "glioma":     0.025,
    "meningioma": 0.000,   # NO border check — meningioma touches brain surface
    "pituitary":  0.020,
}

# Pituitary anatomical position gate.
# Eyes are at cy_frac ≈ 0.21 (upper brain in skull-base axial view).
# True pituitary is deeper — cy_frac ≥ 0.30. X is central (0.25–0.75).
PITUITARY_X_FRAC = (0.25, 0.75)
PITUITARY_Y_FRAC = (0.30, 0.92)   # eyes at 0.21 are excluded; pituitary at ~0.39 passes

# Contour style (blue BGR = matches traced reference style)
CONTOUR_COLOR     = (255, 80, 0)
CONTOUR_THICKNESS = 2

# ── USE-Me Test validation targets ────────────────────────────────────────────

VALIDATION_IMAGES = [
    ("glioma",     str(_ROOT / "Project" / "Testing" / "glioma"     / "Te-gl_0014.jpg")),
    ("meningioma", str(_ROOT / "Project" / "Testing" / "meningioma" / "Te-me_0020.jpg")),
    ("pituitary",  str(_ROOT / "Project" / "Testing" / "pituitary"  / "Te-pi_0021.jpg")),
    ("notumor",    str(_ROOT / "Project" / "Testing" / "notumor"    / "Te-noTr_0000.jpg")),
]


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Brain extraction
# ─────────────────────────────────────────────────────────────────────────────

def _extract_brain(gray: np.ndarray) -> np.ndarray:
    """
    Return a binary uint8 brain mask (0/255) that excludes the dark background
    and erodes the skull boundary so threshold bright pixels are purely intra-cranial.

    Method:
        Heavy Gaussian blur → Otsu → morphological close → fill holes
        → largest CC (= brain) → erode to remove skull ring.
    """
    H, W = gray.shape

    # Aggressive blur to flatten brain texture before thresholding
    blurred = cv2.GaussianBlur(gray, (31, 31), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Close any fine gaps in the brain outline
    kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23, 23))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kc)

    # Fill interior (ventricles, CSF spaces are dark)
    filled = ndimage.binary_fill_holes(closed > 0).astype(np.uint8) * 255

    # Keep the largest connected component — that is the brain
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        filled.astype(np.uint8), connectivity=8)
    if n <= 1:
        return np.ones((H, W), dtype=np.uint8) * 255
    largest = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
    brain = np.zeros((H, W), dtype=np.uint8)
    brain[labels == largest] = 255

    # Erode skull boundary: tumors are well inside the brain
    r_skull = max(6, int(min(H, W) * 0.028))
    ke = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (r_skull * 2 + 1, r_skull * 2 + 1))
    brain = cv2.erode(brain, ke)

    return brain   # 0 or 255


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _compactness(comp_u8: np.ndarray) -> float:
    """
    Compactness  =  4π · area / perimeter²
    = 1.0 for a perfect circle, < 1.0 for irregular shapes.
    Penalises thin elongated structures (blood vessels, white matter tracts).
    """
    area = int(comp_u8.sum())
    if area == 0:
        return 0.0
    cnts, _ = cv2.findContours(comp_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0.0
    perim = cv2.arcLength(cnts[0], True)
    return min(1.0, 4.0 * np.pi * area / (perim ** 2)) if perim > 0 else 0.0


def _touches_border(comp_u8: np.ndarray, margin: int) -> bool:
    """True if any active pixel is within *margin* pixels of the image edge."""
    return bool(
        comp_u8[:margin, :].any()  or
        comp_u8[-margin:, :].any() or
        comp_u8[:, :margin].any()  or
        comp_u8[:, -margin:].any()
    )


def _adaptive_smooth(mask: np.ndarray, H: int, W: int) -> np.ndarray:
    """
    Morphological close then open with kernels proportional to the blob radius.
    This prevents over-smoothing tiny pituitary masses while still rounding
    rough glioma boundaries.
    """
    area = int(mask.sum())
    if area == 0:
        return mask
    r_blob = max(4.0, float(np.sqrt(area / np.pi)))
    r_close = max(3, int(r_blob * 0.30))
    r_open  = max(2, int(r_blob * 0.18))
    kc = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (r_close * 2 + 1, r_close * 2 + 1))
    ko = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (r_open  * 2 + 1, r_open  * 2 + 1))
    b = mask.astype(np.uint8)
    b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, kc)
    b = cv2.morphologyEx(b, cv2.MORPH_OPEN,  ko)
    return b


# ─────────────────────────────────────────────────────────────────────────────
# Step 4-9 — Bright-blob detection
# ─────────────────────────────────────────────────────────────────────────────

def _find_best_blob(
    enhanced:    np.ndarray,   # CLAHE-enhanced grayscale, uint8
    brain:       np.ndarray,   # 0/255 brain mask
    tumor_class: str,
    H: int, W: int,
) -> Optional[np.ndarray]:
    """
    Sweep multiple intensity percentile levels; pick the best-scoring tumor blob.

    Scoring formula
    ---------------
        score = mean_brightness
                × (0.3 + 0.7 × compactness)
                × min(2.0, sqrt(area_frac / min_pct_frac))

    The third term is the SIZE BOOST:
    - Penalises tiny noise blobs that have high brightness/compactness but almost
      no area (e.g. single-pixel artifacts).  This is critical for pituitary where
      isolated bright specks would otherwise outscore the actual 1000+ px tumour.
    - Caps at 2.0 so large meningiomas do not dominate too aggressively.

    Border check
    ------------
    - glioma / pituitary : blobs touching the image edge are skull / artifacts
    - meningioma         : margin = 0 (disabled).  Meningiomas are surface tumours
      that legitimately reach the brain–skull interface; skull was already removed
      by the brain-mask erosion step, so border-touching brain pixels are safe.

    Pituitary position gate
    -----------------------
    Eye orbits are at cy_frac ≈ 0.21 (upper brain in skull-base axial view).
    True pituitary is deeper: cy_frac ≥ 0.30. The y-gate excludes the orbits.
    The x-gate (0.25–0.75) keeps only midline structures.

    Meningioma expansion
    --------------------
    After finding the highest-scoring lobe, dilate by 70% of its radius and
    re-sweep all levels to merge adjacent lobes that belong to the same tumour.
    """
    brain_pixels = enhanced[brain > 0]
    if len(brain_pixels) == 0:
        return None

    margin_frac = CLASS_BORDER_MARGIN.get(tumor_class, 0.025)
    margin      = max(1, int(min(H, W) * margin_frac)) if margin_frac > 0 else 0

    min_pct, max_pct = CLASS_SIZE_BOUNDS.get(tumor_class, (0.3, 35.0))
    min_pct_frac = min_pct / 100.0
    min_area     = max(12, int(H * W * min_pct_frac))
    max_area     = int(H * W * max_pct / 100.0)

    close_k = CLASS_CLOSE_KERNEL.get(tumor_class, 7)
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))

    best_mask  = None
    best_score = -1.0

    for pct in CLASS_PCT_SWEEP.get(tumor_class, [80, 85, 90]):
        thr    = float(np.percentile(brain_pixels, pct))
        binary = np.zeros((H, W), dtype=np.uint8)
        binary[(enhanced >= thr) & (brain > 0)] = 1

        # CRITICAL for glioma: ring-enhancing lesion has bright ring + dark centre.
        # fill_holes BEFORE CC analysis converts the ring into a solid disk.
        binary = ndimage.binary_fill_holes(binary.astype(bool)).astype(np.uint8)

        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_close)

        n, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8)

        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if not (min_area <= area <= max_area):
                continue

            comp = (labels == i).astype(np.uint8)

            # Border check (disabled for meningioma)
            if margin > 0 and _touches_border(comp, margin):
                continue

            # Pituitary: anatomical position filter
            # Eyes at cy_frac ≈ 0.21 are excluded; pituitary at ~0.39 passes.
            if tumor_class == "pituitary":
                cx_f = float(centroids[i][0]) / W
                cy_f = float(centroids[i][1]) / H
                if not (PITUITARY_X_FRAC[0] <= cx_f <= PITUITARY_X_FRAC[1]):
                    logger.debug(
                        f"  pituitary comp {i} cx_frac={cx_f:.2f} outside "
                        f"[{PITUITARY_X_FRAC[0]},{PITUITARY_X_FRAC[1]}] -> skip")
                    continue
                if not (PITUITARY_Y_FRAC[0] <= cy_f <= PITUITARY_Y_FRAC[1]):
                    logger.debug(
                        f"  pituitary comp {i} cy_frac={cy_f:.2f} outside "
                        f"[{PITUITARY_Y_FRAC[0]},{PITUITARY_Y_FRAC[1]}] -> skip")
                    continue

            mean_int   = float(enhanced[comp > 0].mean()) / 255.0
            compact    = _compactness(comp)
            area_frac  = area / (H * W)
            # Size boost: larger blobs preferred; tiny artefacts suppressed
            size_boost = min(2.0, float(np.sqrt(area_frac / max(min_pct_frac, 1e-9))))
            score      = mean_int * (0.3 + 0.7 * compact) * size_boost

            logger.debug(
                f"  pct={pct} comp={i} area={area}({area_frac*100:.2f}%) "
                f"bright={mean_int:.3f} compact={compact:.3f} "
                f"size_boost={size_boost:.2f} score={score:.4f}"
            )

            if score > best_score:
                best_score = score
                best_mask  = comp.copy()

    if best_mask is None:
        return None

    # ── Meningioma: merge adjacent lobes ──────────────────────────────────────
    # Meningiomas can appear as two adjacent lobes separated by a thin dark
    # sulcus.  After finding the best lobe, we look for other lobes whose
    # centroid falls inside the best-lobe's bounding box expanded by 1× radius.
    # This spatial gate prevents accidentally absorbing distant brain structures.
    if tumor_class == "meningioma":
        seed_area = int(best_mask.sum())
        r_blob    = max(10, int(np.sqrt(seed_area / np.pi)))

        bys_m, bxs_m = np.where(best_mask > 0)
        by1 = max(0, int(bys_m.min()) - r_blob)
        by2 = min(H, int(bys_m.max()) + r_blob)
        bx1 = max(0, int(bxs_m.min()) - r_blob)
        bx2 = min(W, int(bxs_m.max()) + r_blob)

        # Sweep ALL levels and keep the LARGEST valid merged result.
        # (Breaking at the first larger result would stop at p72 which only
        #  finds the lower lobe; p76 finds BOTH lobes and is much larger.)
        best_merged: Optional[np.ndarray] = None
        best_merged_pct: float = 0.0

        for pct2 in CLASS_PCT_SWEEP["meningioma"]:
            thr2  = float(np.percentile(brain_pixels, pct2))
            bin2  = np.zeros((H, W), dtype=np.uint8)
            bin2[(enhanced >= thr2) & (brain > 0)] = 1
            bin2  = ndimage.binary_fill_holes(bin2.astype(bool)).astype(np.uint8)
            k9    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            bin2  = cv2.morphologyEx(bin2, cv2.MORPH_CLOSE, k9)

            n2, l2, s2, c2 = cv2.connectedComponentsWithStats(bin2, connectivity=8)
            merged = np.zeros((H, W), dtype=np.uint8)
            for j in range(1, n2):
                area_j = int(s2[j, cv2.CC_STAT_AREA])
                if area_j < int(min_area * 0.30):
                    continue
                if area_j > seed_area * 2.5:
                    continue
                cxj, cyj = float(c2[j][0]), float(c2[j][1])
                if by1 <= cyj <= by2 and bx1 <= cxj <= bx2:
                    merged |= (l2 == j).astype(np.uint8)

            merged     = ndimage.binary_fill_holes(merged.astype(bool)).astype(np.uint8)
            k9b        = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            merged     = cv2.morphologyEx(merged, cv2.MORPH_CLOSE, k9b)
            merged     = ndimage.binary_fill_holes(merged.astype(bool)).astype(np.uint8)
            merged_pct = 100.0 * merged.sum() / (H * W)

            if merged.sum() > seed_area and merged_pct <= max_pct:
                if merged_pct > best_merged_pct:
                    best_merged_pct = merged_pct
                    best_merged     = merged.copy()

        if best_merged is not None:
            logger.debug(
                f"[Classical] meningioma lobes merged: "
                f"{100*seed_area/(H*W):.1f}% -> {best_merged_pct:.1f}%")
            best_mask = best_merged

    logger.debug(f"[Classical] Best blob score={best_score:.4f}")
    return best_mask


# ─────────────────────────────────────────────────────────────────────────────
# Public API — generate_mask
# ─────────────────────────────────────────────────────────────────────────────

def generate_mask(img_bgr: np.ndarray, tumor_class: str) -> np.ndarray:
    """
    Generate a clean binary tumor mask using ONLY classical image processing.

    Parameters
    ----------
    img_bgr     : np.ndarray  uint8 BGR image.
    tumor_class : str         "glioma" | "meningioma" | "pituitary" | "notumor"

    Returns
    -------
    np.ndarray  uint8 binary mask (H, W) with values {0, 1}.
                All-zero for "notumor" or when no valid blob is found.
    """
    H, W  = img_bgr.shape[:2]
    empty = np.zeros((H, W), dtype=np.uint8)

    # ── No-tumor: immediate exit ───────────────────────────────────────────────
    if tumor_class not in TUMOR_CLASSES:
        logger.info("[Classical] notumor — returning zero mask")
        return empty

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # ── Step 2: Brain extraction ───────────────────────────────────────────────
    brain = _extract_brain(gray)

    # ── Step 3: CLAHE for local contrast enhancement ──────────────────────────
    clahe    = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    # Mild blur to suppress speckle noise
    enhanced = cv2.GaussianBlur(enhanced, (5, 5), 1.0)

    # ── Steps 4-9: Multi-percentile bright-blob detection ─────────────────────
    best = _find_best_blob(enhanced, brain, tumor_class, H, W)

    if best is None:
        logger.info("[Classical] No valid tumor blob found — zero mask")
        return empty

    # ── Steps 10-11: Adaptive smooth + final fill ─────────────────────────────
    best = ndimage.binary_fill_holes(best.astype(bool)).astype(np.uint8)
    best = _adaptive_smooth(best, H, W)
    best = ndimage.binary_fill_holes(best.astype(bool)).astype(np.uint8)

    region_pct = 100.0 * best.sum() / (H * W)
    logger.info(f"[Classical] {tumor_class}: region={region_pct:.1f}%")

    return best.astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Step 12 — Boundary drawing
# ─────────────────────────────────────────────────────────────────────────────

def draw_boundary(
    img_bgr:   np.ndarray,
    mask:      np.ndarray,
    color:     tuple = CONTOUR_COLOR,
    thickness: int   = CONTOUR_THICKNESS,
) -> np.ndarray:
    """
    Draw a smooth tumor boundary on a copy of img_bgr.

    Uses cv2.fitEllipse which produces a perfectly smooth, continuous ellipse
    matching the style of the hand-traced reference boundaries.
    Falls back to raw contour for very small blobs (< 5 contour points).
    """
    result  = img_bgr.copy()
    mask_u8 = (mask * 255).astype(np.uint8)

    cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return result

    contour = max(cnts, key=cv2.contourArea)

    if len(contour) >= 5:
        ellipse = cv2.fitEllipse(contour)
        cv2.ellipse(result, ellipse, color, thickness, cv2.LINE_AA)
        (cx, cy), (ax, ay), ang = ellipse
        logger.info(
            f"[Boundary] Ellipse centre=({cx:.1f},{cy:.1f}) "
            f"axes=({ax:.1f},{ay:.1f}) angle={ang:.1f}deg"
        )
    else:
        cv2.drawContours(result, [contour], -1, color, thickness, cv2.LINE_AA)
        logger.info("[Boundary] Blob too small for ellipse — raw contour drawn")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Validation runner
# ─────────────────────────────────────────────────────────────────────────────

def _build_panel(true_cls: str, img_path: str,
                 result_bgr: np.ndarray, has_bnd: bool,
                 region_pct: float, size: int = 320) -> np.ndarray:
    """Two-column panel: [Original | Boundary result] + label strip."""
    bgr  = cv2.imread(img_path)
    if bgr is None:
        return np.zeros((size + 30, size * 2, 3), dtype=np.uint8)
    orig = cv2.resize(bgr,        (size, size))
    bnd  = cv2.resize(result_bgr, (size, size))
    row  = np.concatenate([orig, bnd], axis=1)

    lh  = 30
    lbl = np.full((lh, row.shape[1], 3), 22, dtype=np.uint8)
    if has_bnd:
        info = f"pred={true_cls}  area={region_pct:.1f}%  BOUNDARY"
        col  = (60, 220, 60)
    else:
        info = "NO BOUNDARY"
        col  = (80, 80, 220)
    cv2.putText(lbl, f"True: {true_cls} | {info}",
                (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.52, col, 1, cv2.LINE_AA)

    return np.concatenate([lbl, row], axis=0)


def run_validation(output_dir: str = "outputs/classical/") -> None:
    """Run on the 4 USE-Me Test images and save comparison grid."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 72)
    print(f"{'CLASSICAL MASK GENERATOR — USE-Me Validation':^72}")
    print("=" * 72)
    print()

    panels = []

    for true_cls, img_path in VALIDATION_IMAGES:
        if not Path(img_path).exists():
            print(f"  [SKIP] {Path(img_path).name} not found")
            continue

        bgr  = cv2.imread(img_path)
        if bgr is None:
            print(f"  [SKIP] Cannot read {img_path}")
            continue

        mask       = generate_mask(bgr, true_cls)
        result_bgr = draw_boundary(bgr, mask)
        has_bnd    = bool(mask.any())
        region_pct = 100.0 * mask.sum() / (mask.shape[0] * mask.shape[1])

        stem = Path(img_path).stem
        cv2.imwrite(str(out / f"{stem}_boundary.png"), result_bgr)
        cv2.imwrite(str(out / f"{stem}_mask.png"),
                    (mask * 255).astype(np.uint8))

        if not has_bnd and true_cls == "notumor":
            status = "NO BOUNDARY [correct]"
        elif not has_bnd:
            status = "NO BOUNDARY [missed!]"
        else:
            status = f"BOUNDARY  area={region_pct:.1f}%"

        print(f"  {Path(img_path).name:<28} true={true_cls:<12} -> {status}")

        panel = _build_panel(true_cls, img_path, result_bgr, has_bnd, region_pct)
        panels.append(panel)

    if panels:
        grid     = np.concatenate(panels, axis=0)
        out_path = out / "validation_grid.png"
        cv2.imwrite(str(out_path), grid)
        print(f"\nValidation grid saved -> {out_path}")

    print()
    print("Done.")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Dataset-wide mask generation (for U-Net training)
# ─────────────────────────────────────────────────────────────────────────────

def process_dataset(
    data_root:  str = "",
    output_root: str = "outputs/classical_masks/",
    extensions:  tuple = (".jpg", ".jpeg", ".png"),
) -> None:
    """
    Walk every class subfolder, generate a classical mask for each image,
    and save it to a parallel directory structure under output_root.

    Directory layout produced:
        output_root/
            glioma/        Te-gl_0000_mask.png  ...
            meningioma/    Te-me_0000_mask.png  ...
            pituitary/     Te-pi_0000_mask.png  ...
            notumor/       Te-noTr_0000_mask.png ...  (all-zero)

    These masks can be used directly as pseudo ground-truth for U-Net training.
    """
    # Auto-detect dataset root
    candidates = [
        data_root,
        str(_ROOT / "Project" / "Training"),
        str(_ROOT / "Project" / "Testing"),
        str(_ROOT / "data" / "Training"),
    ]
    src_roots = [Path(c) for c in candidates if c and Path(c).exists()]
    if not src_roots:
        print("[Dataset] ERROR: no dataset root found. "
              "Pass --dataset_root or put data in Project/Training/")
        return

    out_root = Path(output_root)

    total = ok = skipped = 0

    for src in src_roots:
        for cls_dir in sorted(src.iterdir()):
            if not cls_dir.is_dir():
                continue
            cls_name = cls_dir.name.lower()
            if cls_name not in {c.lower() for c in CLASS_NAMES}:
                continue

            # Normalise class name
            cls_norm = next(
                (c for c in CLASS_NAMES if c.lower() == cls_name), cls_name)

            out_cls = out_root / cls_norm
            out_cls.mkdir(parents=True, exist_ok=True)

            images = [
                f for f in cls_dir.iterdir()
                if f.suffix.lower() in extensions
            ]
            print(f"  [{cls_norm}] {len(images)} images from {cls_dir}")

            for img_path in sorted(images):
                bgr = cv2.imread(str(img_path))
                if bgr is None:
                    skipped += 1
                    continue
                try:
                    mask = generate_mask(bgr, cls_norm)
                    save_path = out_cls / (img_path.stem + "_mask.png")
                    cv2.imwrite(str(save_path), (mask * 255).astype(np.uint8))
                    ok += 1
                except Exception as exc:
                    logger.warning(f"  FAILED {img_path.name}: {exc}")
                    skipped += 1
                total += 1

    print(f"\nDataset processing complete: {ok}/{total} masks saved "
          f"({skipped} skipped) -> {out_root}")


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch Dataset class for U-Net training
# ─────────────────────────────────────────────────────────────────────────────

UNET_TRAINING_CODE = '''
# ──────────────────────────────────────────────────────────────────────
# U-Net training with classical pseudo-masks
#
# After running:
#   python refinement/classical_mask_generator.py --dataset
#
# Use this dataset class to train your U-Net:
# ──────────────────────────────────────────────────────────────────────

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
import cv2
import numpy as np

class ClassicalMaskDataset(Dataset):
    """
    Pairs each MRI image with its classical pseudo-mask.

    data_root   : "Project/Training"  (original images)
    mask_root   : "outputs/classical_masks"  (generated by process_dataset)
    use_dist    : bool  If True, use distance-transform soft labels instead of
                        hard binary masks. Soft labels improve boundary sharpness.
    """
    IMG_SIZE = 256

    def __init__(self, data_root: str, mask_root: str,
                 use_dist: bool = True, augment: bool = True):
        self.pairs:   list[tuple[Path, Path]] = []
        self.use_dist = use_dist
        self.augment  = augment

        data_root = Path(data_root)
        mask_root = Path(mask_root)

        for cls_dir in sorted(data_root.iterdir()):
            if not cls_dir.is_dir():
                continue
            for img_path in sorted(cls_dir.glob("*.jpg")) + sorted(cls_dir.glob("*.png")):
                mask_path = mask_root / cls_dir.name / (img_path.stem + "_mask.png")
                if mask_path.exists():
                    self.pairs.append((img_path, mask_path))

        self.img_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        S = self.IMG_SIZE

        bgr  = cv2.imread(str(img_path))
        rgb  = cv2.cvtColor(bgr,  cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        rgb  = cv2.resize(rgb,  (S, S))
        mask = cv2.resize(mask, (S, S), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 127).astype(np.float32)   # binary {0,1}

        if self.use_dist and mask.any():
            # Distance-transform soft label: interior pixels get score > 0
            # This guides U-Net to learn the CENTRE of the tumor, not just edges
            from scipy.ndimage import distance_transform_edt
            dt   = distance_transform_edt(mask)
            dt   = dt / (dt.max() + 1e-6)      # normalise to [0,1]
            mask = 0.5 * mask + 0.5 * dt        # blend hard + soft label

        if self.augment:
            # Random horizontal flip
            if np.random.rand() > 0.5:
                rgb  = rgb[:, ::-1].copy()
                mask = mask[:, ::-1].copy()
            # Random 90-degree rotation
            k = np.random.randint(0, 4)
            rgb  = np.rot90(rgb,  k).copy()
            mask = np.rot90(mask, k).copy()

        img_t  = self.img_tf(rgb)
        mask_t = torch.from_numpy(mask).unsqueeze(0)  # (1, H, W)
        return img_t, mask_t


# Training loop sketch:
# from segmentation.attention_unet import AttentionUNet
# from segmentation.losses import CombinedSegLoss
#
# model    = AttentionUNet(in_channels=3, out_channels=1).cuda()
# dataset  = ClassicalMaskDataset("Project/Training",
#                                 "outputs/classical_masks", use_dist=True)
# loader   = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)
# opt      = torch.optim.AdamW(model.parameters(), lr=1e-4)
# loss_fn  = CombinedSegLoss()
#
# for epoch in range(50):
#     for imgs, masks in loader:
#         imgs, masks = imgs.cuda(), masks.cuda()
#         preds = model(imgs)
#         loss  = loss_fn(preds, masks)
#         opt.zero_grad(); loss.backward(); opt.step()
'''


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Classical image-processing brain tumor mask generator")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--image",   default=None,
                   help="Path to a single MRI image")
    g.add_argument("--dataset", action="store_true",
                   help="Process the entire dataset and save all masks")
    p.add_argument("--cls",          default=None,
                   help="Tumor class for --image mode (glioma/meningioma/"
                        "pituitary/notumor). Auto-detected from parent folder if omitted.")
    p.add_argument("--output_dir",   default="outputs/classical/")
    p.add_argument("--dataset_root", default="",
                   help="Root of the training dataset (for --dataset mode)")
    p.add_argument("--mask_root",    default="outputs/classical_masks/",
                   help="Where to save dataset masks (for --dataset mode)")
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )
    args = _parse()

    if args.dataset:
        process_dataset(
            data_root   = args.dataset_root,
            output_root = args.mask_root,
        )

    elif args.image:
        img_path = args.image
        # Auto-detect class from parent folder name if --cls not given
        if args.cls:
            cls = args.cls.lower()
        else:
            cls = Path(img_path).parent.name.lower()
            if cls not in {c.lower() for c in CLASS_NAMES}:
                cls = "glioma"   # safe default

        bgr  = cv2.imread(img_path)
        if bgr is None:
            print(f"ERROR: Cannot read {img_path}")
            sys.exit(1)

        mask       = generate_mask(bgr, cls)
        result_bgr = draw_boundary(bgr, mask)
        has_bnd    = bool(mask.any())
        pct        = 100.0 * mask.sum() / (mask.shape[0] * mask.shape[1])

        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        stem = Path(img_path).stem
        cv2.imwrite(str(out / f"{stem}_boundary.png"), result_bgr)
        cv2.imwrite(str(out / f"{stem}_mask.png"), (mask * 255).astype(np.uint8))

        print(f"\nImage     : {img_path}")
        print(f"Class     : {cls}")
        bnd = f"YES  area={pct:.1f}%" if has_bnd else "NO (no bright tumor blob)"
        print(f"Boundary  : {bnd}")
        print(f"Saved     : {out / (stem + '_boundary.png')}")

    else:
        # Default: run the 4-image validation demo
        run_validation(output_dir=args.output_dir)
