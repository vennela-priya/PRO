"""
refinement/clean_boundary.py
============================
Definitive tumor boundary extraction pipeline.

Algorithm
---------
1.  Classify      -> predicted class + probability (ResNetCBAM)
2.  No-tumor gate -> suppress boundary if:
        a. pred == notumor
        b. confidence < CLASS_MIN_CONF[pred]   (class-specific)
3.  GradCAM       -> heat-map; centroid of top-25% region = "tumor peak"
4.  Brain mask    -> remove dark MRI background (pixels < p10 brightness)
5.  Intensity     -> keep top (100-P)% brightest brain pixels  [class-specific]
6.  Guided blend  -> multiply by GradCAM weight so off-peak bright spots
                     (e.g. eyes, skull) score low
7.  Connected-components -> score each blob by
        score = cam_mean * sqrt(area) / distance_to_peak
        subject to size bounds [min_pct, max_pct]  [class-specific]
8.  Fill holes    -> handle ring-enhancing glioma's necrotic centre
9.  Morph smooth  -> remove jagged edges
10. cv2.fitEllipse -> perfectly smooth ellipse boundary
11. Draw in blue  -> match traced-reference style

Usage
-----
    python refinement/clean_boundary.py              # USE-Me validation set
    python refinement/clean_boundary.py --image x.jpg
    python refinement/clean_boundary.py --debug      # save intermediate maps
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from scipy import ndimage

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from refinement.gradcam_fusion import _try_load_classifier, _preprocess_image, GradCAM

logger = logging.getLogger(__name__)

# ── class metadata ────────────────────────────────────────────────────────────
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
NOTUMOR_IDX = CLASS_NAMES.index("notumor")

# Minimum classifier confidence to draw a boundary.
# meningioma is higher: notumor images are commonly mispredicted as meningioma
# with conf 0.40-0.48; real meningioma is typically >= 0.50.
CLASS_MIN_CONF: dict[str, float] = {
    "glioma":     0.32,
    "meningioma": 0.50,
    "pituitary":  0.30,
}

# Intensity percentile: keep top (100-P)% brightest brain pixels as candidates.
# Tumors are hyperintense on MRI; higher P = tighter / smaller region.
CLASS_INT_PCT: dict[str, int] = {
    "glioma":     78,   # keep top 22% brightest brain pixels
    "meningioma": 72,   # keep top 28% (meningioma is often large)
    "pituitary":  88,   # keep top 12% (very small, very bright spot)
}

# GradCAM top-% used to define the "guidance region centroid"
CLASS_CAM_TOP_PCT: dict[str, int] = {
    "glioma":     25,
    "meningioma": 30,
    "pituitary":  15,
}

# Mask size sanity bounds (% of image pixels)
CLASS_MIN_PCT: dict[str, float] = {"glioma": 0.5,  "meningioma": 1.0,  "pituitary": 0.1}
CLASS_MAX_PCT: dict[str, float] = {"glioma": 30.0, "meningioma": 45.0, "pituitary": 15.0}

# Boundary style: bright blue (BGR) to match traced references
CONTOUR_COLOR     = (255, 80, 0)
CONTOUR_THICKNESS = 2

# ── USE-Me Test validation targets ───────────────────────────────────────────
VALIDATION_IMAGES = [
    ("glioma",     str(_ROOT / "Project" / "Testing" / "glioma"     / "Te-gl_0014.jpg")),
    ("meningioma", str(_ROOT / "Project" / "Testing" / "meningioma" / "Te-me_0020.jpg")),
    ("pituitary",  str(_ROOT / "Project" / "Testing" / "pituitary"  / "Te-pi_0021.jpg")),
    ("notumor",    str(_ROOT / "Project" / "Testing" / "notumor"    / "Te-noTr_0000.jpg")),
]

DEFAULT_CLF_PATH = str(
    _ROOT / "checkpoints" / "resnet_cbam" / "best_ep008_auc0.9952.pt"
)


# ─────────────────────────────────────────────────────────────────────────────
# Mask helpers
# ─────────────────────────────────────────────────────────────────────────────

def _keep_largest_cc(binary: np.ndarray) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8), connectivity=8)
    if n <= 1:
        return binary
    largest = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
    out = np.zeros_like(binary, dtype=np.uint8)
    out[labels == largest] = 1
    return out


def _fill_holes(binary: np.ndarray) -> np.ndarray:
    return ndimage.binary_fill_holes(binary.astype(bool)).astype(np.uint8)


def _morph_smooth(binary: np.ndarray, H: int, W: int,
                  radius_frac: float = 0.018) -> np.ndarray:
    r = max(4, int(min(H, W) * radius_frac))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r * 2 + 1, r * 2 + 1))
    b = binary.astype(np.uint8)
    b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, k)
    b = cv2.morphologyEx(b, cv2.MORPH_OPEN,  k)
    return b


# ─────────────────────────────────────────────────────────────────────────────
# Hybrid intensity + GradCAM tumor finder
# ─────────────────────────────────────────────────────────────────────────────

def _find_tumor_region(
    img_bgr:   np.ndarray,
    heatmap:   np.ndarray,
    H:         int,
    W:         int,
    pred_name: str,
    debug_dir: Optional[Path] = None,
    stem:      str = "img",
) -> Optional[np.ndarray]:
    """
    Locate the tumor as the brightest brain blob that aligns with the GradCAM peak.

    Parameters
    ----------
    img_bgr   : BGR image (H, W, 3)
    heatmap   : raw GradCAM float array (any size)
    H, W      : target image dimensions
    pred_name : predicted class name
    debug_dir : if given, intermediate maps are saved here
    stem      : filename stem for debug images

    Returns
    -------
    uint8 binary mask (H, W) with {0,1} values, or None on failure.
    """
    # ── Grayscale ────────────────────────────────────────────────────────────
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # ── Resize & normalise heatmap ────────────────────────────────────────────
    hm = cv2.resize(heatmap.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
    hm_min, hm_max = hm.min(), hm.max()
    if hm_max - hm_min < 1e-6:
        logger.warning("[Hybrid] GradCAM heatmap is flat -> skip")
        return None
    hm_norm = (hm - hm_min) / (hm_max - hm_min)

    # ── GradCAM guidance: centroid of top-P% activation ──────────────────────
    cam_top_pct = CLASS_CAM_TOP_PCT.get(pred_name, 25)
    cam_thr     = float(np.percentile(hm_norm, 100 - cam_top_pct))
    top_cam     = hm_norm >= cam_thr
    ys, xs      = np.where(top_cam)
    if len(xs) == 0:
        logger.warning("[Hybrid] empty CAM top region")
        return None
    peak_x = int(xs.mean())
    peak_y = int(ys.mean())
    logger.debug(f"[Hybrid] CAM peak centroid -> ({peak_x}, {peak_y})")

    # ── Brain mask: remove dark background ────────────────────────────────────
    bg_thr     = max(8.0, float(np.percentile(gray, 10)))
    brain_mask = (gray > bg_thr).astype(np.uint8)

    # ── Intensity threshold: brightest brain pixels ───────────────────────────
    int_pct      = CLASS_INT_PCT.get(pred_name, 78)
    brain_pixels = gray[brain_mask > 0]
    if len(brain_pixels) == 0:
        return None
    bright_thr = float(np.percentile(brain_pixels, int_pct))
    bright     = ((gray >= bright_thr) & (brain_mask > 0)).astype(np.uint8)

    # ── Weight bright mask by GradCAM (suppresses off-peak bright spots) ─────
    weighted_score = (gray / 255.0) * hm_norm  # range [0,1]

    # ── Clean the bright mask ─────────────────────────────────────────────────
    bright = _morph_smooth(bright, H, W, radius_frac=0.012)
    bright = _fill_holes(bright)

    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)
        cam_vis = (hm_norm * 255).astype(np.uint8)
        cam_col = cv2.applyColorMap(cam_vis, cv2.COLORMAP_JET)
        cv2.imwrite(str(debug_dir / f"{stem}_cam.png"), cam_col)
        cv2.imwrite(str(debug_dir / f"{stem}_bright.png"),
                    (bright * 255).astype(np.uint8))

    # ── Connected components ──────────────────────────────────────────────────
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(
        bright.astype(np.uint8), connectivity=8)

    if n <= 1:
        logger.warning("[Hybrid] no bright connected components found")
        return None

    total     = H * W
    min_pct   = CLASS_MIN_PCT.get(pred_name, 0.5)
    max_pct   = CLASS_MAX_PCT.get(pred_name, 35.0)
    min_area  = max(10, int(total * min_pct / 100))
    max_area  = int(total * max_pct / 100)

    # Score each component
    best_label = -1
    best_score = -1.0

    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area or area > max_area:
            continue
        cx, cy   = centroids[i]
        dist     = float(np.sqrt((cx - peak_x) ** 2 + (cy - peak_y) ** 2)) + 1.0
        comp_msk = labels == i
        cam_mean = float(hm_norm[comp_msk].mean())
        # Favor blobs with high GradCAM activation, large area, close to peak
        score    = cam_mean * float(np.sqrt(area)) / dist
        logger.debug(f"  comp {i}: area={area} dist={dist:.0f} cam={cam_mean:.3f} "
                     f"score={score:.4f}")
        if score > best_score:
            best_score = score
            best_label = i

    # Fallback: if size filter removed everything, pick highest weighted-score centroid
    if best_label == -1:
        logger.debug("[Hybrid] size filter removed all blobs; using weighted fallback")
        # Create a soft weighted map and threshold it
        wt_map = weighted_score * brain_mask
        wt_thr = float(np.percentile(wt_map[wt_map > 0], 90)) if (wt_map > 0).any() else 0
        fallback = (wt_map >= wt_thr).astype(np.uint8)
        fallback = _morph_smooth(fallback, H, W, radius_frac=0.015)
        fallback = _fill_holes(fallback)
        fallback = _keep_largest_cc(fallback)
        region_pct = 100.0 * fallback.sum() / total
        logger.info(f"[Hybrid-fallback] region={region_pct:.1f}%  class={pred_name}")
        if min_pct <= region_pct <= max_pct:
            return fallback
        return None

    result     = (labels == best_label).astype(np.uint8)
    result     = _fill_holes(result)
    result     = _morph_smooth(result, H, W, radius_frac=0.015)
    result     = _fill_holes(result)

    region_pct = 100.0 * result.sum() / total
    logger.info(
        f"[Hybrid] region={region_pct:.1f}%  class={pred_name}  "
        f"peak=({peak_x},{peak_y})"
    )

    if not (min_pct <= region_pct <= max_pct):
        logger.debug(f"[Hybrid] region {region_pct:.1f}% outside "
                     f"[{min_pct},{max_pct}] -> rejected")
        return None

    return result


# ─────────────────────────────────────────────────────────────────────────────
# No-tumor gate
# ─────────────────────────────────────────────────────────────────────────────

def _no_tumor_gate(
    pred_name:  str,
    confidence: float,
    tumor_mask: Optional[np.ndarray],
) -> tuple[bool, str]:
    if pred_name == "notumor":
        return True, "pred=notumor"
    min_conf = CLASS_MIN_CONF.get(pred_name, 0.35)
    if confidence < min_conf:
        return True, f"conf={confidence:.2f} < min={min_conf} for {pred_name}"
    if tumor_mask is None:
        return True, "no bright tumor blob found"
    return False, ""


# ─────────────────────────────────────────────────────────────────────────────
# Smooth boundary drawing
# ─────────────────────────────────────────────────────────────────────────────

def _draw_smooth_boundary(
    img_bgr:   np.ndarray,
    mask:      np.ndarray,
    color:     tuple = CONTOUR_COLOR,
    thickness: int   = CONTOUR_THICKNESS,
) -> np.ndarray:
    """Fit an ellipse to the tumor mask and draw it on a copy of img_bgr."""
    result  = img_bgr.copy()
    mask_u8 = (mask * 255).astype(np.uint8)

    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        logger.warning("[Boundary] no contours found in mask")
        return result

    contour = max(contours, key=cv2.contourArea)

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
        logger.info("[Boundary] contour too small for ellipse -> raw contour drawn")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main segmentation function
# ─────────────────────────────────────────────────────────────────────────────

def segment_image(
    img_path:   str,
    clf:        nn.Module,
    output_dir: Optional[str] = None,
    save_mask:  bool          = False,
    debug:      bool          = False,
) -> dict:
    """
    Full boundary-extraction pipeline for one MRI image.

    Returns dict with keys:
        result_bgr, mask, predicted, confidence, has_boundary, region_pct
    """
    bgr = cv2.imread(img_path)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read: {img_path}")
    if bgr.ndim == 2:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
    H, W = bgr.shape[:2]
    rgb  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # ── Classify ──────────────────────────────────────────────────────────────
    device = next(clf.parameters()).device
    tensor = _preprocess_image(rgb).to(device)

    with torch.no_grad():
        logits = clf(tensor)
        probs  = torch.softmax(logits, dim=-1)[0]
        pred   = int(probs.argmax())

    pred_name  = CLASS_NAMES[pred]
    confidence = float(probs[pred])
    logger.info(
        f"[Seg] {Path(img_path).name:<30} pred={pred_name:<12} "
        f"conf={confidence:.2f}  "
        f"probs=[{' '.join(f'{p:.2f}' for p in probs.tolist())}]"
    )

    stem      = Path(img_path).stem
    debug_dir = Path(output_dir) / "debug" if (output_dir and debug) else None

    empty = np.zeros((H, W), dtype=np.uint8)

    # ── Early exit: notumor predicted ────────────────────────────────────────
    if pred_name == "notumor":
        logger.info("[Seg] -> NO BOUNDARY  reason: pred=notumor")
        _save(bgr, empty, img_path, output_dir, save_mask)
        return _make_result(bgr, empty, pred_name, confidence, False, 0.0)

    # ── GradCAM ───────────────────────────────────────────────────────────────
    gcam = GradCAM(clf)
    hm   = gcam.generate_heatmap(tensor, target_class=pred)
    gcam.remove_hooks()

    # ── Hybrid: find bright tumor blob guided by GradCAM ─────────────────────
    tumor_mask = _find_tumor_region(bgr, hm, H, W, pred_name,
                                    debug_dir=debug_dir, stem=stem)

    # ── No-tumor gate ─────────────────────────────────────────────────────────
    suppress, reason = _no_tumor_gate(pred_name, confidence, tumor_mask)

    if suppress:
        logger.info(f"[Seg] -> NO BOUNDARY  reason: {reason}")
        _save(bgr, empty, img_path, output_dir, save_mask)
        return _make_result(bgr, empty, pred_name, confidence, False, 0.0)

    # ── Draw smooth ellipse boundary ──────────────────────────────────────────
    result     = _draw_smooth_boundary(bgr, tumor_mask)
    region_pct = float(tumor_mask.sum() / (H * W) * 100)
    logger.info(f"[Seg] -> BOUNDARY  region={region_pct:.1f}%")

    _save(result, tumor_mask, img_path, output_dir, save_mask)
    return _make_result(result, tumor_mask, pred_name, confidence, True, region_pct)


def _make_result(bgr, mask, pred, conf, has_bnd, pct):
    return dict(result_bgr=bgr, mask=mask, predicted=pred,
                confidence=conf, has_boundary=has_bnd, region_pct=pct)


def _save(result_bgr, mask, img_path, output_dir, save_mask):
    if output_dir is None:
        return
    out  = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(img_path).stem
    cv2.imwrite(str(out / f"{stem}_boundary.png"), result_bgr)
    if save_mask:
        cv2.imwrite(str(out / f"{stem}_mask.png"),
                    (mask * 255).astype(np.uint8))


# ─────────────────────────────────────────────────────────────────────────────
# Comparison panel + validation runner
# ─────────────────────────────────────────────────────────────────────────────

def _build_panel(true_cls: str, img_path: str, result: dict, size: int = 320) -> np.ndarray:
    bgr = cv2.imread(img_path)
    if bgr is None:
        return np.zeros((size + 28, size * 2, 3), dtype=np.uint8)

    orig = cv2.resize(bgr,                  (size, size))
    bnd  = cv2.resize(result["result_bgr"], (size, size))
    row  = np.concatenate([orig, bnd], axis=1)

    lh  = 28
    lbl = np.full((lh, row.shape[1], 3), 25, dtype=np.uint8)

    if result["has_boundary"]:
        info = (f"pred={result['predicted']} conf={result['confidence']:.2f} "
                f"area={result['region_pct']:.1f}%")
    else:
        info = "NO BOUNDARY"
    text = f"True: {true_cls} | {info}"
    cv2.putText(lbl, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.50, (230, 230, 230), 1, cv2.LINE_AA)

    return np.concatenate([lbl, row], axis=0)


def run_validation(
    clf_path:   str  = DEFAULT_CLF_PATH,
    output_dir: str  = "outputs/validation/",
    debug:      bool = False,
) -> None:
    clf = _try_load_classifier(clf_path)
    clf.eval()

    panels = []
    print()
    print("=" * 72)
    print(f"{'CLEAN BOUNDARY -- USE-Me Validation':^72}")
    print("=" * 72)
    print()

    for true_cls, img_path in VALIDATION_IMAGES:
        if not Path(img_path).exists():
            print(f"  [SKIP] {Path(img_path).name} not found")
            continue

        res   = segment_image(img_path, clf,
                              output_dir=output_dir, save_mask=True, debug=debug)
        panel = _build_panel(true_cls, img_path, res)
        panels.append(panel)

        if not res["has_boundary"] and true_cls == "notumor":
            status = "NO BOUNDARY [correct]"
        elif not res["has_boundary"]:
            status = "NO BOUNDARY [missed]"
        else:
            status = f"BOUNDARY area={res['region_pct']:.1f}%"

        print(f"  {Path(img_path).name:<28} true={true_cls:<12} "
              f"pred={res['predicted']:<12} conf={res['confidence']:.2f}  "
              f"-> {status}")

    if panels:
        grid     = np.concatenate(panels, axis=0)
        out_path = Path(output_dir) / "validation_grid.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), grid)
        print(f"\nValidation grid saved -> {out_path}")

    print()
    print("Done.")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Hybrid intensity+GradCAM tumor boundary extraction")
    p.add_argument("--image",      default=None,
                   help="Single image path. Omit to run USE-Me validation set.")
    p.add_argument("--classifier", default=DEFAULT_CLF_PATH)
    p.add_argument("--output_dir", default="outputs/validation/")
    p.add_argument("--debug",      action="store_true",
                   help="Save intermediate GradCAM and bright-mask images")
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )
    args = _parse()

    if args.image:
        clf = _try_load_classifier(args.classifier)
        clf.eval()
        r = segment_image(args.image, clf,
                          output_dir=args.output_dir,
                          save_mask=True, debug=args.debug)
        print(f"\nImage     : {args.image}")
        print(f"Predicted : {r['predicted']}  (conf={r['confidence']:.2f})")
        bnd = (f"YES  area={r['region_pct']:.1f}%"
               if r["has_boundary"] else "NO (no-tumor gate)")
        print(f"Boundary  : {bnd}")
        saved = Path(args.output_dir) / (Path(args.image).stem + "_boundary.png")
        print(f"Saved     : {saved}")
    else:
        run_validation(clf_path=args.classifier,
                       output_dir=args.output_dir,
                       debug=args.debug)
