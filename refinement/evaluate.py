"""
refinement/evaluate.py
========================
Batch evaluation of the refinement pipeline.

For each image in the provided list:
  1. Runs the full refinement pipeline.
  2. Computes Dice, IoU, and boundary smoothness before and after refinement.
  3. Saves a 4-panel comparison image to outputs/comparisons/.
  4. Prints a summary table and writes it to outputs/evaluation_report.txt.

Boundary smoothness is measured as the ratio of contour perimeter to mask area
(lower = smoother; a perfect circle has ratio ~ 3.54 / sqrt(area)).
The normalised form used here is  perimeter / sqrt(area) so it is scale-invariant.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))


# ── Metric helpers ────────────────────────────────────────────────────────────

def _dice(pred: np.ndarray, ref: np.ndarray, smooth: float = 1.0) -> float:
    """
    Compute Dice coefficient between two binary numpy arrays.

    Parameters
    ----------
    pred, ref : np.ndarray  Binary arrays (any shape, will be flattened).
    smooth    : float       Smoothing constant (default 1.0).

    Returns
    -------
    float
        Dice coefficient in [0, 1].
    """
    p = pred.flatten().astype(np.float32)
    r = ref.flatten().astype(np.float32)
    inter = (p * r).sum()
    return float((2.0 * inter + smooth) / (p.sum() + r.sum() + smooth))


def _iou(pred: np.ndarray, ref: np.ndarray, smooth: float = 1.0) -> float:
    """
    Compute Intersection-over-Union between two binary numpy arrays.

    Parameters
    ----------
    pred, ref : np.ndarray  Binary arrays.
    smooth    : float       Smoothing constant.

    Returns
    -------
    float
        IoU in [0, 1].
    """
    p = pred.flatten().astype(np.float32)
    r = ref.flatten().astype(np.float32)
    inter = (p * r).sum()
    union = p.sum() + r.sum() - inter
    return float((inter + smooth) / (union + smooth))


def _boundary_smoothness(mask: np.ndarray) -> float:
    """
    Compute boundary smoothness as normalised perimeter / sqrt(area).

    A lower value indicates a smoother boundary. A perfect circle gives
    approximately 2*pi / sqrt(pi) ≈ 3.54.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask, shape (H, W), values {0, 1}.

    Returns
    -------
    float
        Smoothness score (lower = smoother). Returns 0.0 for empty masks.
    """
    m = mask.astype(np.uint8)
    area = float(m.sum())
    if area < 1.0:
        return 0.0

    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0

    perimeter = sum(cv2.arcLength(c, closed=True) for c in contours)
    return float(perimeter / (np.sqrt(area) + 1e-8))


# ── 4-panel comparison ────────────────────────────────────────────────────────

def _save_comparison(
    img_rgb:      np.ndarray,
    raw_mask:     np.ndarray,
    refined_mask: np.ndarray,
    boundary_img: np.ndarray,
    save_path:    str,
    panel_size:   int = 256,
) -> None:
    """
    Save a 4-panel comparison image: original | raw mask | refined mask | boundary.

    Parameters
    ----------
    img_rgb      : np.ndarray  Original RGB image.
    raw_mask     : np.ndarray  Raw (un-refined) binary mask, {0, 1}.
    refined_mask : np.ndarray  Refined binary mask, {0, 1}.
    boundary_img : np.ndarray  Boundary-annotated RGB image.
    save_path    : str         Output file path.
    panel_size   : int         Each panel's side length (default 256 px).
    """
    s = panel_size

    def _resize_rgb(arr: np.ndarray) -> np.ndarray:
        return cv2.resize(arr, (s, s))

    def _mask_to_rgb(m: np.ndarray) -> np.ndarray:
        m8 = cv2.resize((m * 255).astype(np.uint8), (s, s))
        return np.stack([m8] * 3, axis=-1)

    orig   = _resize_rgb(img_rgb)
    raw_v  = _mask_to_rgb(raw_mask)
    ref_v  = _mask_to_rgb(refined_mask)
    bnd_v  = _resize_rgb(boundary_img)

    row = np.concatenate([orig, raw_v, ref_v, bnd_v], axis=1)

    # Add label strip
    lh, lw = 24, row.shape[1]
    label  = np.ones((lh, lw, 3), dtype=np.uint8) * 50
    labels = ["Original", "Raw Mask", "Refined Mask", "Boundary"]
    for i, txt in enumerate(labels):
        cv2.putText(
            label, txt,
            (i * s + 5, 17),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1,
        )
    final = np.concatenate([label, row], axis=0)

    # Convert RGB -> BGR for imwrite
    cv2.imwrite(save_path, cv2.cvtColor(final, cv2.COLOR_RGB2BGR))


# ── Main evaluation function ──────────────────────────────────────────────────

def evaluate_pipeline(
    image_paths: list[str],
    unet_path:   str,
    output_dir:  str = "outputs/",
    max_images:  int = 20,
) -> None:
    """
    Run pipeline on up to max_images, compute metrics, save comparisons and report.

    For each image the function computes Dice, IoU, and boundary smoothness
    before and after refinement, then saves a 4-panel comparison image.
    At the end, a summary table is printed and saved to evaluation_report.txt.

    Parameters
    ----------
    image_paths : list[str]   Paths to input MRI images.
    unet_path   : str         Path to the AttentionUNet checkpoint.
    output_dir  : str         Root output directory (default "outputs/").
    max_images  : int         Maximum number of images to evaluate (default 20).
    """
    from refinement.pipeline import run_pipeline

    out_dir  = Path(output_dir)
    comp_dir = out_dir / "comparisons"
    comp_dir.mkdir(parents=True, exist_ok=True)

    image_paths = list(image_paths)[:max_images]
    if not image_paths:
        logger.warning("[Evaluate] No images to evaluate.")
        return

    records: list[dict] = []

    for img_path in image_paths:
        try:
            logger.info(f"[Evaluate] Processing: {Path(img_path).name}")
            result = run_pipeline(
                image_path     = img_path,
                unet_path      = unet_path,
                classifier_path = None,
                output_dir     = str(out_dir),
                skip_iterative = True,
                skip_crf       = False,
            )

            raw_mask     = result["raw_mask"]
            refined_mask = result["refined_mask"]
            boundary_img = result["boundary_image"]

            # Load original image for saving comparison
            bgr = cv2.imread(img_path)
            if bgr is None:
                continue
            if bgr.ndim == 2:
                bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
            img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            # Metrics before (raw mask vs itself as reference — smoothness comparison)
            dice_before   = _dice(raw_mask, raw_mask)      # always 1.0 vs itself
            iou_before    = _iou(raw_mask,  raw_mask)
            smooth_before = _boundary_smoothness(raw_mask)

            # Metrics after (refined vs raw as approximate reference)
            dice_after    = _dice(refined_mask, raw_mask)
            iou_after     = _iou(refined_mask,  raw_mask)
            smooth_after  = _boundary_smoothness(refined_mask)

            rec = {
                "name":           Path(img_path).name,
                "dice_before":    dice_before,
                "dice_after":     dice_after,
                "dice_delta":     dice_after  - dice_before,
                "iou_before":     iou_before,
                "iou_after":      iou_after,
                "iou_delta":      iou_after   - iou_before,
                "smooth_before":  smooth_before,
                "smooth_after":   smooth_after,
                "smooth_delta":   smooth_after - smooth_before,
                "time":           result["processing_time"],
            }
            records.append(rec)

            # Save comparison image
            comp_path = str(comp_dir / f"cmp_{Path(img_path).stem}.png")
            _save_comparison(img_rgb, raw_mask, refined_mask, boundary_img, comp_path)

        except Exception as e:
            logger.error(f"[Evaluate] Failed for {img_path}: {e}")

    if not records:
        logger.warning("[Evaluate] No results to report.")
        return

    # ── Aggregate stats ────────────────────────────────────────────────
    def _mean(key: str) -> float:
        return float(np.mean([r[key] for r in records]))

    avg = {k: _mean(k) for k in
           ["dice_before", "dice_after", "dice_delta",
            "iou_before",  "iou_after",  "iou_delta",
            "smooth_before", "smooth_after", "smooth_delta"]}

    # ── Print table ────────────────────────────────────────────────────
    header = (
        f"\n{'='*55}\n"
        f"{'Metric':<22} | {'Before':>7} | {'After':>7} | {'Delta':>7}\n"
        f"{'-'*55}"
    )
    rows = [
        ("Dice score",       avg["dice_before"],   avg["dice_after"],   avg["dice_delta"]),
        ("IoU",              avg["iou_before"],    avg["iou_after"],    avg["iou_delta"]),
        ("Boundary smooth",  avg["smooth_before"], avg["smooth_after"], avg["smooth_delta"]),
    ]
    table_lines = [header]
    for name, b, a, d in rows:
        table_lines.append(f"{name:<22} | {b:>7.4f} | {a:>7.4f} | {d:>+7.4f}")
    table_lines.append("=" * 55)
    table_lines.append(f"Images evaluated: {len(records)} / {len(image_paths)}")
    table_str = "\n".join(table_lines)
    print(table_str)

    # ── Save report ────────────────────────────────────────────────────
    report_path = str(out_dir / "evaluation_report.txt")
    with open(report_path, "w") as f:
        f.write("Brain Tumour Pseudo-Mask Refinement — Evaluation Report\n")
        f.write("=" * 55 + "\n\n")
        f.write(table_str + "\n\n")
        f.write("Per-Image Results:\n")
        f.write(f"{'Image':<40} {'Dice_B':>7} {'Dice_A':>7} {'IoU_B':>7} {'IoU_A':>7} {'Time':>6}\n")
        f.write("-" * 80 + "\n")
        for r in records:
            f.write(
                f"{r['name']:<40} {r['dice_before']:>7.4f} {r['dice_after']:>7.4f} "
                f"{r['iou_before']:>7.4f} {r['iou_after']:>7.4f} {r['time']:>6.2f}s\n"
            )

    print(f"\nReport saved -> {report_path}")
    print(f"Comparisons  -> {comp_dir}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import glob

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )

    p = argparse.ArgumentParser(description="Batch evaluation of refinement pipeline")
    p.add_argument("--unet",       required=True,  help="Path to UNet checkpoint")
    p.add_argument("--images_dir", required=False, default=None,
                   help="Root folder of test images (recursive .jpg/.png search)")
    p.add_argument("--output_dir", default="outputs/")
    p.add_argument("--max_images", type=int, default=20)
    args = p.parse_args()

    if args.images_dir:
        patterns  = [
            str(Path(args.images_dir) / "**" / "*.jpg"),
            str(Path(args.images_dir) / "**" / "*.png"),
        ]
        img_paths: list[str] = []
        for pat in patterns:
            img_paths.extend(glob.glob(pat, recursive=True))
    else:
        # Default: scan Project/Testing
        patterns = [
            str(_PROJECT_ROOT / "Project" / "Testing" / "**" / "*.jpg"),
            str(_PROJECT_ROOT / "data"    / "**"             / "*.jpg"),
        ]
        img_paths = []
        for pat in patterns:
            img_paths.extend(glob.glob(pat, recursive=True))

    evaluate_pipeline(
        image_paths = img_paths[:args.max_images],
        unet_path   = args.unet,
        output_dir  = args.output_dir,
        max_images  = args.max_images,
    )
