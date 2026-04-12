"""
segmentation/evaluate_seg.py
============================
Evaluation script — computes Dice, IoU, Precision, Recall per image and
saves a side-by-side comparison grid (Green GT / Red Prediction).

Usage (from project root)
--------------------------
    python -m segmentation.evaluate_seg \\
        --image_dir  data/test/images \\
        --mask_dir   data/test/masks  \\
        --checkpoint checkpoints/unet/unet_best.pt \\
        --out_dir    results/eval \\
        --threshold  0.45 \\
        --erode      2

Outputs
-------
    results/eval/
        ├── metrics.csv          — per-image Dice, IoU, Prec, Rec
        ├── summary.txt          — mean / std of each metric
        └── overlays/
              ├── img_001.png    — side-by-side comparison
              └── ...
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# ── Metrics ───────────────────────────────────────────────────────────────

def compute_metrics(pred: np.ndarray, gt: np.ndarray, smooth: float = 1.0) -> dict:
    """
    Binary segmentation metrics.

    Parameters
    ----------
    pred, gt : (H, W) uint8 {0, 1}

    Returns
    -------
    dict — dice, iou, precision, recall, f1 (all in [0, 1])
    """
    pred = pred.astype(bool)
    gt   = gt.astype(bool)

    TP = float((pred & gt).sum())
    FP = float((pred & ~gt).sum())
    FN = float((~pred & gt).sum())
    TN = float((~pred & ~gt).sum())

    dice      = (2 * TP + smooth) / (2 * TP + FP + FN + smooth)
    iou       = (TP + smooth)     / (TP + FP + FN + smooth)
    precision = (TP + smooth)     / (TP + FP + smooth)
    recall    = (TP + smooth)     / (TP + FN + smooth)

    return {
        "dice":      round(dice,      4),
        "iou":       round(iou,       4),
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
    }


# ── Model loading ──────────────────────────────────────────────────────────

def _load_model(checkpoint_path: str, device: torch.device):
    """Load AttentionUNet or fallback UNet from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg  = ckpt.get("config", {})

    arch = cfg.get("architecture", "unet").lower()
    try:
        if "attention" in arch:
            from segmentation.attention_unet import AttentionUNet
            model = AttentionUNet(
                in_channels  = cfg.get("in_channels",  3),
                base_filters = cfg.get("base_filters", 32),
                bilinear     = cfg.get("bilinear",     True),
            )
        else:
            from segmentation.unet import UNet
            model = UNet(
                in_channels  = cfg.get("in_channels",  3),
                base_filters = cfg.get("base_filters", 32),
                bilinear     = cfg.get("bilinear",     True),
            )
    except Exception:
        from segmentation.unet import UNet
        model = UNet(
            in_channels  = cfg.get("in_channels",  3),
            base_filters = cfg.get("base_filters", 32),
            bilinear     = cfg.get("bilinear",     True),
        )

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval().to(device)
    img_size = cfg.get("image_size", 256)
    logger.info(f"Loaded checkpoint: {Path(checkpoint_path).name}  "
                f"arch={arch}  img_size={img_size}")
    return model, img_size, cfg.get("in_channels", 3)


def _preprocess(img_rgb: np.ndarray, img_size: int, in_channels: int,
                device: torch.device) -> torch.Tensor:
    img = cv2.resize(img_rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    img = (img - _MEAN) / _STD
    t   = torch.from_numpy(img.transpose(2, 0, 1))
    if in_channels == 1:
        t = t.mean(dim=0, keepdim=True)
    return t.unsqueeze(0).float().to(device)


# ── Visualisation ──────────────────────────────────────────────────────────

def _make_comparison(
    img_rgb:   np.ndarray,
    pred_mask: np.ndarray,  # {0, 1}
    gt_mask:   np.ndarray,  # {0, 1}
) -> np.ndarray:
    """
    Three-panel side-by-side:
      [Original]  [GT overlay (green)]  [Pred overlay (red)]
    """
    H, W = img_rgb.shape[:2]
    alpha = 0.40

    # Green GT overlay
    gt_layer  = np.zeros_like(img_rgb, dtype=np.float32)
    gt_layer[gt_mask == 1] = (50, 220, 50)
    gt_img = np.clip(
        img_rgb.astype(np.float32) * (1 - alpha) + gt_layer * alpha,
        0, 255).astype(np.uint8)
    cnts_gt, _ = cv2.findContours(gt_mask.astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(gt_img, cnts_gt, -1, (0, 200, 0), 2)

    # Red Prediction overlay
    pr_layer = np.zeros_like(img_rgb, dtype=np.float32)
    pr_layer[pred_mask == 1] = (220, 50, 50)
    pr_img = np.clip(
        img_rgb.astype(np.float32) * (1 - alpha) + pr_layer * alpha,
        0, 255).astype(np.uint8)
    cnts_pr, _ = cv2.findContours(pred_mask.astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(pr_img, cnts_pr, -1, (200, 0, 0), 2)

    # Combined (both contours on same image)
    combo = img_rgb.copy()
    cv2.drawContours(combo, cnts_gt, -1, (0, 200, 0), 2)
    cv2.drawContours(combo, cnts_pr, -1, (200, 0, 0), 2)

    # Stack horizontally: Original | GT | Pred | Combo
    return np.concatenate([img_rgb, gt_img, pr_img, combo], axis=1)


# ── Main evaluation loop ───────────────────────────────────────────────────

def evaluate(
    image_dir:  str,
    mask_dir:   str,
    checkpoint: str,
    out_dir:    str,
    threshold:  float = 0.45,
    erode_px:   int   = 2,
    device:     Optional[str] = None,
):
    from segmentation.postprocess import refine_mask

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"Device: {dev}")

    model, img_size, in_ch = _load_model(checkpoint, dev)

    img_dir  = Path(image_dir)
    msk_dir  = Path(mask_dir)
    out_path = Path(out_dir)
    ov_path  = out_path / "overlays"
    ov_path.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        p for p in img_dir.rglob("*") if p.suffix.lower() in _IMG_EXTS)
    logger.info(f"Found {len(image_paths)} images in {img_dir}")

    all_metrics = []
    t0_total = time.time()

    for img_path in image_paths:
        stem = img_path.stem

        # Load image
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            logger.warning(f"Cannot read: {img_path}")
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        H, W    = img_rgb.shape[:2]

        # Load GT mask
        gt_mask = None
        for ext in [".png", ".jpg", ".jpeg", ".bmp"]:
            mp = msk_dir / (stem + ext)
            if mp.exists():
                m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
                if m is not None:
                    gt_mask = (m > 0).astype(np.uint8)
                    break
        if gt_mask is None:
            logger.warning(f"No mask for: {stem}")
            continue

        # Inference
        tensor = _preprocess(img_rgb, img_size, in_ch, dev)
        with torch.no_grad():
            logits   = model(tensor)
            prob_map = torch.sigmoid(logits).squeeze().cpu().numpy()
        prob_map = cv2.resize(prob_map.astype(np.float32), (W, H),
                              interpolation=cv2.INTER_LINEAR)
        gt_mask  = cv2.resize(gt_mask, (W, H), interpolation=cv2.INTER_NEAREST)

        # Post-process
        pred_mask = refine_mask(prob_map, threshold=threshold, erode_px=erode_px)

        # Metrics
        m = compute_metrics(pred_mask, gt_mask)
        m["filename"] = img_path.name
        all_metrics.append(m)

        # Overlay
        cmp_img = _make_comparison(img_rgb, pred_mask, gt_mask)
        cv2.imwrite(str(ov_path / (stem + ".png")),
                    cv2.cvtColor(cmp_img, cv2.COLOR_RGB2BGR))

        logger.info(f"  {stem:30s}  dice={m['dice']:.4f}  iou={m['iou']:.4f}  "
                    f"prec={m['precision']:.4f}  rec={m['recall']:.4f}")

    elapsed = time.time() - t0_total
    logger.info(f"\nEvaluated {len(all_metrics)} images in {elapsed:.1f}s")

    if not all_metrics:
        logger.error("No valid images evaluated.")
        return

    # ── Save CSV ───────────────────────────────────────────────────────
    csv_path = out_path / "metrics.csv"
    fieldnames = ["filename", "dice", "iou", "precision", "recall"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_metrics)
    logger.info(f"Metrics saved → {csv_path}")

    # ── Summary ────────────────────────────────────────────────────────
    keys = ["dice", "iou", "precision", "recall"]
    summary_lines = ["=" * 50, "EVALUATION SUMMARY", "=" * 50]
    for k in keys:
        vals = [m[k] for m in all_metrics]
        summary_lines.append(
            f"  {k:12s}  mean={np.mean(vals):.4f}  std={np.std(vals):.4f}  "
            f"min={np.min(vals):.4f}  max={np.max(vals):.4f}"
        )
    summary_lines.append(f"\n  N images : {len(all_metrics)}")
    summary_lines.append(f"  Threshold: {threshold}")
    summary_lines.append(f"  Erode px : {erode_px}")
    summary_lines.append("=" * 50)
    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)

    summary_path = out_path / "summary.txt"
    with open(summary_path, "w") as f:
        f.write(summary_text)
    logger.info(f"Summary saved → {summary_path}")
    logger.info(f"Overlays saved → {ov_path}")


# ── CLI ───────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate segmentation model and generate comparison overlays.")
    p.add_argument("--image_dir",  required=True, help="Directory of test images")
    p.add_argument("--mask_dir",   required=True, help="Directory of GT binary masks")
    p.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    p.add_argument("--out_dir",    default="results/eval", help="Output directory")
    p.add_argument("--threshold",  type=float, default=0.45,
                   help="Probability threshold (default 0.45)")
    p.add_argument("--erode",      type=int,   default=2,
                   help="Boundary erosion pixels (default 2, 0=off)")
    p.add_argument("--device",     default=None,
                   help="cuda or cpu (auto-detect if omitted)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    evaluate(
        image_dir  = args.image_dir,
        mask_dir   = args.mask_dir,
        checkpoint = args.checkpoint,
        out_dir    = args.out_dir,
        threshold  = args.threshold,
        erode_px   = args.erode,
        device     = args.device,
    )
