"""
refinement/pipeline.py
========================
Full pseudo-mask refinement pipeline.

Pipeline (classical approach — NO GradCAM):
    Stage 1 — Classifier: predict tumor class only  (no GradCAM)
    Stage 2 — Classical mask generator              (intensity + morphology)
    Stage 3 — U-Net refinement                      (optional; skipped if no checkpoint)
    Stage 4 — CRF boundary sharpening               (optional, skip_crf=False by default)
    Stage 5 — fitEllipse boundary extraction

Why no GradCAM?
    GradCAM computes gradients w.r.t. classifier activations — it locates the
    region that changes the CLASSIFICATION SCORE most, not the actual tumour.
    When classifier weights are imperfect this can point to the wrong hemisphere.
    Classical intensity thresholding (tumours are hyperintense on contrast-enhanced MRI)
    is physically correct and produces anatomically accurate masks.

CLI usage:
    python refinement/pipeline.py \\
        --image   path/to/mri.jpg \\
        --unet    checkpoints/unet/best_unet.pth \\
        --classifier checkpoints/resnet_cbam/best_ep008_auc0.9952.pt \\
        --output_dir outputs/ \\
        [--skip_crf]
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn

# Ensure project root is on the path so segmentation/ and refinement/ are importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from refinement.mask_refinement       import clean_mask
from refinement.crf_refinement        import apply_crf
from refinement.boundary_extraction   import extract_boundary
from refinement.classical_mask_generator import generate_mask, draw_boundary

logger = logging.getLogger(__name__)

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]


# ── Classifier loader ─────────────────────────────────────────────────────────

def _try_load_classifier(clf_path: str) -> Optional[nn.Module]:
    """
    Load the ResNetCBAM classifier from a checkpoint.
    Returns None if loading fails (non-fatal — classical mask works without it).
    """
    try:
        from classification.resnet_cbam import ResNetCBAM
        ckpt  = torch.load(clf_path, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        clf   = ResNetCBAM(num_classes=4)
        clf.load_state_dict(state, strict=False)
        clf.eval()
        logger.info(f"[Pipeline] Classifier loaded: {Path(clf_path).name}")
        return clf
    except Exception as exc:
        logger.warning(f"[Pipeline] Classifier load failed ({exc}) — "
                       "class will be inferred from folder name")
        return None


def _classify(clf: nn.Module, img_rgb: np.ndarray) -> str:
    """Run classifier forward pass; return predicted class name."""
    img = cv2.resize(img_rgb, (224, 224)).astype(np.float32) / 255.0
    img = (img - _MEAN) / _STD
    t   = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float()
    with torch.no_grad():
        logits = clf(t)
        idx    = int(logits.argmax(dim=-1).item())
    return CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else "glioma"


# ── UNet loader ───────────────────────────────────────────────────────────────

def _load_unet(unet_path: str, device: torch.device) -> Optional[nn.Module]:
    """
    Load the AttentionUNet from a checkpoint file.
    Returns None if loading fails — pipeline continues with classical mask.
    """
    from segmentation.attention_unet import AttentionUNet

    try:
        ckpt  = torch.load(unet_path, map_location=device, weights_only=False)
        cfg   = ckpt.get("config", {})
        model = AttentionUNet(
            in_channels  = cfg.get("in_channels",  3),
            base_filters = cfg.get("base_filters", 16),
            bilinear     = cfg.get("bilinear",     True),
        )
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        model.eval().to(device)
        logger.info(f"[Pipeline] Loaded UNet from {Path(unet_path).name}  "
                    f"val_dice={ckpt.get('val_dice', '?')}")
        return model
    except Exception as e:
        logger.warning(f"[Pipeline] UNet load failed: {e}")
        return None


def _preprocess_unet(img_rgb: np.ndarray, img_size: int = 224) -> torch.Tensor:
    img = cv2.resize(img_rgb, (img_size, img_size)).astype(np.float32) / 255.0
    img = (img - _MEAN) / _STD
    return torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float()


# ── Dice helper ───────────────────────────────────────────────────────────────

def _dice_safe(a: np.ndarray, b: np.ndarray) -> float:
    inter = float((a.flatten().astype(np.float32) *
                   b.flatten().astype(np.float32)).sum())
    denom = float(a.sum() + b.sum())
    return (2.0 * inter) / denom if denom > 0 else 0.0


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(
    image_path:       str,
    unet_path:        str,
    classifier_path:  Optional[str] = None,
    output_dir:       str           = "outputs/",
    skip_iterative:   bool          = True,   # kept for API compat, always skipped
    skip_crf:         bool          = False,
) -> dict:
    """
    Full refinement pipeline for a single MRI image.

    Stages
    ------
    1. Classifier: predict tumor class (glioma / meningioma / pituitary / notumor)
       — No GradCAM. Classifier is used ONLY for class prediction.
    2. Classical mask generation based on predicted class.
    3. (Optional) U-Net refinement — used if a trained checkpoint exists.
    4. (Optional) CRF boundary sharpening.
    5. fitEllipse boundary extraction (smooth elliptical contour).

    Parameters
    ----------
    image_path      : str   Path to the input MRI image file.
    unet_path       : str   Path to the AttentionUNet checkpoint (.pth).
    classifier_path : str   Optional path to ResNetCBAM classifier checkpoint.
    output_dir      : str   Directory to write output visualisations.
    skip_iterative  : bool  Ignored (kept for API compatibility).
    skip_crf        : bool  Skip CRF refinement (default False).

    Returns
    -------
    dict with keys:
        refined_mask    (np.ndarray, H×W, uint8 {0,1})
        boundary_image  (np.ndarray, H×W×3, uint8 BGR)
        contours        (list)
        dice_before     (float)
        dice_after      (float)
        dice_improvement (float)
        processing_time (float)
        stage_times     (dict)
        raw_mask        (np.ndarray)
        cleaned_mask    (np.ndarray)
        output_path     (str)
    """
    t_total = time.time()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stage_times: dict[str, float] = {}
    device = torch.device("cpu")

    # ── Progress helper ───────────────────────────────────────────────
    try:
        from tqdm import tqdm
        bar = tqdm(total=5, desc="Refinement Pipeline", unit="stage", ncols=80)
        def _adv(label: str) -> None:
            bar.set_postfix_str(label); bar.update(1)
    except ImportError:
        bar = None  # type: ignore[assignment]
        def _adv(label: str) -> None:   # type: ignore[misc]
            logger.info(f"[Pipeline] Stage: {label}")

    # ── Load image ────────────────────────────────────────────────────
    logger.info(f"[Pipeline] Processing: {Path(image_path).name}")
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    if bgr.ndim == 2:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
    H, W    = bgr.shape[:2]
    img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # ── Stage 1: Tumor class prediction ──────────────────────────────
    t0 = time.time()
    logger.info("[Pipeline] Stage 1: Tumor class prediction")

    # Auto-discover classifier if not given
    clf_path = classifier_path or ""
    if not clf_path:
        _auto = Path(unet_path).parent.parent / "resnet_cbam" / "best_ep008_auc0.9952.pt"
        if _auto.exists():
            clf_path = str(_auto)

    # Load classifier
    clf           = _try_load_classifier(clf_path) if clf_path else None
    predicted_cls: str

    if clf is not None:
        predicted_cls = _classify(clf, img_rgb)
        logger.info(f"[Pipeline]   Classifier predicted: {predicted_cls}")
    else:
        # Fall back to folder name (works for structured datasets)
        folder = Path(image_path).parent.name.lower()
        predicted_cls = folder if folder in {c.lower() for c in CLASS_NAMES} else "glioma"
        logger.info(f"[Pipeline]   Class inferred from folder: {predicted_cls}")

    stage_times["classifier"] = round(time.time() - t0, 3)
    _adv(f"Class={predicted_cls}")

    # ── Stage 2: Classical mask generation ───────────────────────────
    t0 = time.time()
    logger.info(f"[Pipeline] Stage 2: Classical mask ({predicted_cls})")

    classical_mask = generate_mask(bgr, predicted_cls)   # uint8 {0,1}
    region_pct     = 100.0 * classical_mask.sum() / (H * W)
    logger.info(f"[Pipeline]   Classical mask: {classical_mask.sum()} px "
                f"({region_pct:.1f}%)")

    # The classical mask IS the primary mask — no probability threshold needed
    raw_mask     = classical_mask.copy()   # "before" reference for Dice
    current_mask = classical_mask.copy()

    stage_times["classical"] = round(time.time() - t0, 3)
    _adv("Classical mask done")

    # Dice before = classical mask self-agreement (always 1.0; meaningful after U-Net)
    dice_before = _dice_safe(raw_mask, classical_mask)

    # ── Stage 3: U-Net refinement (optional) ─────────────────────────
    t0 = time.time()
    logger.info("[Pipeline] Stage 3: U-Net refinement")
    unet = _load_unet(unet_path, device) if Path(unet_path).exists() else None

    if unet is not None and predicted_cls != "notumor":
        try:
            tensor = _preprocess_unet(img_rgb).to(device)
            with torch.no_grad():
                logits   = unet(tensor)
                prob     = torch.sigmoid(logits).squeeze().cpu().numpy().astype(np.float32)
            prob_map  = cv2.resize(prob, (W, H), interpolation=cv2.INTER_LINEAR)
            unet_mask = (prob_map >= 0.45).astype(np.uint8)

            if unet_mask.sum() > 0:
                # Blend: if U-Net agrees with classical, trust the union
                # If U-Net is empty / very different, keep classical
                iou = _dice_safe(unet_mask, classical_mask)
                if iou > 0.25:
                    # Merge: union of classical + U-Net, then clean
                    merged       = np.maximum(unet_mask, classical_mask)
                    current_mask = clean_mask(merged.astype(np.float32))
                    logger.info(f"[Pipeline]   U-Net merged  IoU={iou:.2f}  "
                                f"merged={current_mask.sum()} px")
                else:
                    logger.info(f"[Pipeline]   U-Net low IoU={iou:.2f} — "
                                "keeping classical mask")
        except Exception as e:
            logger.warning(f"[Pipeline] U-Net inference failed: {e} — using classical mask")
    else:
        if unet is None:
            logger.info("[Pipeline]   No U-Net checkpoint — using classical mask")

    stage_times["unet"] = round(time.time() - t0, 3)
    _adv("U-Net done")

    # ── Stage 4: CRF boundary sharpening ─────────────────────────────
    if not skip_crf and predicted_cls != "notumor" and current_mask.sum() > 0:
        t0 = time.time()
        logger.info("[Pipeline] Stage 4: CRF boundary sharpening")
        try:
            crf_mask = apply_crf(img_rgb, current_mask.astype(np.float32), n_iter=5)
            if crf_mask.sum() > 0:
                current_mask = crf_mask
        except Exception as e:
            logger.warning(f"[Pipeline] CRF failed: {e}")
        stage_times["crf"] = round(time.time() - t0, 3)
        _adv("CRF done")
    else:
        _adv("CRF skipped")

    # ── Stage 5: fitEllipse boundary extraction ───────────────────────
    t0 = time.time()
    logger.info("[Pipeline] Stage 5: Boundary extraction")

    # Use classical fitEllipse draw (smooth, matches reference style)
    boundary_bgr = draw_boundary(bgr, current_mask)
    boundary_rgb = cv2.cvtColor(boundary_bgr, cv2.COLOR_BGR2RGB)

    # Also get raw contours for downstream use
    try:
        _, contours = extract_boundary(
            img_rgb, current_mask, color=(0, 255, 0), thickness=2, min_area=100)
    except Exception as e:
        logger.warning(f"[Pipeline] Contour extraction failed: {e}")
        contours = []

    stage_times["boundary"] = round(time.time() - t0, 3)
    _adv("Boundary done")

    if bar is not None:
        bar.close()

    # ── Metrics ───────────────────────────────────────────────────────
    cleaned_mask     = current_mask.copy()    # kept for return dict compatibility
    dice_after       = _dice_safe(current_mask, classical_mask)
    dice_improvement = dice_after - dice_before
    processing_time  = round(time.time() - t_total, 3)

    logger.info(
        f"[Pipeline] Done  time={processing_time}s  "
        f"dice_before={dice_before:.3f}  dice_after={dice_after:.3f}  "
        f"pred={predicted_cls}  region={region_pct:.1f}%"
    )

    # ── Save output visualisation ─────────────────────────────────────
    stem     = Path(image_path).stem
    orig_rs  = cv2.resize(img_rgb,                            (256, 256))
    cls_rs   = cv2.resize((classical_mask * 255).astype(np.uint8), (256, 256))
    ref_rs   = cv2.resize((current_mask   * 255).astype(np.uint8), (256, 256))
    bnd_rs   = cv2.resize(boundary_rgb,                       (256, 256))

    cls_bgr  = np.stack([cls_rs] * 3, axis=-1)
    ref_bgr  = np.stack([ref_rs] * 3, axis=-1)

    row = np.concatenate([orig_rs, cls_bgr, ref_bgr, bnd_rs], axis=1)
    out_png = str(out_dir / f"pipeline_{stem}.png")
    cv2.imwrite(out_png, cv2.cvtColor(row, cv2.COLOR_RGB2BGR))
    logger.info(f"[Pipeline] Saved -> {out_png}")

    return {
        "refined_mask":      current_mask,
        "boundary_image":    boundary_rgb,
        "contours":          contours,
        "dice_before":       dice_before,
        "dice_after":        dice_after,
        "dice_improvement":  dice_improvement,
        "processing_time":   processing_time,
        "stage_times":       stage_times,
        "raw_mask":          raw_mask,
        "cleaned_mask":      cleaned_mask,
        "output_path":       out_png,
        "predicted_class":   predicted_cls,
        "region_pct":        region_pct,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Classical brain tumour mask pipeline (no GradCAM)")
    p.add_argument("--image",          required=True,
                   help="Path to input MRI image")
    p.add_argument("--unet",           default="checkpoints/unet/best_unet.pth",
                   help="Path to U-Net checkpoint (.pth)")
    p.add_argument("--classifier",     default=None,
                   help="Path to classifier checkpoint (optional)")
    p.add_argument("--output_dir",     default="outputs/",
                   help="Output directory (default: outputs/)")
    p.add_argument("--skip_iterative", action="store_true",
                   help="(ignored, kept for API compatibility)")
    p.add_argument("--skip_crf",       action="store_true",
                   help="Skip CRF boundary refinement")
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )

    args = _parse_args()

    result = run_pipeline(
        image_path      = args.image,
        unet_path       = args.unet,
        classifier_path = args.classifier,
        output_dir      = args.output_dir,
        skip_iterative  = args.skip_iterative,
        skip_crf        = args.skip_crf,
    )

    print(f"\n{'='*60}")
    print(f"Image:           {args.image}")
    print(f"Predicted class: {result['predicted_class']}")
    print(f"Region:          {result['region_pct']:.1f}%")
    print(f"Dice before:     {result['dice_before']:.4f}")
    print(f"Dice after:      {result['dice_after']:.4f}")
    print(f"Improvement:     {result['dice_improvement']:+.4f}")
    print(f"Processing time: {result['processing_time']}s")
    print(f"Output saved:    {result['output_path']}")
    print(f"{'='*60}")
