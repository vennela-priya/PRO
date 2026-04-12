"""
inference/segmenter.py
======================
Production inference wrapper for the U-Net tumour segmentation model.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)

_CKPT_SEARCH = [
    "./checkpoints/unet",
    "./BrainTumorAI/checkpoints/unet",
    "./models/unet",
    "../checkpoints/unet",
    "./checkpoints",
]

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ── Helpers ───────────────────────────────────────────────────────────────

def _find_checkpoint() -> Optional[str]:
    candidates: list[Path] = []
    for base in _CKPT_SEARCH:
        p = Path(base)
        if p.exists():
            candidates.extend(p.glob("*.pt"))
            candidates.extend(p.glob("*.pth"))
    # Deduplicate (same file via different patterns)
    seen: set[Path] = set()
    unique: list[Path] = []
    for c in candidates:
        r = c.resolve()
        if r not in seen:
            seen.add(r)
            unique.append(c)
    if not unique:
        return None
    # Prefer files with "best" in the name, then pick newest by mtime
    best = [c for c in unique if "best" in c.name.lower()]
    pool = best or unique
    return str(max(pool, key=lambda f: f.stat().st_mtime))


def _load_unet(checkpoint_path: Optional[str], device: torch.device) -> tuple:
    path = checkpoint_path or _find_checkpoint()
    if path is None or not Path(path).exists():
        logger.info("[Segmenter] No checkpoint found — demo mode")
        return None, {}
    try:
        ckpt  = torch.load(path, map_location=device, weights_only=False)
        cfg   = ckpt.get("config", {})
        arch  = cfg.get("architecture", "unet").lower()

        # Auto-select AttentionUNet or vanilla UNet based on saved config
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

        model.load_state_dict(ckpt["model_state_dict"])
        model.eval().to(device)
        dice = ckpt.get("val_dice", "?")
        logger.info(f"[Segmenter] Loaded {Path(path).name}  arch={arch}  val_dice={dice:.4f}"
                    if isinstance(dice, float) else
                    f"[Segmenter] Loaded {Path(path).name}  arch={arch}")
        return model, cfg
    except Exception as e:
        logger.warning(f"[Segmenter] Load failed: {e} — demo mode")
        return None, {}


def _postprocess(prob_map: np.ndarray, threshold: float = 0.5,
                 min_area_pct: float = 0.3, erode_px: int = 2) -> np.ndarray:
    """
    Convert probability map to clean binary mask.

    Steps:
      1. Threshold
      2. Morphological open  (remove salt noise, radius ~1 % of image)
      3. Morphological close (fill holes,        radius ~2 % of image)
      4. Remove components smaller than min_area_pct % of image pixels
      5. Erode by erode_px pixels to tighten boundary (reduce over-segmentation)
    """
    H, W  = prob_map.shape
    binary = (prob_map >= threshold).astype(np.uint8)

    r_open  = max(3, int(min(H, W) * 0.012))
    r_close = max(5, int(min(H, W) * 0.025))

    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r_open*2+1,  r_open*2+1))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r_close*2+1, r_close*2+1))

    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  k_open)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_close)

    # Remove tiny components
    min_px  = max(1, int(H * W * min_area_pct / 100))
    n_comp, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    clean = np.zeros_like(binary)
    for lbl in range(1, n_comp):
        if stats[lbl, cv2.CC_STAT_AREA] >= min_px:
            clean[labels == lbl] = 1

    # Boundary erosion — tighten contour to reduce over-segmentation
    if erode_px > 0:
        k_erode = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (erode_px * 2 + 1, erode_px * 2 + 1))
        clean = cv2.erode(clean, k_erode, iterations=1)

    return clean


def _synthetic_mask(img_np: np.ndarray) -> np.ndarray:
    """Seeded Gaussian-blob pseudo-mask for demo mode."""
    h, w  = img_np.shape[:2]
    seed  = int(img_np.mean() * 137 + img_np.std() * 31) % 9999
    rng   = np.random.RandomState(seed)
    mask  = np.zeros((h, w), dtype=np.float32)
    for _ in range(rng.randint(1, 3)):
        cx  = rng.randint(w // 4,     3 * w // 4)
        cy  = rng.randint(h // 5,     3 * h // 4)
        rx  = rng.randint(w // 8,     w // 3)
        ry  = rng.randint(h // 8,     h // 3)
        Y, X = np.ogrid[:h, :w]
        blob = np.clip(1.0 - ((X-cx)/rx)**2 - ((Y-cy)/ry)**2, 0, 1)
        mask = np.maximum(mask, blob)
    if mask.max() > mask.min():
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
    return mask


# ── TumorSegmenter ────────────────────────────────────────────────────────

class TumorSegmenter:
    """
    Wraps U-Net inference for use in the NeuroScan AI Gradio app.

    Parameters
    ----------
    checkpoint_path : explicit .pt path. Pass None to auto-discover.
    device          : torch.device (defaults to CUDA if available)
    threshold       : probability threshold for binary mask (default 0.45)
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device:          Optional[torch.device] = None,
        threshold:       float = 0.45,   # probability cut-off
        erode_px:        int   = 2,      # boundary erosion pixels (0 = off)
    ):
        self.device    = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = threshold
        self.erode_px  = erode_px
        self.model, self.config = _load_unet(checkpoint_path, self.device)
        self.demo      = self.model is None
        self._img_size = self.config.get("image_size", 256)
        self._in_ch    = self.config.get("in_channels", 3)

    # ── Pre-processing ────────────────────────────────────────────────

    def _preprocess(self, img_np: np.ndarray) -> torch.Tensor:
        """uint8 RGB (H,W,3) → normalised tensor (1,C,S,S)"""
        s   = self._img_size
        img = cv2.resize(img_np, (s, s), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        img = (img - _MEAN) / _STD
        t   = torch.from_numpy(img.transpose(2, 0, 1))     # (3, S, S)
        if self._in_ch == 1:
            t = t.mean(dim=0, keepdim=True)                # (1, S, S)
        return t.unsqueeze(0).float().to(self.device)      # (1, C, S, S)

    # ── Visualisation helpers ─────────────────────────────────────────

    @staticmethod
    def _make_overlay(img_np: np.ndarray, binary_mask: np.ndarray,
                      color: tuple = (220, 50, 50), alpha: float = 0.38) -> np.ndarray:
        overlay     = img_np.copy()
        color_layer = np.zeros_like(img_np)
        color_layer[binary_mask == 1] = color
        return cv2.addWeighted(overlay, 1.0 - alpha, color_layer, alpha, 0)

    @staticmethod
    def _make_contour(img_np: np.ndarray, binary_mask: np.ndarray,
                      color: tuple = (0, 230, 80), thickness: int = 2) -> np.ndarray:
        out  = img_np.copy()
        cnts, _ = cv2.findContours(
            binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, color, thickness)
        return out

    # ── Public API ────────────────────────────────────────────────────

    def segment(
        self,
        img_np:      np.ndarray,
        threshold:   Optional[float] = None,
        cam_map:     Optional[np.ndarray] = None,
        cam_thr:     float = 0.35,
        cam_min_pct: float = 0.5,
    ) -> dict:
        """
        Segment a single MRI image.

        Parameters
        ----------
        img_np      : uint8 RGB numpy array (H, W, 3)
        threshold   : override instance default probability threshold
        cam_map     : optional Grad-CAM heatmap (H×W float [0,1]).
                      When the U-Net mask is empty (< cam_min_pct % area)
                      AND cam_map is supplied, the CAM is thresholded and
                      used as a fallback segmentation mask.  This avoids
                      showing blank results when the U-Net model has not
                      been trained on reliable pseudo-masks.
        cam_thr     : CAM binarisation threshold for fallback
        cam_min_pct : if U-Net area_pct < this value, activate CAM fallback

        Returns
        -------
        dict with keys: prob_map, binary_mask, overlay, contour,
                        area_pct, elapsed, demo, dice_vs_cam, iou_vs_cam,
                        cam_mask, cam_fallback_used
        """
        thr  = threshold if threshold is not None else self.threshold
        t0   = time.time()
        H, W = img_np.shape[:2]

        cam_fallback_used = False

        if self.demo:
            prob_map = _synthetic_mask(img_np)
            prob_map = cv2.resize(prob_map, (W, H), interpolation=cv2.INTER_LINEAR)
            binary   = _postprocess(prob_map, thr, erode_px=self.erode_px)
        else:
            tensor = self._preprocess(img_np)
            with torch.no_grad():
                logits   = self.model(tensor)
                prob_map = (torch.sigmoid(logits)
                            .squeeze().cpu().numpy().astype(np.float32))
            prob_map = cv2.resize(prob_map, (W, H), interpolation=cv2.INTER_LINEAR)
            binary   = _postprocess(prob_map, thr, erode_px=self.erode_px)

            # ── CAM fallback ───────────────────────────────────────────────
            # If the trained U-Net gives an empty mask (collapsed model) and
            # a Grad-CAM heatmap is available, use the CAM as a proxy mask.
            unet_area_pct = float(binary.mean() * 100)
            if cam_map is not None and unet_area_pct < cam_min_pct:
                cam_r  = cv2.resize(
                    cam_map.astype(np.float32), (W, H),
                    interpolation=cv2.INTER_LINEAR)
                # Normalise to [0,1]
                mn, mx = cam_r.min(), cam_r.max()
                if mx > mn:
                    cam_r = (cam_r - mn) / (mx - mn)
                cam_bin = _postprocess(
                    cam_r, cam_thr,
                    min_area_pct=0.1,
                    erode_px=self.erode_px)
                if cam_bin.sum() > 0:
                    binary            = cam_bin
                    prob_map          = cam_r
                    cam_fallback_used = True
                    logger.debug("[Segmenter] U-Net empty — using CAM fallback")

        return {
            "prob_map":         prob_map,
            "binary_mask":      binary,
            "overlay":          self._make_overlay(img_np, binary),
            "contour":          self._make_contour(img_np, binary),
            "area_pct":         float(binary.mean() * 100),
            "elapsed":          round(time.time() - t0, 3),
            "demo":             self.demo,
            "cam_fallback":     cam_fallback_used,
            "dice_vs_cam":      None,
            "iou_vs_cam":       None,
            "cam_mask":         None,
        }

    def compare_with_gradcam(self, seg_result: dict, cam: np.ndarray,
                             cam_threshold: float = 0.5) -> dict:
        """Compute Dice + IoU agreement between U-Net mask and Grad-CAM."""
        seg  = seg_result["binary_mask"]
        H, W = seg.shape
        cam_r    = cv2.resize(cam.astype(np.float32), (W, H),
                              interpolation=cv2.INTER_LINEAR)
        cam_mask = (cam_r >= cam_threshold).astype(np.uint8)
        smooth   = 1.0
        inter    = float((seg & cam_mask).sum())
        dice     = (2.0*inter + smooth) / (float(seg.sum()) + float(cam_mask.sum()) + smooth)
        iou      = (inter + smooth) / (float((seg | cam_mask).sum()) + smooth)
        seg_result["dice_vs_cam"] = round(dice, 4)
        seg_result["iou_vs_cam"]  = round(iou,  4)
        seg_result["cam_mask"]    = cam_mask
        return seg_result
