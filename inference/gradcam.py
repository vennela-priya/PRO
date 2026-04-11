"""
inference/gradcam.py
====================
Grad-CAM engine for NeuroScan AI (PyTorch).
Supports: EfficientNet-B3, ResNet50+CBAM, DenseNet121.
Demo mode: synthetic Gaussian heatmaps when no real model is loaded.

Usage
-----
    engine = EnsembleGradCAM(model_obj, model_mode)
    result = engine.generate(tensor, pred_result_dict, original_img_np)
    # result keys: cam, visuals, stats, elapsed, demo, model_cams
"""
from __future__ import annotations

import logging
import time
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]


# ---------------------------------------------------------------------------
# Target-layer auto-detection
# ---------------------------------------------------------------------------

def _get_target_layer(model: nn.Module, mname: str) -> nn.Module:
    """
    Return the last spatial conv layer for a given model architecture.
    Falls back to auto-scanning named modules when the expected attribute
    is absent (e.g., different timm version).
    """
    candidates = {
        "efficientnet": ["conv_head"],
        "resnet_cbam":  ["layer4"],
        "densenet":     ["features.denseblock4", "features.transition3"],
    }
    for attr_path in candidates.get(mname, []):
        try:
            obj = model
            for part in attr_path.split("."):
                obj = getattr(obj, part)
            if isinstance(obj, nn.Module):
                return obj
        except AttributeError:
            continue

    # Generic fallback: last Conv2d layer
    last_conv = None
    for _, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    if last_conv is not None:
        logger.warning(f"[GradCAM] {mname}: using generic last-conv fallback")
        return last_conv

    raise RuntimeError(f"[GradCAM] Cannot find target conv layer for '{mname}'")


# ---------------------------------------------------------------------------
# Hook context for capturing activations + gradients
# ---------------------------------------------------------------------------

class _Hooks:
    """Forward/backward hooks on a target layer."""

    def __init__(self, layer: nn.Module) -> None:
        self.activations: Optional[torch.Tensor] = None
        self.gradients:   Optional[torch.Tensor] = None
        self._fh = layer.register_forward_hook(self._fwd_hook)
        self._bh = layer.register_full_backward_hook(self._bwd_hook)

    def _fwd_hook(self, module, inp, out) -> None:
        self.activations = out  # [1, C, h, w]

    def _bwd_hook(self, module, grad_in, grad_out) -> None:
        self.gradients = grad_out[0]  # [1, C, h, w]

    def remove(self) -> None:
        self._fh.remove()
        self._bh.remove()


# ---------------------------------------------------------------------------
# Single-model Grad-CAM
# ---------------------------------------------------------------------------

def _single_gradcam(
    model:        nn.Module,
    target_layer: nn.Module,
    tensor:       torch.Tensor,   # [1, 3, H, W]
    pred_class:   int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run standard Grad-CAM for one model / one target class.

    Returns
    -------
    cam  : float32 [h, w] normalised to [0, 1] (spatial resolution of target layer)
    probs: float32 [NUM_CLASSES] softmax probabilities
    """
    hooks = _Hooks(target_layer)
    model.eval()
    model.zero_grad()

    try:
        with torch.enable_grad():
            output = model(tensor)                        # [1, NC]
            probs  = (F.softmax(output.float(), dim=1)
                      .squeeze(0).detach().cpu().numpy()) # [NC]
            score  = output[0, pred_class]
            score.backward()
    finally:
        hooks.remove()

    grads = hooks.gradients    # [1, C, h, w]
    acts  = hooks.activations  # [1, C, h, w]

    if grads is None or acts is None:
        logger.warning("[GradCAM] hooks captured nothing")
        return np.zeros((7, 7), dtype=np.float32), probs

    grads = grads.detach()
    acts  = acts.detach()

    # Global-average-pool gradients -> channel weights [1, C, 1, 1]
    weights = grads.mean(dim=[2, 3], keepdim=True)

    # Weighted sum of activation maps
    cam = (weights * acts).sum(dim=1, keepdim=True)  # [1, 1, h, w]
    cam = F.relu(cam).squeeze().detach().cpu().numpy()  # [h, w]

    # Normalise
    mn, mx = float(cam.min()), float(cam.max())
    if mx - mn > 1e-8:
        cam = (cam - mn) / (mx - mn)
    else:
        cam = np.zeros_like(cam)

    return cam.astype(np.float32), probs


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def make_visuals(
    orig_np:  np.ndarray,  # [H, W, 3] uint8 RGB
    cam_full: np.ndarray,  # [H, W] float32 [0,1]
) -> Dict[str, np.ndarray]:
    """
    Produce four visualisation images (all [H, W, 3] uint8 RGB):
      original  – input MRI unchanged
      raw_hm    – jet colourmap on black background
      overlay   – 55% original + 45% heatmap
      boundary  – original with green contours at CAM > 0.5
    Plus a composite (horizontal concat of all four).
    """
    H, W = orig_np.shape[:2]

    # Ensure uint8 RGB
    if orig_np.dtype != np.uint8:
        orig_u8 = np.clip(orig_np, 0, 255).astype(np.uint8)
    else:
        orig_u8 = orig_np.copy()
    if orig_u8.ndim == 2:
        orig_u8 = cv2.cvtColor(orig_u8, cv2.COLOR_GRAY2RGB)
    elif orig_u8.shape[2] == 4:
        orig_u8 = cv2.cvtColor(orig_u8, cv2.COLOR_RGBA2RGB)

    cam_r    = cv2.resize(cam_full, (W, H), interpolation=cv2.INTER_CUBIC)
    cam_r    = np.clip(cam_r, 0.0, 1.0)
    hm_u8    = (cam_r * 255).astype(np.uint8)
    hm_bgr   = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)
    raw_hm   = cv2.cvtColor(hm_bgr, cv2.COLOR_BGR2RGB)

    orig_bgr    = cv2.cvtColor(orig_u8, cv2.COLOR_RGB2BGR)
    overlay_bgr = cv2.addWeighted(orig_bgr, 0.55, hm_bgr, 0.45, 0)
    overlay     = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

    mask      = (cam_r > 0.50).astype(np.uint8) * 255
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boundary  = orig_u8.copy()
    cv2.drawContours(boundary, contours, -1, (57, 255, 20), 2)

    # Side-by-side panel (each tile 224x224)
    tile = 224
    def _t(im): return cv2.resize(im, (tile, tile))
    composite = np.hstack([_t(orig_u8), _t(raw_hm), _t(overlay), _t(boundary)])

    return {
        "original":    orig_u8,
        "raw_heatmap": raw_hm,
        "overlay":     overlay,
        "boundary":    boundary,
        "composite":   composite,   # [224, 224*4, 3]
    }


def region_stats(cam: np.ndarray, threshold: float = 0.50) -> dict:
    """
    Extract tumour-region statistics from a normalised CAM [H, W].

    Returns keys: area_pct, centroid, bbox, lateralization,
                  intensity_score.
    """
    H, W   = cam.shape
    mask   = cam > threshold
    tot_px = H * W
    tum_px = int(mask.sum())

    area_pct = round(tum_px / tot_px * 100, 1)

    ys, xs = np.where(mask)
    if len(xs) > 0:
        cx = int(xs.mean())
        cy = int(ys.mean())
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        intensity = int(cam[mask].mean() * 100)
    else:
        cx, cy = W // 2, H // 2
        x1, y1, x2, y2 = 0, 0, W, H
        intensity = 0

    mid = W / 2
    if cx < mid * 0.43:
        lateral = "LEFT hemisphere"
    elif cx > mid * 1.57:
        lateral = "RIGHT hemisphere"
    else:
        lateral = "CENTRAL"

    return {
        "area_pct":        area_pct,
        "centroid":        (cx, cy),
        "bbox":            (x1, y1, x2, y2),
        "lateralization":  lateral,
        "intensity_score": intensity,
    }


# ---------------------------------------------------------------------------
# Synthetic demo CAM
# ---------------------------------------------------------------------------

def _synthetic_cam(img_np: np.ndarray) -> np.ndarray:
    """
    Gaussian blob heatmap for demo mode (no real model).
    Shape: [14, 14] float32 [0,1].
    """
    seed = int(abs(float(img_np.mean()) * 1000)) % 99991
    rng  = np.random.RandomState(seed)
    h, w = 14, 14
    cam  = np.zeros((h, w), dtype=np.float32)

    n_blobs = rng.randint(1, 3)
    for _ in range(n_blobs):
        cx  = rng.uniform(0.20, 0.80) * w
        cy  = rng.uniform(0.20, 0.80) * h
        sx  = rng.uniform(1.5, 3.5)
        sy  = rng.uniform(1.5, 3.5)
        amp = rng.uniform(0.55, 1.0)
        Y, X = np.ogrid[:h, :w]
        blob = amp * np.exp(
            -((X - cx) ** 2 / (2 * sx ** 2) + (Y - cy) ** 2 / (2 * sy ** 2))
        )
        cam += blob

    cam += rng.normal(0, 0.04, cam.shape).astype(np.float32)
    cam  = np.clip(cam, 0.0, None)
    if cam.max() > 1e-8:
        cam /= cam.max()
    return cam.astype(np.float32)


# ---------------------------------------------------------------------------
# Main ensemble GradCAM class
# ---------------------------------------------------------------------------

class EnsembleGradCAM:
    """
    Compute Grad-CAM for every loaded model in the ensemble, then produce
    a confidence-weighted average heatmap.

    Parameters
    ----------
    model_obj  : Ensemble | MockEnsemble from app.py
    model_mode : "live" | "partial" | "demo"
    """

    def __init__(self, model_obj, model_mode: str) -> None:
        self.model_obj  = model_obj
        self.model_mode = model_mode
        self.is_demo    = (
            model_mode == "demo"
            or not hasattr(model_obj, "models")
        )

    # ------------------------------------------------------------------
    def generate(
        self,
        tensor:         torch.Tensor,   # [1, 3, 224, 224]
        pred_result:    dict,
        original_img:   np.ndarray,     # [H, W, 3] uint8 RGB
    ) -> dict:
        """
        Full GradCAM pipeline.  Never raises — falls back to demo CAM on error.

        Returns
        -------
        dict with keys:
          cam        : np.ndarray [H, W] float32 [0,1]
          model_cams : dict {mname: cam_small}
          visuals    : dict {original, raw_heatmap, overlay, boundary, composite}
          stats      : dict (area_pct, centroid, bbox, lateralization, intensity)
          elapsed    : float (seconds)
          demo       : bool
        """
        t0 = time.time()
        pred_idx = CLASS_NAMES.index(pred_result["class"])

        # ── per-model GradCAM ──────────────────────────────────────────
        model_cams: Dict[str, np.ndarray] = {}
        if self.is_demo:
            model_cams["demo"] = _synthetic_cam(original_img)
        else:
            for mname, model in self.model_obj.models.items():
                try:
                    layer = _get_target_layer(model, mname)
                    cam_s, _ = _single_gradcam(
                        model, layer, tensor.clone().detach(), pred_idx
                    )
                    model_cams[mname] = cam_s
                    logger.info(f"[GradCAM] {mname}: OK "
                                f"({cam_s.shape[0]}x{cam_s.shape[1]})")
                except Exception as exc:
                    logger.warning(f"[GradCAM] {mname} failed: {exc}")

        # Fallback if everything failed
        if not model_cams:
            logger.warning("[GradCAM] all models failed, using synthetic cam")
            model_cams["fallback"] = _synthetic_cam(original_img)

        # ── weighted ensemble merge ────────────────────────────────────
        ensemble_cam = self._weighted_merge(model_cams, pred_result)

        # ── resize to original image resolution ───────────────────────
        H, W = original_img.shape[:2]
        ensemble_cam = cv2.resize(
            ensemble_cam, (W, H), interpolation=cv2.INTER_CUBIC
        )
        ensemble_cam = np.clip(ensemble_cam, 0.0, 1.0)

        # Normalize once more after resize
        mn, mx = ensemble_cam.min(), ensemble_cam.max()
        if mx - mn > 1e-8:
            ensemble_cam = (ensemble_cam - mn) / (mx - mn)

        # ── visualisations & stats ────────────────────────────────────
        visuals = make_visuals(original_img, ensemble_cam)
        stats   = region_stats(ensemble_cam)

        return {
            "cam":        ensemble_cam,
            "model_cams": model_cams,
            "visuals":    visuals,
            "stats":      stats,
            "elapsed":    round(time.time() - t0, 2),
            "demo":       self.is_demo or "fallback" in model_cams,
        }

    # ------------------------------------------------------------------
    def _weighted_merge(
        self,
        model_cams: Dict[str, np.ndarray],
        pred_result: dict,
    ) -> np.ndarray:
        """
        Merge per-model CAMs using each model's predicted confidence as weight.
        """
        individual = pred_result.get("individual", {})
        pred_cls   = pred_result["class"]

        raw_w: Dict[str, float] = {}
        for name in model_cams:
            if name in individual:
                raw_w[name] = float(individual[name].get(pred_cls, 0.0)) + 0.05
            else:
                raw_w[name] = 1.0

        total = sum(raw_w.values()) or 1.0
        norm_w = {k: v / total for k, v in raw_w.items()}

        # Find common smallest size (for safe broadcasting)
        all_cams = list(model_cams.values())
        h_min = min(c.shape[0] for c in all_cams)
        w_min = min(c.shape[1] for c in all_cams)

        merged = np.zeros((h_min, w_min), dtype=np.float32)
        for name, cam in model_cams.items():
            cam_r   = cv2.resize(cam, (w_min, h_min),
                                 interpolation=cv2.INTER_CUBIC)
            merged += norm_w[name] * cam_r

        # Renormalise
        mx = merged.max()
        if mx > 1e-8:
            merged /= mx
        return merged.astype(np.float32)
