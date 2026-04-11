"""
gradcam.py
==========
Grad-CAM++ implementation for all three backbone models.
Produces per-class activation maps with heatmap overlays.

Reference: Chattopadhay et al., WACV 2018.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GradCAM++ engine
# ---------------------------------------------------------------------------

class GradCAMPlusPlus:
    """
    Grad-CAM++ for an arbitrary nn.Module.

    Parameters
    ----------
    model        : the model (EfficientNet / ResNet-CBAM / ViT)
    target_layer : the layer to hook (e.g. model.target_layer)
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None
        self._hooks: List = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        def forward_hook(module, input, output):
            self._activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self._gradients = grad_output[0].detach()

        self._hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self._hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> Tuple[np.ndarray, int, float]:
        """
        Generate Grad-CAM++ heatmap.

        Parameters
        ----------
        input_tensor : [1, C, H, W] preprocessed image
        target_class : class index to explain (None → argmax)

        Returns
        -------
        cam      : H x W float32 array in [0, 1]
        pred_cls : predicted class index
        prob     : confidence of predicted class
        """
        self.model.eval()
        self.model.zero_grad()

        output = self.model(input_tensor)
        probs = F.softmax(output, dim=-1)

        if target_class is None:
            target_class = output.argmax(dim=-1).item()
        pred_prob = probs[0, target_class].item()

        # Backprop target class score
        score = output[0, target_class]
        score.backward()

        grads = self._gradients          # [1, C, H, W]
        acts = self._activations         # [1, C, H, W]

        if grads is None or acts is None:
            raise RuntimeError("Gradients or activations not captured. Check target_layer.")

        # Grad-CAM++ alpha computation
        grads_power_2 = grads ** 2
        grads_power_3 = grads ** 3
        sum_acts = acts.sum(dim=(2, 3), keepdim=True)

        eps = 1e-8
        alpha_num = grads_power_2
        alpha_denom = 2 * grads_power_2 + sum_acts * grads_power_3 + eps
        alpha = alpha_num / alpha_denom

        relu_grad = F.relu(score.exp() * grads)
        weights = (alpha * relu_grad).sum(dim=(2, 3), keepdim=True)

        cam = (weights * acts).sum(dim=1, keepdim=True)  # [1, 1, H, W]
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        # Normalise to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam, int(target_class), float(pred_prob)

    def __del__(self) -> None:
        self.remove_hooks()


# ---------------------------------------------------------------------------
# Overlay utility
# ---------------------------------------------------------------------------

def overlay_heatmap(
    original_image: np.ndarray,
    cam: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Blend Grad-CAM heatmap onto original image.

    Parameters
    ----------
    original_image : H x W x 3 uint8 RGB image
    cam            : H x W float32 heatmap in [0, 1]
    alpha          : overlay transparency

    Returns
    -------
    blended : H x W x 3 uint8 RGB image
    """
    h, w = original_image.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    heatmap = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), colormap)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Convert original to uint8 if needed
    if original_image.dtype != np.uint8:
        orig = (original_image * 255).clip(0, 255).astype(np.uint8)
    else:
        orig = original_image

    blended = cv2.addWeighted(orig, 1 - alpha, heatmap_rgb, alpha, 0)
    return blended


# ---------------------------------------------------------------------------
# Multi-model GradCAM runner
# ---------------------------------------------------------------------------

class MultiModelGradCAM:
    """
    Run Grad-CAM++ for each model in the ensemble and produce per-model overlays.

    Parameters
    ----------
    models       : dict {name: model}
    target_layers: dict {name: target_layer}  (optional; uses model.target_layer)
    output_dir   : directory to save heatmap PNGs
    """

    def __init__(
        self,
        models: Dict[str, nn.Module],
        target_layers: Optional[Dict[str, nn.Module]] = None,
        output_dir: str = "outputs/heatmaps",
    ) -> None:
        self.models = models
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.gcam_engines: Dict[str, GradCAMPlusPlus] = {}
        for name, model in models.items():
            layer = (target_layers or {}).get(name) or model.target_layer
            self.gcam_engines[name] = GradCAMPlusPlus(model, layer)

    def explain(
        self,
        input_tensor: torch.Tensor,
        original_image: np.ndarray,
        target_class: Optional[int] = None,
        sample_id: str = "sample",
        alpha: float = 0.5,
    ) -> Dict[str, Dict]:
        """
        Generate and save Grad-CAM++ heatmaps for all models.

        Parameters
        ----------
        input_tensor   : [1, C, H, W] preprocessed tensor
        original_image : H x W x 3 uint8 array
        target_class   : class to explain (None → argmax)
        sample_id      : filename prefix

        Returns
        -------
        dict per model: {cam, pred_class, prob, overlay_path}
        """
        results = {}
        for name, engine in self.gcam_engines.items():
            try:
                cam, pred_cls, prob = engine.generate(input_tensor.clone(), target_class)
                overlay = overlay_heatmap(original_image, cam, alpha=alpha)

                out_path = self.output_dir / f"{sample_id}_{name}_gradcam.png"
                import cv2 as _cv2
                _cv2.imwrite(str(out_path), _cv2.cvtColor(overlay, _cv2.COLOR_RGB2BGR))

                results[name] = {
                    "cam": cam,
                    "pred_class": pred_cls,
                    "prob": prob,
                    "overlay_path": str(out_path),
                }
                logger.info(f"[GradCAM++] {name}: class={pred_cls}, prob={prob:.3f} → {out_path}")

            except Exception as exc:
                logger.warning(f"[GradCAM++] Failed for {name}: {exc}")

        return results

    def cleanup(self) -> None:
        for engine in self.gcam_engines.values():
            engine.remove_hooks()
