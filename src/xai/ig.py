"""
ig.py
=====
Integrated Gradients via Captum for fine-grained pixel attribution.
Reference: Sundararajan et al., 2017 — "Axiomatic Attribution for Deep Networks".
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class IntegratedGradientsExplainer:
    """
    Integrated Gradients explainer using Captum.

    Parameters
    ----------
    model      : trained nn.Module (must support autograd)
    device     : compute device
    n_steps    : interpolation steps (50 recommended)
    output_dir : directory to save overlay PNGs
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        n_steps: int = 50,
        output_dir: str = "outputs/ig",
    ) -> None:
        try:
            from captum.attr import IntegratedGradients
            self._IG = IntegratedGradients
        except ImportError:
            raise ImportError("captum not installed. Run: pip install captum")

        self.model = model.eval()
        self.device = device
        self.n_steps = n_steps
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ig = self._IG(model)

    def explain(
        self,
        input_tensor: torch.Tensor,
        original_image: np.ndarray,
        target_class: Optional[int] = None,
        class_names: Optional[List[str]] = None,
        sample_id: str = "sample",
        baseline: Optional[torch.Tensor] = None,
    ) -> Dict:
        """
        Compute Integrated Gradients attribution.

        Parameters
        ----------
        input_tensor   : [1, C, H, W] preprocessed tensor
        original_image : H x W x 3 uint8 array
        target_class   : class to explain (None → argmax prediction)
        baseline       : [1, C, H, W] baseline tensor (None → black image)

        Returns
        -------
        dict with keys: attributions (np.ndarray), overlay_path, json_path
        """
        input_tensor = input_tensor.to(self.device).requires_grad_(True)

        if baseline is None:
            baseline = torch.zeros_like(input_tensor).to(self.device)

        if target_class is None:
            with torch.no_grad():
                logits = self.model(input_tensor)
                target_class = logits.argmax(dim=-1).item()

        try:
            attributions, delta = self.ig.attribute(
                input_tensor,
                baselines=baseline,
                target=target_class,
                n_steps=self.n_steps,
                return_convergence_delta=True,
            )
        except Exception as e:
            logger.warning(f"Integrated Gradients failed: {e}")
            return {}

        attr_np = attributions.squeeze().detach().cpu().numpy()  # [C, H, W]
        attr_magnitude = np.abs(attr_np).mean(axis=0)            # [H, W]

        overlay = self._make_overlay(original_image, attr_magnitude)
        overlay_path = self.output_dir / f"{sample_id}_ig_class{target_class}.png"
        cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        # JSON attribution summary
        summary = {
            "sample_id": sample_id,
            "target_class": target_class,
            "class_name": class_names[target_class] if class_names else str(target_class),
            "mean_abs_attribution": float(attr_magnitude.mean()),
            "max_attribution": float(attr_magnitude.max()),
            "convergence_delta": float(delta.mean().item()),
        }
        json_path = self.output_dir / f"{sample_id}_ig_attribution.json"
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"[IG] target_class={target_class}  conv_delta={delta.mean():.4f} → {overlay_path}")

        return {
            "attributions": attr_np,
            "attr_magnitude": attr_magnitude,
            "target_class": target_class,
            "overlay_path": str(overlay_path),
            "json_path": str(json_path),
            "convergence_delta": float(delta.mean().item()),
        }

    @staticmethod
    def _make_overlay(
        original: np.ndarray,
        attr_map: np.ndarray,
        alpha: float = 0.6,
        colormap: int = cv2.COLORMAP_PLASMA,
    ) -> np.ndarray:
        h, w = original.shape[:2]
        norm = (attr_map - attr_map.min()) / (attr_map.max() - attr_map.min() + 1e-8)
        resized = cv2.resize(norm, (w, h))
        heatmap = cv2.applyColorMap((resized * 255).astype(np.uint8), colormap)
        heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        orig = (original * 255).clip(0, 255).astype(np.uint8) if original.dtype != np.uint8 else original
        return cv2.addWeighted(orig, 1 - alpha, heatmap_rgb, alpha, 0)
