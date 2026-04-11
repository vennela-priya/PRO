"""
shap_explainer.py
=================
SHAP DeepExplainer for feature attribution on CNN models.
Outputs per-pixel SHAP value maps and aggregated JSON attribution scores.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class SHAPExplainer:
    """
    Wraps SHAP DeepExplainer for brain tumor CNN models.

    Parameters
    ----------
    model      : trained nn.Module
    background : background tensor [N, C, H, W] (subset of training data)
    device     : compute device
    output_dir : directory to save outputs
    """

    def __init__(
        self,
        model: nn.Module,
        background: torch.Tensor,
        device: torch.device,
        output_dir: str = "outputs/shap",
    ) -> None:
        try:
            import shap
            self.shap = shap
        except ImportError:
            raise ImportError("shap not installed. Run: pip install shap")

        self.model = model.eval()
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        background = background.to(device)
        self.explainer = shap.DeepExplainer(model, background)
        logger.info(f"SHAP DeepExplainer initialised with background of shape {tuple(background.shape)}")

    def explain(
        self,
        input_tensor: torch.Tensor,
        original_image: np.ndarray,
        class_names: List[str],
        sample_id: str = "sample",
    ) -> Dict:
        """
        Compute SHAP values for a single image.

        Parameters
        ----------
        input_tensor   : [1, C, H, W] preprocessed tensor
        original_image : H x W x 3 uint8 array
        class_names    : list of class name strings
        sample_id      : output filename prefix

        Returns
        -------
        dict with keys: shap_values, attribution_json_path, overlay_paths
        """
        input_tensor = input_tensor.to(self.device)

        try:
            shap_values = self.explainer.shap_values(input_tensor)  # list of [1, C, H, W] per class
        except Exception as e:
            logger.warning(f"SHAP computation failed: {e}")
            return {}

        # Aggregate channel-wise SHAP magnitude per class → mean abs
        attribution_scores = {}
        for i, name in enumerate(class_names):
            sv = np.abs(shap_values[i][0]).mean(axis=0)  # [H, W]
            attribution_scores[name] = float(sv.mean())   # scalar summary

        json_path = self.output_dir / f"{sample_id}_shap_attribution.json"
        with open(json_path, "w") as f:
            json.dump({"sample_id": sample_id, "attributions": attribution_scores}, f, indent=2)

        overlay_paths = {}
        for i, name in enumerate(class_names):
            sv_map = np.abs(shap_values[i][0]).mean(axis=0)
            overlay = self._make_overlay(original_image, sv_map)
            img_path = self.output_dir / f"{sample_id}_shap_{name}.png"
            import cv2
            cv2.imwrite(str(img_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            overlay_paths[name] = str(img_path)

        logger.info(f"SHAP attribution saved → {json_path}")
        return {
            "shap_values": shap_values,
            "attribution_scores": attribution_scores,
            "attribution_json_path": str(json_path),
            "overlay_paths": overlay_paths,
        }

    @staticmethod
    def _make_overlay(
        original: np.ndarray,
        shap_map: np.ndarray,
        alpha: float = 0.5,
    ) -> np.ndarray:
        import cv2
        h, w = original.shape[:2]
        sv_norm = (shap_map - shap_map.min()) / (shap_map.max() - shap_map.min() + 1e-8)
        sv_resized = cv2.resize(sv_norm, (w, h))
        heatmap = cv2.applyColorMap((sv_resized * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
        heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        orig = (original * 255).clip(0, 255).astype(np.uint8) if original.dtype != np.uint8 else original
        return cv2.addWeighted(orig, 1 - alpha, heatmap_rgb, alpha, 0)

    @staticmethod
    def get_background_samples(
        dataset,
        n_samples: int = 50,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Randomly sample n background images from a dataset."""
        indices = np.random.choice(len(dataset), min(n_samples, len(dataset)), replace=False)
        samples = torch.stack([dataset[i]["image"] for i in indices])
        if device is not None:
            samples = samples.to(device)
        return samples
