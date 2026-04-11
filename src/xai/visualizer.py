"""
visualizer.py
=============
Master XAI visualiser — orchestrates Grad-CAM++, SHAP, and Integrated Gradients
for a set of test cases. Generates combined multi-panel figures.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from src.xai.gradcam import MultiModelGradCAM, overlay_heatmap
from src.xai.shap_explainer import SHAPExplainer
from src.xai.ig import IntegratedGradientsExplainer

logger = logging.getLogger(__name__)

CLASS_NAMES = ["meningioma", "glioma", "pituitary"]


class XAIVisualizer:
    """
    High-level interface to run all XAI methods for a batch of samples.

    Parameters
    ----------
    models          : dict {model_name: nn.Module}
    device          : compute device
    xai_config      : xai section from config.yaml
    background_data : tensor for SHAP background [N, C, H, W]
    class_names     : list of class name strings
    """

    def __init__(
        self,
        models: Dict[str, nn.Module],
        device: torch.device,
        xai_config: dict,
        background_data: Optional[torch.Tensor] = None,
        class_names: List[str] = CLASS_NAMES,
    ) -> None:
        self.models = models
        self.device = device
        self.config = xai_config
        self.class_names = class_names

        # Grad-CAM++
        self.gradcam = MultiModelGradCAM(
            models=models,
            output_dir=xai_config.get("gradcam", {}).get("output_dir", "outputs/heatmaps"),
        )

        # Integrated Gradients (use first model by default)
        primary_name = list(models.keys())[0]
        self.ig = IntegratedGradientsExplainer(
            model=models[primary_name],
            device=device,
            n_steps=xai_config.get("integrated_gradients", {}).get("n_steps", 50),
            output_dir=xai_config.get("integrated_gradients", {}).get("output_dir", "outputs/ig"),
        )

        # SHAP (only if background provided)
        self.shap_explainer = None
        if background_data is not None and xai_config.get("shap", {}).get("enabled", False):
            self.shap_explainer = SHAPExplainer(
                model=models[primary_name],
                background=background_data,
                device=device,
                output_dir=xai_config.get("shap", {}).get("output_dir", "outputs/shap"),
            )

    def explain_sample(
        self,
        input_tensor: torch.Tensor,
        original_image: np.ndarray,
        true_label: Optional[int] = None,
        target_class: Optional[int] = None,
        sample_id: str = "sample",
    ) -> Dict:
        """
        Run all XAI methods on a single sample and produce combined panel.

        Returns dict with paths to all outputs.
        """
        results = {"sample_id": sample_id}

        # Grad-CAM++
        gcam_results = self.gradcam.explain(
            input_tensor=input_tensor,
            original_image=original_image,
            target_class=target_class,
            sample_id=sample_id,
            alpha=self.config.get("gradcam", {}).get("alpha", 0.5),
        )
        results["gradcam"] = gcam_results

        # Integrated Gradients
        ig_results = self.ig.explain(
            input_tensor=input_tensor,
            original_image=original_image,
            target_class=target_class,
            class_names=self.class_names,
            sample_id=sample_id,
        )
        results["ig"] = ig_results

        # SHAP (optional)
        if self.shap_explainer is not None:
            shap_results = self.shap_explainer.explain(
                input_tensor=input_tensor,
                original_image=original_image,
                class_names=self.class_names,
                sample_id=sample_id,
            )
            results["shap"] = shap_results

        # Combined panel
        panel_path = self._make_combined_panel(
            original_image=original_image,
            gcam_results=gcam_results,
            ig_results=ig_results,
            true_label=true_label,
            sample_id=sample_id,
        )
        results["panel_path"] = panel_path
        return results

    def _make_combined_panel(
        self,
        original_image: np.ndarray,
        gcam_results: Dict,
        ig_results: Dict,
        true_label: Optional[int],
        sample_id: str,
    ) -> str:
        """Generate a multi-panel figure: original + Grad-CAM per model + IG."""
        n_models = len(gcam_results)
        total_panels = 1 + n_models + (1 if ig_results else 0)

        fig, axes = plt.subplots(1, total_panels, figsize=(4 * total_panels, 4))
        if total_panels == 1:
            axes = [axes]

        # Panel 0: original
        ax = axes[0]
        ax.imshow(original_image)
        title = f"Original\nTrue: {self.class_names[true_label]}" if true_label is not None else "Original"
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.axis("off")

        # Panels 1..n: Grad-CAM per model
        for i, (model_name, res) in enumerate(gcam_results.items(), start=1):
            ax = axes[i]
            if "overlay_path" in res and Path(res["overlay_path"]).exists():
                overlay = cv2.cvtColor(cv2.imread(res["overlay_path"]), cv2.COLOR_BGR2RGB)
                ax.imshow(overlay)
            pred_name = self.class_names[res.get("pred_class", 0)]
            ax.set_title(
                f"GradCAM++ [{model_name}]\nPred: {pred_name} ({res.get('prob', 0):.2%})",
                fontsize=8,
            )
            ax.axis("off")

        # Last panel: IG
        if ig_results and "overlay_path" in ig_results:
            ax = axes[-1]
            if Path(ig_results["overlay_path"]).exists():
                ig_overlay = cv2.cvtColor(cv2.imread(ig_results["overlay_path"]), cv2.COLOR_BGR2RGB)
                ax.imshow(ig_overlay)
            ax.set_title("Integrated Gradients", fontsize=9)
            ax.axis("off")

        plt.suptitle(f"XAI Panel — {sample_id}", fontsize=11, fontweight="bold")
        plt.tight_layout()

        gcam_dir = self.config.get("gradcam", {}).get("output_dir", "outputs/heatmaps")
        panel_path = Path(gcam_dir) / f"{sample_id}_xai_panel.png"
        fig.savefig(str(panel_path), dpi=150, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"Combined XAI panel saved → {panel_path}")
        return str(panel_path)

    def explain_test_cases(
        self,
        test_samples: List[Dict],
    ) -> List[Dict]:
        """
        Run XAI on a list of test case dicts.
        Each dict: {input_tensor, original_image, true_label, sample_id}
        """
        all_results = []
        for sample in test_samples:
            res = self.explain_sample(**sample)
            all_results.append(res)
        return all_results

    def cleanup(self) -> None:
        self.gradcam.cleanup()
