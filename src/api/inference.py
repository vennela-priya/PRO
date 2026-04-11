"""
inference.py
============
Production inference pipeline with:
- ≤500ms latency budget (including XAI)
- TTA (test-time augmentation, 5 variants)
- Ensemble soft-voting
- Grad-CAM++ heatmap generation
- Structured prediction output
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import yaml
from PIL import Image

from src.data.augmentation import build_val_transforms, build_tta_transforms
from src.models.ensemble import BrainTumorEnsemble
from src.models.unet import BrainTumorUNet
from src.xai.gradcam import MultiModelGradCAM

logger = logging.getLogger(__name__)

CLASS_NAMES = ["meningioma", "glioma", "pituitary"]


class BrainTumorInference:
    """
    End-to-end inference engine.

    Parameters
    ----------
    ensemble     : loaded BrainTumorEnsemble
    unet         : loaded BrainTumorUNet (optional)
    config       : full config dict
    device       : compute device
    gradcam      : MultiModelGradCAM (optional, for XAI)
    """

    def __init__(
        self,
        ensemble: BrainTumorEnsemble,
        device: torch.device,
        config: dict,
        unet: Optional[BrainTumorUNet] = None,
        gradcam: Optional[MultiModelGradCAM] = None,
    ) -> None:
        self.ensemble = ensemble.to(device).eval()
        self.unet = unet.to(device).eval() if unet else None
        self.device = device
        self.config = config
        self.gradcam = gradcam
        self.class_names = config["data"]["class_names"]
        self.image_size = config["data"]["image_size"]

        img_cfg = config["augmentation"]["val"]
        self.val_transform = build_val_transforms(
            image_size=self.image_size,
            mean=tuple(img_cfg["normalize"]["mean"]),
            std=tuple(img_cfg["normalize"]["std"]),
        )

        self.tta_transforms = None
        if config["inference"]["tta_enabled"]:
            self.tta_transforms = build_tta_transforms(
                image_size=self.image_size,
                mean=tuple(img_cfg["normalize"]["mean"]),
                std=tuple(img_cfg["normalize"]["std"]),
            )

        self.latency_budget_ms = config["inference"]["latency_budget_ms"]

    def preprocess(self, image: np.ndarray) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Preprocess an RGB numpy image.

        Returns
        -------
        tensor        : [1, C, H, W] tensor
        original_resized : resized uint8 RGB image for display
        """
        h, w = self.image_size, self.image_size
        resized = cv2.resize(image, (w, h))
        augmented = self.val_transform(image=resized)
        tensor = augmented["image"].unsqueeze(0).to(self.device)
        return tensor, resized

    @torch.no_grad()
    def classify(
        self,
        tensor: torch.Tensor,
        use_tta: bool = True,
    ) -> Tuple[int, float, np.ndarray]:
        """
        Run ensemble classification.

        Returns
        -------
        pred_class : int
        confidence : float
        probs      : [num_classes] numpy array
        """
        if use_tta and self.tta_transforms:
            tta_probs = []
            img_np = tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            img_np = (img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
            img_np = (img_np * 255).clip(0, 255).astype(np.uint8)

            for tfm in self.tta_transforms:
                aug = tfm(image=img_np)
                aug_tensor = aug["image"].unsqueeze(0).to(self.device)
                probs = self.ensemble(aug_tensor)
                tta_probs.append(probs.cpu().numpy())

            probs_np = np.mean(tta_probs, axis=0)[0]
        else:
            probs = self.ensemble(tensor)
            probs_np = probs.cpu().numpy()[0]

        pred_class = int(probs_np.argmax())
        confidence = float(probs_np[pred_class])
        return pred_class, confidence, probs_np

    @torch.no_grad()
    def segment(self, tensor: torch.Tensor) -> Optional[Dict]:
        """Run U-Net segmentation if available."""
        if self.unet is None:
            return None
        logits = self.unet(tensor)
        mask = (torch.sigmoid(logits) > 0.5).float()
        bbox = self.unet.predict_bbox(tensor)
        return {
            "mask": mask.squeeze().cpu().numpy(),
            "bbox": bbox[0],
        }

    def explain(
        self,
        tensor: torch.Tensor,
        original_image: np.ndarray,
        pred_class: int,
        sample_id: str = "inference",
    ) -> Optional[Dict]:
        """Run Grad-CAM++ for the predicted class."""
        if self.gradcam is None:
            return None
        return self.gradcam.explain(
            input_tensor=tensor,
            original_image=original_image,
            target_class=pred_class,
            sample_id=sample_id,
        )

    def predict(
        self,
        image: np.ndarray,
        sample_id: str = "sample",
        include_xai: bool = True,
        include_seg: bool = True,
    ) -> Dict:
        """
        Full prediction pipeline: preprocess → classify → segment → XAI.

        Parameters
        ----------
        image      : H x W x 3 uint8 RGB image
        sample_id  : identifier for output files
        include_xai: run Grad-CAM++
        include_seg: run U-Net segmentation

        Returns
        -------
        Structured prediction dict with all results + latency info.
        """
        t_start = time.perf_counter()

        # 1. Preprocess
        tensor, orig_resized = self.preprocess(image)
        t_prep = time.perf_counter()

        # 2. Classify
        pred_class, confidence, probs = self.classify(tensor, use_tta=True)
        t_cls = time.perf_counter()

        # 3. Segmentation
        seg_result = self.segment(tensor) if include_seg else None
        t_seg = time.perf_counter()

        # 4. XAI
        xai_result = None
        if include_xai:
            xai_result = self.explain(tensor, orig_resized, pred_class, sample_id)
        t_xai = time.perf_counter()

        total_ms = (t_xai - t_start) * 1000
        if total_ms > self.latency_budget_ms:
            logger.warning(f"Latency budget exceeded: {total_ms:.1f}ms > {self.latency_budget_ms}ms")

        return {
            "sample_id": sample_id,
            "prediction": {
                "class_index": pred_class,
                "class_name": self.class_names[pred_class],
                "confidence": round(confidence, 4),
                "probabilities": {
                    name: round(float(p), 4)
                    for name, p in zip(self.class_names, probs)
                },
            },
            "segmentation": seg_result,
            "xai": {
                "gradcam_paths": xai_result if xai_result else {},
            },
            "latency_ms": {
                "preprocessing": round((t_prep - t_start) * 1000, 2),
                "classification": round((t_cls - t_prep) * 1000, 2),
                "segmentation": round((t_seg - t_cls) * 1000, 2),
                "xai": round((t_xai - t_seg) * 1000, 2),
                "total": round(total_ms, 2),
            },
        }

    @classmethod
    def from_config(
        cls,
        config_path: str,
        checkpoint_dir: str,
        device: Optional[torch.device] = None,
    ) -> "BrainTumorInference":
        """
        Load full inference pipeline from config + checkpoint directory.
        """
        from src.models import (
            EfficientNetV2Classifier,
            ResNetCBAMClassifier,
            ViTClassifier,
            BrainTumorEnsemble,
            BrainTumorUNet,
        )

        with open(config_path) as f:
            config = yaml.safe_load(f)

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ckpt_dir = Path(checkpoint_dir)
        num_classes = config["data"]["num_classes"]

        # Load individual models
        model_classes = {
            "efficientnet": EfficientNetV2Classifier,
            "resnet_cbam": ResNetCBAMClassifier,
            "vit": ViTClassifier,
        }
        model_kwargs = {
            "efficientnet": {"num_classes": num_classes, "pretrained": False},
            "resnet_cbam": {"num_classes": num_classes, "pretrained": False},
            "vit": {"num_classes": num_classes, "pretrained": False},
        }
        checkpoint_paths = {
            name: str(next(ckpt_dir.glob(f"{name}_*.pt")))
            for name in model_classes
        }

        ensemble = BrainTumorEnsemble.from_checkpoints(
            model_classes=model_classes,
            model_kwargs=model_kwargs,
            checkpoint_paths=checkpoint_paths,
            weights=config["models"]["ensemble"]["weights"],
            device=device,
        )

        return cls(ensemble=ensemble, device=device, config=config)
