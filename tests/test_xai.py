"""
tests/test_xai.py — Smoke tests for XAI modules.
"""

import numpy as np
import pytest
import torch

from src.models.efficientnet import EfficientNetV2Classifier
from src.xai.gradcam import GradCAMPlusPlus, overlay_heatmap

DEVICE = torch.device("cpu")


class TestGradCAM:
    @pytest.fixture
    def model_and_layer(self):
        model = EfficientNetV2Classifier(num_classes=3, pretrained=False).to(DEVICE)
        layer = model.target_layer
        return model, layer

    def test_cam_output_range(self, model_and_layer):
        model, layer = model_and_layer
        engine = GradCAMPlusPlus(model, layer)
        x = torch.randn(1, 3, 224, 224).to(DEVICE)
        cam, pred_cls, prob = engine.generate(x)

        assert cam.ndim == 2, "CAM must be 2D"
        assert cam.min() >= 0.0 - 1e-6, f"CAM min < 0: {cam.min()}"
        assert cam.max() <= 1.0 + 1e-6, f"CAM max > 1: {cam.max()}"
        assert 0 <= pred_cls < 3
        assert 0.0 <= prob <= 1.0
        engine.remove_hooks()

    def test_overlay_shape(self):
        original = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        cam = np.random.rand(28, 28).astype(np.float32)
        overlay = overlay_heatmap(original, cam)
        assert overlay.shape == original.shape
        assert overlay.dtype == np.uint8
