"""
tests/test_models.py — Smoke tests for all model architectures.
"""

import pytest
import torch
import torch.nn as nn

from src.models.efficientnet import EfficientNetV2Classifier
from src.models.resnet_cbam import ResNetCBAMClassifier, CBAM
from src.models.vit import ViTClassifier
from src.models.ensemble import BrainTumorEnsemble, TemperatureScaler
from src.models.unet import BrainTumorUNet, DiceBCELoss, dice_score

DEVICE = torch.device("cpu")
BATCH = 2
IMG_SIZE = 224
NUM_CLASSES = 3


def make_input() -> torch.Tensor:
    return torch.randn(BATCH, 3, IMG_SIZE, IMG_SIZE)


# ---------------------------------------------------------------------------
# CBAM tests
# ---------------------------------------------------------------------------

class TestCBAM:
    def test_cbam_output_shape(self):
        cbam = CBAM(in_channels=64)
        x = torch.randn(2, 64, 28, 28)
        out = cbam(x)
        assert out.shape == x.shape, "CBAM must preserve spatial dimensions"


# ---------------------------------------------------------------------------
# EfficientNetV2 tests
# ---------------------------------------------------------------------------

class TestEfficientNet:
    @pytest.fixture
    def model(self):
        return EfficientNetV2Classifier(num_classes=NUM_CLASSES, pretrained=False).to(DEVICE)

    def test_forward_shape(self, model):
        x = make_input()
        out = model(x)
        assert out.shape == (BATCH, NUM_CLASSES)

    def test_freeze_unfreeze(self, model):
        model.freeze_backbone()
        frozen = all(not p.requires_grad for p in model.backbone.parameters())
        assert frozen, "Backbone should be frozen"

        model.unfreeze_backbone()
        unfrozen = all(p.requires_grad for p in model.backbone.parameters())
        assert unfrozen, "Backbone should be unfrozen"

    def test_output_is_logits(self, model):
        x = make_input()
        out = model(x)
        # Logits can be any value; softmax should sum to 1
        probs = torch.softmax(out, dim=-1)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(BATCH), atol=1e-5)


# ---------------------------------------------------------------------------
# ResNet-CBAM tests
# ---------------------------------------------------------------------------

class TestResNetCBAM:
    @pytest.fixture
    def model(self):
        return ResNetCBAMClassifier(num_classes=NUM_CLASSES, pretrained=False).to(DEVICE)

    def test_forward_shape(self, model):
        x = make_input()
        out = model(x)
        assert out.shape == (BATCH, NUM_CLASSES)

    def test_freeze_unfreeze(self, model):
        model.freeze_backbone()
        assert not model.backbone.layer1[0].conv1.weight.requires_grad
        model.unfreeze_backbone()
        assert model.backbone.layer1[0].conv1.weight.requires_grad


# ---------------------------------------------------------------------------
# ViT tests
# ---------------------------------------------------------------------------

class TestViT:
    @pytest.fixture
    def model(self):
        return ViTClassifier(num_classes=NUM_CLASSES, pretrained=False).to(DEVICE)

    def test_forward_shape(self, model):
        x = make_input()
        out = model(x)
        assert out.shape == (BATCH, NUM_CLASSES)


# ---------------------------------------------------------------------------
# Ensemble tests
# ---------------------------------------------------------------------------

class TestEnsemble:
    @pytest.fixture
    def ensemble(self):
        models = {
            "eff": EfficientNetV2Classifier(num_classes=NUM_CLASSES, pretrained=False),
            "res": ResNetCBAMClassifier(num_classes=NUM_CLASSES, pretrained=False),
            "vit": ViTClassifier(num_classes=NUM_CLASSES, pretrained=False),
        }
        weights = {"eff": 0.4, "res": 0.3, "vit": 0.3}
        return BrainTumorEnsemble(models=models, weights=weights, use_temp_scaling=False).to(DEVICE)

    def test_forward_shape(self, ensemble):
        x = make_input()
        out = ensemble(x)
        assert out.shape == (BATCH, NUM_CLASSES)

    def test_probs_sum_to_one(self, ensemble):
        x = make_input()
        probs = ensemble(x)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(BATCH), atol=1e-5)

    def test_invalid_weights_raise(self):
        models = {"a": nn.Linear(10, 3)}
        with pytest.raises(ValueError):
            BrainTumorEnsemble(models=models, weights={"a": 0.5})  # doesn't sum to 1

    def test_predict_returns_class_and_prob(self, ensemble):
        x = make_input()
        classes, probs = ensemble.predict(x)
        assert classes.shape == (BATCH,)
        assert probs.shape == (BATCH,)
        assert all(0 <= c < NUM_CLASSES for c in classes.tolist())


# ---------------------------------------------------------------------------
# U-Net tests
# ---------------------------------------------------------------------------

class TestUNet:
    @pytest.fixture
    def model(self):
        return BrainTumorUNet(encoder_weights=None).to(DEVICE)

    def test_forward_shape(self, model):
        x = make_input()
        out = model(x)
        assert out.shape == (BATCH, 1, IMG_SIZE, IMG_SIZE)

    def test_predict_mask_binary(self, model):
        x = make_input()
        mask = model.predict_mask(x)
        unique = mask.unique().tolist()
        assert all(v in [0.0, 1.0] for v in unique), "Mask must be binary"

    def test_dice_bce_loss(self, model):
        criterion = DiceBCELoss()
        x = make_input()
        logits = model(x)
        targets = torch.zeros(BATCH, 1, IMG_SIZE, IMG_SIZE)
        loss = criterion(logits, targets)
        assert loss.item() >= 0

    def test_dice_score(self):
        pred = torch.ones(2, 1, 64, 64)
        true = torch.ones(2, 1, 64, 64)
        dsc = dice_score(pred, true)
        assert abs(dsc - 1.0) < 1e-5, "Perfect prediction should give DSC=1"
