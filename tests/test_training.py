"""
tests/test_training.py — Smoke tests for training components.
"""

import pytest
import torch
import torch.nn as nn

from src.training.losses import WeightedLabelSmoothingCE, FocalLoss, CombinedClassificationLoss
from src.training.scheduler import get_cosine_with_warmup


class TestLosses:
    def test_ce_loss_shape(self):
        loss_fn = WeightedLabelSmoothingCE(num_classes=3, smoothing=0.1)
        logits = torch.randn(8, 3)
        labels = torch.randint(0, 3, (8,))
        loss = loss_fn(logits, labels)
        assert loss.dim() == 0, "Loss must be a scalar"
        assert loss.item() > 0

    def test_focal_loss_shape(self):
        loss_fn = FocalLoss(gamma=2.0)
        logits = torch.randn(8, 3)
        labels = torch.randint(0, 3, (8,))
        loss = loss_fn(logits, labels)
        assert loss.item() >= 0

    def test_combined_loss_shape(self):
        loss_fn = CombinedClassificationLoss(num_classes=3)
        logits = torch.randn(8, 3)
        labels = torch.randint(0, 3, (8,))
        loss = loss_fn(logits, labels)
        assert loss.item() > 0

    def test_class_weights_applied(self):
        weights = torch.tensor([1.0, 2.0, 3.0])
        loss_fn = WeightedLabelSmoothingCE(num_classes=3, class_weights=weights)
        logits = torch.randn(4, 3)
        labels = torch.zeros(4, dtype=torch.long)
        # Should run without error
        loss = loss_fn(logits, labels)
        assert not torch.isnan(loss)


class TestScheduler:
    def test_cosine_warmup_runs(self):
        model = nn.Linear(10, 3)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = get_cosine_with_warmup(
            optimizer, warmup_epochs=2, total_epochs=10, steps_per_epoch=50
        )
        # Simulate a few steps
        for _ in range(100):
            optimizer.step()
            scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        assert 0 <= lr <= 1e-3, f"LR out of range: {lr}"
