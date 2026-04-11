"""
scheduler.py
============
Learning rate schedulers:
- CosineAnnealingWithWarmup
- WarmupScheduler wrapper
- get_scheduler factory
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LambdaLR,
    SequentialLR,
    OneCycleLR,
)


def linear_warmup_schedule(warmup_steps: int) -> Any:
    """Lambda for linear warmup over ``warmup_steps`` steps."""
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return 1.0
    return lr_lambda


def get_cosine_with_warmup(
    optimizer: Optimizer,
    warmup_epochs: int,
    total_epochs: int,
    steps_per_epoch: int,
    eta_min: float = 1e-7,
) -> SequentialLR:
    """
    Cosine annealing with linear warmup.

    First ``warmup_epochs`` epochs: linear ramp 0 → lr.
    Remaining epochs: cosine decay lr → eta_min.
    """
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = total_epochs * steps_per_epoch
    cosine_steps = total_steps - warmup_steps

    warmup = LambdaLR(optimizer, lr_lambda=linear_warmup_schedule(warmup_steps))
    cosine = CosineAnnealingLR(optimizer, T_max=cosine_steps, eta_min=eta_min)

    return SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_steps],
    )


def get_onecycle(
    optimizer: Optimizer,
    max_lr: float,
    steps_per_epoch: int,
    epochs: int,
    pct_start: float = 0.3,
) -> OneCycleLR:
    """1-Cycle LR schedule (Smith & Topin, 2019)."""
    return OneCycleLR(
        optimizer,
        max_lr=max_lr,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        pct_start=pct_start,
        anneal_strategy="cos",
    )


def get_scheduler(
    name: str,
    optimizer: Optimizer,
    **kwargs: Any,
) -> Any:
    """
    Factory for creating schedulers by name.

    Supported: 'cosine_annealing', 'onecycle', 'step'
    """
    if name == "cosine_annealing":
        return get_cosine_with_warmup(optimizer, **kwargs)
    elif name == "onecycle":
        return get_onecycle(optimizer, **kwargs)
    elif name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, **kwargs)
    else:
        raise ValueError(f"Unknown scheduler: {name}. Choose from [cosine_annealing, onecycle, step]")
