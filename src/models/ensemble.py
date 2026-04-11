"""
ensemble.py
===========
Soft-voting ensemble with learned temperature scaling.

Combines EfficientNetV2-M, ResNet-50+CBAM, and ViT-B/16 via
weighted soft-voting. Temperature scaling calibrates the logits
of each model before averaging probabilities.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Temperature Scaling wrapper
# ---------------------------------------------------------------------------

class TemperatureScaler(nn.Module):
    """
    A single learned scalar T that scales pre-softmax logits:
        calibrated_prob = softmax(logits / T)

    T > 1 → softer (more uncertain) predictions
    T < 1 → sharper (more confident) predictions
    Optimised on validation set with NLL loss.
    """

    def __init__(self) -> None:
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature.clamp(min=1e-3)

    def calibrate(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        device: torch.device,
        lr: float = 0.01,
        max_iter: int = 50,
    ) -> float:
        """Fit temperature on validation set. Returns optimal T."""
        model.eval()
        self.to(device)

        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        nll_criterion = nn.CrossEntropyLoss()

        all_logits, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(device)
                labels = batch["label"].to(device)
                logits = model(imgs)
                all_logits.append(logits.cpu())
                all_labels.append(labels.cpu())

        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)

        def eval_fn():
            optimizer.zero_grad()
            scaled = self.forward(all_logits.to(device))
            loss = nll_criterion(scaled, all_labels.to(device))
            loss.backward()
            return loss

        optimizer.step(eval_fn)
        t = self.temperature.item()
        logger.info(f"Calibrated temperature T = {t:.4f}")
        return t


# ---------------------------------------------------------------------------
# Ensemble Model
# ---------------------------------------------------------------------------

class BrainTumorEnsemble(nn.Module):
    """
    Weighted soft-voting ensemble of three classifiers.

    Parameters
    ----------
    models   : dict mapping name → nn.Module
    weights  : dict mapping name → float (must sum to 1.0)
    use_temp_scaling : whether to apply per-model temperature scaling
    """

    def __init__(
        self,
        models: Dict[str, nn.Module],
        weights: Optional[Dict[str, float]] = None,
        use_temp_scaling: bool = True,
    ) -> None:
        super().__init__()

        self.model_names = list(models.keys())
        self.models = nn.ModuleDict(models)

        if weights is None:
            n = len(models)
            weights = {name: 1.0 / n for name in self.model_names}

        self._validate_weights(weights)
        self.weights = weights

        self.use_temp_scaling = use_temp_scaling
        if use_temp_scaling:
            self.temp_scalers = nn.ModuleDict(
                {name: TemperatureScaler() for name in self.model_names}
            )

    @staticmethod
    def _validate_weights(weights: Dict[str, float]) -> None:
        total = sum(weights.values())
        if abs(total - 1.0) > 1e-4:
            raise ValueError(f"Ensemble weights must sum to 1.0 (got {total:.4f})")

    def calibrate_temperatures(
        self,
        val_loader: DataLoader,
        device: torch.device,
    ) -> Dict[str, float]:
        """Calibrate temperature of each sub-model on validation data."""
        temperatures = {}
        for name in self.model_names:
            logger.info(f"Calibrating temperature for {name}…")
            t = self.temp_scalers[name].calibrate(
                self.models[name], val_loader, device
            )
            temperatures[name] = t
        return temperatures

    def forward(
        self,
        x: torch.Tensor,
        return_individual: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass — compute weighted average of calibrated softmax probs.

        Parameters
        ----------
        x                : input tensor [B, C, H, W]
        return_individual: if True, also return per-model logits dict

        Returns
        -------
        ensemble_probs : [B, num_classes]
        individual     : (optional) dict of per-model logits
        """
        weighted_probs = None
        individual_logits: Dict[str, torch.Tensor] = {}

        for name in self.model_names:
            logits = self.models[name](x)

            if self.use_temp_scaling:
                logits = self.temp_scalers[name](logits)

            probs = F.softmax(logits, dim=-1)
            w = self.weights[name]

            if weighted_probs is None:
                weighted_probs = w * probs
            else:
                weighted_probs = weighted_probs + w * probs

            individual_logits[name] = logits

        if return_individual:
            return weighted_probs, individual_logits
        return weighted_probs

    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (predicted_class, max_probability)."""
        probs = self.forward(x)
        return probs.argmax(dim=-1), probs.max(dim=-1).values

    def freeze_all(self) -> None:
        for m in self.models.values():
            for p in m.parameters():
                p.requires_grad = False

    def unfreeze_all(self) -> None:
        for m in self.models.values():
            for p in m.parameters():
                p.requires_grad = True

    @classmethod
    def from_checkpoints(
        cls,
        model_classes: Dict[str, type],
        model_kwargs: Dict[str, dict],
        checkpoint_paths: Dict[str, str],
        weights: Optional[Dict[str, float]] = None,
        device: Optional[torch.device] = None,
    ) -> "BrainTumorEnsemble":
        """
        Instantiate ensemble from saved checkpoints.

        Parameters
        ----------
        model_classes    : {name: ModelClass}
        model_kwargs     : {name: kwargs for ModelClass}
        checkpoint_paths : {name: path_to_ckpt}
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        models = {}
        for name, ModelCls in model_classes.items():
            model = ModelCls(**model_kwargs.get(name, {}))
            ckpt = torch.load(checkpoint_paths[name], map_location=device)
            state_dict = ckpt.get("model_state_dict", ckpt)
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            models[name] = model
            logger.info(f"Loaded {name} from {checkpoint_paths[name]}")

        return cls(models=models, weights=weights)
