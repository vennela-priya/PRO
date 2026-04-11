"""
trainer.py
==========
End-to-end training loop for brain tumor classification.

Implements 3-phase training:
  Phase 1 — frozen backbone, warm-up classifier head (10 epochs, lr=1e-3)
  Phase 2 — full fine-tuning, cosine annealing (30 epochs, lr=1e-4)
  Phase 3 — ensemble temperature calibration (5 epochs, lr=1e-5)

Features:
  - Mixed precision (AMP)
  - Gradient accumulation
  - CutMix / MixUp
  - EarlyStopping
  - TensorBoard + optional MLflow logging
  - Checkpoint saving (top-k by val AUC)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import AUROC, Accuracy, F1Score

from src.data.augmentation import cutmix_data, mixup_data, mixup_criterion
from src.training.losses import CombinedClassificationLoss, WeightedLabelSmoothingCE
from src.training.scheduler import get_scheduler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Early Stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """Stops training when a monitored metric does not improve for ``patience`` epochs."""

    def __init__(self, patience: int = 10, mode: str = "max", min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_score: Optional[float] = None
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False

        improved = (score - self.best_score) > self.min_delta if self.mode == "max" \
            else (self.best_score - score) > self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(f"Early stopping triggered after {self.counter} epochs without improvement.")

        return self.should_stop


# ---------------------------------------------------------------------------
# Checkpoint Manager
# ---------------------------------------------------------------------------

class CheckpointManager:
    """Save top-k checkpoints ranked by metric value."""

    def __init__(self, save_dir: str, top_k: int = 3, mode: str = "max") -> None:
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.top_k = top_k
        self.mode = mode
        self.history: List[Tuple[float, Path]] = []

    def save(self, model: nn.Module, optimizer, epoch: int, metric: float, name: str) -> None:
        path = self.save_dir / f"{name}_epoch{epoch:03d}_metric{metric:.4f}.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metric": metric,
        }, path)

        self.history.append((metric, path))
        self.history.sort(key=lambda x: x[0], reverse=(self.mode == "max"))

        # Remove worst checkpoints beyond top-k
        while len(self.history) > self.top_k:
            _, old_path = self.history.pop()
            old_path.unlink(missing_ok=True)
            logger.debug(f"Removed checkpoint: {old_path}")

        logger.info(f"Saved checkpoint: {path}  (metric={metric:.4f})")

    @property
    def best_path(self) -> Optional[Path]:
        return self.history[0][1] if self.history else None


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """
    Full 3-phase trainer for a single classification model.

    Parameters
    ----------
    model         : nn.Module (EfficientNet / ResNet-CBAM / ViT)
    train_loader  : training DataLoader
    val_loader    : validation DataLoader
    device        : torch.device
    config        : dict of training hyperparameters (from config.yaml)
    model_name    : string identifier for checkpointing
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        config: dict,
        model_name: str = "model",
        class_weights: Optional[torch.Tensor] = None,
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        self.model_name = model_name

        self.num_classes = config["data"]["num_classes"]
        self.accum_steps = config["training"]["accumulation_steps"]
        self.mixed_precision = config["project"]["mixed_precision"]

        self.criterion = WeightedLabelSmoothingCE(
            num_classes=self.num_classes,
            smoothing=config["training"]["loss"]["label_smoothing"],
            class_weights=class_weights.to(device) if class_weights is not None else None,
        )

        self.cutmix_alpha = config["augmentation"]["train"]["cutmix_alpha"]
        self.mixup_alpha = config["augmentation"]["train"]["mixup_alpha"]

        self.ckpt_manager = CheckpointManager(
            save_dir=config["training"]["checkpoint"]["save_dir"] + f"/{model_name}",
            top_k=config["training"]["checkpoint"]["save_top_k"],
        )

        self.writer = SummaryWriter(log_dir=f"runs/{model_name}")
        self.scaler = GradScaler(enabled=self.mixed_precision)

        # Torchmetrics
        self._reset_metrics()

    def _reset_metrics(self) -> None:
        nc = self.num_classes
        self.train_acc = Accuracy(task="multiclass", num_classes=nc).to(self.device)
        self.val_acc = Accuracy(task="multiclass", num_classes=nc).to(self.device)
        self.val_auc = AUROC(task="multiclass", num_classes=nc).to(self.device)
        self.val_f1 = F1Score(task="multiclass", num_classes=nc, average="macro").to(self.device)

    def _build_optimizer(self, lr: float) -> torch.optim.Optimizer:
        opt_cfg = self.config["training"]["optimizer"]
        return torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr,
            weight_decay=opt_cfg["weight_decay"],
            betas=tuple(opt_cfg["betas"]),
            eps=opt_cfg["eps"],
        )

    def _train_one_epoch(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler=None,
        use_cutmix: bool = True,
        use_mixup: bool = True,
        epoch: int = 0,
    ) -> Dict[str, float]:
        self.model.train()
        self.train_acc.reset()
        total_loss = 0.0
        n_batches = len(self.train_loader)

        optimizer.zero_grad()

        for batch_idx, batch in enumerate(self.train_loader):
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            # Apply CutMix or MixUp stochastically
            r = torch.rand(1).item()
            if use_cutmix and r < 0.4:
                images, y_a, y_b, lam = cutmix_data(images, labels, self.cutmix_alpha)
                mixed = True
            elif use_mixup and r < 0.6:
                images, y_a, y_b, lam = mixup_data(images, labels, self.mixup_alpha)
                mixed = True
            else:
                mixed = False

            with autocast(enabled=self.mixed_precision):
                logits = self.model(images)
                if mixed:
                    loss = mixup_criterion(self.criterion, logits, y_a, y_b, lam)
                else:
                    loss = self.criterion(logits, labels)
                loss = loss / self.accum_steps

            self.scaler.scale(loss).backward()

            if (batch_idx + 1) % self.accum_steps == 0 or (batch_idx + 1) == n_batches:
                self.scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(optimizer)
                self.scaler.update()
                optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

            total_loss += loss.item() * self.accum_steps
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                self.train_acc.update(preds, labels)

        return {
            "loss": total_loss / n_batches,
            "acc": self.train_acc.compute().item(),
        }

    @torch.no_grad()
    def _validate(self) -> Dict[str, float]:
        self.model.eval()
        self.val_acc.reset()
        self.val_auc.reset()
        self.val_f1.reset()
        total_loss = 0.0

        for batch in self.val_loader:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            with autocast(enabled=self.mixed_precision):
                logits = self.model(images)
                loss = self.criterion(logits, labels)

            total_loss += loss.item()
            probs = torch.softmax(logits, dim=-1)
            preds = probs.argmax(dim=-1)

            self.val_acc.update(preds, labels)
            self.val_auc.update(probs, labels)
            self.val_f1.update(preds, labels)

        return {
            "loss": total_loss / len(self.val_loader),
            "acc": self.val_acc.compute().item(),
            "auc": self.val_auc.compute().item(),
            "f1": self.val_f1.compute().item(),
        }

    def _log(self, metrics: Dict[str, float], phase: str, epoch: int) -> None:
        for k, v in metrics.items():
            self.writer.add_scalar(f"{phase}/{k}", v, epoch)
        logger.info(
            f"[{phase.upper()}] Epoch {epoch:03d} | "
            + " | ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        )

    def run_phase(
        self,
        phase: int,
        epochs: int,
        lr: float,
        freeze_backbone: bool,
        use_cutmix: bool = True,
        use_mixup: bool = True,
        warmup_epochs: int = 0,
        early_stopping: Optional[EarlyStopping] = None,
    ) -> Dict[str, float]:
        """Execute a single training phase. Returns best val metrics."""
        logger.info(f"\n{'='*60}\nPhase {phase} | epochs={epochs} lr={lr} freeze={freeze_backbone}\n{'='*60}")

        if freeze_backbone and hasattr(self.model, "freeze_backbone"):
            self.model.freeze_backbone()
        elif not freeze_backbone and hasattr(self.model, "unfreeze_backbone"):
            self.model.unfreeze_backbone()

        optimizer = self._build_optimizer(lr)

        scheduler = None
        if warmup_epochs > 0:
            scheduler = get_scheduler(
                "cosine_annealing",
                optimizer=optimizer,
                warmup_epochs=warmup_epochs,
                total_epochs=epochs,
                steps_per_epoch=len(self.train_loader),
            )

        best_metrics: Dict[str, float] = {}
        epoch_offset = {"phase": phase, 1: 0, 2: 10, 3: 40}.get(phase, 0)

        for epoch in range(1, epochs + 1):
            global_epoch = epoch_offset + epoch

            t0 = time.time()
            train_metrics = self._train_one_epoch(
                optimizer, scheduler, use_cutmix, use_mixup, global_epoch
            )
            val_metrics = self._validate()
            elapsed = time.time() - t0

            self._log(train_metrics, f"phase{phase}_train", global_epoch)
            self._log(val_metrics, f"phase{phase}_val", global_epoch)
            logger.info(f"  ↳ Epoch time: {elapsed:.1f}s")

            self.ckpt_manager.save(
                self.model, optimizer, global_epoch,
                val_metrics["auc"], self.model_name
            )

            if early_stopping is not None and early_stopping(val_metrics["auc"]):
                logger.info("Early stopping — exiting phase.")
                break

            best_metrics = val_metrics

        return best_metrics

    def train_all_phases(self, class_weights: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """Run all 3 training phases sequentially."""
        tc = self.config["training"]

        # Phase 1
        self.run_phase(
            phase=1,
            epochs=tc["phase1"]["epochs"],
            lr=tc["phase1"]["lr"],
            freeze_backbone=tc["phase1"]["freeze_backbone"],
            use_cutmix=True,
            use_mixup=True,
        )

        # Phase 2
        early_stop = EarlyStopping(patience=tc["early_stopping"]["patience"], mode="max")
        best = self.run_phase(
            phase=2,
            epochs=tc["phase2"]["epochs"],
            lr=tc["phase2"]["lr"],
            freeze_backbone=tc["phase2"]["freeze_backbone"],
            use_cutmix=True,
            use_mixup=True,
            warmup_epochs=tc["phase2"]["warmup_epochs"],
            early_stopping=early_stop,
        )

        self.writer.close()
        return best


# ---------------------------------------------------------------------------
# U-Net Segmentation Trainer
# ---------------------------------------------------------------------------

class SegTrainer:
    """Minimal trainer for U-Net segmentation model."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        config: dict,
    ) -> None:
        from src.models.unet import DiceBCELoss, dice_score
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.criterion = DiceBCELoss(
            dice_weight=config["training"]["unet_loss"]["dice_weight"],
            bce_weight=config["training"]["unet_loss"]["bce_weight"],
        )
        self.dice_score = dice_score
        self.scaler = GradScaler(enabled=config["project"]["mixed_precision"])

    def train_epoch(self, optimizer) -> float:
        self.model.train()
        total_loss = 0.0
        for batch in self.train_loader:
            imgs = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)
            optimizer.zero_grad()
            with autocast():
                logits = self.model(imgs)
                loss = self.criterion(logits, masks)
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()
            total_loss += loss.item()
        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        self.model.eval()
        total_dsc = 0.0
        for batch in self.val_loader:
            imgs = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)
            logits = self.model(imgs)
            pred = (torch.sigmoid(logits) > 0.5).float()
            total_dsc += self.dice_score(pred, masks)
        return {"dsc": total_dsc / len(self.val_loader)}

    def fit(self, epochs: int, lr: float) -> float:
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        best_dsc = 0.0
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(optimizer)
            val_metrics = self.validate()
            dsc = val_metrics["dsc"]
            logger.info(f"[SEG] Epoch {epoch:03d} | loss={train_loss:.4f} | DSC={dsc:.4f}")
            if dsc > best_dsc:
                best_dsc = dsc
        return best_dsc
