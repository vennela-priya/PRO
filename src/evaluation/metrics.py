"""
metrics.py
==========
Comprehensive evaluation metrics for brain tumor classification:
- Per-class accuracy, precision, recall (sensitivity), specificity, F1
- Macro / weighted averages
- AUC-ROC per class and macro
- Confusion matrix
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

CLASS_NAMES = ["meningioma", "glioma", "pituitary"]


# ---------------------------------------------------------------------------
# Core compute functions
# ---------------------------------------------------------------------------

def compute_specificity(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    """Per-class specificity = TN / (TN + FP)."""
    specs = []
    for c in range(num_classes):
        binary_true = (y_true == c).astype(int)
        binary_pred = (y_pred == c).astype(int)
        tn = ((binary_true == 0) & (binary_pred == 0)).sum()
        fp = ((binary_true == 0) & (binary_pred == 1)).sum()
        specs.append(tn / (tn + fp + 1e-8))
    return np.array(specs)


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    num_classes: int = 3,
    class_names: Optional[List[str]] = None,
) -> Dict:
    """
    Compute full suite of classification metrics.

    Parameters
    ----------
    y_true     : 1-D array of true integer labels [N]
    y_pred     : 1-D array of predicted integer labels [N]
    y_prob     : 2-D array of class probabilities [N, C]
    num_classes: number of classes

    Returns
    -------
    dict with keys: accuracy, auc, sensitivity, specificity, f1, precision,
                    per_class (dict per class name), confusion_matrix
    """
    if class_names is None:
        class_names = [f"class_{i}" for i in range(num_classes)]

    acc = accuracy_score(y_true, y_pred)

    # AUC (macro OvR)
    try:
        auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
        per_class_auc = roc_auc_score(
            y_true, y_prob, multi_class="ovr", average=None
        )
    except ValueError:
        auc = float("nan")
        per_class_auc = [float("nan")] * num_classes

    sensitivity = recall_score(y_true, y_pred, average=None, zero_division=0)
    macro_sensitivity = sensitivity.mean()

    specificity = compute_specificity(y_true, y_pred, num_classes)
    macro_specificity = specificity.mean()

    precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    per_class = {}
    for i, name in enumerate(class_names):
        per_class[name] = {
            "sensitivity": float(sensitivity[i]),
            "specificity": float(specificity[i]),
            "precision": float(precision[i]),
            "f1": float(f1_per_class[i]),
            "auc": float(per_class_auc[i]) if not isinstance(per_class_auc, float) else float("nan"),
        }

    results = {
        "accuracy": float(acc),
        "auc": float(auc),
        "sensitivity": float(macro_sensitivity),    # = macro recall
        "specificity": float(macro_specificity),
        "f1_macro": float(macro_f1),
        "f1_weighted": float(weighted_f1),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }

    _log_metrics(results, class_names)
    return results


def _log_metrics(metrics: Dict, class_names: List[str]) -> None:
    logger.info("\n" + "=" * 50)
    logger.info(f"  Accuracy   : {metrics['accuracy']:.4f}")
    logger.info(f"  AUC (macro): {metrics['auc']:.4f}")
    logger.info(f"  Sensitivity: {metrics['sensitivity']:.4f}")
    logger.info(f"  Specificity: {metrics['specificity']:.4f}")
    logger.info(f"  F1 (macro) : {metrics['f1_macro']:.4f}")
    logger.info("\n  Per-class breakdown:")
    for name, vals in metrics["per_class"].items():
        logger.info(f"    {name:15s}: sens={vals['sensitivity']:.3f}  "
                    f"spec={vals['specificity']:.3f}  "
                    f"f1={vals['f1']:.3f}  auc={vals['auc']:.3f}")
    logger.info("=" * 50)


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    tta_transforms: Optional[List] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run inference over a DataLoader.

    Parameters
    ----------
    model          : trained nn.Module (or ensemble)
    loader         : test DataLoader
    device         : compute device
    tta_transforms : list of TTA albumentations transforms (optional)

    Returns
    -------
    y_true : [N]    true labels
    y_pred : [N]    predicted labels
    y_prob : [N, C] class probabilities
    """
    model.eval()
    all_true, all_probs = [], []

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"]

        if tta_transforms:
            # Average predictions over TTA variants
            tta_probs = []
            imgs_np = images.cpu().numpy().transpose(0, 2, 3, 1)  # [B, H, W, C]
            for tfm in tta_transforms:
                augmented = torch.stack([
                    tfm(image=(img * 255).astype(np.uint8))["image"]
                    for img in imgs_np
                ]).to(device)
                out = model(augmented)
                if isinstance(out, tuple):
                    out = out[0]
                tta_probs.append(torch.softmax(out, dim=-1))
            probs = torch.stack(tta_probs).mean(dim=0)
        else:
            out = model(images)
            if isinstance(out, tuple):
                out = out[0]
            probs = torch.softmax(out, dim=-1) if out.shape[-1] > 1 else out

        all_true.append(labels.numpy())
        all_probs.append(probs.cpu().numpy())

    y_true = np.concatenate(all_true)
    y_prob = np.concatenate(all_probs)
    y_pred = y_prob.argmax(axis=1)
    return y_true, y_pred, y_prob


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int = 3,
    class_names: Optional[List[str]] = None,
    tta_transforms: Optional[List] = None,
) -> Dict:
    """Full evaluation pipeline: inference + metrics."""
    y_true, y_pred, y_prob = predict_loader(model, loader, device, tta_transforms)
    return compute_all_metrics(y_true, y_pred, y_prob, num_classes, class_names)
