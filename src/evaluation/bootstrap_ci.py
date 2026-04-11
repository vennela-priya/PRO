"""
bootstrap_ci.py
===============
Bootstrap 95% confidence intervals (n=1000) for classification metrics.
Reports CI for: accuracy, AUC, sensitivity, specificity, F1.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.utils import resample

from src.evaluation.metrics import compute_specificity

logger = logging.getLogger(__name__)


def _sensitivity(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    """Macro sensitivity = mean recall per class."""
    specs = []
    for c in range(num_classes):
        tp = ((y_true == c) & (y_pred == c)).sum()
        fn = ((y_true == c) & (y_pred != c)).sum()
        specs.append(tp / (tp + fn + 1e-8))
    return float(np.mean(specs))


def _specificity_macro(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    return float(compute_specificity(y_true, y_pred, num_classes).mean())


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    n_iterations: int = 1000,
    confidence: float = 0.95,
    num_classes: int = 3,
    random_state: int = 42,
) -> Dict[str, Dict[str, float]]:
    """
    Compute bootstrap confidence intervals for key metrics.

    Parameters
    ----------
    y_true       : true labels [N]
    y_pred       : predicted labels [N]
    y_prob       : class probabilities [N, C]
    n_iterations : number of bootstrap samples
    confidence   : confidence level (default 0.95 → 95% CI)
    num_classes  : number of classes

    Returns
    -------
    dict mapping metric_name → {"mean": float, "lower": float, "upper": float, "std": float}
    """
    alpha = 1.0 - confidence
    rng = np.random.RandomState(random_state)
    n = len(y_true)

    metric_samples: Dict[str, List[float]] = {
        "accuracy": [],
        "auc": [],
        "sensitivity": [],
        "specificity": [],
        "f1_macro": [],
    }

    logger.info(f"Running bootstrap CI with {n_iterations} iterations…")

    for i in range(n_iterations):
        idx = rng.randint(0, n, n)
        bt_true = y_true[idx]
        bt_pred = y_pred[idx]
        bt_prob = y_prob[idx]

        # Skip degenerate samples (missing class)
        if len(np.unique(bt_true)) < 2:
            continue

        metric_samples["accuracy"].append(accuracy_score(bt_true, bt_pred))
        metric_samples["f1_macro"].append(
            f1_score(bt_true, bt_pred, average="macro", zero_division=0)
        )
        metric_samples["sensitivity"].append(_sensitivity(bt_true, bt_pred, num_classes))
        metric_samples["specificity"].append(_specificity_macro(bt_true, bt_pred, num_classes))

        try:
            auc = roc_auc_score(bt_true, bt_prob, multi_class="ovr", average="macro")
            metric_samples["auc"].append(auc)
        except ValueError:
            pass

    results = {}
    lower_pct = 100 * alpha / 2
    upper_pct = 100 * (1 - alpha / 2)

    for metric, samples in metric_samples.items():
        if not samples:
            results[metric] = {"mean": float("nan"), "lower": float("nan"),
                               "upper": float("nan"), "std": float("nan")}
            continue

        arr = np.array(samples)
        results[metric] = {
            "mean": float(arr.mean()),
            "lower": float(np.percentile(arr, lower_pct)),
            "upper": float(np.percentile(arr, upper_pct)),
            "std": float(arr.std()),
        }

    _log_ci(results, confidence)
    return results


def _log_ci(results: Dict, confidence: float) -> None:
    pct = int(confidence * 100)
    logger.info(f"\n{'='*55}")
    logger.info(f"  Bootstrap {pct}% Confidence Intervals (n=1000)")
    logger.info(f"  {'Metric':<15} {'Mean':>7} {'Lower':>7} {'Upper':>7} {'Std':>7}")
    logger.info(f"  {'-'*50}")
    for metric, vals in results.items():
        logger.info(
            f"  {metric:<15} {vals['mean']:>7.4f} {vals['lower']:>7.4f} "
            f"{vals['upper']:>7.4f} {vals['std']:>7.4f}"
        )
    logger.info(f"{'='*55}\n")


def format_ci_table(results: Dict[str, Dict[str, float]]) -> str:
    """Return a formatted markdown table of CI results."""
    lines = ["| Metric | Mean | 95% CI | Std |", "|---|---|---|---|"]
    for metric, vals in results.items():
        ci = f"[{vals['lower']:.4f}, {vals['upper']:.4f}]"
        lines.append(f"| {metric} | {vals['mean']:.4f} | {ci} | {vals['std']:.4f} |")
    return "\n".join(lines)
