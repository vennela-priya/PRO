"""
confusion_viz.py
================
Visualisation utilities:
- Confusion matrix heatmap
- ROC curves (per-class + macro)
- Precision-Recall curves
- Per-class metric bar chart
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.preprocessing import label_binarize


CLASS_NAMES = ["meningioma", "glioma", "pituitary"]
PALETTE = ["#2196F3", "#4CAF50", "#FF5722"]


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str] = CLASS_NAMES,
    save_path: Optional[str] = None,
    normalize: bool = True,
    title: str = "Confusion Matrix",
) -> plt.Figure:
    """Plot normalized or raw confusion matrix."""
    if normalize:
        cm_display = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fmt = ".2%"
    else:
        cm_display = cm
        fmt = "d"

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_roc_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str] = CLASS_NAMES,
    save_path: Optional[str] = None,
    title: str = "ROC Curves",
) -> plt.Figure:
    """Plot per-class + macro-average ROC curves."""
    num_classes = len(class_names)
    y_bin = label_binarize(y_true, classes=list(range(num_classes)))

    fig, ax = plt.subplots(figsize=(8, 6))

    macro_tpr_list = []
    mean_fpr = np.linspace(0, 1, 200)

    for i, (name, color) in enumerate(zip(class_names, PALETTE)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        auc = roc_auc_score(y_bin[:, i], y_prob[:, i])
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={auc:.3f})")
        macro_tpr_list.append(np.interp(mean_fpr, fpr, tpr))

    macro_tpr = np.mean(macro_tpr_list, axis=0)
    macro_auc = roc_auc_score(y_bin, y_prob, average="macro")
    ax.plot(mean_fpr, macro_tpr, color="black", lw=2.5,
            linestyle="--", label=f"Macro avg (AUC={macro_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k:", lw=1, alpha=0.4)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_pr_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str] = CLASS_NAMES,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot precision-recall curves per class."""
    num_classes = len(class_names)
    y_bin = label_binarize(y_true, classes=list(range(num_classes)))

    fig, ax = plt.subplots(figsize=(8, 6))

    for i, (name, color) in enumerate(zip(class_names, PALETTE)):
        precision, recall, _ = precision_recall_curve(y_bin[:, i], y_prob[:, i])
        ap = average_precision_score(y_bin[:, i], y_prob[:, i])
        ax.plot(recall, precision, color=color, lw=2, label=f"{name} (AP={ap:.3f})")

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curves", fontsize=14, fontweight="bold")
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_metric_bars(
    per_class_metrics: Dict[str, Dict[str, float]],
    metrics: List[str] = ["sensitivity", "specificity", "f1", "auc"],
    save_path: Optional[str] = None,
    title: str = "Per-Class Metrics",
) -> plt.Figure:
    """Grouped bar chart of per-class metrics."""
    class_names = list(per_class_metrics.keys())
    n_metrics = len(metrics)
    n_classes = len(class_names)
    x = np.arange(n_classes)
    width = 0.8 / n_metrics

    fig, ax = plt.subplots(figsize=(10, 6))
    metric_colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]

    for i, (metric, color) in enumerate(zip(metrics, metric_colors)):
        values = [per_class_metrics[cls].get(metric, 0.0) for cls in class_names]
        offset = (i - n_metrics / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=metric.capitalize(), color=color, alpha=0.85)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(class_names, fontsize=12)
    ax.set_ylim([0, 1.10])
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.4)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


def save_all_plots(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    metrics_dict: Dict,
    output_dir: str = "results",
    prefix: str = "ensemble",
) -> None:
    """Save all evaluation visualisations to disk."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cm = np.array(metrics_dict["confusion_matrix"])
    plot_confusion_matrix(cm, save_path=str(out / f"{prefix}_confusion_matrix.png"))
    plot_roc_curves(y_true, y_prob, save_path=str(out / f"{prefix}_roc_curves.png"))
    plot_pr_curves(y_true, y_prob, save_path=str(out / f"{prefix}_pr_curves.png"))
    plot_metric_bars(
        metrics_dict["per_class"],
        save_path=str(out / f"{prefix}_per_class_metrics.png"),
        title=f"{prefix.capitalize()} Per-Class Metrics",
    )
    print(f"Saved evaluation plots to {out}/")
