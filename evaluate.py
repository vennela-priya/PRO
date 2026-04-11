"""
evaluate.py
===========
Standalone evaluation script.
Loads ensemble from checkpoints and runs full test-set evaluation
including bootstrap CI, confusion matrix, ROC curves, and XAI samples.

Usage:
    python evaluate.py --config config.yaml --checkpoint-dir checkpoints
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s — %(message)s")
logger = logging.getLogger("evaluate")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--output-dir", default="results")
    p.add_argument("--n-xai-samples", type=int, default=5)
    p.add_argument("--gpu", default="0")
    return p.parse_args()


def main():
    args = parse_args()

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load engine ────────────────────────────────────────────────
    from src.api.inference import BrainTumorInference
    engine = BrainTumorInference.from_config(args.config, args.checkpoint_dir, device)

    # ── Build test loader ──────────────────────────────────────────
    import pandas as pd
    from src.data.dataset import build_classification_loaders, BrainTumorDataset
    from src.data.augmentation import build_val_transforms
    from src.data.preprocessing import patient_stratified_split

    df = pd.read_csv(Path(config["data"]["processed_dir"]) / "metadata.csv")
    _, _, test_df = patient_stratified_split(df, seed=config["project"]["seed"])

    val_tfm = build_val_transforms(image_size=config["data"]["image_size"])
    _, _, test_loader = build_classification_loaders(
        df, df, test_df,
        train_transform=val_tfm,
        val_transform=val_tfm,
        batch_size=32,
        num_workers=config["data"]["num_workers"],
    )

    # ── Metrics ────────────────────────────────────────────────────
    from src.evaluation.metrics import predict_loader, compute_all_metrics
    from src.evaluation.bootstrap_ci import bootstrap_ci, format_ci_table
    from src.evaluation.confusion_viz import save_all_plots
    from src.data.augmentation import build_tta_transforms

    tta_transforms = build_tta_transforms(image_size=config["data"]["image_size"])

    logger.info("Running inference with TTA…")
    y_true, y_pred, y_prob = predict_loader(
        engine.ensemble, test_loader, device, tta_transforms=tta_transforms
    )

    metrics = compute_all_metrics(
        y_true, y_pred, y_prob,
        num_classes=config["data"]["num_classes"],
        class_names=config["data"]["class_names"],
    )

    logger.info("\nComputing 95% bootstrap CI (n=1000)…")
    ci = bootstrap_ci(y_true, y_pred, y_prob, n_iterations=1000)
    print("\n" + format_ci_table(ci))

    save_all_plots(y_true, y_pred, y_prob, metrics, output_dir=args.output_dir)

    # ── XAI on 5 test cases ───────────────────────────────────────
    logger.info(f"\nGenerating XAI for {args.n_xai_samples} test cases…")
    import cv2
    from src.xai.visualizer import XAIVisualizer
    from src.data.dataset import BrainTumorDataset

    test_ds = BrainTumorDataset(test_df, transform=val_tfm)

    # Select: 1 per class + 1 FP + 1 FN
    sample_indices = []
    for cls in range(config["data"]["num_classes"]):
        cls_idx = np.where(y_true == cls)[0]
        correct = cls_idx[y_pred[cls_idx] == cls]
        if len(correct):
            sample_indices.append(int(correct[0]))

    fp_idx = np.where((y_pred != y_true) & (y_pred == 0))[0]
    if len(fp_idx):
        sample_indices.append(int(fp_idx[0]))
    fn_idx = np.where((y_pred != y_true) & (y_true == 0))[0]
    if len(fn_idx):
        sample_indices.append(int(fn_idx[0]))

    sample_indices = sample_indices[:args.n_xai_samples]

    xai = XAIVisualizer(
        models={name: engine.ensemble.models[name] for name in engine.ensemble.model_names},
        device=device,
        xai_config=config["xai"],
        class_names=config["data"]["class_names"],
    )

    for i, idx in enumerate(sample_indices):
        sample = test_ds[idx]
        img_tensor = sample["image"].unsqueeze(0)
        img_np = cv2.cvtColor(
            cv2.imread(test_ds.df.iloc[idx]["path"]), cv2.COLOR_BGR2RGB
        )
        xai.explain_sample(
            input_tensor=img_tensor,
            original_image=img_np,
            true_label=int(y_true[idx]),
            target_class=None,
            sample_id=f"test_case_{i+1}",
        )

    xai.cleanup()
    logger.info("\nEvaluation complete. Results saved to results/")


if __name__ == "__main__":
    main()
