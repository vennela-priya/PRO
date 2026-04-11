"""
final_test.py
=============
Open the locked test holdout and report final unbiased metrics.
Run ONLY after training is complete and val accuracy is satisfactory.

Usage:  python final_test.py
"""
from __future__ import annotations
import logging, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s — %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("final_test")


def main():
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    nc     = config["data"]["num_classes"]
    names  = config["data"]["class_names"]
    size   = config["data"]["image_size"]
    seed   = config["project"]["seed"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── Load holdout CSV ───────────────────────────────────────────────────
    holdout_csv = Path(config["data"]["holdout_csv"])
    if not holdout_csv.exists():
        logger.error(f"Holdout CSV not found: {holdout_csv}\nRun: python prepare_data.py")
        sys.exit(1)

    test_df = pd.read_csv(holdout_csv)
    logger.info(f"Test holdout: {len(test_df)} images")
    logger.info(f"{test_df['class_name'].value_counts().to_string()}")

    # ── Rebuild models & load weights ─────────────────────────────────────
    from train import (MRIDataset, val_tfm, make_efficientnet,
                       make_resnet_cbam, make_densenet, Ensemble, evaluate)

    model_fns = {"efficientnet": make_efficientnet,
                 "resnet_cbam":  make_resnet_cbam,
                 "densenet":     make_densenet}

    loaded = {}
    for mname, mfn in model_fns.items():
        ckpt_dir = Path("checkpoints") / mname
        ckpts = sorted(ckpt_dir.glob("best_*.pt"), reverse=True) if ckpt_dir.exists() else []
        if not ckpts:
            logger.error(f"No checkpoint found for {mname}. Run train.py first.")
            sys.exit(1)
        m = mfn(nc, pretrained=False).to(device)
        ckpt = torch.load(ckpts[0], map_location=device)
        m.load_state_dict(ckpt["model_state_dict"])
        m.eval()
        loaded[mname] = m
        logger.info(f"Loaded {mname} from {ckpts[0].name}  (val_auc={ckpt.get('auc',0):.4f})")

    ens_weights = {"efficientnet": 0.4, "resnet_cbam": 0.3, "densenet": 0.3}
    ensemble = Ensemble(loaded, ens_weights).to(device)

    ens_ckpt = Path("checkpoints/ensemble_final.pt")
    if ens_ckpt.exists():
        ensemble.load_state_dict(torch.load(ens_ckpt, map_location=device))
        logger.info("Loaded calibrated ensemble temperatures.")

    # ── Test DataLoader ────────────────────────────────────────────────────
    test_ds = MRIDataset(test_df, val_tfm(size))
    nw = config["data"]["num_workers"]
    pm = config["data"]["pin_memory"]
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False,
                             num_workers=nw, pin_memory=pm)

    # ── Per-model evaluation ───────────────────────────────────────────────
    rows = []
    for mname, m in loaded.items():
        res = evaluate(m, test_loader, device, nc, names, label=f"{mname} (TEST)")
        rows.append({"model":mname,"split":"test",
                     "accuracy":res["accuracy"],"auc":res["auc"],
                     "sensitivity":res["sensitivity"],"specificity":res["specificity"],
                     "f1_macro":res["f1_macro"]})

    # ── Ensemble evaluation ────────────────────────────────────────────────
    res = evaluate(ensemble, test_loader, device, nc, names, label="ENSEMBLE (TEST)")
    rows.append({"model":"ensemble","split":"test",
                 "accuracy":res["accuracy"],"auc":res["auc"],
                 "sensitivity":res["sensitivity"],"specificity":res["specificity"],
                 "f1_macro":res["f1_macro"]})

    # ── Bootstrap CI ───────────────────────────────────────────────────────
    from src.evaluation.bootstrap_ci import bootstrap_ci, format_ci_table
    logger.info("Computing 95% Bootstrap CI (n=1000)…")
    ci = bootstrap_ci(res["y_true"], res["y_pred"], res["y_prob"],
                      n_iterations=1000, num_classes=nc)
    print("\n" + format_ci_table(ci))

    # ── Plots ──────────────────────────────────────────────────────────────
    from src.evaluation.confusion_viz import save_all_plots
    Path("results").mkdir(exist_ok=True)
    save_all_plots(res["y_true"], res["y_pred"], res["y_prob"],
                   {"confusion_matrix": res["cm"].tolist(),
                    "per_class": {n: {"sensitivity":0,"specificity":0,"f1":0,"auc":0}
                                  for n in names}},
                   output_dir="results", prefix="final_test")

    # ── Save CSV ───────────────────────────────────────────────────────────
    pd.DataFrame(rows).to_csv("results/final_test_metrics.csv", index=False)
    logger.info("Saved results/final_test_metrics.csv")

    # ── Base paper comparison ──────────────────────────────────────────────
    base = {"accuracy":0.971,"auc":0.980,"sensitivity":0.919,"specificity":0.980}
    ens = rows[-1]
    print(f"\n{'='*55}")
    print(f"  FINAL TEST SET — ENSEMBLE RESULTS")
    print(f"{'='*55}")
    print(f"  Accuracy    : {ens['accuracy']:.4f}  ({ens['accuracy']*100:.2f}%)")
    print(f"  AUC (macro) : {ens['auc']:.4f}")
    print(f"  Sensitivity : {ens['sensitivity']:.4f}  ({ens['sensitivity']*100:.2f}%)")
    print(f"  Specificity : {ens['specificity']:.4f}  ({ens['specificity']*100:.2f}%)")
    print(f"  F1 (macro)  : {ens['f1_macro']:.4f}")
    print(f"\n  vs. Base Paper (Amin et al. SVM+GLCM):")
    for m, bv in base.items():
        ov = ens.get(m, 0)
        print(f"    {m:<15}: {bv:.3f} → {ov:.4f}  Δ={ov-bv:+.4f}")
    print(f"{'='*55}")
    print("\nAll plots saved to results/")


if __name__ == "__main__":
    main()
