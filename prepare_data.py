"""
prepare_data.py
===============
Scans the MainProject/Project Training & Testing folders,
builds metadata CSVs, and locks the test holdout.

Structure expected:
  Training/
    glioma/      meningioma/    notumor/    pituitary/
  Testing/
    glioma/      meningioma/    notumor/    pituitary/

Outputs:
  data/processed/trainval.csv      → 80% train / 20% val (used in train.py)
  data/processed/test_holdout.csv  → Testing folder (locked, used in final_test.py)
"""

from __future__ import annotations
import logging
import sys
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s — %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("prepare_data")


def scan_folder(root: Path, class_names: list[str]) -> pd.DataFrame:
    """Scan a folder of class subdirectories → DataFrame."""
    records = []
    for label, cls in enumerate(class_names):
        cls_dir = root / cls
        if not cls_dir.exists():
            logger.warning(f"Class folder not found: {cls_dir}")
            continue
        for img_path in cls_dir.glob("*"):
            if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                records.append({
                    "path": str(img_path),
                    "label": label,
                    "class_name": cls,
                    "patient_id": f"{cls}_{img_path.stem}",
                    "source": root.name,
                })
    return pd.DataFrame(records)


def main() -> None:
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    train_dir  = Path(config["data"]["train_dir"])
    test_dir   = Path(config["data"]["test_dir"])
    class_names = config["data"]["class_names"]
    val_split   = config["data"]["val_split"]
    seed        = config["project"]["seed"]
    out_dir     = Path(config["data"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    if not train_dir.exists():
        logger.error(f"Training folder not found: {train_dir}"); sys.exit(1)
    if not test_dir.exists():
        logger.error(f"Testing folder not found: {test_dir}"); sys.exit(1)

    # ── Build DataFrames ───────────────────────────────────────────────────
    trainval_df = scan_folder(train_dir, class_names)
    holdout_df  = scan_folder(test_dir,  class_names)

    logger.info(f"Training folder — {len(trainval_df)} images found")
    logger.info(f"{trainval_df['class_name'].value_counts().to_string()}")
    logger.info(f"Testing folder  — {len(holdout_df)} images found")
    logger.info(f"{holdout_df['class_name'].value_counts().to_string()}")

    # ── Split trainval into train/val ──────────────────────────────────────
    train_df, val_df = train_test_split(
        trainval_df,
        test_size=val_split,
        stratify=trainval_df["label"],
        random_state=seed,
    )
    train_df = train_df.copy()
    train_df["split"] = "train"
    val_df = val_df.copy()
    val_df["split"] = "val"
    trainval_tagged = pd.concat([train_df, val_df]).reset_index(drop=True)

    # ── Save CSVs ──────────────────────────────────────────────────────────
    trainval_tagged.to_csv(out_dir / "trainval.csv", index=False)
    holdout_df.to_csv(out_dir / "test_holdout.csv", index=False)

    logger.info(f"""
{'='*60}
  DATASET PREPARATION COMPLETE
{'='*60}
  Train      : {len(train_df)} images
  Val        : {len(val_df)} images
  Test (held): {len(holdout_df)} images  ← LOCKED

  Saved:
    {out_dir}/trainval.csv
    {out_dir}/test_holdout.csv

  Next → python train.py
{'='*60}""")


if __name__ == "__main__":
    main()
