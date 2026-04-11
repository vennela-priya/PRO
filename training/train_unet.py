"""
training/train_unet.py
======================
Complete U-Net training pipeline for brain tumour segmentation.

Quickstart
----------
# With real masks (BraTS or hand-labelled):
python training/train_unet.py \
    --images_dir  data/images \
    --masks_dir   data/masks \
    --epochs 50   --batch_size 8

# With pseudo masks (Grad-CAM derived):
python training/train_unet.py \
    --images_dir      data/Testing \
    --pseudo_cache    data/pseudo_masks \
    --epochs 30       --batch_size 8 --lr 3e-4

# Generate pseudo masks first (run once):
python -c "
import sys; sys.path.insert(0,'.')
from app import load_model, preprocess
from inference.gradcam import EnsembleGradCAM
from segmentation.pseudo_masks import build_pseudo_dataset
import numpy as np, torch

model, mode = load_model()
engine = EnsembleGradCAM(model, mode)

def gradcam_fn(img_np):
    t = preprocess(img_np)
    pred = {'class':'glioma','class_label':'Glioma','confidence':0.9,
            'probabilities':{'glioma':0.9,'meningioma':0.03,'notumor':0.04,'pituitary':0.03},
            'individual':{}}
    r = engine.generate(t, pred, img_np)
    return r['cam']

n = build_pseudo_dataset('data/Testing', 'data/pseudo_masks', gradcam_fn)
print(f'Generated {n} masks')
"

Features
--------
* BCEDiceLoss (configurable alpha)
* ReduceLROnPlateau scheduler
* Best-model checkpoint saved by val Dice
* Per-epoch visualisation saved to <out_dir>/vis/
* Training curves saved to <out_dir>/curves.png
* Final model exported to <out_dir>/unet_final.pt
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

# Make sure project root is on PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from segmentation.unet    import UNet
from segmentation.losses  import BCEDiceLoss
from segmentation.metrics import compute_all_metrics
from segmentation.dataset import BrainSegDataset

# ── Logging ───────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("unet_train.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("UNetTrainer")


# ── CLI ───────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train U-Net for brain tumour segmentation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Data
    p.add_argument("--images_dir",    required=True,
                   help="Directory containing MRI images")
    p.add_argument("--masks_dir",     default=None,
                   help="Directory containing binary masks (real labels)")
    p.add_argument("--pseudo_cache",  default=None,
                   help="Directory of pseudo masks (.npy or .png) from Grad-CAM")
    p.add_argument("--image_size",    type=int,   default=224,
                   help="Resize all images to image_size × image_size")
    p.add_argument("--val_split",     type=float, default=0.15,
                   help="Fraction of data for validation")

    # Model
    p.add_argument("--in_channels",   type=int,   default=1,
                   choices=[1, 3],
                   help="1=grayscale input, 3=RGB input")
    p.add_argument("--base_filters",  type=int,   default=32,
                   help="Filters at first conv level (doubles each level). "
                        "32 → ~8 M params, 64 → ~31 M params")
    p.add_argument("--bilinear",      action="store_true", default=True,
                   help="Use bilinear upsampling (True) vs transposed conv (False)")

    # Training
    p.add_argument("--epochs",        type=int,   default=50)
    p.add_argument("--batch_size",    type=int,   default=8)
    p.add_argument("--lr",            type=float, default=1e-4)
    p.add_argument("--bce_alpha",     type=float, default=0.5,
                   help="Weight for BCE in BCEDiceLoss (rest goes to Dice)")
    p.add_argument("--threshold",     type=float, default=0.5,
                   help="Binarisation threshold for metrics evaluation")
    p.add_argument("--num_workers",   type=int,   default=0)
    p.add_argument("--seed",          type=int,   default=42)

    # Output
    p.add_argument("--out_dir",       default="./checkpoints/unet",
                   help="Directory to save checkpoints and visualisations")
    p.add_argument("--vis_every",     type=int,   default=5,
                   help="Save prediction images every N epochs (0=never)")
    p.add_argument("--device",        default="cuda" if torch.cuda.is_available()
                                                else "cpu")

    args = p.parse_args()

    if args.masks_dir is None and args.pseudo_cache is None:
        p.error("Provide at least one of --masks_dir or --pseudo_cache")

    return args


# ── Visualisation ─────────────────────────────────────────────────────────

_DARK = "#0f172a"
_CYAN = "#00d4ff"


def denorm(t: torch.Tensor) -> np.ndarray:
    """
    Reverse ImageNet normalisation and return float32 RGB [0,1].
    t: (C, H, W) tensor
    """
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr  = t.cpu().float().numpy().transpose(1, 2, 0)   # (H,W,C)
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    arr  = arr * std + mean
    return arr.clip(0, 1)


def save_vis(
    model:     UNet,
    loader:    DataLoader,
    epoch:     int,
    out_dir:   Path,
    device:    torch.device,
    threshold: float,
    n_samples: int = 4,
) -> None:
    """Save a grid of (input | ground-truth | prediction) for quick inspection."""
    model.eval()
    samples: list[tuple] = []

    with torch.no_grad():
        for batch in loader:
            imgs  = batch["image"].to(device)
            masks = batch["mask"].to(device)
            logits = model(imgs)
            probs  = torch.sigmoid(logits)
            preds  = (probs >= threshold).float()

            for i in range(min(n_samples - len(samples), imgs.size(0))):
                img_np  = denorm(imgs[i])
                gt_np   = masks[i, 0].cpu().numpy()
                pred_np = probs[i, 0].cpu().numpy()
                samples.append((img_np, gt_np, pred_np))
            if len(samples) >= n_samples:
                break

    if not samples:
        return

    fig, axes = plt.subplots(
        len(samples), 3,
        figsize=(9, 3 * len(samples)),
        facecolor=_DARK,
    )
    if len(samples) == 1:
        axes = [axes]   # ensure iterable rows

    col_titles = ["MRI Input", "Ground Truth", "Predicted Prob"]
    for row, (img, gt, pred) in enumerate(samples):
        for col, (arr, title, cmap) in enumerate(
            [(img, col_titles[0], None),
             (gt,  col_titles[1], "gray"),
             (pred,col_titles[2], "RdYlGn")]):
            ax = axes[row][col]
            ax.imshow(arr, cmap=cmap, vmin=0, vmax=1)
            if row == 0:
                ax.set_title(title, color=_CYAN, fontsize=9, fontweight="bold")
            ax.axis("off")

    plt.suptitle(f"Epoch {epoch:03d}", color="white", fontsize=11)
    plt.tight_layout(pad=0.4)

    vis_dir = out_dir / "vis"
    vis_dir.mkdir(exist_ok=True)
    fig.savefig(vis_dir / f"ep{epoch:03d}.png", dpi=90,
                bbox_inches="tight", facecolor=_DARK)
    plt.close(fig)


def plot_curves(history: dict, out_dir: Path) -> None:
    """Save training / validation loss + dice + IoU curves."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), facecolor=_DARK)
    for ax, (key, color) in zip(
        axes,
        [("loss", "#ef4444"), ("dice", "#10b981"), ("iou", "#3b82f6")],
    ):
        ax.plot(history[f"tr_{key}"],  color=color, lw=2, label="Train")
        ax.plot(history[f"val_{key}"], color=color, lw=2, ls="--",
                alpha=0.7, label="Val")
        ax.set_title(key.upper(), color="white", fontsize=11)
        ax.set_facecolor(_DARK)
        ax.tick_params(colors="#94a3b8")
        ax.legend(facecolor="#1e293b", labelcolor="white", fontsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor("#334155")

    plt.suptitle("U-Net Training Curves", color="white", fontsize=13)
    plt.tight_layout()
    fig.savefig(out_dir / "curves.png", dpi=100,
                bbox_inches="tight", facecolor=_DARK)
    plt.close(fig)


# ── One epoch routines ────────────────────────────────────────────────────

def run_epoch(
    model:     UNet,
    loader:    DataLoader,
    criterion: BCEDiceLoss,
    optimizer: torch.optim.Optimizer | None,
    device:    torch.device,
    threshold: float,
    train:     bool,
) -> dict[str, float]:
    """Run one training or validation epoch and return aggregated metrics."""
    model.train() if train else model.eval()

    totals = {"loss": 0.0, "dice": 0.0, "iou": 0.0, "px_acc": 0.0}
    n = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            imgs  = batch["image"].to(device)
            masks = batch["mask"].to(device)

            logits = model(imgs)
            loss, _ = criterion(logits, masks)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            bs     = imgs.size(0)
            met    = compute_all_metrics(logits.detach(), masks, threshold)
            totals["loss"]   += loss.item() * bs
            totals["dice"]   += met["dice"]   * bs
            totals["iou"]    += met["iou"]    * bs
            totals["px_acc"] += met["pixel_acc"] * bs
            n += bs

    return {k: v / max(n, 1) for k, v in totals.items()}


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    args   = parse_args()
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    log.info("=" * 62)
    log.info("  U-Net  Brain Tumour Segmentation Trainer")
    log.info(f"  Device      : {device}")
    log.info(f"  Image size  : {args.image_size}  |  Channels: {args.in_channels}")
    log.info(f"  Base filters: {args.base_filters}")
    log.info(f"  Epochs: {args.epochs}  |  Batch: {args.batch_size}  |  LR: {args.lr}")
    log.info(f"  BCE alpha   : {args.bce_alpha}")
    log.info("=" * 62)

    # ── Build datasets ────────────────────────────────────────────────
    def make_ds(train: bool) -> BrainSegDataset:
        return BrainSegDataset(
            image_dir    = args.images_dir,
            mask_dir     = args.masks_dir,
            pseudo_cache = args.pseudo_cache,
            image_size   = args.image_size,
            train        = train,
        )

    full_ds  = make_ds(train=True)
    n_total  = len(full_ds)
    n_val    = max(1, int(n_total * args.val_split))
    n_train  = n_total - n_val

    gen = torch.Generator().manual_seed(args.seed)
    tr_sub, va_sub = random_split(full_ds, [n_train, n_val], generator=gen)

    # Swap val subset to the non-augmented version
    val_full_ds  = make_ds(train=False)
    va_sub.dataset = val_full_ds

    log.info(f"Train: {n_train}   Val: {n_val}   Total: {n_total}")

    loader_kw = dict(
        batch_size  = args.batch_size,
        num_workers = args.num_workers,
        pin_memory  = torch.cuda.is_available(),
    )
    train_loader = DataLoader(tr_sub, shuffle=True,  **loader_kw)
    val_loader   = DataLoader(va_sub, shuffle=False, **loader_kw)

    # ── Model ─────────────────────────────────────────────────────────
    model = UNet(
        in_channels  = args.in_channels,
        base_filters = args.base_filters,
        bilinear     = args.bilinear,
    ).to(device)

    n_params = model.count_parameters()
    log.info(f"U-Net parameters: {n_params:,}")

    # ── Optimiser + loss + scheduler ──────────────────────────────────
    criterion = BCEDiceLoss(alpha=args.bce_alpha)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=6, min_lr=1e-7,
    )

    # ── Training loop ─────────────────────────────────────────────────
    history: dict[str, list[float]] = {
        k: [] for k in
        ["tr_loss", "tr_dice", "tr_iou",
         "val_loss", "val_dice", "val_iou"]
    }
    best_dice = 0.0
    best_ckpt = None

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        tr  = run_epoch(model, train_loader, criterion, optimizer,
                        device, args.threshold, train=True)
        val = run_epoch(model, val_loader,   criterion, None,
                        device, args.threshold, train=False)

        scheduler.step(val["dice"])

        for k in ["loss", "dice", "iou"]:
            history[f"tr_{k}"].append(tr[k])
            history[f"val_{k}"].append(val[k])

        elapsed = time.time() - t0
        lr_now  = optimizer.param_groups[0]["lr"]
        log.info(
            f"Ep {epoch:03d}/{args.epochs}  "
            f"TR  loss={tr['loss']:.4f}  dice={tr['dice']:.4f}  iou={tr['iou']:.4f}  |  "
            f"VAL loss={val['loss']:.4f}  dice={val['dice']:.4f}  iou={val['iou']:.4f}  "
            f"lr={lr_now:.2e}  [{elapsed:.1f}s]"
        )

        # ── Save best checkpoint ──────────────────────────────────────
        if val["dice"] > best_dice:
            best_dice = val["dice"]
            ckpt_name = f"unet_best_ep{epoch:03d}_dice{best_dice:.4f}.pt"
            ckpt_path = out_dir / ckpt_name
            torch.save({
                "epoch":            epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state":  optimizer.state_dict(),
                "val_dice":         best_dice,
                "val_iou":          val["iou"],
                "config": {
                    "in_channels":  args.in_channels,
                    "base_filters": args.base_filters,
                    "bilinear":     args.bilinear,
                    "image_size":   args.image_size,
                },
            }, ckpt_path)
            # Remove previous best to avoid clutter
            if best_ckpt and best_ckpt.exists() and best_ckpt != ckpt_path:
                best_ckpt.unlink(missing_ok=True)
            best_ckpt = ckpt_path
            log.info(f"  *** New best checkpoint: {ckpt_name}  (dice={best_dice:.4f})")

        # ── Visualise ─────────────────────────────────────────────────
        if args.vis_every > 0 and (epoch % args.vis_every == 0 or epoch == 1):
            save_vis(model, val_loader, epoch, out_dir, device,
                     args.threshold, n_samples=4)

    # ── Post-training ─────────────────────────────────────────────────
    plot_curves(history, out_dir)

    # Save final model
    final_path = out_dir / "unet_final.pt"
    torch.save({
        "epoch":            args.epochs,
        "model_state_dict": model.state_dict(),
        "val_dice":         best_dice,
        "config": {
            "in_channels":  args.in_channels,
            "base_filters": args.base_filters,
            "bilinear":     args.bilinear,
            "image_size":   args.image_size,
        },
    }, final_path)

    log.info("=" * 62)
    log.info(f"  Training complete")
    log.info(f"  Best Val Dice : {best_dice:.4f}")
    log.info(f"  Best ckpt     : {best_ckpt}")
    log.info(f"  Final model   : {final_path}")
    log.info(f"  Curves        : {out_dir / 'curves.png'}")
    log.info("=" * 62)


if __name__ == "__main__":
    main()
