"""
refinement/iterative_refinement.py
====================================
Iterative self-training refinement of the AttentionUNet segmentation model.

Pipeline per iteration:
    1. Train the U-Net (decoder only — encoder frozen) on current pseudo masks.
    2. Run inference on all images to generate new (refined) pseudo masks.
    3. Repeat for N iterations or until Dice improvement < min_improvement.

The encoder layers (enc1-enc4, bottleneck) are frozen throughout;
only decoder layers are updated with lr=1e-5 to prevent catastrophic forgetting.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_IMG_SIZE = 224


# ── Freeze / unfreeze ─────────────────────────────────────────────────────────

def freeze_encoder(model: nn.Module) -> None:
    """
    Freeze all encoder layers (enc1-enc4, bottleneck) in-place.

    Layers named enc1, enc2, enc3, enc4, and bottleneck have their
    parameters' requires_grad set to False so they are not updated
    during fine-tuning.

    Parameters
    ----------
    model : torch.nn.Module
        AttentionUNet (or UNet) instance to modify in-place.
    """
    frozen_prefixes = ("enc1", "enc2", "enc3", "enc4", "bottleneck")
    frozen_count = 0
    for name, param in model.named_parameters():
        if any(name.startswith(pfx) for pfx in frozen_prefixes):
            param.requires_grad = False
            frozen_count += 1
    logger.info(f"[IterRef] Frozen {frozen_count} encoder parameters.")


# ── Loss ─────────────────────────────────────────────────────────────────────

def dice_bce_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1.0,
    bce_weight: float = 0.5,
) -> torch.Tensor:
    """
    Combined Dice + Binary Cross-Entropy loss.

    Parameters
    ----------
    pred       : torch.Tensor  Raw logits, shape (B, 1, H, W).
    target     : torch.Tensor  Binary float targets, shape (B, 1, H, W), values in {0, 1}.
    smooth     : float         Smoothing factor for Dice (default 1.0).
    bce_weight : float         Weight on BCE term (Dice weight = 1 - bce_weight).

    Returns
    -------
    torch.Tensor
        Scalar combined loss.
    """
    probs = torch.sigmoid(pred)
    p_f   = probs.view(probs.size(0),   -1)
    t_f   = target.view(target.size(0), -1)

    inter = (p_f * t_f).sum(dim=1)
    dice  = 1.0 - (2.0 * inter + smooth) / (p_f.sum(dim=1) + t_f.sum(dim=1) + smooth)
    dice_loss = dice.mean()

    bce_loss = F.binary_cross_entropy_with_logits(pred, target)

    return (1.0 - bce_weight) * dice_loss + bce_weight * bce_loss


# ── Dataset ──────────────────────────────────────────────────────────────────

class _SegDataset(Dataset):
    """
    Simple paired image + mask dataset for iterative refinement.

    Parameters
    ----------
    image_paths : list[str]   Paths to input MRI images.
    mask_arrays : list[np.ndarray]  Corresponding binary masks (H, W) uint8.
    img_size    : int         Target image size for resizing (default 224).
    """

    def __init__(
        self,
        image_paths: list[str],
        mask_arrays: list[np.ndarray],
        img_size: int = _IMG_SIZE,
    ) -> None:
        assert len(image_paths) == len(mask_arrays), \
            "image_paths and mask_arrays must have the same length"
        self.image_paths = image_paths
        self.mask_arrays = mask_arrays
        self.img_size    = img_size

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Load image
        bgr = cv2.imread(self.image_paths[idx])
        if bgr is None:
            # Return blank tensors if file unreadable
            s = self.img_size
            return (torch.zeros(3, s, s), torch.zeros(1, s, s))
        if bgr.ndim == 2:
            bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        s   = self.img_size
        img = cv2.resize(rgb, (s, s)).astype(np.float32) / 255.0
        img = (img - _MEAN) / _STD
        t   = torch.from_numpy(img.transpose(2, 0, 1)).float()   # (3, H, W)

        # Load mask
        mask = self.mask_arrays[idx]
        mask = cv2.resize(mask.astype(np.float32), (s, s),
                          interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0.5).astype(np.float32)
        m    = torch.from_numpy(mask).unsqueeze(0).float()        # (1, H, W)
        return t, m


# ── Inference helper ──────────────────────────────────────────────────────────

def _infer_mask(
    model: nn.Module,
    img_path: str,
    device: torch.device,
    img_size: int = _IMG_SIZE,
    threshold: float = 0.45,
) -> np.ndarray:
    """
    Run segmentation inference on a single image and return binary mask.

    Parameters
    ----------
    model     : Trained AttentionUNet model.
    img_path  : Path to the input image file.
    device    : torch.device to run inference on.
    img_size  : Spatial resolution for model input.
    threshold : Probability threshold for binarisation.

    Returns
    -------
    np.ndarray
        Binary mask, dtype=uint8, shape (H, W) matching original image size.
    """
    bgr = cv2.imread(img_path)
    if bgr is None:
        return np.zeros((224, 224), dtype=np.uint8)
    if bgr.ndim == 2:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
    H, W = bgr.shape[:2]
    rgb  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    img = cv2.resize(rgb, (img_size, img_size)).astype(np.float32) / 255.0
    img = (img - _MEAN) / _STD
    t   = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

    with torch.no_grad():
        logits = model(t)
        prob   = torch.sigmoid(logits).squeeze().cpu().numpy()

    prob_rs = cv2.resize(prob.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
    return (prob_rs >= threshold).astype(np.uint8)


# ── Dice metric ───────────────────────────────────────────────────────────────

def _dice_np(pred: np.ndarray, target: np.ndarray, smooth: float = 1.0) -> float:
    """
    Compute Dice coefficient between two binary numpy arrays.

    Parameters
    ----------
    pred   : np.ndarray  Predicted binary mask.
    target : np.ndarray  Target binary mask.
    smooth : float       Smoothing factor (default 1.0).

    Returns
    -------
    float
        Dice coefficient in [0, 1].
    """
    p = pred.flatten().astype(np.float32)
    t = target.flatten().astype(np.float32)
    inter = (p * t).sum()
    return float((2.0 * inter + smooth) / (p.sum() + t.sum() + smooth))


# ── Gaussian blob mask (synthetic GT) ─────────────────────────────────────────

def _gaussian_blob_mask(h: int, w: int) -> np.ndarray:
    """
    Create a synthetic Gaussian blob mask at the image centre.

    Parameters
    ----------
    h, w : int  Image height and width.

    Returns
    -------
    np.ndarray
        Binary uint8 mask, shape (H, W), values {0, 1}.
    """
    cx, cy = w // 2, h // 2
    radius = 0.30 * min(h, w)
    Y, X   = np.ogrid[:h, :w]
    dist   = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    blob   = (dist <= radius).astype(np.uint8)
    return blob


# ── Main iterative refinement ─────────────────────────────────────────────────

def run_iterative_refinement(
    unet_path:       str,
    image_paths:     list[str],
    mask_dir:        str,
    output_dir:      str,
    n_iterations:    int = 3,
    epochs_per_iter: int = 5,
    lr:              float = 1e-5,
    batch_size:      int = 4,
) -> dict:
    """
    Iterative self-training: train U-Net on best masks, generate new masks, repeat.

    At each iteration:
      1. Fine-tune decoder-only on current pseudo masks (encoder frozen).
      2. Infer new masks on all images.
      3. Check Dice improvement; stop early if < 0.01.

    Parameters
    ----------
    unet_path       : str    Path to AttentionUNet checkpoint (.pth).
    image_paths     : list   List of image file paths (will use first 10 max).
    mask_dir        : str    Directory to load/save pseudo masks (unused if
                             masks don't exist — synthetic Gaussian masks used).
    output_dir      : str    Where to save checkpoint and loss curve.
    n_iterations    : int    Maximum number of self-training iterations (default 3).
    epochs_per_iter : int    Training epochs per iteration (default 5).
    lr              : float  Learning rate for decoder (default 1e-5).
    batch_size      : int    DataLoader batch size (default 4).

    Returns
    -------
    dict
        Keys: best_dice (float), loss_history (list[float]),
              checkpoint_path (str), n_iterations_run (int).
    """
    # ── Setup ─────────────────────────────────────────────────────────
    out_dir  = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(__file__).resolve().parent / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"[IterRef] Device: {device}")

    # Limit to first 10 images for speed
    image_paths = list(image_paths)[:10]
    if not image_paths:
        logger.warning("[IterRef] No image paths provided.")
        return {"best_dice": 0.0, "loss_history": [], "checkpoint_path": "", "n_iterations_run": 0}

    # ── Load model ────────────────────────────────────────────────────
    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root))
    from segmentation.attention_unet import AttentionUNet

    try:
        ckpt = torch.load(unet_path, map_location="cpu", weights_only=False)
        cfg  = ckpt.get("config", {})
        model = AttentionUNet(
            in_channels  = cfg.get("in_channels",  3),
            base_filters = cfg.get("base_filters", 16),
            bilinear     = cfg.get("bilinear",     True),
        )
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        logger.info(f"[IterRef] Loaded AttentionUNet from {Path(unet_path).name}")
    except Exception as e:
        logger.warning(f"[IterRef] Could not load checkpoint: {e} — using random weights")
        model = AttentionUNet(in_channels=3, base_filters=16)

    freeze_encoder(model)
    model.to(device)

    # ── Build initial pseudo masks ────────────────────────────────────
    # Try to load from mask_dir; fall back to Gaussian blobs
    mask_dir_p = Path(mask_dir)
    current_masks: list[np.ndarray] = []

    for img_path in image_paths:
        stem      = Path(img_path).stem
        mask_file = mask_dir_p / (stem + ".npy")
        png_file  = mask_dir_p / (stem + ".png")

        if mask_file.exists():
            m = np.load(str(mask_file))
            current_masks.append((m > 0.5).astype(np.uint8))
        elif png_file.exists():
            m = cv2.imread(str(png_file), cv2.IMREAD_GRAYSCALE)
            current_masks.append((m > 127).astype(np.uint8) if m is not None
                                  else np.zeros((224, 224), dtype=np.uint8))
        else:
            # Synthetic Gaussian blob
            bgr = cv2.imread(img_path)
            h, w = (bgr.shape[:2] if bgr is not None else (224, 224))
            current_masks.append(_gaussian_blob_mask(h, w))

    # ── Optimiser (decoder parameters only) ──────────────────────────
    trainable = [p for p in model.parameters() if p.requires_grad]
    logger.info(f"[IterRef] Trainable params: {sum(p.numel() for p in trainable):,}")
    optimizer = torch.optim.Adam(trainable, lr=lr)

    # ── Iterative loop ────────────────────────────────────────────────
    loss_history: list[float] = []
    best_dice     = 0.0
    best_state    = None
    best_ckpt_path = str(ckpt_dir / "best_refined.pth")

    for iteration in range(n_iterations):
        logger.info(f"[IterRef] === Iteration {iteration + 1} / {n_iterations} ===")

        # Build DataLoader with current masks
        dataset    = _SegDataset(image_paths, current_masks, img_size=_IMG_SIZE)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                num_workers=0, drop_last=False)

        # Train for epochs_per_iter epochs
        model.train()
        iter_losses: list[float] = []

        for epoch in range(epochs_per_iter):
            epoch_loss = 0.0
            n_batches  = 0
            for imgs, masks in dataloader:
                imgs  = imgs.to(device)
                masks = masks.to(device)
                optimizer.zero_grad()
                logits = model(imgs)
                loss   = dice_bce_loss(logits, masks)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches  += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            iter_losses.append(avg_loss)
            loss_history.append(avg_loss)
            logger.info(
                f"[IterRef]   Epoch {epoch + 1}/{epochs_per_iter}  loss={avg_loss:.4f}")

        # ── Generate new masks ──────────────────────────────────────
        model.eval()
        new_masks: list[np.ndarray] = []
        dice_scores: list[float] = []

        for img_path, prev_mask in zip(image_paths, current_masks):
            new_mask = _infer_mask(model, img_path, device)
            new_masks.append(new_mask)
            dice_scores.append(_dice_np(new_mask, prev_mask))

        mean_dice = float(np.mean(dice_scores)) if dice_scores else 0.0
        logger.info(f"[IterRef]   Mean Dice vs prev masks: {mean_dice:.4f}")

        # Save best checkpoint
        if mean_dice > best_dice:
            best_dice  = mean_dice
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}

        # Update masks for next iteration
        improvement = mean_dice - best_dice + (mean_dice > best_dice) * (mean_dice - best_dice)
        current_masks = new_masks

        # Early stop
        if iteration > 0 and (mean_dice - best_dice) < 0.01:
            logger.info(f"[IterRef] Early stop: dice improvement < 0.01")
            break

    # ── Save best checkpoint ──────────────────────────────────────────
    if best_state is not None:
        torch.save(
            {"model_state_dict": best_state,
             "best_dice": best_dice,
             "config": {"architecture": "attention_unet",
                        "in_channels": 3, "base_filters": 16}},
            best_ckpt_path,
        )
        logger.info(f"[IterRef] Saved best checkpoint -> {best_ckpt_path}")

    # ── Loss curve ────────────────────────────────────────────────────
    _save_loss_curve(loss_history, str(out_dir / "stage4_loss_curve.png"))

    return {
        "best_dice":        best_dice,
        "loss_history":     loss_history,
        "checkpoint_path":  best_ckpt_path,
        "n_iterations_run": min(n_iterations, iteration + 1),  # type: ignore[possibly-undefined]
    }


def _save_loss_curve(loss_history: list[float], save_path: str) -> None:
    """
    Save a simple loss curve plot as a PNG image using OpenCV (no matplotlib needed).

    Parameters
    ----------
    loss_history : list[float]  Loss values per epoch across all iterations.
    save_path    : str          Output file path for the PNG image.
    """
    if not loss_history:
        return

    h, w  = 300, 600
    canvas = np.ones((h, w, 3), dtype=np.uint8) * 240  # light grey background

    mn, mx = min(loss_history), max(loss_history)
    span   = mx - mn + 1e-8
    n      = len(loss_history)

    pts = []
    for i, v in enumerate(loss_history):
        x = int(30 + (w - 60) * i / max(n - 1, 1))
        y = int(30 + (h - 60) * (1.0 - (v - mn) / span))
        pts.append((x, y))

    # Draw grid lines
    for yi in range(0, h - 30, 50):
        cv2.line(canvas, (30, yi + 30), (w - 30, yi + 30), (200, 200, 200), 1)

    # Draw loss curve
    for i in range(len(pts) - 1):
        cv2.line(canvas, pts[i], pts[i + 1], (0, 100, 200), 2)

    # Mark points
    for pt in pts:
        cv2.circle(canvas, pt, 4, (200, 50, 50), -1)

    # Labels
    cv2.putText(canvas, "Training Loss", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 1)
    cv2.putText(canvas, f"min={mn:.4f}", (w - 130, h - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (50, 50, 50), 1)
    cv2.putText(canvas, f"max={mx:.4f}", (30, h - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (50, 50, 50), 1)

    cv2.imwrite(save_path, canvas)
    logger.info(f"[IterRef] Loss curve saved -> {save_path}")


# ── Test / Demo ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import glob
    from pathlib import Path

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root))

    unet_path = str(project_root / "checkpoints" / "unet" / "best_unet.pth")
    output_dir = str(project_root / "outputs")
    mask_dir   = str(project_root / "pseudo_masks")   # likely doesn't exist

    patterns = [
        str(project_root / "Project" / "Testing" / "**" / "*.jpg"),
        str(project_root / "data"    / "**"            / "*.jpg"),
    ]
    image_paths: list[str] = []
    for pat in patterns:
        image_paths.extend(glob.glob(pat, recursive=True))
        if len(image_paths) >= 10:
            break
    image_paths = image_paths[:10]

    print(f"Running iterative refinement on {len(image_paths)} images ...")
    result = run_iterative_refinement(
        unet_path       = unet_path,
        image_paths     = image_paths,
        mask_dir        = mask_dir,
        output_dir      = output_dir,
        n_iterations    = 2,
        epochs_per_iter = 2,
        lr              = 1e-5,
        batch_size      = 4,
    )
    print(f"Done. Best Dice: {result['best_dice']:.4f}  "
          f"Iterations: {result['n_iterations_run']}")
    print(f"Checkpoint: {result['checkpoint_path']}")
