"""
refinement/gradcam_fusion.py
=============================
GradCAM attention extraction and fusion with pseudo masks.

GradCAM generates a class activation heatmap by computing the gradient of
the target class score with respect to the last convolutional feature map.
The heatmap highlights the regions that most influenced the classification.

Fusing the heatmap with a pseudo mask removes non-attended (spurious) regions.
"""
from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ── GradCAM ──────────────────────────────────────────────────────────────────

class GradCAM:
    """
    GradCAM with forward/backward hooks on the last convolutional layer.

    Parameters
    ----------
    model : torch.nn.Module
        Classifier model (must have at least one Conv2d layer).

    Usage
    -----
    >>> gcam = GradCAM(model)
    >>> heatmap = gcam.generate_heatmap(image_tensor, target_class=0)
    """

    def __init__(self, model: torch.nn.Module) -> None:
        """Attach forward and backward hooks to the last Conv2d layer."""
        self.model = model
        self.model.eval()

        self._feature_maps: Optional[torch.Tensor] = None
        self._gradients:    Optional[torch.Tensor] = None
        self._hooks:        list = []

        target_layer = self._find_last_conv()
        if target_layer is None:
            raise RuntimeError("No Conv2d layer found in model — GradCAM cannot attach hooks.")

        # Forward hook: capture feature maps
        self._hooks.append(
            target_layer.register_forward_hook(self._save_features)
        )
        # Backward hook: capture gradients
        self._hooks.append(
            target_layer.register_full_backward_hook(self._save_gradients)
        )
        logger.info(f"[GradCAM] Attached hooks to: {type(target_layer).__name__}")

    def _find_last_conv(self) -> Optional[nn.Module]:
        """
        Traverse all named modules and return the last Conv2d found.

        Returns
        -------
        torch.nn.Module or None
            The last Conv2d module in the model, or None if none found.
        """
        last_conv = None
        for _, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module
        return last_conv

    def _save_features(
        self,
        module: nn.Module,
        inp: tuple,
        output: torch.Tensor,
    ) -> None:
        """Forward hook: save the output feature maps."""
        self._feature_maps = output.detach()

    def _save_gradients(
        self,
        module: nn.Module,
        grad_input: tuple,
        grad_output: tuple,
    ) -> None:
        """Backward hook: save gradients flowing back through the layer."""
        self._gradients = grad_output[0].detach()

    def generate_heatmap(
        self,
        image_tensor: torch.Tensor,
        target_class: int = 0,
    ) -> np.ndarray:
        """
        Compute the GradCAM heatmap for the given image and target class.

        Parameters
        ----------
        image_tensor : torch.Tensor
            Preprocessed image tensor, shape (1, C, H, W).
        target_class : int
            Index of the class to explain (default 0).

        Returns
        -------
        np.ndarray
            Normalised heatmap in [0, 1], shape (H, W) at the model's
            spatial output resolution (will be upsampled by the caller).
        """
        self.model.zero_grad()
        device = next(self.model.parameters()).device
        x = image_tensor.to(device)

        # Forward pass
        logits = self.model(x)                    # (1, num_classes)

        # Safety: clamp target class to valid range
        n_cls = logits.shape[-1] if logits.ndim > 1 else 1
        tgt   = min(target_class, n_cls - 1)

        if logits.ndim == 1:
            score = logits[tgt]
        else:
            score = logits[0, tgt]

        # Backward pass
        self.model.zero_grad()
        score.backward(retain_graph=False)

        if self._gradients is None or self._feature_maps is None:
            logger.warning("[GradCAM] No gradients/features captured — returning blank heatmap")
            h_in, w_in = image_tensor.shape[-2:]
            return np.zeros((h_in // 16, w_in // 16), dtype=np.float32)

        # Global average pooling of gradients → channel weights
        grads  = self._gradients         # (1, C, H', W')
        feats  = self._feature_maps      # (1, C, H', W')
        weights = grads.mean(dim=(2, 3), keepdim=True)   # (1, C, 1, 1)

        # Weighted sum of feature maps
        cam = (weights * feats).sum(dim=1).squeeze(0)    # (H', W')
        cam = torch.relu(cam).cpu().numpy()              # keep only positive activations

        # Normalise to [0, 1]
        mn, mx = cam.min(), cam.max()
        if mx > mn:
            cam = (cam - mn) / (mx - mn + 1e-8)
        else:
            cam = np.zeros_like(cam)

        return cam.astype(np.float32)

    def remove_hooks(self) -> None:
        """Remove all registered hooks (call when done)."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ── Fusion ───────────────────────────────────────────────────────────────────

def cam_to_mask(
    heatmap:   np.ndarray,
    h:         int,
    w:         int,
    threshold: float = 0.40,
) -> np.ndarray:
    """
    Convert a GradCAM heatmap directly to a binary mask.

    Use this when the U-Net mask is empty (collapsed model) — the CAM
    becomes the primary segmentation source rather than a filter.

    Parameters
    ----------
    heatmap   : np.ndarray  GradCAM float [0,1], any spatial size.
    h, w      : int         Target height/width for the output mask.
    threshold : float       Pixels above this become foreground (default 0.40).

    Returns
    -------
    np.ndarray  Binary uint8 mask {0,1}, shape (H, W).
    """
    heat_rs = cv2.resize(heatmap.astype(np.float32), (w, h),
                         interpolation=cv2.INTER_LINEAR)
    mn, mx = heat_rs.min(), heat_rs.max()
    if mx > mn:
        heat_rs = (heat_rs - mn) / (mx - mn + 1e-8)
    return (heat_rs >= threshold).astype(np.uint8)


def fuse_gradcam_with_mask(
    pseudo_mask: np.ndarray,
    heatmap:     np.ndarray,
    threshold:   float = 0.40,
) -> np.ndarray:
    """
    Fuse a GradCAM heatmap with a pseudo mask.

    Behaviour:
    - If pseudo_mask is non-empty  → keep only mask pixels inside the CAM region
      (prune false positives outside the classifier's attention).
    - If pseudo_mask is empty      → use the CAM directly as the mask
      (recover signal when the U-Net has collapsed to all-zero output).

    Parameters
    ----------
    pseudo_mask : np.ndarray
        Binary or float mask, shape (H, W).  Values in {0,1} or [0,1].
    heatmap : np.ndarray
        GradCAM heatmap, shape (H', W'), values in [0, 1].
        Resized automatically to match pseudo_mask.
    threshold : float
        Minimum heatmap activation to keep a region (default 0.40).

    Returns
    -------
    np.ndarray
        Fused binary mask, dtype=uint8, values in {0, 1}, shape (H, W).
    """
    h, w = pseudo_mask.shape[:2]

    heat_rs = cv2.resize(heatmap.astype(np.float32), (w, h),
                         interpolation=cv2.INTER_LINEAR)
    mn, mx = heat_rs.min(), heat_rs.max()
    if mx > mn:
        heat_rs = (heat_rs - mn) / (mx - mn + 1e-8)

    attention_gate = (heat_rs >= threshold).astype(np.uint8)

    mask_binary = (pseudo_mask > 0.5).astype(np.uint8) \
                  if pseudo_mask.max() <= 1.0 else \
                  (pseudo_mask > 127).astype(np.uint8)

    # ── Key fix: if mask is empty, promote CAM directly ──────────────────
    if mask_binary.sum() == 0:
        logger.debug("[GradCAM] Pseudo mask empty — using CAM as primary mask")
        return attention_gate

    fused = (mask_binary * attention_gate).astype(np.uint8)
    # If fusion kills everything, fall back to raw attention gate
    if fused.sum() == 0:
        logger.debug("[GradCAM] Fusion zeroed mask — using CAM gate as fallback")
        return attention_gate
    return fused


# ── Classifier loader ─────────────────────────────────────────────────────────

def _try_load_classifier(ckpt_path: str, num_classes: int = 4) -> Optional[nn.Module]:
    """
    Attempt to load the ResNetCBAM classifier from a checkpoint file.

    Falls back to a dummy model (random weights) if loading fails, logging a warning.

    Parameters
    ----------
    ckpt_path   : str  Path to the .pt checkpoint file.
    num_classes : int  Number of output classes (default 4).

    Returns
    -------
    torch.nn.Module
        Loaded (or dummy) classifier in eval mode, on CPU.
    """
    # Build a minimal inline classifier that mirrors the ResNetCBAM structure
    # from app.py without requiring timm at import time.
    try:
        import timm  # type: ignore
        _has_timm = True
    except ImportError:
        _has_timm = False

    if _has_timm:
        # Reproduce the ResNetCBAM structure from app.py
        import torch.nn as nn

        class _ChannelAttention(nn.Module):
            def __init__(self, ch: int, r: int = 16) -> None:
                super().__init__()
                self.avg = nn.AdaptiveAvgPool2d(1)
                self.max = nn.AdaptiveMaxPool2d(1)
                self.fc  = nn.Sequential(
                    nn.Conv2d(ch, ch // r, 1, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(ch // r, ch, 1, bias=False),
                )
                self.sig = nn.Sigmoid()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x * self.sig(self.fc(self.avg(x)) + self.fc(self.max(x)))

        class _SpatialAttention(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)
                self.sig  = nn.Sigmoid()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                avg = x.mean(dim=1, keepdim=True)
                mx, _ = x.max(dim=1, keepdim=True)
                return x * self.sig(self.conv(torch.cat([avg, mx], dim=1)))

        class _CBAM(nn.Module):
            def __init__(self, ch: int, r: int = 16) -> None:
                super().__init__()
                self.ca = _ChannelAttention(ch, r)
                self.sa = _SpatialAttention()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.sa(self.ca(x))

        class _ResNetCBAM(nn.Module):
            def __init__(self, nc: int) -> None:
                super().__init__()
                base       = timm.create_model(
                    "resnet50", pretrained=False, num_classes=0, global_pool="")
                self.conv1   = base.conv1
                self.bn1     = base.bn1
                self.act1    = base.act1
                self.maxpool = base.maxpool
                self.layer1  = base.layer1
                self.layer2  = base.layer2
                self.layer3  = base.layer3
                self.layer4  = base.layer4
                self.cbam3   = _CBAM(1024)
                self.cbam4   = _CBAM(2048)
                self.pool    = nn.AdaptiveAvgPool2d(1)
                self.head    = nn.Sequential(
                    nn.Dropout(0.4),
                    nn.Linear(2048, 512),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.2),
                    nn.Linear(512, nc),
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = self.act1(self.bn1(self.conv1(x)))
                x = self.maxpool(x)
                x = self.layer1(x)
                x = self.layer2(x)
                x = self.cbam3(self.layer3(x))
                x = self.cbam4(self.layer4(x))
                return self.head(self.pool(x).flatten(1))

        try:
            model = _ResNetCBAM(num_classes)
            ckpt  = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            raw   = ckpt.get("model_state_dict", ckpt)

            # ── Key remapping ──────────────────────────────────────────
            # The checkpoint was saved with a 'backbone.' wrapper around the
            # ResNet50 body (old training code).  Strip it so keys like
            # 'backbone.conv1.weight' map to the flat 'conv1.weight' we have.
            # Also convert Conv2d CBAM weights [out,in,1,1] → they're fine as-is
            # since our _ChannelAttention also uses Conv2d.
            state: dict = {}
            for k, v in raw.items():
                nk = k
                if nk.startswith("backbone."):
                    nk = nk[len("backbone."):]
                state[nk] = v

            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing:
                logger.warning(f"[GradCAM] Still missing after remap: {missing[:5]}")
            if not missing:
                logger.info(f"[GradCAM] Loaded classifier from {Path(ckpt_path).name} (0 missing keys)")
            model.eval()
            return model
        except Exception as e:
            logger.warning(f"[GradCAM] Classifier load failed: {e} — using dummy model")

    # Fallback: tiny dummy conv model (always produces valid heatmap)
    logger.warning("[GradCAM] Using dummy classifier (no timm or load failed)")
    dummy = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(32, num_classes),
    )
    dummy.eval()
    return dummy


# ── Preprocessing ─────────────────────────────────────────────────────────────

def _preprocess_image(img_rgb: np.ndarray, img_size: int = 224) -> torch.Tensor:
    """
    Preprocess an RGB image for the classifier.

    Parameters
    ----------
    img_rgb  : np.ndarray  uint8 RGB image, shape (H, W, 3).
    img_size : int         Target spatial size (square).

    Returns
    -------
    torch.Tensor
        Normalised tensor, shape (1, 3, img_size, img_size).
    """
    img = cv2.resize(img_rgb, (img_size, img_size)).astype(np.float32) / 255.0
    img = (img - _MEAN) / _STD
    t   = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float()
    return t


# ── Test / Demo ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import glob
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root))
    output_dir   = project_root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = str(
        project_root / "checkpoints" / "resnet_cbam" / "best_ep008_auc0.9952.pt"
    )

    # Load classifier
    classifier = _try_load_classifier(ckpt_path)

    # Try to attach GradCAM
    gcam: Optional[GradCAM] = None
    try:
        gcam = GradCAM(classifier)
    except Exception as e:
        print(f"[WARNING] GradCAM init failed: {e} — skipping heatmap generation")

    # Find 3 images
    patterns = [
        str(project_root / "Project" / "Testing" / "**" / "*.jpg"),
        str(project_root / "data"    / "**"            / "*.jpg"),
    ]
    image_paths: list[str] = []
    for pat in patterns:
        image_paths.extend(glob.glob(pat, recursive=True))
        if len(image_paths) >= 3:
            break
    image_paths = image_paths[:3]

    panels = []
    for img_path in image_paths:
        try:
            bgr = cv2.imread(img_path)
            if bgr is None:
                continue
            if bgr.ndim == 2:
                bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
            h, w = bgr.shape[:2]
            rgb  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            # Synthetic pseudo mask
            from refinement.mask_refinement import _make_gaussian_blob_mask, clean_mask
            raw_mask = _make_gaussian_blob_mask(h, w, 0.30)
            pseudo   = clean_mask(raw_mask)

            # GradCAM heatmap
            if gcam is not None:
                try:
                    tensor  = _preprocess_image(rgb)
                    heatmap = gcam.generate_heatmap(tensor, target_class=0)
                    heatmap = cv2.resize(heatmap, (w, h))
                except Exception as e:
                    print(f"  [WARNING] GradCAM generate failed for {Path(img_path).name}: {e}")
                    heatmap = np.zeros((h, w), dtype=np.float32)
            else:
                heatmap = np.zeros((h, w), dtype=np.float32)

            fused = fuse_gradcam_with_mask(pseudo, heatmap, threshold=0.40)

            # Visualise
            heat_vis = cv2.applyColorMap(
                (heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
            heat_vis = cv2.cvtColor(heat_vis, cv2.COLOR_BGR2RGB)

            orig_rs  = cv2.resize(rgb,       (256, 256))
            mask_rs  = cv2.resize(
                (pseudo * 255).astype(np.uint8), (256, 256))
            mask_rgb = np.stack([mask_rs] * 3, axis=-1)

            heat_rs  = cv2.resize(heat_vis, (256, 256))

            fused_rs = cv2.resize(
                (fused * 255).astype(np.uint8), (256, 256))
            fused_rgb = np.stack([fused_rs] * 3, axis=-1)

            row = np.concatenate([orig_rs, mask_rgb, heat_rs, fused_rgb], axis=1)
            panels.append(row)
            print(
                f"  {Path(img_path).name:40s}  "
                f"mask={pseudo.mean()*100:.1f}%  "
                f"fused={fused.mean()*100:.1f}%"
            )
        except Exception as e:
            print(f"  Error processing {img_path}: {e}")

    if gcam is not None:
        gcam.remove_hooks()

    if panels:
        grid     = np.concatenate(panels, axis=0)
        out_path = str(output_dir / "stage3_gradcam.png")
        cv2.imwrite(out_path, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
        print(f"\nSaved -> {out_path}")
    else:
        print("No panels to save.")
