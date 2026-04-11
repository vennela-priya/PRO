"""
visualization/seg_charts.py
===========================
Visualization helpers for U-Net segmentation output in NeuroScan AI.

Public API
----------
    plot_seg_panel(...)       → uint8 RGB numpy array  (4-panel strip)
    plot_seg_vs_gradcam(...)  → uint8 RGB numpy array  (3-panel comparison)
    seg_stats_html(...)       → styled HTML string for gr.HTML

All figure functions return numpy arrays so they slot directly into
gr.Image components without any extra PIL/cv2 conversion.
"""
from __future__ import annotations

import io
from typing import Optional

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image


# ── Dark theme RC params (matches existing charts.py) ────────────────────
_DARK_RC: dict = {
    "figure.facecolor": "#0f172a",
    "axes.facecolor":   "#0f172a",
    "axes.edgecolor":   "#334155",
    "text.color":       "#e2e8f0",
    "axes.labelcolor":  "#94a3b8",
    "xtick.color":      "#64748b",
    "ytick.color":      "#64748b",
    "grid.color":       "#1e293b",
}

_CYAN  = "#00d4ff"
_RED   = "#ef4444"
_GREEN = "#10b981"
_AMBER = "#f59e0b"
_BLUE  = "#3b82f6"
_PURP  = "#a855f7"


# ── Internal helpers ──────────────────────────────────────────────────────

def _fig_to_numpy(fig: plt.Figure, dpi: int = 100) -> np.ndarray:
    """Render a matplotlib figure to uint8 RGB numpy array."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    arr = np.array(Image.open(buf).convert("RGB"), dtype=np.uint8)
    buf.close()
    return arr


def _label(ax, text: str, fontsize: int = 9) -> None:
    ax.set_title(text, color=_CYAN, fontsize=fontsize,
                 fontweight="bold", pad=4)
    ax.axis("off")


# ── 4-panel segmentation strip ────────────────────────────────────────────

def plot_seg_panel(
    original:    np.ndarray,
    prob_map:    np.ndarray,
    overlay:     np.ndarray,
    contour:     np.ndarray,
    area_pct:    float,
    dice_vs_cam: Optional[float] = None,
    iou_vs_cam:  Optional[float] = None,
    demo:        bool = False,
) -> np.ndarray:
    """
    4-panel horizontal strip:
      [Original MRI] | [Prob Map] | [Seg Overlay] | [Tumour Contour]

    Parameters
    ----------
    original  : uint8 RGB (H, W, 3) — raw MRI
    prob_map  : float32 (H, W) [0,1] — U-Net sigmoid output
    overlay   : uint8 RGB (H, W, 3) — red-tinted tumour region
    contour   : uint8 RGB (H, W, 3) — green contour on MRI
    area_pct  : tumour area as % of image
    dice_vs_cam : Dice overlap with Grad-CAM (None if not computed)
    iou_vs_cam  : IoU  overlap with Grad-CAM (None if not computed)
    demo      : whether U-Net is in demo mode

    Returns
    -------
    uint8 RGB numpy array — ready for gr.Image
    """
    with plt.rc_context(_DARK_RC):
        fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))

        panels = [
            (original,  "Original MRI",        None,       {}),
            (prob_map,  "U-Net Prob Map",       "RdYlGn_r", {"vmin": 0, "vmax": 1}),
            (overlay,   "Segmentation Overlay", None,       {}),
            (contour,   "Tumour Boundary",      None,       {}),
        ]

        for ax, (img, title, cmap, kw) in zip(axes, panels):
            ax.imshow(img, cmap=cmap, **kw)
            _label(ax, title)

        # Colour-bar under the probability panel
        sm = plt.cm.ScalarMappable(
            cmap="RdYlGn_r", norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cb = fig.colorbar(sm, ax=axes[1], orientation="horizontal",
                          fraction=0.055, pad=0.06)
        cb.ax.tick_params(colors="#94a3b8", labelsize=7)
        cb.set_label("Tumour probability", color="#94a3b8", fontsize=7)

        # Bottom annotation bar
        mode_tag = "  [Demo Mode]" if demo else "  [Live U-Net]"
        dice_str = (f"   Dice/CAM: {dice_vs_cam:.3f}"
                    if dice_vs_cam is not None else "")
        iou_str  = (f"   IoU/CAM: {iou_vs_cam:.3f}"
                    if iou_vs_cam  is not None else "")
        fig.text(
            0.5, 0.01,
            f"Tumour area: {area_pct:.1f}%{mode_tag}{dice_str}{iou_str}",
            ha="center", va="bottom",
            color="#94a3b8", fontsize=8.5,
        )

        plt.tight_layout(pad=0.5)
        arr = _fig_to_numpy(fig, dpi=100)
        plt.close(fig)
    return arr


# ── 3-panel Grad-CAM vs U-Net comparison ─────────────────────────────────

def plot_seg_vs_gradcam(
    cam:         np.ndarray,
    cam_mask:    np.ndarray,
    seg_mask:    np.ndarray,
    original:    np.ndarray,
    dice:        Optional[float] = None,
    iou:         Optional[float] = None,
) -> np.ndarray:
    """
    3-panel comparison showing where Grad-CAM and U-Net agree/disagree.

    Panels
    ------
    Left   : Grad-CAM heatmap (jet colormap)
    Centre : U-Net binary mask (grey)
    Right  : Agreement map
               🟢 Green — both methods agree (true overlap)
               🔴 Red   — Grad-CAM only (no U-Net support)
               🔵 Blue  — U-Net only (no Grad-CAM support)

    Parameters
    ----------
    cam      : float32 (H, W) Grad-CAM in [0, 1]
    cam_mask : uint8   (H, W) binarised CAM {0, 1}
    seg_mask : uint8   (H, W) U-Net binary mask {0, 1}
    original : uint8   (H, W, 3) reference image for resizing

    Returns
    -------
    uint8 RGB numpy array — ready for gr.Image
    """
    H, W = original.shape[:2]

    # Resize all masks to original resolution
    cam_r  = cv2.resize(cam.astype(np.float32), (W, H),
                        interpolation=cv2.INTER_LINEAR)
    cmk_r  = cv2.resize(cam_mask.astype(np.float32), (W, H),
                        interpolation=cv2.INTER_NEAREST)
    smk_r  = cv2.resize(seg_mask.astype(np.float32), (W, H),
                        interpolation=cv2.INTER_NEAREST)

    # Agreement overlay (R=CAM-only, G=both, B=Seg-only)
    agree = np.zeros((H, W, 3), dtype=np.float32)
    agree[..., 1] = (cmk_r > 0.5) & (smk_r > 0.5)   # green  = both
    agree[..., 0] = (cmk_r > 0.5) & (smk_r <= 0.5)  # red    = CAM only
    agree[..., 2] = (cmk_r <= 0.5) & (smk_r > 0.5)  # blue   = Seg only

    with plt.rc_context(_DARK_RC):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))

        axes[0].imshow(cam_r, cmap="jet", vmin=0, vmax=1)
        _label(axes[0], "Grad-CAM Heatmap")

        axes[1].imshow(smk_r, cmap="gray", vmin=0, vmax=1)
        _label(axes[1], "U-Net Mask")

        axes[2].imshow(agree)
        _label(axes[2], "Agreement Map")

        # Legend
        patches = [
            mpatches.Patch(color="lime",       label="Both agree"),
            mpatches.Patch(color="red",         label="Grad-CAM only"),
            mpatches.Patch(color="cornflowerblue", label="U-Net only"),
        ]
        axes[2].legend(
            handles=patches, loc="lower right",
            facecolor="#1e293b", labelcolor="white",
            fontsize=7.5, framealpha=0.85,
        )

        # Metrics annotation
        if dice is not None or iou is not None:
            parts = []
            if dice is not None: parts.append(f"Dice: {dice:.3f}")
            if iou  is not None: parts.append(f"IoU: {iou:.3f}")
            fig.text(
                0.5, 0.01, "  ".join(parts),
                ha="center", color="#94a3b8", fontsize=9,
            )

        plt.tight_layout(pad=0.5)
        arr = _fig_to_numpy(fig, dpi=100)
        plt.close(fig)
    return arr


# ── HTML stats card ───────────────────────────────────────────────────────

def seg_stats_html(seg_result: dict) -> str:
    """
    Build a styled dark-theme HTML card summarising segmentation results.
    Compatible with gr.HTML in the NeuroScan AI Gradio app.

    Parameters
    ----------
    seg_result : dict returned by TumorSegmenter.segment()
                 optionally enriched by compare_with_gradcam()

    Returns
    -------
    HTML string
    """
    area    = seg_result.get("area_pct",    0.0)
    elapsed = seg_result.get("elapsed",     0.0)
    demo    = seg_result.get("demo",        True)
    dice    = seg_result.get("dice_vs_cam")
    iou     = seg_result.get("iou_vs_cam")

    # Area colour coding
    if area > 20:
        area_color = _RED
    elif area > 8:
        area_color = _AMBER
    else:
        area_color = _GREEN

    # Mode badge
    if demo:
        badge = (
            '<span style="background:#1e40af22;border:1px solid #3b82f6;'
            'color:#3b82f6;padding:3px 10px;border-radius:12px;'
            'font-size:10px;font-weight:600;">🔵 Demo Mode</span>'
        )
    else:
        badge = (
            '<span style="background:#05966922;border:1px solid #10b981;'
            'color:#10b981;padding:3px 10px;border-radius:12px;'
            'font-size:10px;font-weight:600;">🟢 Live U-Net</span>'
        )

    def _box(value: str, label: str, color: str = _CYAN) -> str:
        return (
            f'<div style="background:#0f172a;border:1px solid #334155;'
            f'border-radius:8px;padding:8px 16px;text-align:center;'
            f'min-width:90px;flex:1;">'
            f'<div style="font-size:18px;font-weight:700;color:{color};">{value}</div>'
            f'<div style="font-size:9px;color:#475569;margin-top:2px;">{label}</div>'
            f'</div>'
        )

    dice_box = _box(
        f"{dice:.3f}" if dice is not None else "N/A",
        "Dice vs CAM", _PURP,
    )
    iou_box = _box(
        f"{iou:.3f}" if iou is not None else "N/A",
        "IoU vs CAM", _AMBER,
    )

    return f"""
<div style="background:linear-gradient(135deg,#0f172a,#1a1040);
            padding:18px 22px;border-radius:14px;margin:8px 0;
            border:1px solid #334155;
            font-family:'Inter',system-ui,sans-serif;">

  <div style="display:flex;align-items:center;
              justify-content:space-between;margin-bottom:14px;
              flex-wrap:wrap;gap:8px;">
    <div style="font-size:14px;font-weight:700;color:#e2e8f0;">
      🔬 U-Net Segmentation Results
    </div>
    {badge}
  </div>

  <div style="display:flex;gap:10px;flex-wrap:wrap;">
    {_box(f"{area:.1f}%",   "Tumour Area",  area_color)}
    {_box(f"{elapsed:.2f}s","Inference",    _BLUE)}
    {dice_box}
    {iou_box}
  </div>

  <div style="margin-top:12px;font-size:10px;color:#475569;line-height:1.6;">
    <b style="color:#64748b;">Agreement map legend:</b>
    &nbsp; <span style="color:lime;">█</span> Both methods agree &nbsp;
    <span style="color:#ef4444;">█</span> Grad-CAM only &nbsp;
    <span style="color:cornflowerblue;">█</span> U-Net only
  </div>
</div>"""
