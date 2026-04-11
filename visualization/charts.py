"""
visualization/charts.py
========================
Matplotlib chart helpers for GradCAM visualisation in NeuroScan AI.

All functions return either a numpy array (uint8 RGB) or a matplotlib Figure.
Use fig_to_numpy() to convert a Figure to an array suitable for gr.Image.
"""
from __future__ import annotations

import io
import logging
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

logger = logging.getLogger(__name__)

# ── Dark theme constants ───────────────────────────────────────────────────
_BG       = "#0f172a"
_PANEL    = "#1e293b"
_GRID     = "#334155"
_ACCENT   = "#00d4ff"
_WHITE    = "#e2e8f0"
_FONT     = "DejaVu Sans"

_DARK_RC = {
    "figure.facecolor":  _BG,
    "axes.facecolor":    _PANEL,
    "axes.edgecolor":    _GRID,
    "axes.labelcolor":   _WHITE,
    "text.color":        _WHITE,
    "xtick.color":       "#94a3b8",
    "ytick.color":       "#94a3b8",
    "grid.color":        _GRID,
    "grid.alpha":        0.5,
    "font.family":       _FONT,
    "lines.linewidth":   1.8,
}

# Custom jet colourmap for colour bar
_JET = plt.get_cmap("jet")


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def fig_to_numpy(fig: plt.Figure, dpi: int = 120) -> np.ndarray:
    """Render a matplotlib Figure to a uint8 RGB numpy array."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    arr = np.array(Image.open(buf).convert("RGB"))
    plt.close(fig)
    return arr


def _add_label(ax, text: str, fontsize: int = 9) -> None:
    ax.set_title(text, color=_WHITE, fontsize=fontsize,
                 pad=4, fontfamily=_FONT, fontweight="bold")


# ---------------------------------------------------------------------------
# 1. GradCAM 4-panel figure
# ---------------------------------------------------------------------------

def plot_gradcam_panel(
    original:   np.ndarray,     # [H, W, 3] uint8 RGB
    raw_hm:     np.ndarray,     # [H, W, 3] uint8 RGB – jet colourmap
    overlay:    np.ndarray,     # [H, W, 3] uint8 RGB – blended
    boundary:   np.ndarray,     # [H, W, 3] uint8 RGB – contour
    pred_label: str = "",
    confidence: float = 0.0,
    demo: bool = False,
) -> np.ndarray:
    """
    2x2 grid: Original | Raw CAM | Overlay | Boundary Detection.
    Returns uint8 RGB numpy array.
    """
    with plt.rc_context(_DARK_RC):
        fig, axes = plt.subplots(1, 4, figsize=(16, 4.2),
                                 facecolor=_BG)
        fig.patch.set_facecolor(_BG)

        panels = [
            (original, "Original MRI"),
            (raw_hm,   "Activation Map (Jet)"),
            (overlay,  f"GradCAM Overlay{' [DEMO]' if demo else ''}"),
            (boundary, "Tumour Boundary (CAM>0.5)"),
        ]
        for ax, (img, title) in zip(axes, panels):
            ax.imshow(img)
            ax.set_axis_off()
            _add_label(ax, title, fontsize=9)

        # Colour bar under the raw heatmap panel (axes[1])
        sm = plt.cm.ScalarMappable(cmap="jet",
                                   norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes[1],
                            orientation="horizontal",
                            fraction=0.046, pad=0.12, shrink=0.85)
        cbar.set_label("Activation intensity", color="#94a3b8",
                       fontsize=7, labelpad=2)
        cbar.ax.tick_params(labelsize=6, colors="#94a3b8")
        cbar.ax.set_xticks([0, 0.5, 1.0])
        cbar.ax.set_xticklabels(["Low", "Mid", "High"])

        # Super-title
        title_str = (f"Tumour Localisation — {pred_label}  "
                     f"({confidence*100:.1f}% conf)")
        fig.suptitle(title_str, color=_ACCENT, fontsize=11,
                     fontweight="bold", y=1.01)

        # Legend row at the bottom
        legend_items = [
            mpatches.Patch(color="#0000aa", label="Low activation"),
            mpatches.Patch(color="#00aa00", label="Mid activation"),
            mpatches.Patch(color="#ff0000", label="High activation"),
            mpatches.Patch(color="#39ff14", label="Tumour boundary"),
        ]
        fig.legend(handles=legend_items,
                   loc="lower center", ncol=4,
                   frameon=False, fontsize=7.5,
                   labelcolor=_WHITE,
                   bbox_to_anchor=(0.5, -0.04))

        plt.tight_layout(pad=0.4)
        return fig_to_numpy(fig, dpi=110)


# ---------------------------------------------------------------------------
# 2. Tumour location brain diagram
# ---------------------------------------------------------------------------

def plot_tumor_location(
    centroid:      Tuple[int, int],  # (cx, cy) in original image px
    bbox:          Tuple[int, int, int, int],  # (x1,y1,x2,y2)
    lateralization: str,
    img_shape:     Tuple[int, int],  # (H, W)
    cam:           Optional[np.ndarray] = None,  # [H, W] for background
) -> np.ndarray:
    """
    Simplified brain diagram with tumour centroid crosshair and bbox.
    Returns uint8 RGB numpy array.
    """
    H, W = img_shape
    cx, cy = centroid
    x1, y1, x2, y2 = bbox

    # Normalise to [0, 1]
    nx  = cx / W;  ny  = cy / H
    nx1 = x1 / W; ny1 = y1 / H
    nx2 = x2 / W; ny2 = y2 / H

    with plt.rc_context(_DARK_RC):
        fig, ax = plt.subplots(1, 1, figsize=(5, 5),
                               facecolor=_BG)
        ax.set_facecolor(_BG)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(1.02, -0.02)   # image coords (y flipped)
        ax.set_aspect("equal")

        # Draw brain outline (ellipse)
        from matplotlib.patches import Ellipse, FancyArrowPatch, Rectangle
        brain_outline = Ellipse(
            xy=(0.5, 0.5), width=0.88, height=0.95,
            edgecolor="#64748b", facecolor="#1e293b",
            linewidth=2, linestyle="--", zorder=1,
        )
        ax.add_patch(brain_outline)

        # Hemisphere midline
        ax.axvline(0.5, color="#334155", linewidth=1,
                   linestyle=":", zorder=2)

        # Hemisphere labels
        ax.text(0.22, 0.06, "LEFT",  color="#475569",
                fontsize=9, ha="center", fontweight="bold", zorder=5)
        ax.text(0.78, 0.06, "RIGHT", color="#475569",
                fontsize=9, ha="center", fontweight="bold", zorder=5)

        # Background CAM (optional)
        if cam is not None:
            cam_rgb = cv2.applyColorMap(
                (cam * 255).astype(np.uint8), cv2.COLORMAP_JET
            )
            cam_rgb = cv2.cvtColor(cam_rgb, cv2.COLOR_BGR2RGB)
            ax.imshow(cam_rgb, extent=[0, 1, 1, 0],
                      alpha=0.25, zorder=1, aspect="auto")

        # Bounding box
        bw = nx2 - nx1
        bh = ny2 - ny1
        rect = Rectangle(
            (nx1, ny1), bw, bh,
            edgecolor="#facc15", facecolor="none",
            linewidth=1.5, linestyle="--", zorder=3,
        )
        ax.add_patch(rect)

        # Crosshair at centroid
        hlen = 0.06
        ax.plot([nx - hlen, nx + hlen], [ny, ny],
                color="#ef4444", linewidth=2.5, zorder=4)
        ax.plot([nx, nx], [ny - hlen, ny + hlen],
                color="#ef4444", linewidth=2.5, zorder=4)
        ax.scatter([nx], [ny], s=60, color="#ef4444",
                   zorder=5, marker="o")

        # Annotation
        side_offset = 0.14 if nx < 0.5 else -0.14
        ax.annotate(
            f"Tumour\ncentroid\n({cx},{cy})px",
            xy=(nx, ny),
            xytext=(nx + side_offset, ny - 0.18),
            color="#fca5a5", fontsize=7.5,
            arrowprops=dict(arrowstyle="->",
                            color="#fca5a5", lw=1.2),
            ha="center", zorder=6,
        )

        # Legend items
        from matplotlib.lines import Line2D
        legend_elems = [
            Line2D([0], [0], color="#ef4444", linewidth=2,
                   label="Tumour centroid"),
            mpatches.Patch(edgecolor="#facc15", facecolor="none",
                           linestyle="--", label="Activation bbox"),
            mpatches.Patch(edgecolor="#64748b", facecolor="none",
                           linestyle="--", label="Brain outline"),
        ]
        ax.legend(handles=legend_elems, loc="lower right",
                  frameon=True, framealpha=0.3,
                  labelcolor=_WHITE, fontsize=7.5,
                  facecolor=_PANEL, edgecolor=_GRID)

        ax.set_title(
            f"Tumour Location — {lateralization}",
            color=_ACCENT, fontsize=10, fontweight="bold", pad=8,
        )
        ax.set_xticks([]); ax.set_yticks([])
        ax.spines[:].set_color(_GRID)

        plt.tight_layout()
        return fig_to_numpy(fig, dpi=110)


# ---------------------------------------------------------------------------
# 3. Activation intensity profile
# ---------------------------------------------------------------------------

def plot_activation_profile(
    cam: np.ndarray,   # [H, W] float32 [0,1]
    pred_label: str = "",
) -> np.ndarray:
    """
    Column-wise max-activation profile chart.
    Shows where horizontal activation is concentrated across image width.
    Returns uint8 RGB numpy array.
    """
    # Horizontal profile (column-wise max and mean)
    col_max  = cam.max(axis=0)   # [W]
    col_mean = cam.mean(axis=0)  # [W]
    row_max  = cam.max(axis=1)   # [H]

    x_cols = np.arange(len(col_max))
    x_rows = np.arange(len(row_max))

    with plt.rc_context(_DARK_RC):
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(10, 3.4), facecolor=_BG
        )

        # -- Column profile (horizontal sweep) --
        ax1.fill_between(x_cols, col_max,
                         alpha=0.30, color="#ef4444")
        ax1.plot(x_cols, col_max,  color="#ef4444",
                 linewidth=1.6, label="Peak")
        ax1.plot(x_cols, col_mean, color="#f59e0b",
                 linewidth=1.2, linestyle="--", label="Mean")

        peak_col = int(col_max.argmax())
        ax1.axvline(peak_col, color="#ef4444", alpha=0.6,
                    linewidth=1, linestyle=":")
        ax1.annotate(f"Peak\nx={peak_col}",
                     xy=(peak_col, col_max[peak_col]),
                     xytext=(peak_col + len(x_cols) * 0.08,
                             col_max[peak_col] * 0.82),
                     color="#fca5a5", fontsize=7.5,
                     arrowprops=dict(arrowstyle="->",
                                     color="#fca5a5", lw=1))
        ax1.set_xlim(0, len(x_cols) - 1)
        ax1.set_ylim(-0.02, 1.08)
        ax1.set_xlabel("Image column (px)", fontsize=8)
        ax1.set_ylabel("Activation", fontsize=8)
        ax1.set_title("Horizontal Activation Profile",
                      color=_ACCENT, fontsize=9, fontweight="bold")
        ax1.legend(frameon=False, fontsize=7.5,
                   labelcolor=_WHITE, loc="upper right")
        ax1.grid(True, axis="y")
        ax1.axhline(0.5, color=_GRID, linewidth=1, linestyle="--")
        ax1.text(len(x_cols) - 2, 0.52, "threshold=0.5",
                 color="#475569", fontsize=6.5, ha="right")

        # -- Row profile (vertical sweep) --
        ax2.fill_betweenx(x_rows, row_max,
                          alpha=0.30, color="#a855f7")
        ax2.plot(row_max, x_rows, color="#a855f7",
                 linewidth=1.6, label="Peak")
        peak_row = int(row_max.argmax())
        ax2.axhline(peak_row, color="#a855f7", alpha=0.6,
                    linewidth=1, linestyle=":")
        ax2.annotate(f"Peak\ny={peak_row}",
                     xy=(row_max[peak_row], peak_row),
                     xytext=(row_max[peak_row] * 0.55,
                             peak_row + len(x_rows) * 0.08),
                     color="#d8b4fe", fontsize=7.5,
                     arrowprops=dict(arrowstyle="->",
                                     color="#d8b4fe", lw=1))
        ax2.set_xlim(-0.02, 1.08)
        ax2.set_ylim(len(x_rows) - 1, 0)    # flipped
        ax2.set_xlabel("Activation", fontsize=8)
        ax2.set_ylabel("Image row (px)", fontsize=8)
        ax2.set_title("Vertical Activation Profile",
                      color=_ACCENT, fontsize=9, fontweight="bold")
        ax2.legend(frameon=False, fontsize=7.5,
                   labelcolor=_WHITE)
        ax2.grid(True, axis="x")

        if pred_label:
            fig.suptitle(
                f"Activation Profile — {pred_label}",
                color=_WHITE, fontsize=9, y=1.01,
            )

        plt.tight_layout(pad=0.5)
        return fig_to_numpy(fig, dpi=110)


# ---------------------------------------------------------------------------
# 4. Stats HTML card (for gr.HTML)
# ---------------------------------------------------------------------------

def gradcam_stats_html(stats: dict, elapsed: float, demo: bool) -> str:
    """Return styled HTML summarising the tumour region statistics."""
    area    = stats["area_pct"]
    cx, cy  = stats["centroid"]
    lateral = stats["lateralization"]
    score   = stats["intensity_score"]
    x1, y1, x2, y2 = stats["bbox"]

    risk_color = ("#ef4444" if area > 20
                  else "#f59e0b" if area > 10
                  else "#10b981")
    score_color = ("#ef4444" if score > 65
                   else "#f59e0b" if score > 35
                   else "#10b981")
    demo_badge = (
        '<span style="background:#1e40af22;border:1px solid #3b82f644;'
        'color:#60a5fa;padding:2px 8px;border-radius:10px;font-size:9px;">'
        'Demo Mode</span> '
        if demo else ""
    )

    rows = [
        ("Estimated Area",      f"{area:.1f}% of visible scan", risk_color),
        ("Hemisphere",          lateral,                         "#e2e8f0"),
        ("Centroid (px)",       f"({cx}, {cy})",                 "#e2e8f0"),
        ("Bounding Region",     f"[{x1},{y1}] → [{x2},{y2}]",  "#e2e8f0"),
        ("Activation Score",    f"{score} / 100",               score_color),
        ("GradCAM time",        f"{elapsed}s",                  "#64748b"),
    ]

    row_html = "".join(
        f'<div style="display:flex;justify-content:space-between;'
        f'align-items:center;padding:5px 0;border-bottom:1px solid #1e293b;">'
        f'<span style="font-size:11px;color:#94a3b8;">{k}</span>'
        f'<span style="font-size:11px;font-weight:600;color:{c};">{v}</span>'
        f'</div>'
        for k, v, c in rows
    )

    return f"""
<div style="background:#0f172a;border:1px solid #1e293b;border-radius:10px;
            padding:14px 16px;font-family:'Inter',system-ui,sans-serif;
            margin-top:6px;">
  <div style="display:flex;align-items:center;gap:8px;margin-bottom:10px;">
    <span style="font-size:13px;font-weight:700;color:#00d4ff;">
      📍 Tumour Region Analysis
    </span>
    {demo_badge}
  </div>
  {row_html}
  <div style="margin-top:8px;padding:6px 8px;background:#1e293b44;
              border-radius:6px;border-left:3px solid #ef4444;">
    <span style="font-size:9.5px;color:#94a3b8;">
      🔴 Red areas indicate high activation probability &nbsp;|&nbsp;
      🟡 Yellow = moderate &nbsp;|&nbsp; 🔵 Blue = low
    </span>
  </div>
</div>"""
