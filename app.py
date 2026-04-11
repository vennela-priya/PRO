#!/usr/bin/env python3
"""
NeuroScan AI — Brain Tumor Diagnostic Application v2.0
EfficientNet-B3 + ResNet50+CBAM + DenseNet121 Ensemble with TTA
"""
# ── stdlib ────────────────────────────────────────────────────────────────
import os, sys, uuid, logging, time, warnings, io, math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

warnings.filterwarnings("ignore")

# ── logging ───────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s — %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("NeuroScanAI")

# ── numeric / image ───────────────────────────────────────────────────────
import numpy as np
import cv2
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import plotly.graph_objects as go

# ── torch ─────────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch.cuda.amp import autocast
    _AMP = True
except ImportError:
    _AMP = False

try:
    import timm
    _TIMM = True
except ImportError:
    _TIMM = False
    logger.warning("timm not installed — demo mode only")

# ── gradio ────────────────────────────────────────────────────────────────
import gradio as gr

# ── reportlab ─────────────────────────────────────────────────────────────
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors as rlc
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table,
        TableStyle, Image as RLImage, HRFlowable,
    )
    _PDF = True
except ImportError:
    _PDF = False
    logger.warning("reportlab not installed — PDF disabled (pip install reportlab)")

# ── GradCAM + visualization charts + enhanced report ─────────────────────
try:
    from inference.gradcam import EnsembleGradCAM
    from visualization.charts import (
        plot_gradcam_panel, plot_tumor_location,
        plot_activation_profile, gradcam_stats_html,
    )
    from report.report_generator import generate_report as _generate_report_v2
    _GRADCAM = True
    logger.info("GradCAM modules loaded")
except ImportError as _gc_err:
    _GRADCAM = False
    logger.warning(f"GradCAM modules not available: {_gc_err}")

# ── U-Net Segmentation ────────────────────────────────────────────────────
try:
    from inference.segmenter import TumorSegmenter
    from visualization.seg_charts import (
        plot_seg_panel, plot_seg_vs_gradcam, seg_stats_html as _seg_html_fn,
    )
    _SEG = True
    logger.info("Segmentation modules loaded")
except ImportError as _seg_err:
    _SEG = False
    logger.warning(f"Segmentation modules not available: {_seg_err}")

# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════
APP_VERSION  = "2.0.0"
APP_NAME     = "NeuroScan AI"
CLASS_NAMES  = ["glioma", "meningioma", "notumor", "pituitary"]
CLASS_LABELS = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
NUM_CLASSES  = 4
IMAGE_SIZE   = 224
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_COLORS = {
    "glioma":     "#ef4444",
    "meningioma": "#f59e0b",
    "notumor":    "#10b981",
    "pituitary":  "#3b82f6",
}
RISK_MAP = {
    "glioma":     ("CRITICAL", "#ef4444"),
    "meningioma": ("HIGH",     "#f59e0b"),
    "pituitary":  ("MODERATE", "#3b82f6"),
    "notumor":    ("LOW",      "#10b981"),
}
CLINICAL_NOTES = {
    "glioma": (
        "Findings suggest a possible gliomatous lesion. Immediate referral to "
        "neurosurgery and neuro-oncology is strongly recommended. MRI with contrast "
        "enhancement, spectroscopy, and perfusion imaging should be obtained. "
        "Biopsy and molecular profiling (IDH status, MGMT methylation) may be "
        "required for definitive grading."
    ),
    "meningioma": (
        "Imaging pattern is consistent with a meningioma. Contrast-enhanced MRI "
        "is advised to characterize the lesion. Neurosurgical consultation and "
        "close radiological follow-up are recommended. Most meningiomas are "
        "benign (WHO Grade I) but grading requires histopathological confirmation."
    ),
    "pituitary": (
        "Findings may indicate a pituitary adenoma or sellar lesion. Dedicated "
        "pituitary MRI protocol and endocrinology referral are suggested. "
        "Hormonal panel (prolactin, GH, ACTH, TSH, FSH/LH) should be evaluated. "
        "Visual field assessment is recommended if the lesion approximates the optic chiasm."
    ),
    "notumor": (
        "No significant intracranial mass lesion identified on the presented image. "
        "Routine clinical follow-up is advised as clinically indicated. If symptoms "
        "persist or new neurological signs develop, repeat imaging should be considered. "
        "This result does not exclude all pathologies — clinical correlation is essential."
    ),
}

SEARCH_PATHS = [
    "./checkpoints",
    "./BrainTumorAI/checkpoints",
    "./models",
    str(Path.home() / "BrainTumorAI/checkpoints"),
    "/content/drive/MyDrive/BrainTumorAI/checkpoints",
    "../checkpoints",
]
ENS_WEIGHTS = {"efficientnet": 0.4, "resnet_cbam": 0.3, "densenet": 0.3}

# ═══════════════════════════════════════════════════════════════════════════
# MODEL ARCHITECTURE  (must match training exactly)
# ═══════════════════════════════════════════════════════════════════════════

class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        avg  = x.mean(dim=[2, 3])
        mx   = x.amax(dim=[2, 3])
        attn = torch.sigmoid(self.fc(avg) + self.fc(mx))
        return x * attn.view(B, C, 1, 1)


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        avg  = x.mean(dim=1, keepdim=True)
        mx   = x.amax(dim=1, keepdim=True)
        attn = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * attn


class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention()

    def forward(self, x):
        return self.sa(self.ca(x))


class ResNetCBAM(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = False):
        super().__init__()
        base = timm.create_model(
            "resnet50", pretrained=pretrained, num_classes=0, global_pool="")
        self.conv1   = base.conv1
        self.bn1     = base.bn1
        self.act1    = base.act1
        self.maxpool = base.maxpool
        self.layer1  = base.layer1
        self.layer2  = base.layer2
        self.layer3  = base.layer3
        self.layer4  = base.layer4
        self.cbam3   = CBAM(1024)
        self.cbam4   = CBAM(2048)
        self.pool    = nn.AdaptiveAvgPool2d(1)
        self.head    = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.cbam3(self.layer3(x))
        x = self.cbam4(self.layer4(x))
        return self.head(self.pool(x).flatten(1))


def _make_efficientnet(nc: int) -> nn.Module:
    return timm.create_model(
        "efficientnet_b3.ra2_in1k", pretrained=False, num_classes=nc)

def _make_resnet_cbam(nc: int) -> nn.Module:
    return ResNetCBAM(nc, pretrained=False)

def _make_densenet(nc: int) -> nn.Module:
    return timm.create_model("densenet121", pretrained=False, num_classes=nc)


class Ensemble(nn.Module):
    def __init__(self, models: Dict[str, nn.Module], weights: Dict[str, float]):
        super().__init__()
        self.models  = nn.ModuleDict(models)
        self.weights = weights
        self.temps   = nn.ParameterDict(
            {k: nn.Parameter(torch.ones(1)) for k in models})

    def forward(self, x):
        total = None
        for k, m in self.models.items():
            logits = m(x) / self.temps[k].clamp(min=0.1)
            prob   = F.softmax(logits.float(), dim=1)
            w      = self.weights.get(k, 1.0)
            total  = w * prob if total is None else total + w * prob
        return total

    def individual_probs(self, x) -> Dict[str, np.ndarray]:
        out = {}
        for k, m in self.models.items():
            logits = m(x) / self.temps[k].clamp(min=0.1)
            out[k] = F.softmax(logits.float(), dim=1).cpu().numpy()
        return out

# ═══════════════════════════════════════════════════════════════════════════
# DEMO / MOCK MODEL
# ═══════════════════════════════════════════════════════════════════════════

class MockEnsemble:
    """Realistic demo predictions when no checkpoint is available."""
    def predict(self, tensor: torch.Tensor) -> Tuple[np.ndarray, Dict]:
        arr  = tensor.cpu().numpy()
        seed = int((arr.mean() * 1e4 + arr.std() * 1e3) % 1e6)
        rng  = np.random.RandomState(seed % 100000)
        dom  = rng.randint(0, NUM_CLASSES)
        base = rng.dirichlet(np.ones(NUM_CLASSES) * 0.3)
        base[dom] += rng.uniform(0.40, 0.65)
        probs = base / base.sum()
        individual = {}
        for name in ["efficientnet", "resnet_cbam", "densenet"]:
            noise = rng.dirichlet(np.ones(NUM_CLASSES) * 3.0) * 0.08
            ind   = probs * 0.92 + noise
            individual[name] = ind / ind.sum()
        return probs, individual

# ═══════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════

def _best_ckpt(folder: Path) -> Optional[Path]:
    ckpts = list(folder.glob("best_*.pt"))
    if not ckpts:
        return None
    def _auc(p):
        try:    return float(p.stem.split("auc")[-1])
        except: return 0.0
    return max(ckpts, key=_auc)


def _remap_keys(raw_sd: dict, mname: str) -> dict:
    """
    Remap checkpoint saved by OLD train.py (backbone.* prefix, Conv2d CBAM)
    to match the current model architecture.

    Old format:
      - EfficientNet/DenseNet: backbone.* (num_classes=0) + head.1 Linear(feat→512)
                               + head.4 Linear(512→nc)  [shapes don't match new classifier]
      - ResNet+CBAM: backbone.* + cbam*.ca.fc.*.weight (4-D Conv2d)
                     + head.1 Linear(2048→512) + head.4 Linear(512→nc)  [MATCH current]
    New format:
      - EfficientNet: flat timm keys + classifier Linear(1536→nc)
      - DenseNet:     flat timm keys + classifier Linear(1024→nc)
      - ResNet+CBAM:  flat keys + cbam*.ca.fc.* (2-D Linear) + head.1/4 (unchanged)
    """
    new_sd = {}
    for k, v in raw_sd.items():
        nk = k
        # Strip backbone. prefix
        if nk.startswith("backbone."):
            nk = nk[len("backbone."):]
        # EfficientNet / DenseNet: old head (512-dim bottle-neck) shape doesn't match
        # the new timm classifier (direct feat→nc linear).  Drop all head.* keys and
        # let strict=False leave classifier randomly initialised — backbone still loads.
        if mname in ("efficientnet", "densenet"):
            if nk.startswith("head."):
                continue
        # ResNet+CBAM: CBAM channel-attention weights were saved as Conv2d [out,in,1,1];
        # current ChannelAttention uses Linear [out,in] — squeeze spatial dims.
        if "cbam" in nk and "ca.fc" in nk and v.dim() == 4:
            v = v.squeeze(-1).squeeze(-1)
        new_sd[nk] = v
    return new_sd


def load_model() -> Tuple[Union[Ensemble, MockEnsemble], str]:
    if not _TIMM:
        logger.info("timm missing — demo mode")
        return MockEnsemble(), "demo"

    fns = {
        "efficientnet": _make_efficientnet,
        "resnet_cbam":  _make_resnet_cbam,
        "densenet":     _make_densenet,
    }

    for base_str in SEARCH_PATHS:
        base = Path(base_str)
        if not base.exists():
            continue

        # ── Try full ensemble checkpoint ───────────────────────────────
        ens_ckpt = base / "ensemble_final.pt"
        if ens_ckpt.exists():
            try:
                mdls = {k: fn(NUM_CLASSES) for k, fn in fns.items()}
                ens  = Ensemble(mdls, ENS_WEIGHTS).to(DEVICE)
                ens.load_state_dict(
                    torch.load(ens_ckpt, map_location=DEVICE, weights_only=False))
                ens.eval()
                logger.info(f"Loaded full ensemble: {ens_ckpt}")
                return ens, "live"
            except Exception as e:
                logger.warning(f"Ensemble load failed: {e}")

        # ── Try individual checkpoints (with key remapping) ────────────
        loaded: Dict[str, nn.Module] = {}
        for mname, mfn in fns.items():
            ckpt = _best_ckpt(base / mname)
            if ckpt is None:
                continue
            try:
                m      = mfn(NUM_CLASSES)
                raw    = torch.load(ckpt, map_location=DEVICE, weights_only=False)
                raw_sd = raw["model_state_dict"]

                # Try direct load first (Colab-trained checkpoints)
                try:
                    m.load_state_dict(raw_sd, strict=True)
                    logger.info(f"  {mname}: {ckpt.name} (direct)")
                except Exception:
                    # Remap keys for old train.py checkpoints
                    remapped = _remap_keys(raw_sd, mname)
                    m.load_state_dict(remapped, strict=False)
                    missing  = [k for k in m.state_dict() if k not in remapped]
                    extra    = [k for k in remapped if k not in m.state_dict()]
                    logger.info(f"  {mname}: {ckpt.name} (remapped) "
                                f"missing={len(missing)} extra={len(extra)}")
                    # Skip if the output layer wasn't loaded (random classifier = useless)
                    _out_key = {"efficientnet": "classifier.weight",
                                "densenet":     "classifier.weight",
                                "resnet_cbam":  "head.4.weight"}.get(mname, "")
                    if _out_key and _out_key in missing:
                        logger.warning(f"  {mname}: output layer missing after remap "
                                       f"(old checkpoint format, shapes incompatible) — skipping")
                        continue

                m.eval()
                loaded[mname] = m.to(DEVICE)
            except Exception as e:
                logger.warning(f"  {mname} failed: {e}")

        if loaded:
            w    = {k: ENS_WEIGHTS[k] for k in loaded}
            ens  = Ensemble(loaded, w).to(DEVICE)
            ens.eval()
            mode = "live" if len(loaded) == 3 else "partial"
            logger.info(f"Assembled {mode} ensemble: {list(loaded.keys())}")
            return ens, mode

    logger.info("No checkpoints found — demo mode")
    return MockEnsemble(), "demo"

# ═══════════════════════════════════════════════════════════════════════════
# PREPROCESSING & TTA INFERENCE
# ═══════════════════════════════════════════════════════════════════════════

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(image) -> Optional[torch.Tensor]:
    try:
        if isinstance(image, np.ndarray):
            img = image.copy().astype(np.uint8)
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        elif isinstance(image, Image.Image):
            img = np.array(image.convert("RGB"), dtype=np.uint8)
        else:
            return None
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32) / 255.0
        img = (img - _MEAN) / _STD
        return torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float()
    except Exception as e:
        logger.error(f"Preprocess: {e}")
        return None


def _safe(arr: np.ndarray) -> np.ndarray:
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    s   = arr.sum(keepdims=True)
    return arr / max(float(s), 1e-8)


def _tta_views(t: torch.Tensor) -> List[torch.Tensor]:
    return [
        t,
        torch.flip(t, dims=[3]),
        torch.flip(t, dims=[2]),
        torch.rot90(t, 1, [2, 3]),
        torch.rot90(t, 2, [2, 3]),
        torch.rot90(t, 3, [2, 3]),
    ]


def run_inference(model_obj, image) -> Dict:
    t0     = time.time()
    tensor = preprocess(image)
    if tensor is None:
        return {"error": "Preprocessing failed — check image format"}
    tensor = tensor.to(DEVICE)

    if isinstance(model_obj, MockEnsemble):
        probs, individual = model_obj.predict(tensor)
    else:
        tta_probs, ind_acc = [], {k: [] for k in model_obj.models}
        model_obj.eval()
        with torch.no_grad():
            for view in _tta_views(tensor):
                if _AMP:
                    with autocast():
                        p = model_obj(view)
                else:
                    p = model_obj(view)
                tta_probs.append(_safe(p.cpu().numpy()[0]))
                for k, v in model_obj.individual_probs(view).items():
                    ind_acc[k].append(_safe(v[0]))

        probs      = _safe(np.mean(tta_probs, axis=0))
        individual = {k: _safe(np.mean(v, axis=0))
                      for k, v in ind_acc.items() if v}

    pred_idx   = int(probs.argmax())
    pred_class = CLASS_NAMES[pred_idx]

    return {
        "class":         pred_class,
        "class_label":   CLASS_LABELS[pred_idx],
        "confidence":    float(probs[pred_idx]),
        "probabilities": {CLASS_NAMES[i]: float(probs[i]) for i in range(NUM_CLASSES)},
        "individual":    {k: {CLASS_NAMES[i]: float(v[i]) for i in range(NUM_CLASSES)}
                          for k, v in individual.items()},
        "risk_level":    RISK_MAP[pred_class][0],
        "risk_color":    RISK_MAP[pred_class][1],
        "elapsed":       round(time.time() - t0, 3),
    }

# ═══════════════════════════════════════════════════════════════════════════
# CHARTS
# ═══════════════════════════════════════════════════════════════════════════

_BG   = "#0f172a"
_CARD = "#1e293b"
_GRID = "#334155"
_FONT = "Inter, system-ui, sans-serif"


def chart_prob_bars(probs: Dict[str, float]) -> go.Figure:
    vals  = [probs.get(c, 0) * 100 for c in CLASS_NAMES]
    clrs  = [CLASS_COLORS[c] for c in CLASS_NAMES]
    fig = go.Figure(go.Bar(
        x=vals, y=CLASS_LABELS, orientation="h",
        marker=dict(color=clrs),
        text=[f"  {v:.1f}%" for v in vals],
        textposition="outside",
        textfont=dict(color="white", size=13, family=_FONT),
        cliponaxis=False,
    ))
    fig.update_layout(
        paper_bgcolor=_BG, plot_bgcolor=_CARD,
        font=dict(color="white", family=_FONT),
        title=dict(text="Class Probabilities", font=dict(size=14, color="#94a3b8")),
        xaxis=dict(range=[0, 118], showgrid=True, gridcolor=_GRID,
                   ticksuffix="%", color="#94a3b8", zeroline=False),
        yaxis=dict(showgrid=False, color="#e2e8f0", tickfont=dict(size=13)),
        margin=dict(l=10, r=50, t=44, b=20),
        height=250,
    )
    return fig


def chart_gauge(confidence: float, pred_class: str) -> go.Figure:
    pct   = confidence * 100
    color = CLASS_COLORS.get(pred_class, "#94a3b8")
    bc    = "#64748b" if pct < 60 else "#f59e0b" if pct < 85 else color
    fig   = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number=dict(suffix="%", font=dict(size=36, color="white", family=_FONT)),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor="#94a3b8",
                      tickfont=dict(color="#94a3b8", size=10)),
            bar=dict(color=bc, thickness=0.28),
            bgcolor=_CARD, bordercolor=_GRID,
            steps=[
                dict(range=[0,  60],  color="#1e293b"),
                dict(range=[60, 85],  color="#292524"),
                dict(range=[85, 100], color="#1c1917"),
            ],
            threshold=dict(line=dict(color=color, width=4),
                           thickness=0.75, value=85),
        ),
        title=dict(text="Confidence", font=dict(size=13, color="#94a3b8", family=_FONT)),
        domain=dict(x=[0, 1], y=[0, 1]),
    ))
    fig.update_layout(
        paper_bgcolor=_BG,
        font=dict(color="white", family=_FONT),
        margin=dict(l=20, r=20, t=44, b=10),
        height=240,
    )
    return fig


def chart_radar(individual: Dict[str, Dict[str, float]]) -> go.Figure:
    cats  = CLASS_LABELS + [CLASS_LABELS[0]]
    clrs  = {"efficientnet": "#00d4ff", "resnet_cbam": "#a855f7", "densenet": "#f59e0b"}
    names = {"efficientnet": "EfficientNet-B3",
             "resnet_cbam":  "ResNet50+CBAM",
             "densenet":     "DenseNet121"}
    fig = go.Figure()
    for mname, pd_ in individual.items():
        vals = [pd_.get(c, 0) * 100 for c in CLASS_NAMES] + [pd_.get(CLASS_NAMES[0], 0) * 100]
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=cats, fill="toself", opacity=0.35,
            name=names.get(mname, mname),
            line=dict(color=clrs.get(mname, "#fff"), width=2.5),
            fillcolor=clrs.get(mname, "#fff"),
        ))
    fig.update_layout(
        paper_bgcolor=_BG,
        font=dict(color="white", family=_FONT),
        title=dict(text="Ensemble Member Agreement",
                   font=dict(size=14, color="#94a3b8")),
        polar=dict(
            bgcolor=_CARD,
            radialaxis=dict(visible=True, range=[0, 100],
                            color="#94a3b8", gridcolor=_GRID,
                            ticksuffix="%", tickfont=dict(size=9)),
            angularaxis=dict(color="#e2e8f0", gridcolor=_GRID,
                             tickfont=dict(size=11)),
        ),
        legend=dict(bgcolor=_CARD, bordercolor=_GRID, font=dict(color="white", size=10)),
        margin=dict(l=40, r=40, t=54, b=30),
        height=360,
    )
    return fig

# ═══════════════════════════════════════════════════════════════════════════
# DIAGNOSIS HTML CARD
# ═══════════════════════════════════════════════════════════════════════════

def diagnosis_card(result: Dict) -> str:
    cls    = result["class"]
    label  = result["class_label"]
    conf   = result["confidence"] * 100
    risk   = result["risk_level"]
    rcolor = result["risk_color"]
    ela    = result["elapsed"]
    color  = CLASS_COLORS.get(cls, "#94a3b8")
    ctxt   = ("High Confidence Diagnosis" if conf >= 85
              else "Moderate Confidence" if conf >= 60
              else "Uncertain — Recommend Clinical Review")
    icons  = {"CRITICAL": "🔴", "HIGH": "🟠", "MODERATE": "🔵", "LOW": "🟢"}

    bars = "".join(
        f"""<div style="display:flex;align-items:center;gap:8px;margin:5px 0;">
              <div style="width:88px;font-size:11px;color:#94a3b8;">{CLASS_LABELS[i]}</div>
              <div style="flex:1;background:#0f172a;border-radius:4px;height:8px;overflow:hidden;">
                <div style="width:{result['probabilities'][CLASS_NAMES[i]]*100:.1f}%;
                            height:8px;background:{CLASS_COLORS[CLASS_NAMES[i]]};
                            border-radius:4px;"></div></div>
              <div style="width:44px;font-size:11px;color:#e2e8f0;text-align:right;">
                {result['probabilities'][CLASS_NAMES[i]]*100:.1f}%</div>
            </div>"""
        for i in range(NUM_CLASSES)
    )

    return f"""
    <div style="background:linear-gradient(135deg,#1e293b,#0f172a);
                border:1px solid {color}44;border-radius:16px;padding:22px;
                font-family:'Inter',system-ui,sans-serif;">

      <div style="text-align:center;margin-bottom:18px;">
        <div style="font-size:10px;letter-spacing:3px;text-transform:uppercase;
                    color:#64748b;margin-bottom:6px;">AI Diagnosis Result</div>
        <div style="font-size:38px;font-weight:800;color:{color};
                    text-shadow:0 0 28px {color}55;">{label.upper()}</div>
        <div style="font-size:20px;font-weight:700;color:#10b981;margin-top:4px;">
          {conf:.2f}% Confidence</div>
        <div style="font-size:10px;color:#475569;margin-top:3px;">
          ⏱ {ela:.2f}s · TTA 6-view · 3-model ensemble</div>
      </div>

      <div style="display:flex;gap:8px;justify-content:center;flex-wrap:wrap;margin-bottom:16px;">
        <span style="background:{color}22;border:1px solid {color}55;color:{color};
                     padding:4px 12px;border-radius:16px;font-size:11px;font-weight:600;">
          {icons.get(risk,'⚪')} Risk: {risk}</span>
        <span style="background:#1e293b;border:1px solid #334155;color:#94a3b8;
                     padding:4px 12px;border-radius:16px;font-size:11px;">{ctxt}</span>
      </div>

      <div style="background:#0f172a55;border-radius:8px;padding:12px;margin-bottom:14px;">
        <div style="font-size:9px;color:#64748b;letter-spacing:1px;
                    text-transform:uppercase;margin-bottom:8px;">Probability Distribution</div>
        {bars}
      </div>

      <div style="border-left:3px solid {color};background:#0f172a55;
                  border-radius:0 8px 8px 0;padding:12px 14px;margin-bottom:12px;">
        <div style="font-size:9px;color:#64748b;letter-spacing:1px;
                    text-transform:uppercase;margin-bottom:5px;">Clinical Interpretation</div>
        <div style="font-size:12px;color:#cbd5e1;line-height:1.7;">
          {CLINICAL_NOTES.get(cls, '')}</div>
      </div>

      <div style="text-align:center;padding:8px;background:#1e293b55;
                  border-radius:6px;border:1px solid #334155;">
        <span style="font-size:9px;color:#475569;">
          ⚕ For screening purposes only. Does not replace professional clinical diagnosis.
          Always consult a qualified neurologist or radiologist.
        </span>
      </div>
    </div>"""

# ═══════════════════════════════════════════════════════════════════════════
# EXAMPLE IMAGES (synthetic MRI-like)
# ═══════════════════════════════════════════════════════════════════════════

def _synth_mri(cls: str, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    img = np.zeros((224, 224), dtype=np.float32)
    Y, X = np.ogrid[:224, :224]
    cx, cy = 112, 112
    skull = ((X-cx)**2/95**2 + (Y-cy)**2/105**2) <= 1
    brain = ((X-cx)**2/82**2 + (Y-cy)**2/92**2)  <= 1
    img[skull] = 0.25
    img[brain] = 0.45 + rng.uniform(0, 0.04, img[brain].shape)
    for _ in range(5):
        ang = rng.uniform(0, math.pi)
        xc  = int(cx + rng.uniform(-40, 40))
        yc  = int(cy + rng.uniform(-40, 40))
        ln  = rng.randint(20, 55)
        cv2.line(img, (int(xc-ln*math.cos(ang)), int(yc-ln*math.sin(ang))),
                      (int(xc+ln*math.cos(ang)), int(yc+ln*math.sin(ang))), 0.36, 1)
    img += rng.normal(0, 0.025, img.shape).astype(np.float32)

    if cls == "glioma":
        mx, my = int(cx+rng.uniform(-20,20)), int(cy+rng.uniform(-20,20))
        cv2.ellipse(img, (mx,my), (28,21), int(rng.uniform(0,180)), 0, 360, 0.73, 4)
        cv2.ellipse(img, (mx,my), (20,15), int(rng.uniform(0,180)), 0, 360, 0.16, -1)
        cv2.ellipse(img, (mx,my), (45,38), 0, 0, 360, 0.55, 2)
    elif cls == "meningioma":
        sx = int(cx + rng.choice([-1,1]) * rng.uniform(58, 72))
        sy = int(cy + rng.choice([-1,1]) * rng.uniform(8, 25))
        r  = rng.randint(20, 30)
        cv2.circle(img, (sx, sy), r, 0.78, -1)
        cv2.circle(img, (sx, sy), r+2, 0.87, 2)
    elif cls == "pituitary":
        cv2.ellipse(img, (cx, cy+30), (13, 9), 0, 0, 360, 0.83, -1)
        cv2.circle (img, (cx, cy+30),  5, 0.92, 2)

    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)


def example_images() -> List[np.ndarray]:
    return [_synth_mri(cls, seed=i*7+42) for i, cls in enumerate(CLASS_NAMES)]

# ═══════════════════════════════════════════════════════════════════════════
# PDF REPORT
# ═══════════════════════════════════════════════════════════════════════════

def generate_pdf(result: Dict, image, patient_name: str, patient_id: str) -> Optional[str]:
    if not _PDF or result is None:
        return None

    rid       = str(uuid.uuid4())[:8].upper()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    date_str  = datetime.now().strftime("%B %d, %Y")
    out_path  = f"NeuroScan_Report_{rid}.pdf"

    cls   = result["class"]
    label = result["class_label"]
    conf  = result["confidence"] * 100
    risk  = result["risk_level"]
    probs = result["probabilities"]
    ela   = result.get("elapsed", 0)

    hex_cls  = CLASS_COLORS[cls].lstrip("#")
    cls_col  = rlc.HexColor(f"#{hex_cls}")
    hex_risk = result["risk_color"].lstrip("#")
    risk_col = rlc.HexColor(f"#{hex_risk}")

    NAVY  = rlc.HexColor("#0f172a")
    SLATE = rlc.HexColor("#1e293b")
    CYAN  = rlc.HexColor("#00d4ff")
    LGRAY = rlc.HexColor("#cbd5e1")
    GRAY  = rlc.HexColor("#94a3b8")

    doc   = SimpleDocTemplate(out_path, pagesize=A4,
                              topMargin=1.5*cm, bottomMargin=1.5*cm,
                              leftMargin=2*cm,  rightMargin=2*cm)
    story = []

    def sty(name, **kw): return ParagraphStyle(name, **kw)

    sec_sty  = sty("sec",  fontName="Helvetica-Bold", fontSize=11,
                   textColor=CYAN, spaceBefore=8, spaceAfter=4)
    body_sty = sty("body", fontName="Helvetica", fontSize=9,
                   textColor=LGRAY, leading=14, spaceAfter=4, alignment=TA_JUSTIFY)
    sm_sty   = sty("sm",   fontName="Helvetica", fontSize=7,
                   textColor=GRAY, alignment=TA_CENTER)

    # Header
    hdr_data = [[
        Paragraph("<b>🧠 NeuroScan AI</b>",
                  sty("h1", fontName="Helvetica-Bold", fontSize=22, textColor=rlc.white)),
        Paragraph("DIAGNOSTIC REPORT",
                  sty("h2", fontName="Helvetica-Bold", fontSize=12,
                      textColor=CYAN, alignment=TA_RIGHT)),
    ]]
    hdr_tbl = Table(hdr_data, colWidths=[10*cm, 7*cm])
    hdr_tbl.setStyle(TableStyle([
        ("BACKGROUND",   (0,0),(-1,-1), NAVY),
        ("LINEBELOW",    (0,0),(-1,-1), 2,   CYAN),
        ("TOPPADDING",   (0,0),(-1,-1), 12),
        ("BOTTOMPADDING",(0,0),(-1,-1), 12),
        ("LEFTPADDING",  (0,0),(-1,-1), 12),
        ("RIGHTPADDING", (0,0),(-1,-1), 12),
        ("VALIGN",       (0,0),(-1,-1), "MIDDLE"),
    ]))
    story.append(hdr_tbl)
    story.append(Spacer(1, 0.3*cm))

    # Patient meta
    meta = [
        ["Report ID:", rid,                  "Date:",    date_str],
        ["Patient ID:", patient_id or "N/A", "Time:",    timestamp.split()[1]],
        ["Patient:",   patient_name or "N/A","Model:",   f"Ensemble v{APP_VERSION}"],
    ]
    meta_tbl = Table(meta, colWidths=[2.8*cm, 5.2*cm, 2.5*cm, 6.5*cm])
    meta_tbl.setStyle(TableStyle([
        ("FONTNAME",     (0,0),(0,-1), "Helvetica-Bold"),
        ("FONTNAME",     (2,0),(2,-1), "Helvetica-Bold"),
        ("FONTNAME",     (1,0),(1,-1), "Helvetica"),
        ("FONTNAME",     (3,0),(3,-1), "Helvetica"),
        ("FONTSIZE",     (0,0),(-1,-1), 8),
        ("TEXTCOLOR",    (0,0),(0,-1), CYAN),
        ("TEXTCOLOR",    (2,0),(2,-1), CYAN),
        ("TEXTCOLOR",    (1,0),(1,-1), LGRAY),
        ("TEXTCOLOR",    (3,0),(3,-1), LGRAY),
        ("BACKGROUND",   (0,0),(-1,-1), SLATE),
        ("GRID",         (0,0),(-1,-1), 0.25, rlc.HexColor("#334155")),
        ("TOPPADDING",   (0,0),(-1,-1), 5),
        ("BOTTOMPADDING",(0,0),(-1,-1), 5),
        ("LEFTPADDING",  (0,0),(-1,-1), 7),
        ("ROWBACKGROUNDS",(0,0),(-1,-1), [SLATE, rlc.HexColor("#0f172a")]),
    ]))
    story.append(meta_tbl)
    story.append(Spacer(1, 0.3*cm))
    story.append(HRFlowable(width="100%", thickness=1, color=CYAN, spaceAfter=8))

    # Primary finding
    story.append(Paragraph("PRIMARY FINDING", sec_sty))
    find_data = [[
        Paragraph(f"<b>{label.upper()}</b>",
                  sty("fn", fontName="Helvetica-Bold", fontSize=26,
                      textColor=cls_col, alignment=TA_CENTER)),
        Paragraph(
            f"Confidence: <b>{conf:.1f}%</b><br/>"
            f"Risk Level: <b>{risk}</b><br/>"
            f"Processing: <b>{ela:.2f}s (TTA 6-view)</b>",
            sty("fi", fontName="Helvetica", fontSize=10,
                textColor=LGRAY, leading=16)),
    ]]
    find_tbl = Table(find_data, colWidths=[7*cm, 10*cm])
    find_tbl.setStyle(TableStyle([
        ("BACKGROUND",   (0,0),(0,0), rlc.HexColor("#0f172a")),
        ("BACKGROUND",   (1,0),(1,0), SLATE),
        ("GRID",         (0,0),(-1,-1), 0.5, rlc.HexColor("#334155")),
        ("LINEAFTER",    (0,0),(0,-1), 2, cls_col),
        ("TOPPADDING",   (0,0),(-1,-1), 10),
        ("BOTTOMPADDING",(0,0),(-1,-1), 10),
        ("LEFTPADDING",  (0,0),(-1,-1), 10),
        ("VALIGN",       (0,0),(-1,-1), "MIDDLE"),
    ]))
    story.append(find_tbl)
    story.append(Spacer(1, 0.3*cm))

    # Image + probability table side-by-side
    prob_data = [["Class", "Probability", "Bar"]]
    for cn, cl in zip(CLASS_NAMES, CLASS_LABELS):
        p   = probs.get(cn, 0) * 100
        bar = "█" * int(p / 5)
        prob_data.append([cl, f"{p:.1f}%", bar])
    prob_tbl = Table(prob_data, colWidths=[3.5*cm, 2.5*cm, 10*cm])
    prob_tbl.setStyle(TableStyle([
        ("BACKGROUND",   (0,0),(-1,0), SLATE),
        ("FONTNAME",     (0,0),(-1,0), "Helvetica-Bold"),
        ("TEXTCOLOR",    (0,0),(-1,0), CYAN),
        ("FONTNAME",     (0,1),(-1,-1),"Helvetica"),
        ("TEXTCOLOR",    (0,1),(1,-1), LGRAY),
        ("TEXTCOLOR",    (2,1),(2,-1), rlc.HexColor("#10b981")),
        ("FONTSIZE",     (0,0),(-1,-1), 8),
        ("GRID",         (0,0),(-1,-1), 0.25, rlc.HexColor("#334155")),
        ("TOPPADDING",   (0,0),(-1,-1), 4),
        ("BOTTOMPADDING",(0,0),(-1,-1), 4),
        ("LEFTPADDING",  (0,0),(-1,-1), 6),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[rlc.HexColor("#0f172a"), SLATE]),
    ]))

    try:
        if image is not None:
            pil_img = (Image.fromarray(image.astype(np.uint8))
                       if isinstance(image, np.ndarray)
                       else image)
            tmp = f"_tmp_{rid}.jpg"
            pil_img.convert("RGB").resize((300, 300)).save(tmp, "JPEG", quality=90)
            rl_img  = RLImage(tmp, width=5.5*cm, height=5.5*cm)
            lay_tbl = Table([[rl_img, prob_tbl]], colWidths=[6.5*cm, 10.5*cm])
            lay_tbl.setStyle(TableStyle([
                ("VALIGN",      (0,0),(-1,-1),"TOP"),
                ("LEFTPADDING", (0,0),(-1,-1), 0),
                ("RIGHTPADDING",(0,0),(-1,-1), 4),
            ]))
            story.append(lay_tbl)
            if Path(tmp).exists():
                os.remove(tmp)
        else:
            story.append(prob_tbl)
    except Exception as e:
        logger.warning(f"Image embed: {e}")
        story.append(prob_tbl)

    story.append(Spacer(1, 0.3*cm))
    story.append(HRFlowable(width="100%", thickness=0.5,
                            color=rlc.HexColor("#334155"), spaceAfter=6))

    # Clinical notes
    story.append(Paragraph("CLINICAL INTERPRETATION", sec_sty))
    story.append(Paragraph(CLINICAL_NOTES.get(cls, ""), body_sty))
    story.append(Spacer(1, 0.2*cm))

    # Disclaimer
    story.append(HRFlowable(width="100%", thickness=0.5,
                            color=rlc.HexColor("#334155"), spaceAfter=6))
    disc = (
        "<b>DISCLAIMER:</b> This report is generated by NeuroScan AI, an "
        "AI-assisted screening tool. It is intended for informational and research "
        "purposes only and does not constitute a medical diagnosis. All findings must "
        "be reviewed by a qualified radiologist or neurologist. Individual performance "
        "may vary. Reported model metrics: AUC 0.9996, Accuracy 98.78%, "
        "Sensitivity 98.68%, Specificity 99.59%."
    )
    story.append(Paragraph(disc, sty("disc", fontName="Helvetica-Oblique",
                                     fontSize=7, textColor=GRAY,
                                     leading=11, alignment=TA_JUSTIFY)))
    story.append(Spacer(1, 0.2*cm))

    # Footer
    ft_data = [[
        Paragraph(f"NeuroScan AI v{APP_VERSION} · AUC 0.9996 · Acc 98.78%", sm_sty),
        Paragraph(f"Generated: {timestamp}",
                  sty("fr", fontName="Helvetica", fontSize=7,
                      textColor=GRAY, alignment=TA_RIGHT)),
    ]]
    ft_tbl = Table(ft_data, colWidths=[9*cm, 8*cm])
    ft_tbl.setStyle(TableStyle([
        ("LINEABOVE", (0,0),(-1,-1), 0.5, CYAN),
        ("TOPPADDING",(0,0),(-1,-1), 6),
    ]))
    story.append(ft_tbl)

    doc.build(story)
    logger.info(f"PDF: {out_path}")
    return out_path

# ═══════════════════════════════════════════════════════════════════════════
# GRADIO UI
# ═══════════════════════════════════════════════════════════════════════════

CSS = """
body,.gradio-container{background:#0a0f1e!important;color:#e2e8f0!important;}
.gradio-container{max-width:1180px!important;margin:0 auto!important;}
.tab-nav button{background:#1e293b!important;color:#94a3b8!important;
    border:1px solid #334155!important;border-radius:8px 8px 0 0!important;
    font-weight:600;transition:all .2s;}
.tab-nav button.selected{background:linear-gradient(135deg,#1e40af,#7c3aed)!important;
    color:#fff!important;border-color:#3b82f6!important;}
.gr-panel,.panel{background:#0f172a!important;border:1px solid #1e293b!important;
    border-radius:12px!important;}
.gr-input,.gr-textbox input,.gr-textbox textarea{background:#1e293b!important;
    color:#e2e8f0!important;border:1px solid #334155!important;border-radius:8px!important;}
.gr-button-primary{background:linear-gradient(135deg,#1e40af,#7c3aed)!important;
    color:#fff!important;border:none!important;border-radius:10px!important;
    font-weight:700!important;font-size:15px!important;padding:11px 28px!important;
    box-shadow:0 4px 20px #3b82f644!important;transition:all .2s;}
.gr-button-primary:hover{transform:translateY(-1px);
    box-shadow:0 6px 28px #3b82f666!important;}
.gr-button-secondary{background:#1e293b!important;color:#94a3b8!important;
    border:1px solid #334155!important;border-radius:8px!important;
    font-size:12px!important;}
label.svelte-1b6s6vi{color:#64748b!important;font-size:11px!important;
    letter-spacing:1px!important;text-transform:uppercase!important;}
::-webkit-scrollbar{width:5px;}::-webkit-scrollbar-thumb{background:#334155;border-radius:3px;}
"""

STATS_HTML = """
<div style="background:linear-gradient(135deg,#0f172a,#1e1b4b);
            padding:26px 28px;border-radius:14px;margin-bottom:14px;
            border:1px solid #334155;font-family:'Inter',system-ui,sans-serif;">
  <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px;">
    <div>
      <div style="font-size:28px;font-weight:900;
                  background:linear-gradient(135deg,#60a5fa,#a78bfa,#06b6d4);
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
        🧠 NeuroScan AI</div>
      <div style="font-size:12px;color:#64748b;margin-top:3px;">
        EfficientNet-B3 + ResNet50+CBAM + DenseNet121 · TTA Ensemble</div>
    </div>
    <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;">
      <span style="background:{bc}22;border:1px solid {bc}66;color:{bc};
                   padding:5px 12px;border-radius:16px;font-size:11px;font-weight:600;">
        {badge}</span>
      <span style="background:#1e293b;border:1px solid #334155;color:#64748b;
                   padding:5px 12px;border-radius:16px;font-size:10px;">{gpu}</span>
      <span style="background:#1e293b;border:1px solid #334155;color:#475569;
                   padding:5px 12px;border-radius:16px;font-size:10px;">v{ver}</span>
    </div>
  </div>
  <div style="display:flex;gap:14px;margin-top:14px;flex-wrap:wrap;">
    {metrics}
  </div>
</div>
"""

METRIC_BOX = """
<div style="background:#0f172a;border:1px solid #334155;border-radius:8px;
            padding:7px 12px;text-align:center;">
  <div style="font-size:14px;font-weight:700;color:{c};">{v}</div>
  <div style="font-size:9px;color:#475569;">{l}</div>
</div>"""


def build_ui(model_obj, model_mode: str, gradcam_engine=None, segmenter=None) -> gr.Blocks:
    bc_map  = {"live":"#10b981","partial":"#f59e0b","demo":"#3b82f6"}
    bdg_map = {"live":"🟢 Live Model","partial":"🟡 Partial Model","demo":"🔵 Demo Mode"}
    bc      = bc_map.get(model_mode, "#94a3b8")
    badge   = bdg_map.get(model_mode, "⚪ Unknown")
    gpu     = f"GPU: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "CPU"
    metrics = "".join(METRIC_BOX.format(v=v, l=l, c=c) for v, l, c in [
        ("0.9996", "AUC",         "#10b981"),
        ("98.78%", "Accuracy",    "#3b82f6"),
        ("98.68%", "Sensitivity", "#a855f7"),
        ("99.59%", "Specificity", "#f59e0b"),
        ("4 cls",  "Classes",     "#06b6d4"),
    ])
    header_html = STATS_HTML.format(bc=bc, badge=badge, gpu=gpu,
                                    ver=APP_VERSION, metrics=metrics)

    ex_imgs = example_images()

    with gr.Blocks(css=CSS, title="NeuroScan AI", theme=gr.themes.Base()) as demo:
        result_st  = gr.State(None)
        image_st   = gr.State(None)
        gradcam_st = gr.State(None)
        seg_st     = gr.State(None)
        count_st  = gr.State(0)

        gr.HTML(header_html)

        with gr.Tabs():
            # ── Tab 1: Diagnosis ──────────────────────────────────────────
            with gr.Tab("🔬 Diagnosis"):
                with gr.Row():
                    # Left — upload
                    with gr.Column(scale=1, min_width=280):
                        gr.HTML('<div style="font-size:10px;letter-spacing:2px;'
                                'color:#475569;text-transform:uppercase;margin-bottom:6px;">'
                                'Upload MRI</div>')
                        img_in = gr.Image(
                            label="Brain MRI Scan",
                            type="numpy",
                            height=250,
                            sources=["upload", "clipboard"],
                        )
                        pid_in = gr.Textbox(
                            label="Patient ID (optional)",
                            placeholder="PT-00123",
                            max_lines=1,
                        )
                        run_btn = gr.Button("🔬 Start AI Diagnosis",
                                           variant="primary", size="lg")
                        status_out = gr.HTML(
                            '<div style="text-align:center;color:#475569;'
                            'font-size:11px;padding:6px;">Upload an MRI image to begin</div>'
                        )
                        gr.HTML(f"""
                        <div style="background:#0f172a;border:1px solid #1e293b;
                                    border-radius:8px;padding:12px;margin-top:8px;
                                    font-family:'Inter',system-ui,sans-serif;">
                          <div style="font-size:9px;letter-spacing:2px;color:#475569;
                                      text-transform:uppercase;margin-bottom:8px;">
                            Ensemble Weights</div>
                          {"".join(f'<div style="display:flex;justify-content:space-between;'
                                   f'padding:3px 0;border-bottom:1px solid #1e293b;">'
                                   f'<span style="font-size:10px;color:#94a3b8;">{n}</span>'
                                   f'<span style="font-size:10px;font-weight:600;color:{c};">{w}</span>'
                                   f'</div>'
                                   for n, w, c in [("EfficientNet-B3","40%","#00d4ff"),
                                                   ("ResNet50+CBAM",  "30%","#a855f7"),
                                                   ("DenseNet121",    "30%","#f59e0b")])}
                          <div style="margin-top:6px;font-size:9px;color:#475569;">
                            TTA: 6 augmented views averaged</div>
                        </div>""")

                    # Right — results
                    with gr.Column(scale=2, min_width=360):
                        diag_out = gr.HTML(
                            '<div style="background:#0f172a;border:1px solid #1e293b;'
                            'border-radius:14px;padding:36px;text-align:center;'
                            'font-family:Inter,sans-serif;color:#334155;">'
                            '<div style="font-size:44px;margin-bottom:10px;">🧠</div>'
                            '<div style="color:#475569;font-size:13px;">'
                            'Diagnosis results will appear here after analysis</div>'
                            '</div>'
                        )
                        with gr.Row():
                            gauge_out = gr.Plot(show_label=False)
                            prob_out  = gr.Plot(show_label=False)

                # ── Grad-CAM Heatmap Section ──────────────────────────────
                with gr.Accordion("🔥 Grad-CAM Heatmap Analysis", open=False):
                    gradcam_img = gr.Image(
                        label="GradCAM 4-Panel  (Original | Heatmap | Overlay | Boundary)",
                        show_label=True,
                        height=224,
                        interactive=False,
                    )
                    gradcam_html = gr.HTML("")
                    with gr.Row():
                        gradcam_loc  = gr.Image(
                            label="Tumor Location Diagram",
                            height=220,
                            interactive=False,
                        )
                        gradcam_prof = gr.Image(
                            label="Activation Profile",
                            height=220,
                            interactive=False,
                        )

                # ── U-Net Segmentation Section ────────────────────────────
                with gr.Accordion("🔬 Tumour Segmentation (U-Net)", open=False):
                    seg_panel = gr.Image(
                        label="Segmentation Panel  (Prob Map | Overlay | Contour)",
                        show_label=True,
                        height=200,
                        interactive=False,
                    )
                    seg_html_out = gr.HTML("")
                    seg_cmp = gr.Image(
                        label="Grad-CAM vs U-Net Agreement",
                        height=200,
                        interactive=False,
                    )

                # Example gallery
                gr.HTML('<div style="font-size:10px;letter-spacing:2px;color:#475569;'
                        'text-transform:uppercase;margin:14px 0 8px 0;">Example MRI Images</div>')
                with gr.Row():
                    ex_btns = []
                    for i, (cls, lbl) in enumerate(zip(CLASS_NAMES, CLASS_LABELS)):
                        with gr.Column():
                            gr.Image(value=ex_imgs[i], label=f"Example: {lbl}",
                                     height=130, interactive=False, show_label=True)
                            b = gr.Button(f"▶ {lbl}", size="sm", variant="secondary")
                            ex_btns.append((b, ex_imgs[i]))

            # ── Tab 2: Analytics ──────────────────────────────────────────
            with gr.Tab("📊 Analytics"):
                gr.HTML('<div style="color:#475569;font-size:12px;padding:6px 0 14px;'
                        'font-family:Inter,sans-serif;">Run a diagnosis first to see analytics.</div>')
                radar_out = gr.Plot(show_label=False)
                with gr.Row():
                    roc_ref = gr.Image(
                        label="ROC Curve (Reference)",
                        value="./results/roc_ensemble_test.png"
                              if Path("./results/roc_ensemble_test.png").exists()
                              else "./roc_ensemble_test.png"
                              if Path("./roc_ensemble_test.png").exists() else None,
                        height=320, interactive=False,
                    )
                    cm_ref = gr.Image(
                        label="Confusion Matrix (Reference)",
                        value="./results/cm_ensemble_test.png"
                              if Path("./results/cm_ensemble_test.png").exists()
                              else "./cm_ensemble_test.png"
                              if Path("./cm_ensemble_test.png").exists() else None,
                        height=320, interactive=False,
                    )

            # ── Tab 3: Report ─────────────────────────────────────────────
            with gr.Tab("📋 Report"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.HTML('<div style="color:#94a3b8;font-size:12px;'
                                'font-family:Inter,sans-serif;margin-bottom:12px;">'
                                'Run a diagnosis first, then fill in patient details '
                                'and generate a professional PDF report.</div>')
                        rep_name = gr.Textbox(label="Patient Name", placeholder="John Doe")
                        rep_id   = gr.Textbox(label="Patient ID",   placeholder="PT-00123")
                        rep_btn  = gr.Button("📄 Generate PDF Report", variant="primary")
                        rep_st   = gr.HTML("")
                    with gr.Column(scale=1):
                        pdf_dl = gr.File(label="Download Report", visible=False)
                        gr.HTML(f"""
                        <div style="background:#0f172a;border:1px solid #1e293b;
                                    border-radius:10px;padding:14px;
                                    font-family:'Inter',system-ui,sans-serif;">
                          <div style="font-size:9px;letter-spacing:2px;color:#475569;
                                      text-transform:uppercase;margin-bottom:10px;">
                            Report Contents</div>
                          {"".join(f'<div style="display:flex;gap:7px;padding:4px 0;'
                                   f'border-bottom:1px solid #1e293b;">'
                                   f'<span style="color:#10b981;">{ic}</span>'
                                   f'<span style="font-size:11px;color:#94a3b8;">{tx}</span>'
                                   f'</div>'
                                   for ic, tx in [
                                     ("✓","Patient Information Header"),
                                     ("✓","Embedded MRI Image"),
                                     ("✓","Primary Diagnosis & Confidence"),
                                     ("✓","All-Class Probability Table"),
                                     ("✓","Risk Level Indicator"),
                                     ("✓","Clinical Interpretation Notes"),
                                     ("✓","Model Performance Metrics"),
                                     ("✓","Medical Disclaimer & Footer"),
                                   ])}
                          <div style="margin-top:8px;font-size:9px;color:#475569;">
                            PDF via ReportLab {'✓' if _PDF else '✗ (pip install reportlab)'}</div>
                        </div>""")

            # ── Tab 4: About ──────────────────────────────────────────────
            with gr.Tab("ℹ️ About"):
                gr.Markdown(f"""
## NeuroScan AI — Brain Tumor Classification

### Architecture
| Model | Role | Ensemble Weight |
|-------|------|----------------|
| EfficientNet-B3 | Compound-scaled CNN | 40% |
| ResNet50 + CBAM | Channel & Spatial Attention | 30% |
| DenseNet121 | Dense feature reuse | 30% |

**Inference**: TTA (6 augmented views: original, H-flip, V-flip, rot90/180/270) + Temperature Calibration

### Performance (Test Set — 1,311 images)
| Metric | Score | Target | Status |
|--------|-------|--------|--------|
| AUC (macro OvR) | **0.9996** | ≥ 0.995 | ✅ PASS |
| Accuracy        | **98.78%** | ≥ 99.1% | ⚠ 0.32% gap |
| Sensitivity     | **98.68%** | ≥ 96%   | ✅ PASS |
| Specificity     | **99.59%** | ≥ 99%   | ✅ PASS |

**Baseline (Amin et al. 2020 — SVM+GLCM):** AUC 0.980, Accuracy 97.1%

### Classes
- **Glioma** — Malignant glial-cell tumor
- **Meningioma** — Extra-axial meningeal tumor (usually benign)
- **Pituitary** — Sellar/pituitary adenoma
- **No Tumor** — Normal / no significant mass

### Training Details
- Dataset: 4,569 train / 1,143 val / 1,311 test
- Phase 1: Frozen backbone, Adam lr=1e-3, OneCycleLR (5 epochs)
- Phase 2: Full fine-tune, AdamW lr=1e-4, CosineAnnealing (20 epochs)
- Augmentation: MixUp(α=0.3), CutMix(α=0.4) + geometric transforms
- Loss: CrossEntropyLoss + label_smoothing=0.1 + class weights
- AMP (float16) training on NVIDIA T4 GPU

### Version
NeuroScan AI v{APP_VERSION} · Built with PyTorch · timm · Gradio · ReportLab
                """)

        # ── CALLBACKS ──────────────────────────────────────────────────────

        def on_analyze(image, pid, count):
            _gc_none  = (None, "", None, None, None)   # 5 GradCAM placeholders
            _seg_none = (None, "", None, None)          # 4 Segmentation placeholders
            if image is None:
                empty = ('<div style="text-align:center;color:#ef4444;padding:20px;'
                         'font-family:Inter,sans-serif;">⚠ Please upload an MRI image first.</div>')
                return (empty, None, None, None, None, None, count,
                        '<div style="color:#ef4444;font-size:11px;text-align:center;">No image</div>',
                        *_gc_none, *_seg_none)
            try:
                r = run_inference(model_obj, image)
                if "error" in r:
                    raise ValueError(r["error"])
                count = (count or 0) + 1
                st_html = (f'<div style="text-align:center;color:#10b981;font-size:11px;">'
                           f'✓ Analysis complete · Session: {count} image(s) analyzed</div>')

                # ── Build orig_rgb once — shared by GradCAM + Segmenter ──
                if isinstance(image, np.ndarray):
                    orig_rgb = image.copy()
                    if orig_rgb.dtype != np.uint8:
                        orig_rgb = np.clip(orig_rgb, 0, 255).astype(np.uint8)
                    if orig_rgb.ndim == 2:
                        orig_rgb = cv2.cvtColor(orig_rgb, cv2.COLOR_GRAY2RGB)
                    elif orig_rgb.ndim == 3 and orig_rgb.shape[2] == 4:
                        orig_rgb = cv2.cvtColor(orig_rgb, cv2.COLOR_RGBA2RGB)
                else:
                    orig_rgb = np.array(
                        Image.fromarray(np.array(image)).convert("RGB"), dtype=np.uint8)

                # ── Grad-CAM ──────────────────────────────────────────────
                gc_composite = None
                gc_html_str  = ""
                gc_loc_img   = None
                gc_prof_img  = None
                gc_result    = None
                if gradcam_engine is not None and _GRADCAM:
                    try:

                        tensor_gc = preprocess(image)
                        if tensor_gc is not None:
                            tensor_gc = tensor_gc.to(DEVICE)
                            gc_result = gradcam_engine.generate(tensor_gc, r, orig_rgb)
                            vis = gc_result["visuals"]
                            stats = gc_result["stats"]
                            gc_composite = vis["composite"]   # [224, 896, 3] uint8
                            gc_html_str  = gradcam_stats_html(
                                stats, gc_result["elapsed"], gc_result["demo"])
                            gc_loc_img   = plot_tumor_location(
                                stats["centroid"], stats["bbox"],
                                stats["lateralization"], orig_rgb.shape[:2],
                                gc_result["cam"],
                            )
                            gc_prof_img  = plot_activation_profile(
                                gc_result["cam"], r["class_label"])
                            logger.info(
                                f"[GradCAM] done ({gc_result['elapsed']:.2f}s, "
                                f"demo={gc_result['demo']})")
                    except Exception as eg:
                        logger.warning(f"[GradCAM] failed during UI run: {eg}")

                # ── U-Net Segmentation ────────────────────────────────────
                seg_panel_arr = None
                seg_html_str  = ""
                seg_cmp_arr   = None
                seg_result    = None
                if segmenter is not None and _SEG:
                    try:
                        seg_result = segmenter.segment(orig_rgb)

                        # Compare with GradCAM if available
                        if gc_result is not None:
                            seg_result = segmenter.compare_with_gradcam(
                                seg_result, gc_result["cam"])
                            seg_cmp_arr = plot_seg_vs_gradcam(
                                gc_result["cam"],
                                seg_result["cam_mask"],
                                seg_result["binary_mask"],
                                orig_rgb,
                                dice=seg_result.get("dice_vs_cam"),
                                iou=seg_result.get("iou_vs_cam"),
                            )

                        seg_panel_arr = plot_seg_panel(
                            original    = orig_rgb,
                            prob_map    = seg_result["prob_map"],
                            overlay     = seg_result["overlay"],
                            contour     = seg_result["contour"],
                            area_pct    = seg_result["area_pct"],
                            dice_vs_cam = seg_result.get("dice_vs_cam"),
                            iou_vs_cam  = seg_result.get("iou_vs_cam"),
                            demo        = seg_result["demo"],
                        )
                        seg_html_str = _seg_html_fn(seg_result)
                        logger.info(
                            f"[Seg] done ({seg_result['elapsed']:.2f}s, "
                            f"area={seg_result['area_pct']:.1f}%, "
                            f"demo={seg_result['demo']})")
                    except Exception as es:
                        logger.warning(f"[Seg] failed: {es}")

                return (
                    diagnosis_card(r),
                    chart_gauge(r["confidence"], r["class"]),
                    chart_prob_bars(r["probabilities"]),
                    chart_radar(r["individual"]),
                    r, image, count, st_html,
                    gc_composite, gc_html_str, gc_loc_img, gc_prof_img, gc_result,
                    seg_panel_arr, seg_html_str, seg_cmp_arr, seg_result,
                )
            except Exception as e:
                logger.error(f"Analysis: {e}")
                err = (f'<div style="text-align:center;color:#ef4444;padding:20px;'
                       f'font-family:Inter,sans-serif;">⚠ {e}</div>')
                return (err, None, None, None, None, None, count,
                        f'<div style="color:#ef4444;font-size:11px;">Error: {e}</div>',
                        *_gc_none, *_seg_none)

        run_btn.click(
            fn=on_analyze,
            inputs=[img_in, pid_in, count_st],
            outputs=[diag_out, gauge_out, prob_out,
                     radar_out, result_st, image_st, count_st, status_out,
                     gradcam_img, gradcam_html, gradcam_loc, gradcam_prof, gradcam_st,
                     seg_panel, seg_html_out, seg_cmp, seg_st],
        )

        # Example buttons — use closure to avoid numpy-array default-arg bug
        def _make_loader(a):
            def _load(): return a
            return _load
        for btn, arr in ex_btns:
            btn.click(fn=_make_loader(arr), inputs=[], outputs=[img_in])

        # PDF report
        def on_report(r_st, i_st, gc_st, name, pid):
            if r_st is None:
                return ('<div style="color:#ef4444;font-size:11px;">'
                        '⚠ Run a diagnosis first.</div>',
                        gr.update(visible=False))
            if not _PDF:
                return ('<div style="color:#f59e0b;font-size:11px;">'
                        '⚠ Install reportlab: pip install reportlab</div>',
                        gr.update(visible=False))
            try:
                # Use enhanced GradCAM-aware report when available
                if _GRADCAM and gc_st is not None:
                    path = _generate_report_v2(r_st, i_st, gc_st, name, pid)
                else:
                    path = generate_pdf(r_st, i_st, name, pid)
                if path and Path(path).exists():
                    return ('<div style="color:#10b981;font-size:11px;">'
                            '✓ Report generated!</div>',
                            gr.update(visible=True, value=path))
                return ('<div style="color:#ef4444;font-size:11px;">PDF failed.</div>',
                        gr.update(visible=False))
            except Exception as e:
                logger.error(f"PDF: {e}")
                return (f'<div style="color:#ef4444;font-size:11px;">Error: {e}</div>',
                        gr.update(visible=False))

        rep_btn.click(
            fn=on_report,
            inputs=[result_st, image_st, gradcam_st, rep_name, rep_id],
            outputs=[rep_st, pdf_dl],
        )

    return demo

# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    logger.info(f"{'='*50}")
    logger.info(f"  {APP_NAME} v{APP_VERSION}")
    logger.info(f"  Device  : {DEVICE}")
    logger.info(f"  timm    : {_TIMM}")
    logger.info(f"  reportlab: {_PDF}")
    logger.info(f"{'='*50}")

    model_obj, model_mode = load_model()
    logger.info(f"Model mode: {model_mode}")

    gradcam_engine = None
    if _GRADCAM:
        try:
            gradcam_engine = EnsembleGradCAM(model_obj, model_mode)
            logger.info("GradCAM engine ready")
        except Exception as _eg:
            logger.warning(f"GradCAM engine init failed: {_eg}")

    segmenter = None
    if _SEG:
        try:
            segmenter = TumorSegmenter()
            logger.info(
                f"Segmenter ready  (demo={segmenter.demo})")
        except Exception as _es:
            logger.warning(f"Segmenter init failed: {_es}")

    demo = build_ui(model_obj, model_mode, gradcam_engine, segmenter)

    port = int(os.environ.get("GRADIO_SERVER_PORT", 7862))
    logger.info(f"Launching at http://localhost:{port}")
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        inbrowser=True,
        show_error=True,
        share=False,
    )


if __name__ == "__main__":
    main()
