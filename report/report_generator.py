"""
report/report_generator.py
===========================
3-page professional radiology PDF report for NeuroScan AI.

Usage
-----
    from report.report_generator import generate_report
    pdf_path = generate_report(
        result, image_np, gradcam_result,
        patient_name="John Doe", patient_id="PT-001",
    )
    # returns absolute path string

Dependencies: reportlab >= 4.0 (primary)  |  fpdf2 (fallback)
"""
from __future__ import annotations

import io
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ── Dirs ──────────────────────────────────────────────────────────────────
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Try reportlab ─────────────────────────────────────────────────────────
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors as rlc
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table,
        TableStyle, Image as RLImage, HRFlowable, PageBreak,
        KeepTogether,
    )
    _RL = True
except ImportError:
    _RL = False
    logger.warning("reportlab not installed; PDF generation disabled")

# ── Colour palette ────────────────────────────────────────────────────────
_NAVY  = "#0f172a"
_SLATE = "#1e293b"
_CYAN  = "#00d4ff"
_LGRAY = "#cbd5e1"
_GRAY  = "#334155"
_RED   = "#ef4444"
_AMB   = "#f59e0b"
_GRN   = "#10b981"
_BLU   = "#3b82f6"
_PUR   = "#a855f7"

CLASS_COLORS = {
    "glioma":     _RED,
    "meningioma": _AMB,
    "notumor":    _GRN,
    "pituitary":  _BLU,
}

RISK_MAP = {
    "glioma":     ("CRITICAL", _RED),
    "meningioma": ("HIGH",     _AMB),
    "pituitary":  ("MODERATE", _BLU),
    "notumor":    ("LOW",      _GRN),
}

CLINICAL_TEXT = {
    "glioma": (
        "The AI model identifies imaging features consistent with a glial neoplasm. "
        "The Grad-CAM localisation map highlights a region of interest with high "
        "activation intensity. Gliomas represent the most common primary brain tumours "
        "and vary widely in grade (WHO I–IV). Immediate referral to neurosurgery and "
        "neuro-oncology is strongly recommended. MRI with contrast enhancement, "
        "spectroscopy, and perfusion imaging should be obtained. Biopsy and molecular "
        "profiling (IDH status, MGMT methylation) may be required for definitive grading."
    ),
    "meningioma": (
        "Imaging pattern is consistent with a meningeal-based lesion, potentially a "
        "meningioma. These are typically benign (WHO Grade I) but may cause symptoms "
        "due to mass effect. Contrast-enhanced MRI is advised to characterise the lesion. "
        "Neurosurgical consultation and close radiological follow-up are recommended. "
        "Grading requires histopathological confirmation."
    ),
    "pituitary": (
        "Findings raise the possibility of a pituitary region lesion, potentially a "
        "pituitary adenoma or sellar tumour. Dedicated pituitary protocol MRI, visual "
        "field assessment, and endocrinology referral are recommended. Hormonal panel "
        "(prolactin, GH, ACTH, TSH, FSH/LH) should be evaluated. Correlation with "
        "clinical symptoms (headache, visual changes, endocrine dysfunction) is essential."
    ),
    "notumor": (
        "No significant intracranial mass lesion identified by AI analysis. The Grad-CAM "
        "map shows no focal area of high activation consistent with tumour pathology. "
        "Routine clinical follow-up is advised as clinically indicated. If symptoms "
        "persist or new neurological signs develop, repeat imaging should be considered. "
        "This result does not exclude all pathologies — clinical correlation is essential."
    ),
}

RECOMMENDATIONS = {
    "glioma": [
        "Urgent MRI brain with gadolinium contrast enhancement",
        "Neurosurgical consultation within 48 hours",
        "Neuro-oncology multidisciplinary team review",
        "Biopsy / resection planning (IDH, MGMT, EGFR profiling)",
        "Baseline neuropsychological assessment",
    ],
    "meningioma": [
        "Contrast-enhanced MRI for lesion characterisation",
        "Neurosurgical consultation and watchful waiting assessment",
        "Annual imaging follow-up if conservative management chosen",
        "Visual field testing if lesion is near optic structures",
    ],
    "pituitary": [
        "Dedicated pituitary protocol MRI (thin coronal slices)",
        "Endocrinology referral for hormonal evaluation",
        "Visual field examination (Goldman / Humphrey perimetry)",
        "Ophthalmology review if bitemporal hemianopia suspected",
    ],
    "notumor": [
        "Correlate with clinical history and neurological examination",
        "No urgent imaging follow-up required based on AI screening",
        "Repeat MRI in 6–12 months if symptoms persist",
        "Consider EEG if seizures are present",
    ],
}


# ── BytesIO image helper ───────────────────────────────────────────────────

def _np_to_rl_image(
    arr: np.ndarray,
    width_cm: float,
    height_cm: float,
    quality: int = 88,
) -> "RLImage":
    """Convert a numpy RGB array to a ReportLab Image object via BytesIO."""
    from PIL import Image as PILImage
    pil = PILImage.fromarray(arr.astype(np.uint8)).convert("RGB")
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return RLImage(buf, width=width_cm * cm, height=height_cm * cm)


# ── Style helpers ─────────────────────────────────────────────────────────

def _hex(h: str):
    return rlc.HexColor("#" + h.lstrip("#"))


def _sty(name: str, **kw) -> "ParagraphStyle":
    return ParagraphStyle(name, **kw)


# ── Bar string builder ────────────────────────────────────────────────────

def _bar(pct: float, width: int = 20) -> str:
    filled = int(pct / 100 * width)
    return "█" * filled + "░" * (width - filled)


# ── Main generate function ─────────────────────────────────────────────────

def generate_report(
    result:        dict,
    image:         Optional[np.ndarray],
    gradcam_result: Optional[dict],
    patient_name:  str = "",
    patient_id:    str = "",
) -> str:
    """
    Generate a 3-page PDF radiology report.

    Parameters
    ----------
    result         : prediction dict from run_inference()
    image          : original MRI as uint8 RGB numpy array
    gradcam_result : output from EnsembleGradCAM.generate()
    patient_name   : optional patient name string
    patient_id     : optional patient ID string

    Returns
    -------
    Absolute path string of saved PDF.
    Raises RuntimeError if reportlab is unavailable.
    """
    if not _RL:
        raise RuntimeError(
            "reportlab not installed. Run: pip install reportlab"
        )

    # ── Meta ──────────────────────────────────────────────────────────
    rid       = str(uuid.uuid4())[:8].upper()
    now       = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S IST")
    date_str  = now.strftime("%B %d, %Y")
    time_str  = now.strftime("%H:%M:%S")

    cls   = result["class"]
    label = result["class_label"]
    conf  = result["confidence"] * 100
    risk, risk_hex = RISK_MAP.get(cls, ("UNKNOWN", _GRAY))
    probs = result["probabilities"]
    ela   = result.get("elapsed", 0)
    indiv = result.get("individual", {})

    cls_col  = _hex(CLASS_COLORS.get(cls, _GRAY))
    risk_col = _hex(risk_hex)

    NAVY  = _hex(_NAVY)
    SLATE = _hex(_SLATE)
    CYAN  = _hex(_CYAN)
    LGRAY = _hex(_LGRAY)
    GRAY  = _hex(_GRAY)
    RED   = _hex(_RED)
    GRN   = _hex(_GRN)

    pid_clean  = (patient_id or "ANON").replace(" ", "_").replace("/", "-")
    pname_disp = patient_name or "Anonymous"
    pid_disp   = patient_id   or "N/A"

    date_file = now.strftime("%Y%m%d_%H%M")
    filename  = f"NeuroScan_Report_{pid_clean}_{date_file}.pdf"
    out_path  = REPORTS_DIR / filename

    doc   = SimpleDocTemplate(
        str(out_path),
        pagesize=A4,
        topMargin=1.4 * cm, bottomMargin=1.4 * cm,
        leftMargin=1.8 * cm, rightMargin=1.8 * cm,
    )
    W_pt, H_pt = A4                      # 595 x 842 pts
    usable_w   = W_pt - 3.6 * cm        # ~ 16 cm

    story = []

    # ================================================================
    # ── Shared styles ────────────────────────────────────────────────
    # ================================================================
    h1  = _sty("h1",  fontName="Helvetica-Bold", fontSize=22,
                textColor=rlc.white, alignment=TA_LEFT)
    h2  = _sty("h2",  fontName="Helvetica-Bold", fontSize=11,
                textColor=CYAN, alignment=TA_RIGHT)
    sec = _sty("sec", fontName="Helvetica-Bold", fontSize=10,
                textColor=CYAN, spaceBefore=6, spaceAfter=3)
    sub = _sty("sub", fontName="Helvetica-Bold", fontSize=8,
                textColor=LGRAY, spaceBefore=4, spaceAfter=2)
    bod = _sty("bod", fontName="Helvetica", fontSize=8.5,
                textColor=LGRAY, leading=13, spaceAfter=3,
                alignment=TA_JUSTIFY)
    sm  = _sty("sm",  fontName="Helvetica", fontSize=7,
                textColor=GRAY, alignment=TA_CENTER)
    sm_l = _sty("sm_l", fontName="Helvetica", fontSize=7,
                 textColor=GRAY)
    warn = _sty("warn", fontName="Helvetica-Oblique", fontSize=7,
                textColor=GRAY, leading=10.5, alignment=TA_JUSTIFY)
    mono = _sty("mono", fontName="Courier", fontSize=8,
                textColor=_hex(_GRN), leading=12)

    # Helper for horizontal rule
    def _hr(thickness=0.5, color=CYAN):
        return HRFlowable(
            width="100%", thickness=thickness,
            color=color, spaceAfter=5, spaceBefore=3,
        )

    # ================================================================
    # PAGE 1 — HEADER
    # ================================================================

    # ── Banner ────────────────────────────────────────────────────────
    banner_data = [[
        Paragraph("<b>🧠 NEUROSCAN AI</b>", h1),
        Paragraph(
            "DIAGNOSTIC REPORT<br/>"
            "<font size='7' color='#64748b'>Brain MRI Analysis — AI Assisted</font>",
            h2,
        ),
    ]]
    banner_tbl = Table(banner_data,
                       colWidths=[usable_w * 0.60, usable_w * 0.40])
    banner_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), NAVY),
        ("LINEBELOW",     (0, 0), (-1, -1), 2.5, CYAN),
        ("TOPPADDING",    (0, 0), (-1, -1), 14),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 14),
        ("LEFTPADDING",   (0, 0), (-1, -1), 12),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 12),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(banner_tbl)
    story.append(Spacer(1, 0.25 * cm))

    # ── Report + Patient meta ─────────────────────────────────────────
    meta_data = [
        ["Report ID:",  f"RPT-{rid}",            "Generated:", timestamp],
        ["Patient:",    pname_disp,              "Study Date:", date_str],
        ["Patient ID:", pid_disp,                "Modality:",   "MRI Brain (AI Inferred)"],
        ["Model:",      "BrainTumorAI v2.0 Ensemble",
         "Model AUC:",  "0.9996 | Acc 98.78%"],
    ]
    meta_widths = [2.6 * cm, 5.8 * cm, 2.8 * cm, 5.8 * cm]
    meta_tbl = Table(meta_data, colWidths=meta_widths)
    meta_tbl.setStyle(TableStyle([
        ("FONTNAME",      (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME",      (2, 0), (2, -1), "Helvetica-Bold"),
        ("FONTNAME",      (1, 0), (1, -1), "Helvetica"),
        ("FONTNAME",      (3, 0), (3, -1), "Helvetica"),
        ("FONTSIZE",      (0, 0), (-1, -1), 7.5),
        ("TEXTCOLOR",     (0, 0), (0, -1), CYAN),
        ("TEXTCOLOR",     (2, 0), (2, -1), CYAN),
        ("TEXTCOLOR",     (1, 0), (1, -1), LGRAY),
        ("TEXTCOLOR",     (3, 0), (3, -1), LGRAY),
        ("BACKGROUND",    (0, 0), (-1, -1), SLATE),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1),
         [SLATE, _hex(_NAVY)]),
        ("GRID",          (0, 0), (-1, -1), 0.25, _hex("#334155")),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
    ]))
    story.append(meta_tbl)
    story.append(Spacer(1, 0.25 * cm))
    story.append(_hr(1.5, CYAN))

    # ── PRIMARY FINDING ───────────────────────────────────────────────
    story.append(Paragraph("PRIMARY FINDING", sec))

    bar20 = _bar(conf, 20)
    risk_icon = {"CRITICAL": "🔴", "HIGH": "🟠",
                 "MODERATE": "🔵", "LOW": "🟢"}.get(risk, "⚪")

    # Auto-scale font so every tumor name fits on ONE line in the 6.5 cm column
    # (usable ≈ 164 pt after 10 pt padding each side)
    # Helvetica-Bold glyph widths (units/1000 em) used for sizing:
    #   GLIOMA(6)=3835  NO TUMOR(8)=5277  PITUITARY(9)=5060  MENINGIOMA(10)=6448
    label_upper = label.upper()
    _fn_size = (28 if len(label_upper) <= 6 else
                24 if len(label_upper) <= 8 else
                20 if len(label_upper) <= 9 else 18)

    find_data = [[
        Paragraph(
            f"<b>{label_upper}</b>",
            _sty("fn", fontName="Helvetica-Bold", fontSize=_fn_size,
                 textColor=cls_col, alignment=TA_CENTER,
                 leading=_fn_size + 4),
        ),
        Paragraph(
            f"<b>Diagnosis:</b> {label.upper()}<br/>"
            f"<b>Confidence:</b> {bar20}  {conf:.1f}%<br/>"
            f"<b>Risk Level:</b> {risk_icon} {risk}<br/>"
            f"<b>Processing:</b> {ela:.2f}s (TTA 6-view ensemble)<br/>"
            f"<b>Method:</b> EfficientNet-B3 + ResNet50+CBAM + DenseNet121",
            _sty("fi", fontName="Helvetica", fontSize=8.5,
                 textColor=LGRAY, leading=14),
        ),
    ]]
    find_tbl = Table(find_data,
                     colWidths=[6.5 * cm, usable_w - 6.5 * cm])
    find_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (0, 0), NAVY),
        ("BACKGROUND",    (1, 0), (1, 0), SLATE),
        ("LINEAFTER",     (0, 0), (0, -1), 3, cls_col),
        ("GRID",          (0, 0), (-1, -1), 0.25, _hex("#334155")),
        ("TOPPADDING",    (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(find_tbl)
    story.append(Spacer(1, 0.25 * cm))

    # ── DIFFERENTIAL DIAGNOSIS ────────────────────────────────────────
    story.append(Paragraph("DIFFERENTIAL DIAGNOSIS", sec))

    CLASS_NAMES  = ["glioma", "meningioma", "notumor", "pituitary"]
    CLASS_LABELS = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

    diff_rows = [
        [Paragraph("<b>Class</b>", sub),
         Paragraph("<b>Probability</b>", sub),
         Paragraph("<b>Bar Chart</b>", sub),
         Paragraph("<b>Notes</b>", sub)],
    ]
    for cn, cl in zip(CLASS_NAMES, CLASS_LABELS):
        p   = probs.get(cn, 0) * 100
        bar = _bar(p, 16)
        note = "← Primary" if cn == cls else ""
        diff_rows.append([
            Paragraph(cl, _sty(f"d{cn}", fontName="Helvetica",
                                fontSize=8, textColor=_hex(CLASS_COLORS[cn]))),
            Paragraph(f"{p:.1f}%", _sty(f"dp{cn}", fontName="Helvetica-Bold",
                                        fontSize=8, textColor=LGRAY)),
            Paragraph(f'<font color="#{_GRN.lstrip("#")}">{bar}</font>',
                      _sty(f"db{cn}", fontName="Courier", fontSize=7.5)),
            Paragraph(note, _sty(f"dn{cn}", fontName="Helvetica-Oblique",
                                 fontSize=7, textColor=GRAY)),
        ])

    diff_tbl = Table(diff_rows,
                     colWidths=[2.8*cm, 2.2*cm, 8.0*cm, 4.0*cm])
    diff_tbl.setStyle(TableStyle([
        ("BACKGROUND",     (0, 0), (-1, 0), SLATE),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [_hex(_NAVY), SLATE]),
        ("GRID",           (0, 0), (-1, -1), 0.2, _hex("#334155")),
        ("TOPPADDING",     (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",  (0, 0), (-1, -1), 4),
        ("LEFTPADDING",    (0, 0), (-1, -1), 6),
        ("LINEBELOW",      (0, 0), (-1, 0), 1, CYAN),
    ]))
    story.append(diff_tbl)
    story.append(Spacer(1, 0.25 * cm))

    # ── ENSEMBLE AGREEMENT ────────────────────────────────────────────
    if indiv:
        story.append(Paragraph("ENSEMBLE AGREEMENT", sec))
        model_labels = {
            "efficientnet": "EfficientNet-B3",
            "resnet_cbam":  "ResNet50+CBAM",
            "densenet":     "DenseNet121",
        }
        ens_rows = [
            [Paragraph("<b>Model</b>", sub),
             Paragraph("<b>Confidence</b>", sub),
             Paragraph("<b>Bar</b>", sub)],
        ]
        for mname, mprobs in indiv.items():
            p = mprobs.get(cls, 0) * 100
            ens_rows.append([
                Paragraph(model_labels.get(mname, mname),
                          _sty(f"em{mname}", fontName="Helvetica",
                               fontSize=8, textColor=LGRAY)),
                Paragraph(f"{p:.1f}%",
                          _sty(f"ep{mname}", fontName="Helvetica-Bold",
                               fontSize=8, textColor=LGRAY)),
                Paragraph(f'<font color="#{_CYAN.lstrip("#")}">'
                          f'{_bar(p, 14)}</font>',
                          _sty(f"eb{mname}", fontName="Courier", fontSize=7.5)),
            ])
        ens_rows.append([
            Paragraph("<b>Ensemble (TTA)</b>",
                      _sty("eens", fontName="Helvetica-Bold",
                           fontSize=8, textColor=CYAN)),
            Paragraph(f"<b>{conf:.1f}%</b>",
                      _sty("eensp", fontName="Helvetica-Bold",
                           fontSize=8, textColor=CYAN)),
            Paragraph(f'<font color="#{_CYAN.lstrip("#")}">'
                      f'← FINAL</font>',
                      _sty("eensb", fontName="Helvetica-Bold",
                           fontSize=8, textColor=CYAN)),
        ])
        ens_tbl = Table(ens_rows,
                        colWidths=[4.5*cm, 3.0*cm, 9.5*cm])
        ens_tbl.setStyle(TableStyle([
            ("BACKGROUND",     (0, 0), (-1, 0), SLATE),
            ("ROWBACKGROUNDS", (0, 1), (-1, -2),
             [_hex(_NAVY), SLATE]),
            ("BACKGROUND",     (0, -1), (-1, -1), _hex("#1e3a5f")),
            ("GRID",           (0, 0), (-1, -1), 0.2, _hex("#334155")),
            ("LINEBELOW",      (0, 0), (-1, 0), 1, CYAN),
            ("LINEABOVE",      (0, -1), (-1, -1), 1, CYAN),
            ("TOPPADDING",     (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING",  (0, 0), (-1, -1), 4),
            ("LEFTPADDING",    (0, 0), (-1, -1), 6),
        ]))
        story.append(ens_tbl)

    # ================================================================
    # PAGE 2 — CLINICAL INTERPRETATION
    # ================================================================
    story.append(PageBreak())
    story.append(Paragraph("CLINICAL INTERPRETATION", sec))
    story.append(_hr())
    story.append(Paragraph(
        CLINICAL_TEXT.get(cls, "Clinical interpretation unavailable."),
        bod,
    ))
    story.append(Spacer(1, 0.2 * cm))

    # ── RECOMMENDATIONS ───────────────────────────────────────────────
    story.append(Paragraph("RECOMMENDATIONS", sec))
    for i, rec in enumerate(RECOMMENDATIONS.get(cls, []), 1):
        story.append(Paragraph(
            f"&nbsp;&nbsp;{i}.&nbsp; {rec}",
            _sty(f"rec{i}", fontName="Helvetica", fontSize=8.5,
                 textColor=LGRAY, leading=14, leftIndent=10),
        ))
    story.append(Spacer(1, 0.3 * cm))

    # ── DISCLAIMER ────────────────────────────────────────────────────
    story.append(_hr(0.5, _hex("#334155")))
    disc = (
        "<b>⚠ DISCLAIMER:</b> This report is generated by NeuroScan AI, an "
        "AI-assisted screening tool. It is intended for informational and research "
        "purposes only and does <b>NOT</b> constitute a medical diagnosis. All findings "
        "must be reviewed and confirmed by a qualified radiologist or neurologist before "
        "any clinical decision is made. Individual model performance may vary across "
        "patient populations and imaging protocols. "
        f"Reported metrics: AUC 0.9996, Accuracy 98.78%, Sensitivity 98.68%, "
        f"Specificity 99.59% (test set n=1,311 images). "
        f"Report generated: {timestamp} | Report ID: RPT-{rid}"
    )
    story.append(Paragraph(disc, warn))
    story.append(Spacer(1, 0.3 * cm))

    # ── FOOTER ────────────────────────────────────────────────────────
    ft_data = [[
        Paragraph(
            f"NeuroScan AI v2.0 · AUC 0.9996 · Acc 98.78% · "
            f"Report ID: RPT-{rid}",
            sm,
        ),
        Paragraph(
            f"Generated: {timestamp}",
            _sty("ftR", fontName="Helvetica", fontSize=7,
                 textColor=GRAY, alignment=TA_RIGHT),
        ),
    ]]
    ft_tbl = Table(ft_data,
                   colWidths=[usable_w * 0.62, usable_w * 0.38])
    ft_tbl.setStyle(TableStyle([
        ("LINEABOVE",     (0, 0), (-1, -1), 1.5, CYAN),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("LEFTPADDING",   (0, 0), (-1, -1), 0),
    ]))
    story.append(ft_tbl)
    story.append(Paragraph(
        "Confidential — For Medical Use Only",
        _sty("conf", fontName="Helvetica-Oblique", fontSize=7,
             textColor=GRAY, alignment=TA_CENTER),
    ))

    # ── Build ─────────────────────────────────────────────────────────
    doc.build(story)
    logger.info(f"[Report] Saved: {out_path}  ({out_path.stat().st_size // 1024} KB)")
    return str(out_path.resolve())
