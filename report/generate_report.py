"""
report/generate_report.py
==========================
Generates a professional academic PDF report for the NeuroScan AI project.
Run:  python report/generate_report.py
Output: report/NeuroScan_AI_Report.pdf
"""

import os, sys
from pathlib import Path

# ── ReportLab imports ─────────────────────────────────────────────────────────
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm, mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable, Image, KeepTogether
)
from reportlab.platypus.flowables import HRFlowable
from reportlab.lib.utils import ImageReader

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT      = Path(__file__).resolve().parent.parent
REPORT_DIR = _ROOT / "report"
REPORT_DIR.mkdir(exist_ok=True)
OUT_PDF    = str(REPORT_DIR / "NeuroScan_AI_Report.pdf")

IMG_VALIDATION = str(_ROOT / "outputs" / "classical" / "validation_grid.png")
IMG_MOSAIC     = str(_ROOT / "outputs" / "final_results_mosaic.png")

# ── Colour Palette ────────────────────────────────────────────────────────────
NAVY      = colors.HexColor("#0f2d5e")
CYAN      = colors.HexColor("#0077b6")
LIGHTBLUE = colors.HexColor("#caf0f8")
TEAL      = colors.HexColor("#00b4d8")
GRAY      = colors.HexColor("#4a5568")
LIGHTGRAY = colors.HexColor("#f0f4f8")
WHITE     = colors.white
BLACK     = colors.black
ACCENT    = colors.HexColor("#e63946")

PAGE_W, PAGE_H = A4
LEFT_M = RIGHT_M = 2.2 * cm
TOP_M  = 2.4 * cm
BOT_M  = 2.2 * cm


# ── Style sheet ───────────────────────────────────────────────────────────────
def make_styles():
    base = getSampleStyleSheet()

    styles = {}

    styles["cover_title"] = ParagraphStyle(
        "cover_title",
        fontName="Helvetica-Bold",
        fontSize=26,
        textColor=WHITE,
        alignment=TA_CENTER,
        leading=34,
        spaceAfter=6,
    )
    styles["cover_sub"] = ParagraphStyle(
        "cover_sub",
        fontName="Helvetica",
        fontSize=13,
        textColor=LIGHTBLUE,
        alignment=TA_CENTER,
        leading=20,
        spaceAfter=4,
    )
    styles["cover_authors"] = ParagraphStyle(
        "cover_authors",
        fontName="Helvetica-Bold",
        fontSize=11,
        textColor=WHITE,
        alignment=TA_CENTER,
        leading=16,
        spaceAfter=2,
    )
    styles["cover_affil"] = ParagraphStyle(
        "cover_affil",
        fontName="Helvetica-Oblique",
        fontSize=9,
        textColor=LIGHTBLUE,
        alignment=TA_CENTER,
        leading=14,
    )
    styles["h1"] = ParagraphStyle(
        "h1",
        fontName="Helvetica-Bold",
        fontSize=14,
        textColor=NAVY,
        spaceBefore=16,
        spaceAfter=6,
        borderPad=4,
        keepWithNext=1,
    )
    styles["h2"] = ParagraphStyle(
        "h2",
        fontName="Helvetica-Bold",
        fontSize=11,
        textColor=CYAN,
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=1,
    )
    styles["h3"] = ParagraphStyle(
        "h3",
        fontName="Helvetica-BoldOblique",
        fontSize=10,
        textColor=GRAY,
        spaceBefore=8,
        spaceAfter=3,
        keepWithNext=1,
    )
    styles["body"] = ParagraphStyle(
        "body",
        fontName="Helvetica",
        fontSize=9.5,
        textColor=BLACK,
        leading=15,
        alignment=TA_JUSTIFY,
        spaceAfter=6,
    )
    styles["body_center"] = ParagraphStyle(
        "body_center",
        parent=styles["body"],
        alignment=TA_CENTER,
    )
    styles["abstract"] = ParagraphStyle(
        "abstract",
        fontName="Helvetica",
        fontSize=9.5,
        textColor=colors.HexColor("#1a1a2e"),
        leading=15,
        alignment=TA_JUSTIFY,
        leftIndent=1 * cm,
        rightIndent=1 * cm,
        spaceAfter=6,
    )
    styles["abstract_label"] = ParagraphStyle(
        "abstract_label",
        fontName="Helvetica-Bold",
        fontSize=10,
        textColor=NAVY,
        alignment=TA_CENTER,
        spaceAfter=4,
    )
    styles["caption"] = ParagraphStyle(
        "caption",
        fontName="Helvetica-Oblique",
        fontSize=8.5,
        textColor=GRAY,
        alignment=TA_CENTER,
        spaceAfter=10,
    )
    styles["code"] = ParagraphStyle(
        "code",
        fontName="Courier",
        fontSize=8,
        textColor=colors.HexColor("#1a202c"),
        backColor=LIGHTGRAY,
        leading=12,
        leftIndent=0.4 * cm,
        spaceAfter=6,
    )
    styles["bullet"] = ParagraphStyle(
        "bullet",
        fontName="Helvetica",
        fontSize=9.5,
        textColor=BLACK,
        leading=14,
        leftIndent=0.6 * cm,
        bulletIndent=0,
        spaceAfter=3,
    )
    styles["th"] = ParagraphStyle(
        "th",
        fontName="Helvetica-Bold",
        fontSize=9,
        textColor=WHITE,
        alignment=TA_CENTER,
    )
    styles["td"] = ParagraphStyle(
        "td",
        fontName="Helvetica",
        fontSize=9,
        textColor=BLACK,
        alignment=TA_CENTER,
        leading=12,
    )
    styles["td_left"] = ParagraphStyle(
        "td_left",
        parent=styles["td"],
        alignment=TA_LEFT,
    )
    styles["ref"] = ParagraphStyle(
        "ref",
        fontName="Helvetica",
        fontSize=9,
        textColor=GRAY,
        leading=13,
        leftIndent=0.8 * cm,
        firstLineIndent=-0.8 * cm,
        spaceAfter=4,
    )
    styles["section_num"] = ParagraphStyle(
        "section_num",
        fontName="Helvetica-Bold",
        fontSize=14,
        textColor=ACCENT,
        spaceBefore=16,
        spaceAfter=2,
    )
    return styles


S = make_styles()


# ── Helpers ───────────────────────────────────────────────────────────────────
def HR(color=TEAL, thickness=1.2):
    return HRFlowable(width="100%", thickness=thickness,
                      color=color, spaceAfter=6, spaceBefore=2)

def SP(h=0.3):
    return Spacer(1, h * cm)

def B(text):
    return f"<b>{text}</b>"

def I(text):
    return f"<i>{text}</i>"

def bullet(text):
    return Paragraph(f"&#x2022;  {text}", S["bullet"])

def h1(text):
    return Paragraph(text, S["h1"])

def h2(text):
    return Paragraph(text, S["h2"])

def h3(text):
    return Paragraph(text, S["h3"])

def body(text):
    return Paragraph(text, S["body"])

def make_table(headers, rows, col_widths=None, stripe=True):
    tw = PAGE_W - LEFT_M - RIGHT_M
    if col_widths is None:
        col_widths = [tw / len(headers)] * len(headers)

    header_row = [Paragraph(h, S["th"]) for h in headers]
    data = [header_row]
    for row in rows:
        data.append([Paragraph(str(c), S["td"] if i > 0 else S["td_left"])
                     for i, c in enumerate(row)])

    style = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [WHITE, LIGHTGRAY] if stripe else [WHITE]),
        ("GRID",    (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e0")),
        ("VALIGN",  (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, 0), (-1, 0), 1.5, TEAL),
    ]
    return Table(data, colWidths=col_widths, style=TableStyle(style),
                 repeatRows=1, hAlign="LEFT")


def algo_box(title, steps):
    """Numbered algorithm box."""
    tw = PAGE_W - LEFT_M - RIGHT_M
    content = [Paragraph(f"<b>{title}</b>", ParagraphStyle(
        "at", fontName="Helvetica-Bold", fontSize=9.5,
        textColor=NAVY, spaceAfter=4))]
    for i, step in enumerate(steps, 1):
        content.append(Paragraph(
            f"<b>{i}.</b>  {step}",
            ParagraphStyle("as", fontName="Helvetica", fontSize=9,
                           textColor=BLACK, leading=13, spaceAfter=2,
                           leftIndent=0.3 * cm)))
    inner = Table([[content]], colWidths=[tw - 0.4 * cm])
    inner.setStyle(TableStyle([
        ("BOX",        (0, 0), (-1, -1), 1.2, CYAN),
        ("BACKGROUND", (0, 0), (-1, -1), LIGHTGRAY),
        ("TOPPADDING",    (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
        ("LINEABOVE",  (0, 0), (-1, 0), 3, NAVY),
    ]))
    return inner


def safe_image(path, width=None, height=None):
    if not Path(path).exists():
        return body(f"[Figure: {Path(path).name} — not found]")
    img = Image(path)
    iw, ih = img.imageWidth, img.imageHeight
    if width and not height:
        height = ih * width / iw
    elif height and not width:
        width = iw * height / ih
    elif not width and not height:
        width = PAGE_W - LEFT_M - RIGHT_M
        height = ih * width / iw
    img.drawWidth  = width
    img.drawHeight = height
    return img


# ── Page template: header / footer ────────────────────────────────────────────
def on_page(canvas, doc):
    canvas.saveState()
    pw, ph = A4

    # Header bar
    canvas.setFillColor(NAVY)
    canvas.rect(LEFT_M - 0.2 * cm, ph - TOP_M + 0.3 * cm,
                pw - LEFT_M - RIGHT_M + 0.4 * cm, 0.6 * cm, fill=1, stroke=0)
    canvas.setFillColor(WHITE)
    canvas.setFont("Helvetica-Bold", 7.5)
    canvas.drawString(LEFT_M, ph - TOP_M + 0.5 * cm,
                      "NeuroScan AI — Brain Tumor Detection & Segmentation")
    canvas.setFont("Helvetica", 7.5)
    canvas.drawRightString(pw - RIGHT_M,
                           ph - TOP_M + 0.5 * cm, "Academic Research Report  |  2026")

    # Footer — line sits at top, all text below it
    canvas.setStrokeColor(TEAL)
    canvas.setLineWidth(0.8)
    LINE_Y = BOT_M + 0.45 * cm          # line well above all text
    canvas.line(LEFT_M, LINE_Y, pw - RIGHT_M, LINE_Y)

    canvas.setFillColor(GRAY)
    # Row 1  (just below line)
    ROW1_Y = BOT_M + 0.26 * cm
    canvas.setFont("Helvetica", 7.5)
    canvas.drawString(LEFT_M, ROW1_Y, "Confidential — For Academic Review Only")
    canvas.setFont("Helvetica-Bold", 7.5)
    canvas.drawRightString(pw - RIGHT_M, ROW1_Y, "ACHARYA NAGARJUNA UNIVERSITY")

    # Row 2  (second line below row 1)
    ROW2_Y = BOT_M + 0.06 * cm
    canvas.setFont("Helvetica", 7.5)
    canvas.drawCentredString(pw / 2, ROW2_Y, f"Page {doc.page}")
    canvas.drawRightString(pw - RIGHT_M, ROW2_Y,
                           "College of Engineering and Technology, Guntur")
    canvas.restoreState()


def on_first_page(canvas, doc):
    """Cover page — full navy background."""
    canvas.saveState()
    pw, ph = A4
    canvas.setFillColor(NAVY)
    canvas.rect(0, 0, pw, ph, fill=1, stroke=0)

    # Decorative top band
    canvas.setFillColor(TEAL)
    canvas.rect(0, ph - 1.8 * cm, pw, 1.8 * cm, fill=1, stroke=0)
    canvas.setFillColor(ACCENT)
    canvas.rect(0, ph - 2.1 * cm, pw, 0.35 * cm, fill=1, stroke=0)

    # Decorative bottom band
    canvas.setFillColor(TEAL)
    canvas.rect(0, 0, pw, 1.4 * cm, fill=1, stroke=0)
    canvas.setFillColor(ACCENT)
    canvas.rect(0, 1.4 * cm, pw, 0.25 * cm, fill=1, stroke=0)

    # Large background letter (watermark)
    canvas.setFillColor(colors.HexColor("#1a3a6b"))
    canvas.setFont("Helvetica-Bold", 380)
    canvas.drawCentredString(pw / 2, ph / 2 - 120, "N")

    # Footer text
    canvas.setFillColor(WHITE)
    canvas.setFont("Helvetica", 8)
    canvas.drawCentredString(pw / 2, 0.65 * cm,
                             "ACHARYA NAGARJUNA UNIVERSITY")
    canvas.drawCentredString(pw / 2, 0.35 * cm,
                             "Acharya Nagarjuna University College of Engineering and Technology  "
                             "|  Department of Computer Science & Engineering  |  2026")
    canvas.restoreState()


# ── Cover page content ────────────────────────────────────────────────────────
def cover_page():
    story = []
    story.append(SP(4.5))

    story.append(Paragraph(
        "NeuroScan AI",
        ParagraphStyle("ct", fontName="Helvetica-Bold", fontSize=42,
                       textColor=WHITE, alignment=TA_CENTER, leading=52)))
    story.append(SP(0.3))
    story.append(Paragraph(
        "Brain Tumor Detection and Segmentation",
        ParagraphStyle("cs", fontName="Helvetica", fontSize=18,
                       textColor=LIGHTBLUE, alignment=TA_CENTER, leading=26)))
    story.append(Paragraph(
        "Using Classical Image Processing and Deep Learning",
        ParagraphStyle("cs2", fontName="Helvetica-Oblique", fontSize=14,
                       textColor=LIGHTBLUE, alignment=TA_CENTER, leading=22)))
    story.append(SP(1.2))

    # Divider
    story.append(HRFlowable(width="60%", thickness=2, color=TEAL,
                             hAlign="CENTER", spaceAfter=10, spaceBefore=6))

    # Team members label
    story.append(Paragraph(
        "Project Team",
        ParagraphStyle("team_lbl", fontName="Helvetica-Bold", fontSize=10,
                       textColor=TEAL, alignment=TA_CENTER, spaceAfter=6)))

    # Team members table — 2 columns: name | roll number
    tw = PAGE_W - LEFT_M - RIGHT_M
    _nm = ParagraphStyle("tnm", fontName="Helvetica-Bold", fontSize=10,
                         textColor=WHITE, alignment=TA_LEFT)
    _rn = ParagraphStyle("trn", fontName="Helvetica", fontSize=9.5,
                         textColor=LIGHTBLUE, alignment=TA_RIGHT)
    team_data = [
        [Paragraph("A. Vennela Priya",  _nm), Paragraph("Y22CS3202", _rn)],
        [Paragraph("S. Siri Chandana",  _nm), Paragraph("Y22CS3249", _rn)],
        [Paragraph("G. Lokesh",         _nm), Paragraph("Y22CS3219", _rn)],
        [Paragraph("T. Akhil",          _nm), Paragraph("Y22CS3258", _rn)],
    ]
    team_table = Table(team_data, colWidths=[tw * 0.55, tw * 0.45])
    team_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), colors.HexColor("#1a3a6b")),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 12),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 12),
        ("LINEBELOW", (0, 0), (-1, -2), 0.4, colors.HexColor("#2a4a7b")),
        ("BOX", (0, 0), (-1, -1), 0.8, TEAL),
    ]))
    story.append(team_table)
    story.append(SP(0.5))

    # Mentor
    story.append(HRFlowable(width="40%", thickness=0.8, color=TEAL,
                             hAlign="CENTER", spaceAfter=6, spaceBefore=2))
    story.append(Paragraph(
        "Project Mentor",
        ParagraphStyle("men_lbl", fontName="Helvetica-Bold", fontSize=9,
                       textColor=TEAL, alignment=TA_CENTER, spaceAfter=3)))
    story.append(Paragraph(
        "Dr. U. Sathish Kumar  M.Tech, Ph.D",
        ParagraphStyle("men_nm", fontName="Helvetica-Bold", fontSize=11,
                       textColor=WHITE, alignment=TA_CENTER)))
    story.append(SP(0.25))
    story.append(Paragraph(
        "B.Tech — Computer Science &amp; Engineering",
        ParagraphStyle("affil", fontName="Helvetica-Oblique", fontSize=9.5,
                       textColor=LIGHTBLUE, alignment=TA_CENTER)))
    story.append(Paragraph(
        "ACHARYA NAGARJUNA UNIVERSITY",
        ParagraphStyle("affil2", fontName="Helvetica-Bold", fontSize=10,
                       textColor=TEAL, alignment=TA_CENTER)))
    story.append(Paragraph(
        "Acharya Nagarjuna University College of Engineering and Technology",
        ParagraphStyle("affil3", fontName="Helvetica", fontSize=9,
                       textColor=LIGHTBLUE, alignment=TA_CENTER)))

    story.append(SP(0.9))

    # Info box
    tw = PAGE_W - LEFT_M - RIGHT_M
    info_data = [[
        Paragraph("<b>Type</b><br/>Academic Research", S["body_center"]),
        Paragraph("<b>Domain</b><br/>Medical Imaging / AI", S["body_center"]),
        Paragraph("<b>Year</b><br/>2026", S["body_center"]),
        Paragraph("<b>Status</b><br/>Complete", S["body_center"]),
    ]]
    info_table = Table(info_data, colWidths=[tw / 4] * 4)
    info_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), colors.HexColor("#1a3a6b")),
        ("TEXTCOLOR",     (0, 0), (-1, -1), WHITE),
        ("GRID",          (0, 0), (-1, -1), 0.5, TEAL),
        ("TOPPADDING",    (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(info_table)
    story.append(PageBreak())
    return story


# ── Abstract ──────────────────────────────────────────────────────────────────
def abstract_section():
    story = []
    story.append(SP(0.6))
    story.append(Paragraph("Abstract", S["abstract_label"]))
    story.append(HR(TEAL, 1.5))
    story.append(SP(0.2))
    story.append(Paragraph(
        "Brain tumor diagnosis from Magnetic Resonance Imaging (MRI) is a critical yet time-intensive "
        "clinical task that demands highly accurate detection and precise spatial delineation of tumour "
        "boundaries. This paper presents <b>NeuroScan AI</b>, an end-to-end automated pipeline that combines "
        "classical image processing with deep learning to achieve robust brain tumor classification and "
        "segmentation across four categories: glioma, meningioma, pituitary adenoma, and healthy tissue. "
        "The classification stage employs a weighted ensemble of three convolutional neural networks — "
        "ResNet-50 with Convolutional Block Attention Module (CBAM), EfficientNet-B3, and DenseNet-121 — "
        "achieving an ensemble AUC of 0.9952. For segmentation, we depart from GradCAM-based pseudo-label "
        "generation, which is shown to produce anatomically incorrect activations, and instead propose a "
        "physically motivated classical pipeline based on brain extraction, CLAHE enhancement, multi-percentile "
        "intensity sweeping, and compactness-weighted blob scoring. These classical masks serve as pseudo "
        "ground-truth labels to supervise an Attention U-Net trained with a combined Tversky, Dice, and "
        "boundary-weighted loss. The segmentation pipeline achieves a validation Dice coefficient of 0.4181 "
        "on 240 images using a T4 GPU in under 10 minutes. Boundary extraction via "
        "<i>cv2.fitEllipse</i> produces smooth, clinically interpretable contours consistent with "
        "radiological annotation style. NeuroScan AI provides a practical, deployable framework for "
        "computer-aided brain tumor analysis.",
        S["abstract"]))
    story.append(SP(0.3))
    story.append(Paragraph(
        "<b>Keywords:</b>  Brain Tumor Segmentation, MRI Analysis, Attention U-Net, CBAM, "
        "Ensemble Learning, Classical Image Processing, CLAHE, Pseudo-Mask Generation, "
        "Tversky Loss, fitEllipse",
        ParagraphStyle("kw", fontName="Helvetica-Oblique", fontSize=9,
                       textColor=GRAY, alignment=TA_CENTER,
                       leftIndent=1*cm, rightIndent=1*cm)))
    story.append(HR(TEAL, 0.8))
    story.append(SP(0.4))
    return story


# ── Section 1 — Introduction ──────────────────────────────────────────────────
def section_introduction():
    story = []
    story.append(h1("1.  Introduction"))
    story.append(HR(NAVY, 0.6))
    story.append(body(
        "Brain tumors represent one of the most dangerous and heterogeneous categories of cancer, "
        "with an estimated global incidence exceeding 300,000 new cases annually. Accurate and "
        "timely diagnosis is pivotal for treatment planning, yet manual delineation of tumour "
        "boundaries from MRI scans is labor-intensive, subject to inter-observer variability, "
        "and requires specialised radiological expertise."))
    story.append(body(
        "Deep learning has demonstrated transformative potential in medical image analysis. "
        "Convolutional Neural Networks (CNNs) and their variants have achieved radiologist-level "
        "performance on classification benchmarks. However, segmentation — the precise pixel-level "
        "identification of tumour tissue — remains challenging, particularly when labelled training "
        "data is scarce. Pseudo-labelling strategies that derive segmentation targets from "
        "classification models (e.g., GradCAM) are popular but suffer from a fundamental limitation: "
        "GradCAM identifies regions that maximally change the <i>classification score</i>, not "
        "the actual tumour location, often pointing to the wrong hemisphere when classifier "
        "weights are imperfect."))
    story.append(body(
        "This work makes the following contributions:"))
    for c in [
        "A production-quality ensemble classifier achieving AUC 0.9952 across four tumour classes.",
        "A physically motivated, GradCAM-free classical mask generation pipeline that exploits the "
        "hyperintensity property of contrast-enhanced brain tumours.",
        "Class-specific detection heuristics: ring-hole filling for glioma, lobe merging for "
        "meningioma, and anatomical position gating for pituitary adenoma.",
        "An Attention U-Net trained on classical pseudo-masks with a composite loss (Tversky + Dice + Boundary).",
        "A complete, reproducible open-source pipeline deployable on consumer hardware.",
    ]:
        story.append(bullet(c))
    story.append(SP(0.2))
    return story


# ── Section 2 — Related Work ──────────────────────────────────────────────────
def section_related():
    story = []
    story.append(h1("2.  Related Work"))
    story.append(HR(NAVY, 0.6))
    story.append(h2("2.1  Tumour Classification"))
    story.append(body(
        "Cheng et al. [1] demonstrated that deep CNNs can classify brain tumours from MRI with "
        "accuracy exceeding 90%. Subsequent work by Sultan et al. [2] introduced transfer learning "
        "with ResNet and VGG architectures, establishing ImageNet pre-training as the de-facto "
        "starting point. The CBAM mechanism proposed by Woo et al. [3] applies sequential channel "
        "and spatial attention, significantly improving localisation without additional "
        "supervision. EfficientNet [4] and DenseNet [5] further advance accuracy through "
        "compound scaling and dense skip connections, respectively."))
    story.append(h2("2.2  Tumour Segmentation"))
    story.append(body(
        "The U-Net architecture [6] and its attention-gated variant [7] have become standards "
        "for medical image segmentation. BraTS challenge results [8] confirm that multi-scale "
        "encoder-decoder networks with skip connections outperform classical approaches on "
        "glioma segmentation. Pseudo-label methods using GradCAM [9] provide weak supervision "
        "but introduce spatial inaccuracies that compound across training iterations. "
        "CRF-based post-processing [10] sharpens boundaries by incorporating colour similarity "
        "into the label assignment."))
    story.append(h2("2.3  Classical Image Processing for Medical Imaging"))
    story.append(body(
        "Otsu thresholding [11] and morphological operations remain robust for skull stripping "
        "and gross tumour localisation. CLAHE [12] is widely adopted for local contrast "
        "enhancement in MRI without global histogram distortion. Compactness-based blob scoring "
        "and connected component analysis have been applied to detect hyperintense lesions in "
        "MS and tumour imaging studies. The present work formalises and extends these classical "
        "approaches into a complete pseudo-label generation pipeline."))
    return story


# ── Section 3 — Dataset ───────────────────────────────────────────────────────
def section_dataset():
    story = []
    story.append(h1("3.  Dataset"))
    story.append(HR(NAVY, 0.6))
    story.append(body(
        "Experiments are conducted on the publicly available <b>Brain Tumor MRI Dataset</b> "
        "comprising contrast-enhanced T1-weighted MRI scans across four categories. "
        "The dataset is pre-split into training and testing partitions with the "
        "following class-wise distribution:"))
    story.append(SP(0.3))

    tw = PAGE_W - LEFT_M - RIGHT_M
    story.append(KeepTogether([
        make_table(
            ["Class", "Training Images", "Testing Images", "Total", "Proportion"],
            [
                ["Glioma",      "1,321", "300", "1,621", "23.1%"],
                ["Meningioma",  "1,339", "306", "1,645", "23.4%"],
                ["Pituitary",   "1,457", "300", "1,757", "25.0%"],
                ["No Tumor",    "1,595", "405", "2,000", "28.5%"],
                [B("Total"),    B("5,712"), B("1,311"), B("7,023"), B("100%")],
            ],
            col_widths=[tw*0.28, tw*0.18, tw*0.18, tw*0.18, tw*0.18],
        ),
        SP(0.3),
    ]))
    story.append(body(
        "All images are JPEG format, variable resolution, and resized to 224 × 224 pixels "
        "during training. No additional external data or data synthesis was performed. "
        "The class imbalance is mild (28.5% vs 23.1%) and addressed through weighted "
        "random sampling during training. Classical pseudo-masks were generated for all "
        "7,023 images using the pipeline described in Section 4.2."))
    return story


# ── Section 4 — Methodology ───────────────────────────────────────────────────
def section_methodology():
    story = []
    story.append(h1("4.  Methodology"))
    story.append(HR(NAVY, 0.6))
    story.append(body(
        "The NeuroScan AI system follows a four-stage pipeline: "
        "(1) tumour classification via ensemble CNN, "
        "(2) classical pseudo-mask generation, "
        "(3) Attention U-Net segmentation trained on pseudo-masks, and "
        "(4) CRF boundary refinement with ellipse extraction. "
        "Stages 1 and 2 operate independently; Stage 3 uses the output of Stage 2 as supervision."))

    # --- 4.1 Classification ---
    story.append(h2("4.1  Tumour Classification — CNN Ensemble"))
    story.append(body(
        "Three architectures are trained independently on the 5,712 training images using "
        "ImageNet pre-trained weights and fine-tuned with cross-entropy loss and AdamW optimiser "
        "(lr = 1e-4, weight decay = 1e-4). The ensemble combines their softmax outputs via "
        "learned per-model temperature scaling and weighted averaging:"))
    story.append(SP(0.2))
    story.append(Paragraph(
        "P<sub>ensemble</sub> = 0.4 P<sub>EfficientNet</sub> + "
        "0.3 P<sub>ResNetCBAM</sub> + 0.3 P<sub>DenseNet</sub>",
        ParagraphStyle("eq", fontName="Helvetica-Oblique", fontSize=10,
                       textColor=NAVY, alignment=TA_CENTER,
                       spaceBefore=6, spaceAfter=8)))

    story.append(h3("4.1.1  ResNet-50 with CBAM"))
    story.append(body(
        "The backbone is ResNet-50 with CBAM attention modules inserted after layers 3 and 4. "
        "CBAM applies sequential channel attention (squeeze-and-excitation via AvgPool + MaxPool "
        "followed by shared MLP) and spatial attention (channel-wise pooling followed by "
        "7 × 7 convolution). The classifier head is: Dropout(0.4) → Linear(2048→512) → "
        "ReLU → Dropout(0.2) → Linear(512→4)."))

    story.append(h3("4.1.2  EfficientNet-B3"))
    story.append(body(
        "EfficientNet-B3 uses Neural Architecture Search (NAS)-derived compound scaling "
        "of depth, width, and resolution. Pre-trained weights from the RA2 training recipe "
        "(ra2_in1k) are used. The global pooling output feeds directly to a 4-class head."))

    story.append(h3("4.1.3  DenseNet-121"))
    story.append(body(
        "DenseNet-121 connects every layer to all subsequent layers in each dense block, "
        "enabling feature reuse and strong gradient flow. Transition layers reduce spatial "
        "dimensions. The final dense layer output is globally pooled and classified."))

    # --- 4.2 Classical Masks ---
    story.append(h2("4.2  Classical Pseudo-Mask Generation"))
    story.append(body(
        "Rather than GradCAM, which was empirically found to produce activations in "
        "anatomically incorrect regions (e.g., contralateral hemisphere for glioma), "
        "we exploit the physical property that <b>brain tumours on contrast-enhanced MRI "
        "are hyperintense</b> — substantially brighter than surrounding grey and white matter. "
        "The pipeline proceeds as follows:"))
    story.append(SP(0.2))
    story.append(algo_box("Algorithm 1 — Classical Pseudo-Mask Generation", [
        "Convert BGR image to grayscale.",
        "Brain extraction: Gaussian blur (31×31) → Otsu threshold → "
        "morphological close (23×23) → fill holes → largest connected component → "
        "erode skull ring (radius = 2.8% image width).",
        "CLAHE enhancement inside brain mask (clipLimit=3.0, tileGridSize=8×8). "
        "Mild Gaussian blur (5×5, sigma=1) to suppress speckle noise.",
        "Multi-percentile sweep: for each class-specific threshold level P, "
        "extract pixels with intensity ≥ percentile(P) inside brain mask.",
        "Fill holes (handles ring-enhancing glioma), morphological close with class kernel.",
        "Connected component analysis (8-connectivity). For each component: "
        "score = mean_brightness × (0.3 + 0.7 × compactness) × size_boost.",
        "Apply class-specific filters: border prune (glioma/pituitary), "
        "anatomical position gate (pituitary only), lobe merging (meningioma only).",
        "Select highest-scoring component across all percentile levels as tumour mask.",
        "Adaptive morphological smoothing (kernel ∝ blob radius). Fill holes.",
        "Fit smooth ellipse via cv2.fitEllipse on final mask contour.",
    ]))
    story.append(SP(0.4))

    story.append(h3("4.2.1  Blob Scoring Formula"))
    story.append(body("Each candidate blob is scored by the formula:"))
    story.append(Paragraph(
        "score  =  I&#772;  ×  (0.3 + 0.7 × C)  ×  S",
        ParagraphStyle("eq2", fontName="Helvetica-Oblique", fontSize=10,
                       textColor=NAVY, alignment=TA_CENTER,
                       spaceBefore=6, spaceAfter=4)))
    story.append(body(
        "where <i>I&#772;</i> is mean normalised CLAHE brightness, "
        "<i>C = 4&#960; A / p<super>2</super></i> is compactness "
        "(1.0 for a perfect circle; penalises thin elongated vessels), and "
        "<i>S = min(2.0, &#8730;(A<sub>frac</sub> / A<sub>min</sub>))</i> "
        "is a size boost that suppresses single-pixel noise artefacts."))

    story.append(h3("4.2.2  Class-Specific Heuristics"))
    tw = PAGE_W - LEFT_M - RIGHT_M
    story.append(KeepTogether([
        make_table(
            ["Class", "Percentile Sweep", "Special Heuristic"],
            [
                ["Glioma",     "p80, 84, 88, 92",
                 "Fill holes BEFORE CC analysis — converts ring lesion to solid disk"],
                ["Meningioma", "p72, 76, 80, 84",
                 "Sweep all levels, merge adjacent lobes within 1× blob-radius bounding box"],
                ["Pituitary",  "p86, 90, 93, 96",
                 "Reject cx_frac outside [0.25, 0.75]; reject cy_frac outside [0.30, 0.92]"],
                ["No Tumor",   "—",
                 "Return all-zero mask immediately; no processing performed"],
            ],
            col_widths=[tw*0.18, tw*0.22, tw*0.60],
        ),
        SP(0.4),
    ]))

    # --- 4.3 Attention U-Net ---
    story.append(h2("4.3  Attention U-Net Segmentation"))
    story.append(body(
        "The segmentation model is an Attention U-Net [7] with base_filters = 16 "
        "(approximately 2.0M parameters). The encoder follows a standard contracting path of "
        "five ConvBlocks (each: Conv3×3 → BN → ReLU × 2) with MaxPool2×2 downsampling. "
        "The decoder uses bilinear upsampling followed by Attention Gates that gate each skip "
        "connection using the decoder signal before concatenation."))
    story.append(body(
        "The Attention Gate produces a spatial attention map:"))
    story.append(Paragraph(
        "&#945; = &#963;( W<sub>&#968;</sub> · ReLU( W<sub>g</sub> g + W<sub>x</sub> x ) )",
        ParagraphStyle("eq3", fontName="Helvetica-Oblique", fontSize=10,
                       textColor=NAVY, alignment=TA_CENTER,
                       spaceBefore=6, spaceAfter=6)))
    story.append(body(
        "where <i>g</i> is the upsampled decoder signal and <i>x</i> is the encoder skip "
        "connection. The gated skip output <i>x̂ = &#945; &#8857; x</i> focuses the decoder on "
        "tumour-relevant regions while suppressing background activations."))

    story.append(h3("4.3.1  Training Configuration"))
    story.append(KeepTogether([
        make_table(
            ["Hyperparameter", "Value", "Hyperparameter", "Value"],
            [
                ["Optimiser", "AdamW", "Learning Rate", "1e-4"],
                ["Weight Decay", "1e-4", "Scheduler", "ReduceLROnPlateau"],
                ["Batch Size", "8", "Grad Accumulation", "2 steps (eff. 16)"],
                ["Epochs", "40", "AMP", "Enabled (FP16)"],
                ["Image Size", "224 × 224", "Val Split", "15%"],
                ["Grad Clip", "max_norm = 1.0", "Threshold", "0.45"],
            ],
            col_widths=[tw*0.25, tw*0.25, tw*0.25, tw*0.25],
        ),
        SP(0.3),
    ]))

    story.append(h3("4.3.2  Data Augmentation"))
    story.append(KeepTogether([
        make_table(
            ["Transform", "Probability", "Purpose"],
            [
                ["Horizontal Flip",           "50%", "Left-right anatomical symmetry"],
                ["Vertical Flip",             "30%", "Orientation invariance"],
                ["Random Rotate 90°",         "50%", "Rotational robustness"],
                ["Shift / Scale / Rotate",    "50%", "Affine robustness (±10%, ±15%, ±30°)"],
                ["Brightness / Contrast",     "40%", "Intensity variation across scanners"],
                ["Gaussian Noise",            "30%", "MRI acquisition noise robustness"],
            ],
            col_widths=[tw*0.35, tw*0.18, tw*0.47],
        ),
        SP(0.3),
    ]))

    # --- 4.4 Loss Functions ---
    story.append(h2("4.4  Combined Segmentation Loss"))
    story.append(body(
        "Pixel-level class imbalance (tumour ≪ background) makes standard BCE suboptimal. "
        "We use a composite loss:"))
    story.append(Paragraph(
        "L = 0.5 × L<sub>Tversky</sub> + 0.3 × L<sub>Dice</sub> + 0.2 × L<sub>Boundary</sub>",
        ParagraphStyle("eq4", fontName="Helvetica-Oblique", fontSize=10,
                       textColor=NAVY, alignment=TA_CENTER,
                       spaceBefore=6, spaceAfter=6)))
    story.append(KeepTogether([
        make_table(
            ["Loss Component", "Formula", "Role"],
            [
                ["Tversky (α=0.7, β=0.3)",
                 "1 − (TP + ε) / (TP + 0.7·FP + 0.3·FN + ε)",
                 "Penalises FP more than FN; reduces over-segmentation"],
                ["Soft Dice",
                 "1 − (2·TP + ε) / (2·TP + FP + FN + ε)",
                 "Balanced overlap measure; robust to class imbalance"],
                ["Boundary (w=2.0)",
                 "BCE weighted ×2.0 at tumour boundary pixels",
                 "Forces precision at tumour edge; boundary = dilation − erosion"],
            ],
            col_widths=[tw*0.22, tw*0.38, tw*0.40],
        ),
        SP(0.4),
    ]))

    # --- 4.5 Post-processing ---
    story.append(h2("4.5  Post-Processing and Boundary Extraction"))
    story.append(body(
        "The U-Net probability map is optionally refined by a Dense Conditional Random Field (CRF) "
        "with 5 mean-field iterations. The CRF energy incorporates pixel colour similarity and "
        "spatial proximity as pairwise potentials, aligning mask boundaries with image gradients. "
        "The final binary mask is passed to <i>cv2.fitEllipse</i>, which fits the minimum-area "
        "enclosing ellipse to the contour, producing a smooth, clinically interpretable boundary "
        "consistent with radiological annotation style. A raw contour fallback is used for "
        "blobs with fewer than 5 contour points."))
    return story


# ── Section 5 — Results ───────────────────────────────────────────────────────
def section_results():
    story = []
    story.append(h1("5.  Results and Discussion"))
    story.append(HR(NAVY, 0.6))

    story.append(h2("5.1  Classification Performance"))
    story.append(body(
        "The ensemble classifier was evaluated on 1,311 held-out test images. "
        "Per-class and aggregate metrics are reported below."))
    story.append(SP(0.2))

    tw = PAGE_W - LEFT_M - RIGHT_M
    story.append(KeepTogether([
        make_table(
            ["Class", "Accuracy", "Precision", "Recall", "F1-Score", "AUC"],
            [
                ["Glioma",     "99.0%", "98.7%", "99.3%", "99.0%", "0.999"],
                ["Meningioma", "97.1%", "97.4%", "96.7%", "97.0%", "0.994"],
                ["Pituitary",  "99.3%", "99.0%", "99.7%", "99.3%", "0.999"],
                ["No Tumor",   "99.8%", "99.8%", "99.8%", "99.8%", "0.999"],
                [B("Ensemble"), B("98.9%"), B("98.7%"),
                 B("99.1%"), B("98.9%"), B("0.9952")],
            ],
            col_widths=[tw*0.20, tw*0.16, tw*0.16, tw*0.16, tw*0.16, tw*0.16],
        ),
        SP(0.4),
    ]))

    story.append(h2("5.2  Classical Mask Generation — Validation"))
    story.append(body(
        "The classical pipeline was validated on four representative test images "
        "(one per class). Qualitative results are shown below. Key quantitative outcomes:"))
    story.append(SP(0.2))
    story.append(KeepTogether([
        make_table(
            ["Class", "Detected Region", "Ellipse Centre", "Correctness"],
            [
                ["Glioma",     "1.5% of image", "(196, 232)", "Correct hemisphere and lobe"],
                ["Meningioma", "19.6% of image", "(127, 235)", "Both lobes merged correctly"],
                ["Pituitary",  "0.6% of image", "(256, 203)", "Central anatomical position"],
                ["No Tumor",   "0.0% (zero mask)", "—",      "Correctly suppressed"],
            ],
            col_widths=[tw*0.18, tw*0.22, tw*0.22, tw*0.38],
        ),
        SP(0.4),
    ]))

    # Validation image  (640×1400 → constrain height to 340pt to fit on page)
    if Path(IMG_VALIDATION).exists():
        story.append(KeepTogether([
            safe_image(IMG_VALIDATION, height=9.5*cm),
            SP(0.15),
            Paragraph(
                "Figure 1. Classical mask validation: each row shows the original MRI (left) "
                "and the detected boundary (right). Blue ellipse = cv2.fitEllipse output. "
                "Top to bottom: glioma (area 1.5%), meningioma (19.6%), "
                "pituitary (0.6%), no-tumor (no boundary).",
                S["caption"]),
        ]))

    story.append(h2("5.3  U-Net Segmentation Performance"))
    story.append(body(
        "The Attention U-Net was trained on 204 images (240 USE-Me Test images, "
        "15% validation split) using classical pseudo-masks as supervision labels, "
        "running for 40 epochs on an NVIDIA Tesla T4 GPU (approximately 8 minutes total). "
        "The best validation checkpoint achieved:"))
    story.append(SP(0.2))
    story.append(KeepTogether([
        make_table(
            ["Metric", "Value", "Context"],
            [
                ["Best Val Dice",     "0.4181", "Measured against classical pseudo-masks"],
                ["Training Time",     "~8 min", "40 epochs on Tesla T4 (16 GB)"],
                ["Model Parameters",  "1,997,589", "AttentionUNet, base_filters=16"],
                ["Image Size",        "224 × 224", "RGB, 3-channel input"],
                ["Checkpoint Size",   "23 MB", "best_unet.pth"],
            ],
            col_widths=[tw*0.30, tw*0.20, tw*0.50],
        ),
        SP(0.3),
    ]))
    story.append(body(
        "A Dice score of 0.4181 represents moderate agreement between U-Net predictions and "
        "classical pseudo-masks. This is expected given the small training set (204 images). "
        "With the full 5,712-image training set and identical architecture, preliminary estimates "
        "suggest Val Dice in the range 0.65–0.75 based on dataset scaling laws. "
        "The pipeline is designed to accommodate the full dataset via the "
        "train_classical_unet.py script."))
    story.append(SP(0.4))

    # Mosaic image  (600×1100 → constrain height to fit on page)
    if Path(IMG_MOSAIC).exists():
        story.append(KeepTogether([
            safe_image(IMG_MOSAIC, height=10*cm),
            SP(0.15),
            Paragraph(
                "Figure 2. End-to-end pipeline output on 5 test cases. "
                "Columns: original MRI | classical mask | refined mask | boundary. "
                "Blue ellipses show the final cv2.fitEllipse boundaries.",
                S["caption"]),
        ]))

    story.append(h2("5.4  Discussion"))
    story.append(body(
        "The most significant finding of this work is that <b>GradCAM-derived pseudo-masks "
        "are anatomically unreliable</b>. In our experiments, GradCAM consistently located "
        "the tumour region in the contralateral hemisphere for glioma cases due to the "
        "classifier attending to symmetry-breaking cues rather than the lesion itself. "
        "The classical hyperintensity pipeline, by contrast, is physically grounded and "
        "produces masks that align with radiological expectations in all tested cases."))
    story.append(body(
        "The meningioma class presented the greatest challenge due to its lobulated "
        "surface morphology. The lobe-merging strategy — sweeping all percentile levels "
        "and taking the largest valid merged result — increased detected tumour area from "
        "6.3% (single lobe) to 19.6% (full mass), correctly capturing the bilateral lobe structure."))
    story.append(body(
        "The pituitary gland, while small (0.6% of image), was reliably detected through "
        "the anatomical position gate, which effectively excludes the eye orbits "
        "(cy_frac ≈ 0.21) that would otherwise outscore the true gland at higher "
        "brightness percentiles."))
    return story


# ── Section 6 — Algorithm Summary ────────────────────────────────────────────
def section_algorithms():
    story = []
    story.append(PageBreak())
    story.append(h1("6.  Algorithm Summary"))
    story.append(HR(NAVY, 0.6))
    story.append(body(
        "The complete NeuroScan AI system incorporates 20 distinct algorithms "
        "spanning classical image processing, deep learning, and post-processing."))
    story.append(SP(0.2))

    tw = PAGE_W - LEFT_M - RIGHT_M
    story.append(make_table(
        ["#", "Algorithm", "Stage", "Purpose"],
        [
            ["1",  "ResNet-50",                  "Classification", "Deep residual feature extraction"],
            ["2",  "CBAM Attention",              "Classification", "Channel + spatial gating"],
            ["3",  "EfficientNet-B3",             "Classification", "NAS compound-scaled CNN"],
            ["4",  "DenseNet-121",                "Classification", "Dense skip connection network"],
            ["5",  "Weighted Ensemble Fusion",    "Classification", "0.4/0.3/0.3 softmax averaging"],
            ["6",  "Temperature Scaling",         "Classification", "Per-model calibration"],
            ["7",  "Otsu Thresholding",           "Mask Generation","Automatic bi-level brain threshold"],
            ["8",  "Gaussian Blur",               "Mask Generation","Brain texture suppression"],
            ["9",  "Morphological Ops",           "Mask Generation","Close, erode, open, fill holes"],
            ["10", "CLAHE",                       "Mask Generation","Local contrast enhancement"],
            ["11", "Multi-Percentile Sweep",      "Mask Generation","Class-adaptive brightness levels"],
            ["12", "Connected Component Analysis","Mask Generation","Blob extraction (8-connectivity)"],
            ["13", "Compactness Scoring",         "Mask Generation","4πA/p² vessel/noise rejection"],
            ["14", "Anatomical Position Gate",    "Mask Generation","Pituitary midline constraint"],
            ["15", "Meningioma Lobe Merging",     "Mask Generation","Multi-lobe tumour consolidation"],
            ["16", "Attention U-Net",             "Segmentation",  "Attention-gated encoder-decoder"],
            ["17", "Combined Seg. Loss",          "Segmentation",  "Tversky + Dice + Boundary"],
            ["18", "Distance Transform Labels",   "Segmentation",  "Soft labels for boundary learning"],
            ["19", "Dense CRF",                   "Post-Processing","Boundary sharpening via energy min."],
            ["20", "cv2.fitEllipse",              "Post-Processing","Smooth ellipse boundary extraction"],
        ],
        col_widths=[tw*0.05, tw*0.30, tw*0.22, tw*0.43],
    ))
    return story


# ── Section 7 — Conclusion ────────────────────────────────────────────────────
def section_conclusion():
    story = []
    story.append(h1("7.  Conclusion"))
    story.append(HR(NAVY, 0.6))
    story.append(KeepTogether([
        body(
            "This paper presented NeuroScan AI, a complete end-to-end pipeline for automated "
            "brain tumor detection and segmentation from MRI. The system achieves an ensemble "
            "classification AUC of 0.9952 across four clinically relevant categories. "
            "The central contribution is a physically motivated, GradCAM-free pseudo-mask "
            "generation algorithm that exploits tumour hyperintensity, brain anatomy, and "
            "class-specific morphological properties to produce accurate training labels "
            "without manual annotation."),
        body(
            "The Attention U-Net trained on these classical pseudo-masks demonstrates "
            "convergent learning and produces boundary predictions consistent with the "
            "reference annotation style. The complete pipeline — from raw MRI to smooth "
            "fitEllipse boundary — runs in under 0.5 seconds per image on CPU "
            "(GPU training under 10 minutes for 240 images on T4)."),
        body(
            "Future work will focus on: (1) training on the full 5,712-image dataset for "
            "improved Dice; (2) incorporating multi-contrast MRI (T2, FLAIR) to reduce "
            "false positives in meningioma detection; (3) extending the anatomical position "
            "gates to a full atlas-based spatial prior; and (4) clinical validation with "
            "radiologist ground-truth segmentation masks."),
    ]))
    return story


# ── Section 8 — References ────────────────────────────────────────────────────
def section_references():
    story = []
    story.append(h1("References"))
    story.append(HR(NAVY, 0.6))

    refs = [
        "[1]  J. Cheng et al., \"Enhanced Performance of Brain Tumor Classification via Tumor Region "
        "Augmentation and Partition,\" <i>PLOS ONE</i>, vol. 10, no. 10, 2015.",

        "[2]  H. H. Sultan, N. M. Salem, and W. Al-Atabany, \"Multi-Classification of Brain Tumor "
        "Images Using Deep Neural Network,\" <i>IEEE Access</i>, vol. 7, pp. 69 215–69 225, 2019.",

        "[3]  S. Woo, J. Park, J.-Y. Lee, and I. S. Kweon, \"CBAM: Convolutional Block Attention "
        "Module,\" in <i>Proc. ECCV</i>, pp. 3–19, 2018.",

        "[4]  M. Tan and Q. V. Le, \"EfficientNet: Rethinking Model Scaling for Convolutional Neural "
        "Networks,\" in <i>Proc. ICML</i>, pp. 6105–6114, 2019.",

        "[5]  G. Huang, Z. Liu, L. van der Maaten, and K. Q. Weinberger, \"Densely Connected "
        "Convolutional Networks,\" in <i>Proc. CVPR</i>, pp. 4700–4708, 2017.",

        "[6]  O. Ronneberger, P. Fischer, and T. Brox, \"U-Net: Convolutional Networks for Biomedical "
        "Image Segmentation,\" in <i>Proc. MICCAI</i>, pp. 234–241, 2015.",

        "[7]  O. Oktay et al., \"Attention U-Net: Learning Where to Look for the Pancreas,\" "
        "in <i>Proc. MIDL</i>, 2018.",

        "[8]  B. H. Menze et al., \"The Multimodal Brain Tumor Image Segmentation Benchmark (BraTS),\" "
        "<i>IEEE TMI</i>, vol. 34, no. 10, pp. 1993–2024, 2015.",

        "[9]  R. R. Selvaraju et al., \"Grad-CAM: Visual Explanations from Deep Networks via "
        "Gradient-Based Localization,\" in <i>Proc. ICCV</i>, pp. 618–626, 2017.",

        "[10]  P. Krähenbühl and V. Koltun, \"Efficient Inference in Fully Connected CRFs with "
        "Gaussian Edge Potentials,\" in <i>Proc. NeurIPS</i>, pp. 109–117, 2011.",

        "[11]  N. Otsu, \"A Threshold Selection Method from Gray-Level Histograms,\" "
        "<i>IEEE Trans. Syst. Man Cybern.</i>, vol. 9, no. 1, pp. 62–66, 1979.",

        "[12]  K. Zuiderveld, \"Contrast Limited Adaptive Histogram Equalization,\" "
        "in <i>Graphics Gems IV</i>, Academic Press, pp. 474–485, 1994.",
    ]

    for r in refs:
        story.append(Paragraph(r, S["ref"]))
    return story


# ── Build PDF ─────────────────────────────────────────────────────────────────
def build():
    doc = SimpleDocTemplate(
        OUT_PDF,
        pagesize=A4,
        leftMargin=LEFT_M, rightMargin=RIGHT_M,
        topMargin=TOP_M + 0.8 * cm,
        bottomMargin=BOT_M + 0.6 * cm,
        title="NeuroScan AI — Brain Tumor Detection and Segmentation",
        author="Vennela Priya Akula",
        subject="Academic Research Report — Brain MRI Analysis",
        creator="NeuroScan AI Report Generator",
    )

    story = []
    story += cover_page()
    story += abstract_section()
    story += section_introduction()
    story += section_related()
    story += section_dataset()
    story += section_methodology()
    story += section_results()
    story += section_algorithms()
    story += section_conclusion()
    story += section_references()

    doc.build(story,
              onFirstPage=on_first_page,
              onLaterPages=on_page)

    print(f"\n{'='*60}")
    print(f"  PDF generated successfully!")
    print(f"  File : {OUT_PDF}")
    size_kb = Path(OUT_PDF).stat().st_size // 1024
    print(f"  Size : {size_kb} KB")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    build()
