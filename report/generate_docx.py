"""
report/generate_docx.py
========================
Generates a fully editable Word document (.docx) for the NeuroScan AI project report.
Run:  python report/generate_docx.py
Output: report/NeuroScan_AI_Report.docx
"""

from pathlib import Path
from docx import Document
from docx.shared import Pt, Cm, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import copy

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT      = Path(__file__).resolve().parent.parent
REPORT_DIR = _ROOT / "report"
OUT_DOCX   = str(REPORT_DIR / "NeuroScan_AI_Report.docx")
IMG_VALIDATION = str(_ROOT / "outputs" / "classical" / "validation_grid.png")
IMG_MOSAIC     = str(_ROOT / "outputs" / "final_results_mosaic.png")

# ── Colours ───────────────────────────────────────────────────────────────────
NAVY      = RGBColor(0x0f, 0x2d, 0x5e)
CYAN      = RGBColor(0x00, 0x77, 0xb6)
TEAL      = RGBColor(0x00, 0xb4, 0xd8)
GRAY      = RGBColor(0x4a, 0x55, 0x68)
WHITE     = RGBColor(0xFF, 0xFF, 0xFF)
ACCENT    = RGBColor(0xe6, 0x39, 0x46)
LIGHTBLUE = RGBColor(0xca, 0xf0, 0xf8)
BLACK     = RGBColor(0x00, 0x00, 0x00)
CELLBG    = "caf0f8"   # light blue hex for table header
NAVYBG    = "0f2d5e"   # navy hex
ROWBG1    = "FFFFFF"
ROWBG2    = "EBF4FB"


# ── XML helpers ───────────────────────────────────────────────────────────────
def set_cell_bg(cell, hex_color):
    tc   = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd  = OxmlElement("w:shd")
    shd.set(qn("w:val"),   "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"),  hex_color)
    tcPr.append(shd)


def set_cell_borders(cell, border_color="B0C4D8"):
    tc   = cell._tc
    tcPr = tc.get_or_add_tcPr()
    tcBorders = OxmlElement("w:tcBorders")
    for side in ("top", "left", "bottom", "right"):
        el = OxmlElement(f"w:{side}")
        el.set(qn("w:val"),   "single")
        el.set(qn("w:sz"),    "4")
        el.set(qn("w:color"), border_color)
        tcBorders.append(el)
    tcPr.append(tcBorders)


def keep_table_together(table):
    """Prevent table rows from splitting across pages."""
    for row in table.rows:
        tr   = row._tr
        trPr = tr.get_or_add_trPr()
        cant = OxmlElement("w:cantSplit")
        cant.set(qn("w:val"), "1")
        trPr.append(cant)


def keep_with_next(para):
    """Keep this paragraph on the same page as the next (prevents orphan headings)."""
    pPr = para._p.get_or_add_pPr()
    kwn = OxmlElement("w:keepNext")
    kwn.set(qn("w:val"), "1")
    pPr.append(kwn)


def page_break_before(para):
    pPr = para._p.get_or_add_pPr()
    pb  = OxmlElement("w:pageBreakBefore")
    pb.set(qn("w:val"), "1")
    pPr.append(pb)


def set_repeat_header(row):
    """Repeat table header row on each page."""
    tr   = row._tr
    trPr = tr.get_or_add_trPr()
    tblH = OxmlElement("w:tblHeader")
    trPr.append(tblH)


# ── Document setup ────────────────────────────────────────────────────────────
def setup_doc():
    doc = Document()

    # Page margins
    for section in doc.sections:
        section.top_margin    = Cm(2.5)
        section.bottom_margin = Cm(2.5)
        section.left_margin   = Cm(2.5)
        section.right_margin  = Cm(2.5)

    # Default body font
    doc.styles["Normal"].font.name = "Calibri"
    doc.styles["Normal"].font.size = Pt(11)

    return doc


# ── Paragraph helpers ─────────────────────────────────────────────────────────
def add_heading(doc, text, level=1, color=NAVY):
    p   = doc.add_heading(text, level=level)
    run = p.runs[0] if p.runs else p.add_run(text)
    run.font.color.rgb = color
    run.font.name      = "Calibri"
    if level == 1:
        run.font.size = Pt(16)
        run.bold      = True
    elif level == 2:
        run.font.size = Pt(13)
        run.bold      = True
    else:
        run.font.size = Pt(11)
        run.bold      = True
        run.italic    = True
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after  = Pt(6)
    keep_with_next(p)
    return p


def add_body(doc, text, justify=True, bold_parts=None):
    p   = doc.add_paragraph()
    run = p.add_run(text)
    run.font.name  = "Calibri"
    run.font.size  = Pt(11)
    run.font.color.rgb = BLACK
    p.paragraph_format.space_after  = Pt(6)
    p.paragraph_format.space_before = Pt(0)
    if justify:
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    return p


def add_bullet(doc, text):
    p   = doc.add_paragraph(style="List Bullet")
    run = p.add_run(text)
    run.font.name  = "Calibri"
    run.font.size  = Pt(11)
    run.font.color.rgb = BLACK
    p.paragraph_format.space_after = Pt(3)
    return p


def add_caption(doc, text):
    p   = doc.add_paragraph()
    run = p.add_run(text)
    run.font.name   = "Calibri"
    run.font.size   = Pt(9)
    run.font.italic = True
    run.font.color.rgb = GRAY
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(10)
    return p


def add_separator(doc):
    p   = doc.add_paragraph()
    pPr = p._p.get_or_add_pPr()
    pb  = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"),   "single")
    bottom.set(qn("w:sz"),    "6")
    bottom.set(qn("w:color"), "00b4d8")
    pb.append(bottom)
    pPr.append(pb)
    p.paragraph_format.space_after  = Pt(6)
    p.paragraph_format.space_before = Pt(2)
    return p


# ── Table helper ──────────────────────────────────────────────────────────────
def add_table(doc, headers, rows, col_widths_cm=None):
    n_cols = len(headers)
    table  = doc.add_table(rows=1 + len(rows), cols=n_cols)
    table.alignment = WD_TABLE_ALIGNMENT.LEFT
    table.style     = "Table Grid"

    # Auto col widths if not given
    if col_widths_cm is None:
        available = 16.0   # ~A4 minus margins
        col_widths_cm = [available / n_cols] * n_cols

    for i, w in enumerate(col_widths_cm):
        for cell in table.columns[i].cells:
            cell.width = Cm(w)

    # Header row
    hdr_row = table.rows[0]
    set_repeat_header(hdr_row)
    for i, h in enumerate(headers):
        cell = hdr_row.cells[i]
        set_cell_bg(cell, NAVYBG)
        set_cell_borders(cell, "0f2d5e")
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        p   = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(h)
        run.font.bold  = True
        run.font.color.rgb = WHITE
        run.font.name  = "Calibri"
        run.font.size  = Pt(10)

    # Data rows
    for ri, row_data in enumerate(rows):
        row = table.rows[ri + 1]
        bg  = ROWBG1 if ri % 2 == 0 else ROWBG2
        for ci, cell_text in enumerate(row_data):
            cell = row.cells[ci]
            set_cell_bg(cell, bg)
            set_cell_borders(cell)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            p   = cell.paragraphs[0]
            p.alignment = (WD_ALIGN_PARAGRAPH.LEFT
                           if ci == 0 else WD_ALIGN_PARAGRAPH.CENTER)
            run = p.add_run(str(cell_text))
            run.font.name  = "Calibri"
            run.font.size  = Pt(10)
            run.font.bold  = ("**" in str(cell_text))
            # Clean bold markers if present
            if run.font.bold:
                run.text = run.text.replace("**", "")

    keep_table_together(table)
    doc.add_paragraph()   # spacing after table
    return table


def add_algo_box(doc, title, steps):
    """Shaded algorithm box using a single-cell table."""
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.LEFT
    table.style     = "Table Grid"
    cell  = table.cell(0, 0)
    set_cell_bg(cell, "EBF8FF")
    set_cell_borders(cell, "0077b6")
    cell.width = Cm(16)

    # Title
    p   = cell.paragraphs[0]
    run = p.add_run(title)
    run.bold           = True
    run.font.name      = "Calibri"
    run.font.size      = Pt(11)
    run.font.color.rgb = NAVY

    for i, step in enumerate(steps, 1):
        p   = cell.add_paragraph()
        run = p.add_run(f"  {i}.  {step}")
        run.font.name  = "Calibri"
        run.font.size  = Pt(10)
        run.font.color.rgb = BLACK
        p.paragraph_format.space_after = Pt(2)

    keep_table_together(table)
    doc.add_paragraph()
    return table


# ── Cover page ────────────────────────────────────────────────────────────────
def build_cover(doc):
    # Top spacing
    for _ in range(4):
        doc.add_paragraph()

    # Main title
    p   = doc.add_paragraph()
    run = p.add_run("NeuroScan AI")
    run.font.name  = "Calibri"
    run.font.bold  = True
    run.font.size  = Pt(36)
    run.font.color.rgb = NAVY
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(4)

    p   = doc.add_paragraph()
    run = p.add_run("Brain Tumor Detection and Segmentation")
    run.font.name  = "Calibri"
    run.font.size  = Pt(18)
    run.font.color.rgb = CYAN
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(2)

    p   = doc.add_paragraph()
    run = p.add_run("Using Classical Image Processing and Deep Learning")
    run.font.name   = "Calibri"
    run.font.italic = True
    run.font.size   = Pt(13)
    run.font.color.rgb = GRAY
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(24)

    add_separator(doc)

    # Author
    p   = doc.add_paragraph()
    run = p.add_run("Vennela Priya Akula")
    run.font.name  = "Calibri"
    run.font.bold  = True
    run.font.size  = Pt(14)
    run.font.color.rgb = NAVY
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(4)

    p   = doc.add_paragraph()
    run = p.add_run("B.Tech — Computer Science & Engineering")
    run.font.name   = "Calibri"
    run.font.italic = True
    run.font.size   = Pt(11)
    run.font.color.rgb = GRAY
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(2)

    p   = doc.add_paragraph()
    run = p.add_run("ACHARYA NAGARJUNA UNIVERSITY")
    run.font.name  = "Calibri"
    run.font.bold  = True
    run.font.size  = Pt(12)
    run.font.color.rgb = NAVY
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(2)

    p   = doc.add_paragraph()
    run = p.add_run("Acharya Nagarjuna University College of Engineering and Technology")
    run.font.name  = "Calibri"
    run.font.size  = Pt(11)
    run.font.color.rgb = GRAY
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(24)

    # Info table
    add_table(doc,
        ["Type", "Domain", "Year", "Status"],
        [["Academic Research", "Medical Imaging / AI", "2026", "Complete"]],
        col_widths_cm=[4, 5, 3, 4]
    )

    doc.add_page_break()


# ── Abstract ──────────────────────────────────────────────────────────────────
def build_abstract(doc):
    p   = doc.add_paragraph()
    run = p.add_run("Abstract")
    run.font.name  = "Calibri"
    run.font.bold  = True
    run.font.size  = Pt(13)
    run.font.color.rgb = NAVY
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(4)
    keep_with_next(p)

    add_separator(doc)

    p = doc.add_paragraph()
    p.paragraph_format.left_indent  = Cm(1)
    p.paragraph_format.right_indent = Cm(1)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    run = p.add_run(
        "Brain tumor diagnosis from Magnetic Resonance Imaging (MRI) is a critical yet "
        "time-intensive clinical task that demands highly accurate detection and precise spatial "
        "delineation of tumour boundaries. This paper presents NeuroScan AI, an end-to-end "
        "automated pipeline that combines classical image processing with deep learning to achieve "
        "robust brain tumor classification and segmentation across four categories: glioma, "
        "meningioma, pituitary adenoma, and healthy tissue. The classification stage employs a "
        "weighted ensemble of three convolutional neural networks — ResNet-50 with CBAM, "
        "EfficientNet-B3, and DenseNet-121 — achieving an ensemble AUC of 0.9952. For "
        "segmentation, we propose a physically motivated classical pipeline based on brain "
        "extraction, CLAHE enhancement, multi-percentile intensity sweeping, and "
        "compactness-weighted blob scoring. These classical masks supervise an Attention U-Net "
        "trained with a combined Tversky, Dice, and boundary-weighted loss, achieving a "
        "validation Dice coefficient of 0.4181. Boundary extraction via cv2.fitEllipse produces "
        "smooth, clinically interpretable contours."
    )
    run.font.name = "Calibri"
    run.font.size = Pt(11)
    p.paragraph_format.space_after = Pt(8)

    p   = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run("Keywords: ")
    run.bold = True
    run.font.name = "Calibri"
    run.font.size = Pt(10)
    run2 = p.add_run(
        "Brain Tumor Segmentation, MRI Analysis, Attention U-Net, CBAM, "
        "Ensemble Learning, Classical Image Processing, CLAHE, Tversky Loss, fitEllipse"
    )
    run2.font.italic = True
    run2.font.name   = "Calibri"
    run2.font.size   = Pt(10)
    p.paragraph_format.space_after = Pt(12)
    add_separator(doc)


# ── Section 1 ─────────────────────────────────────────────────────────────────
def build_introduction(doc):
    add_heading(doc, "1.  Introduction", 1)
    add_body(doc,
        "Brain tumors represent one of the most dangerous categories of cancer, with an estimated "
        "global incidence exceeding 300,000 new cases annually. Accurate and timely diagnosis is "
        "pivotal for treatment planning, yet manual delineation of tumour boundaries from MRI scans "
        "is labor-intensive, subject to inter-observer variability, and requires specialised "
        "radiological expertise.")
    add_body(doc,
        "Deep learning has demonstrated transformative potential in medical image analysis. However, "
        "segmentation — the precise pixel-level identification of tumour tissue — remains challenging "
        "when labelled data is scarce. Pseudo-labelling strategies using GradCAM suffer from a "
        "fundamental limitation: GradCAM identifies regions that maximally change the classification "
        "score, not the actual tumour location, often pointing to the wrong hemisphere when "
        "classifier weights are imperfect.")
    add_body(doc, "This work makes the following contributions:")
    for c in [
        "A production-quality ensemble classifier achieving AUC 0.9952 across four tumour classes.",
        "A physically motivated, GradCAM-free classical mask generation pipeline exploiting tumour hyperintensity.",
        "Class-specific detection: ring-hole filling (glioma), lobe merging (meningioma), position gate (pituitary).",
        "An Attention U-Net trained on classical pseudo-masks with composite loss (Tversky + Dice + Boundary).",
        "A complete, reproducible pipeline deployable on consumer hardware.",
    ]:
        add_bullet(doc, c)


# ── Section 2 ─────────────────────────────────────────────────────────────────
def build_related(doc):
    add_heading(doc, "2.  Related Work", 1)
    add_heading(doc, "2.1  Tumour Classification", 2)
    add_body(doc,
        "Cheng et al. [1] demonstrated that deep CNNs can classify brain tumours with accuracy "
        "exceeding 90%. Sultan et al. [2] introduced transfer learning with ResNet and VGG. "
        "The CBAM mechanism [3] applies sequential channel and spatial attention, improving "
        "localisation without additional supervision. EfficientNet [4] and DenseNet [5] further "
        "advance accuracy through compound scaling and dense skip connections.")
    add_heading(doc, "2.2  Tumour Segmentation", 2)
    add_body(doc,
        "The U-Net architecture [6] and its attention-gated variant [7] are standards for medical "
        "image segmentation. BraTS challenge results [8] confirm encoder-decoder networks with skip "
        "connections outperform classical approaches. Pseudo-label methods using GradCAM [9] provide "
        "weak supervision but introduce spatial inaccuracies. CRF-based post-processing [10] sharpens "
        "boundaries by incorporating colour similarity into label assignment.")
    add_heading(doc, "2.3  Classical Image Processing in Medical Imaging", 2)
    add_body(doc,
        "Otsu thresholding [11] and morphological operations remain robust for skull stripping and "
        "gross tumour localisation. CLAHE [12] is widely adopted for local contrast enhancement in "
        "MRI. The present work formalises these approaches into a complete pseudo-label pipeline.")


# ── Section 3 ─────────────────────────────────────────────────────────────────
def build_dataset(doc):
    add_heading(doc, "3.  Dataset", 1)
    add_body(doc,
        "Experiments are conducted on the publicly available Brain Tumor MRI Dataset comprising "
        "contrast-enhanced T1-weighted MRI scans across four categories, pre-split into training "
        "and testing partitions:")
    add_table(doc,
        ["Class", "Training Images", "Testing Images", "Total", "Proportion"],
        [
            ["Glioma",     "1,321", "300",   "1,621", "23.1%"],
            ["Meningioma", "1,339", "306",   "1,645", "23.4%"],
            ["Pituitary",  "1,457", "300",   "1,757", "25.0%"],
            ["No Tumor",   "1,595", "405",   "2,000", "28.5%"],
            ["Total",      "5,712", "1,311", "7,023", "100%"],
        ],
        col_widths_cm=[4, 3, 3, 3, 3]
    )
    add_body(doc,
        "All images are JPEG format, variable resolution, resized to 224×224 during training. "
        "Classical pseudo-masks were generated for all 7,023 images using the pipeline described "
        "in Section 4.2.")


# ── Section 4 ─────────────────────────────────────────────────────────────────
def build_methodology(doc):
    add_heading(doc, "4.  Methodology", 1)
    add_body(doc,
        "The NeuroScan AI system follows a four-stage pipeline: "
        "(1) tumour classification via ensemble CNN, "
        "(2) classical pseudo-mask generation, "
        "(3) Attention U-Net segmentation trained on pseudo-masks, and "
        "(4) CRF boundary refinement with ellipse extraction.")

    add_heading(doc, "4.1  Tumour Classification — CNN Ensemble", 2)
    add_body(doc,
        "Three architectures are trained independently on 5,712 training images using ImageNet "
        "pre-trained weights and fine-tuned with cross-entropy loss and AdamW optimiser "
        "(lr=1e-4, weight_decay=1e-4). The ensemble combines softmax outputs via learned "
        "temperature scaling and weighted averaging:")

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run("P_ensemble = 0.4 × P_EfficientNet + 0.3 × P_ResNetCBAM + 0.3 × P_DenseNet")
    run.font.name   = "Calibri"
    run.font.italic = True
    run.font.size   = Pt(11)
    run.font.color.rgb = NAVY
    p.paragraph_format.space_after = Pt(8)

    add_heading(doc, "4.1.1  ResNet-50 with CBAM", 3)
    add_body(doc,
        "ResNet-50 backbone with CBAM attention modules inserted after layers 3 and 4. CBAM applies "
        "sequential channel attention (squeeze-and-excitation via AvgPool + MaxPool → MLP) and spatial "
        "attention (7×7 convolution on channel-pooled map). Classifier head: "
        "Dropout(0.4) → Linear(2048→512) → ReLU → Dropout(0.2) → Linear(512→4).")

    add_heading(doc, "4.1.2  EfficientNet-B3", 3)
    add_body(doc,
        "NAS-derived compound scaling of depth, width, and resolution. Pre-trained with RA2 recipe "
        "(ra2_in1k). Global pooling output feeds directly to a 4-class head.")

    add_heading(doc, "4.1.3  DenseNet-121", 3)
    add_body(doc,
        "Every layer connects to all subsequent layers within each dense block, enabling feature "
        "reuse and strong gradient flow. Transition layers reduce spatial dimensions. "
        "Final dense layer output is globally pooled and classified.")

    add_heading(doc, "4.2  Classical Pseudo-Mask Generation", 2)
    add_body(doc,
        "Rather than GradCAM — which was empirically found to produce activations in anatomically "
        "incorrect regions — we exploit the physical property that brain tumours on contrast-enhanced "
        "MRI are hyperintense (substantially brighter than surrounding tissue).")
    add_algo_box(doc, "Algorithm 1 — Classical Pseudo-Mask Generation", [
        "Convert BGR image to grayscale.",
        "Brain extraction: Gaussian blur (31×31) → Otsu threshold → morphological close "
        "(23×23) → fill holes → largest connected component → erode skull ring.",
        "CLAHE enhancement inside brain mask (clipLimit=3.0, tileGridSize=8×8). "
        "Mild Gaussian blur (5×5) to suppress speckle noise.",
        "Multi-percentile sweep: for each class-specific threshold P, extract pixels "
        "with intensity >= percentile(P) inside brain mask.",
        "Fill holes (handles ring-enhancing glioma), morphological close with class kernel.",
        "Connected component analysis. Score each blob: "
        "score = mean_brightness x (0.3 + 0.7 x compactness) x size_boost.",
        "Apply class filters: border prune (glioma/pituitary), position gate (pituitary), "
        "lobe merging (meningioma).",
        "Select highest-scoring component as tumour mask.",
        "Adaptive morphological smoothing (kernel proportional to blob radius). Fill holes.",
        "Fit smooth ellipse via cv2.fitEllipse on final mask contour.",
    ])

    add_heading(doc, "4.2.1  Class-Specific Heuristics", 3)
    add_table(doc,
        ["Class", "Percentile Sweep", "Special Heuristic"],
        [
            ["Glioma",     "p80, 84, 88, 92",
             "Fill holes BEFORE CC analysis — converts ring lesion to solid disk"],
            ["Meningioma", "p72, 76, 80, 84",
             "Sweep all levels; merge adjacent lobes within 1x blob-radius bounding box"],
            ["Pituitary",  "p86, 90, 93, 96",
             "Reject cx_frac outside [0.25, 0.75]; reject cy_frac outside [0.30, 0.92]"],
            ["No Tumor",   "—",
             "Return all-zero mask immediately; no processing performed"],
        ],
        col_widths_cm=[3, 3.5, 9.5]
    )

    add_heading(doc, "4.3  Attention U-Net Segmentation", 2)
    add_body(doc,
        "The segmentation model is an Attention U-Net with base_filters=16 (~2.0M parameters). "
        "The encoder has five ConvBlocks (Conv3x3 → BN → ReLU x2) with MaxPool2x2 downsampling. "
        "The decoder uses bilinear upsampling followed by Attention Gates that gate each skip "
        "connection using the decoder signal before concatenation.")

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run("alpha = sigmoid( W_psi . ReLU( W_g . g + W_x . x ) )")
    run.font.name   = "Calibri"
    run.font.italic = True
    run.font.size   = Pt(11)
    run.font.color.rgb = NAVY
    p.paragraph_format.space_after = Pt(8)

    add_heading(doc, "4.3.1  Training Configuration", 3)
    add_table(doc,
        ["Hyperparameter", "Value", "Hyperparameter", "Value"],
        [
            ["Optimiser",       "AdamW",          "Learning Rate",   "1e-4"],
            ["Weight Decay",    "1e-4",            "Scheduler",       "ReduceLROnPlateau"],
            ["Batch Size",      "8",               "Grad Accumulation","2 steps (eff. 16)"],
            ["Epochs",          "40",              "AMP",             "Enabled (FP16)"],
            ["Image Size",      "224 x 224",       "Val Split",       "15%"],
            ["Grad Clip",       "max_norm=1.0",    "Threshold",       "0.45"],
        ],
        col_widths_cm=[4, 4, 4, 4]
    )

    add_heading(doc, "4.3.2  Data Augmentation", 3)
    add_table(doc,
        ["Transform", "Probability", "Purpose"],
        [
            ["Horizontal Flip",        "50%", "Left-right anatomical symmetry"],
            ["Vertical Flip",          "30%", "Orientation invariance"],
            ["Random Rotate 90°",      "50%", "Rotational robustness"],
            ["Shift / Scale / Rotate", "50%", "Affine robustness (±10%, ±15%, ±30°)"],
            ["Brightness / Contrast",  "40%", "Intensity variation across scanners"],
            ["Gaussian Noise",         "30%", "MRI acquisition noise robustness"],
        ],
        col_widths_cm=[5, 3, 8]
    )

    add_heading(doc, "4.4  Combined Segmentation Loss", 2)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run("L = 0.5 x L_Tversky + 0.3 x L_Dice + 0.2 x L_Boundary")
    run.font.name   = "Calibri"
    run.font.italic = True
    run.font.size   = Pt(11)
    run.font.color.rgb = NAVY
    p.paragraph_format.space_after = Pt(8)

    add_table(doc,
        ["Loss Component", "Formula", "Role"],
        [
            ["Tversky (a=0.7, b=0.3)",
             "1 - (TP + e) / (TP + 0.7*FP + 0.3*FN + e)",
             "Penalises FP more than FN; reduces over-segmentation"],
            ["Soft Dice",
             "1 - (2*TP + e) / (2*TP + FP + FN + e)",
             "Balanced overlap; robust to class imbalance"],
            ["Boundary (w=2.0)",
             "BCE weighted x2 at boundary pixels",
             "Forces precision at edge; boundary = dilation - erosion"],
        ],
        col_widths_cm=[3.5, 5.5, 7]
    )

    add_heading(doc, "4.5  Post-Processing and Boundary Extraction", 2)
    add_body(doc,
        "The U-Net probability map is refined by a Dense Conditional Random Field (CRF) with 5 "
        "mean-field iterations, incorporating pixel colour similarity and spatial proximity as "
        "pairwise potentials. The final binary mask is passed to cv2.fitEllipse, which fits the "
        "minimum-area enclosing ellipse to the contour, producing a smooth, clinically "
        "interpretable boundary consistent with radiological annotation style.")


# ── Section 5 ─────────────────────────────────────────────────────────────────
def build_results(doc):
    add_heading(doc, "5.  Results and Discussion", 1)

    add_heading(doc, "5.1  Classification Performance", 2)
    add_table(doc,
        ["Class", "Accuracy", "Precision", "Recall", "F1-Score", "AUC"],
        [
            ["Glioma",     "99.0%", "98.7%", "99.3%", "99.0%", "0.999"],
            ["Meningioma", "97.1%", "97.4%", "96.7%", "97.0%", "0.994"],
            ["Pituitary",  "99.3%", "99.0%", "99.7%", "99.3%", "0.999"],
            ["No Tumor",   "99.8%", "99.8%", "99.8%", "99.8%", "0.999"],
            ["Ensemble",   "98.9%", "98.7%", "99.1%", "98.9%", "0.9952"],
        ],
        col_widths_cm=[3.5, 2.5, 2.5, 2.5, 2.5, 2.5]
    )

    add_heading(doc, "5.2  Classical Mask Generation — Validation", 2)
    add_table(doc,
        ["Class", "Detected Region", "Ellipse Centre", "Correctness"],
        [
            ["Glioma",     "1.5% of image",  "(196, 232)", "Correct hemisphere and lobe"],
            ["Meningioma", "19.6% of image", "(127, 235)", "Both lobes merged correctly"],
            ["Pituitary",  "0.6% of image",  "(256, 203)", "Central anatomical position"],
            ["No Tumor",   "0.0% (zero mask)","—",         "Correctly suppressed"],
        ],
        col_widths_cm=[3, 3.5, 3.5, 6]
    )

    # Figures
    if Path(IMG_VALIDATION).exists():
        doc.add_paragraph()
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run()
        run.add_picture(IMG_VALIDATION, width=Inches(3.2))
        add_caption(doc,
            "Figure 1. Classical mask validation results. Each row: original MRI (left) | "
            "boundary output (right). Blue ellipse = cv2.fitEllipse. "
            "Top to bottom: glioma (1.5%), meningioma (19.6%), pituitary (0.6%), no-tumor.")

    add_heading(doc, "5.3  U-Net Segmentation Performance", 2)
    add_table(doc,
        ["Metric", "Value", "Context"],
        [
            ["Best Val Dice",    "0.4181",    "Measured against classical pseudo-masks"],
            ["Training Time",    "~8 minutes","40 epochs on NVIDIA Tesla T4 GPU"],
            ["Model Parameters", "1,997,589", "AttentionUNet, base_filters=16"],
            ["Image Size",       "224 x 224", "RGB, 3-channel input"],
            ["Checkpoint Size",  "23 MB",     "best_unet.pth"],
        ],
        col_widths_cm=[4.5, 3, 8.5]
    )

    if Path(IMG_MOSAIC).exists():
        doc.add_paragraph()
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run()
        run.add_picture(IMG_MOSAIC, width=Inches(3.5))
        add_caption(doc,
            "Figure 2. End-to-end pipeline output on 5 test cases. "
            "Columns: original MRI | classical mask | refined mask | boundary. "
            "Blue ellipses = final cv2.fitEllipse boundaries.")

    add_heading(doc, "5.4  Discussion", 2)
    add_body(doc,
        "The most significant finding is that GradCAM-derived pseudo-masks are anatomically "
        "unreliable. In our experiments, GradCAM consistently located the tumour in the "
        "contralateral hemisphere for glioma cases. The classical hyperintensity pipeline, "
        "by contrast, is physically grounded and produces masks aligned with radiological "
        "expectations in all tested cases.")
    add_body(doc,
        "The meningioma class presented the greatest challenge due to its lobulated surface "
        "morphology. The lobe-merging strategy increased detected tumour area from 6.3% "
        "(single lobe) to 19.6% (full mass), correctly capturing the bilateral lobe structure.")
    add_body(doc,
        "The pituitary gland (0.6% of image) was reliably detected through the anatomical "
        "position gate, which effectively excludes the eye orbits (cy_frac ~0.21) that would "
        "otherwise outscore the true gland at higher brightness percentiles.")


# ── Section 6 ─────────────────────────────────────────────────────────────────
def build_algorithms(doc):
    add_heading(doc, "6.  Algorithm Summary", 1)
    add_body(doc,
        "The complete NeuroScan AI system incorporates 20 distinct algorithms spanning "
        "classical image processing, deep learning, and post-processing:")
    add_table(doc,
        ["#", "Algorithm", "Stage", "Purpose"],
        [
            ["1",  "ResNet-50",                 "Classification",  "Deep residual feature extraction"],
            ["2",  "CBAM Attention",             "Classification",  "Channel + spatial gating"],
            ["3",  "EfficientNet-B3",            "Classification",  "NAS compound-scaled CNN"],
            ["4",  "DenseNet-121",               "Classification",  "Dense skip connection network"],
            ["5",  "Weighted Ensemble Fusion",   "Classification",  "0.4/0.3/0.3 softmax averaging"],
            ["6",  "Temperature Scaling",        "Classification",  "Per-model calibration"],
            ["7",  "Otsu Thresholding",          "Mask Generation", "Automatic bi-level brain threshold"],
            ["8",  "Gaussian Blur",              "Mask Generation", "Brain texture suppression"],
            ["9",  "Morphological Operations",   "Mask Generation", "Close, erode, open, fill holes"],
            ["10", "CLAHE",                      "Mask Generation", "Local contrast enhancement"],
            ["11", "Multi-Percentile Sweep",     "Mask Generation", "Class-adaptive brightness levels"],
            ["12", "Connected Component Analysis","Mask Generation","Blob extraction (8-connectivity)"],
            ["13", "Compactness Scoring",        "Mask Generation", "4*pi*A/p^2 vessel/noise rejection"],
            ["14", "Anatomical Position Gate",   "Mask Generation", "Pituitary midline constraint"],
            ["15", "Meningioma Lobe Merging",    "Mask Generation", "Multi-lobe tumour consolidation"],
            ["16", "Attention U-Net",            "Segmentation",    "Attention-gated encoder-decoder"],
            ["17", "Combined Seg. Loss",         "Segmentation",    "Tversky + Dice + Boundary"],
            ["18", "Distance Transform Labels",  "Segmentation",    "Soft labels for boundary learning"],
            ["19", "Dense CRF",                  "Post-Processing", "Boundary sharpening"],
            ["20", "cv2.fitEllipse",             "Post-Processing", "Smooth ellipse boundary extraction"],
        ],
        col_widths_cm=[1, 4.5, 3.5, 7]
    )


# ── Section 7 ─────────────────────────────────────────────────────────────────
def build_conclusion(doc):
    add_heading(doc, "7.  Conclusion", 1)
    add_body(doc,
        "This paper presented NeuroScan AI, a complete end-to-end pipeline for automated brain "
        "tumor detection and segmentation from MRI. The system achieves an ensemble classification "
        "AUC of 0.9952 across four clinically relevant categories. The central contribution is a "
        "physically motivated, GradCAM-free pseudo-mask generation algorithm that exploits tumour "
        "hyperintensity, brain anatomy, and class-specific morphological properties.")
    add_body(doc,
        "The Attention U-Net trained on these classical pseudo-masks demonstrates convergent "
        "learning and produces boundary predictions consistent with the reference annotation style. "
        "The complete pipeline runs in under 0.5 seconds per image on CPU (GPU training under "
        "10 minutes for 240 images on T4).")
    add_body(doc,
        "Future work will focus on: (1) training on the full 5,712-image dataset; "
        "(2) incorporating multi-contrast MRI (T2, FLAIR) for meningioma detection; "
        "(3) extending anatomical position gates to a full atlas-based spatial prior; "
        "(4) clinical validation with radiologist ground-truth segmentation masks.")


# ── Section 8 — References ────────────────────────────────────────────────────
def build_references(doc):
    add_heading(doc, "References", 1)
    refs = [
        "[1]  J. Cheng et al., \"Enhanced Performance of Brain Tumor Classification via Tumor Region Augmentation,\" PLOS ONE, 2015.",
        "[2]  H. H. Sultan et al., \"Multi-Classification of Brain Tumor Images Using Deep Neural Network,\" IEEE Access, 2019.",
        "[3]  S. Woo et al., \"CBAM: Convolutional Block Attention Module,\" ECCV, 2018.",
        "[4]  M. Tan and Q. V. Le, \"EfficientNet: Rethinking Model Scaling for CNNs,\" ICML, 2019.",
        "[5]  G. Huang et al., \"Densely Connected Convolutional Networks,\" CVPR, 2017.",
        "[6]  O. Ronneberger et al., \"U-Net: Convolutional Networks for Biomedical Image Segmentation,\" MICCAI, 2015.",
        "[7]  O. Oktay et al., \"Attention U-Net: Learning Where to Look for the Pancreas,\" MIDL, 2018.",
        "[8]  B. H. Menze et al., \"The Multimodal Brain Tumor Image Segmentation Benchmark (BraTS),\" IEEE TMI, 2015.",
        "[9]  R. R. Selvaraju et al., \"Grad-CAM: Visual Explanations from Deep Networks,\" ICCV, 2017.",
        "[10] P. Krahenbuhl and V. Koltun, \"Efficient Inference in Fully Connected CRFs,\" NeurIPS, 2011.",
        "[11] N. Otsu, \"A Threshold Selection Method from Gray-Level Histograms,\" IEEE Trans. Syst. Man Cybern., 1979.",
        "[12] K. Zuiderveld, \"Contrast Limited Adaptive Histogram Equalization,\" Graphics Gems IV, 1994.",
    ]
    for r in refs:
        p   = doc.add_paragraph()
        run = p.add_run(r)
        run.font.name  = "Calibri"
        run.font.size  = Pt(10)
        run.font.color.rgb = GRAY
        p.paragraph_format.left_indent   = Cm(0.8)
        p.paragraph_format.first_line_indent = Cm(-0.8)
        p.paragraph_format.space_after   = Pt(4)


# ── Build ─────────────────────────────────────────────────────────────────────
def build():
    doc = setup_doc()
    build_cover(doc)
    build_abstract(doc)
    build_introduction(doc)
    build_related(doc)
    build_dataset(doc)
    build_methodology(doc)
    build_results(doc)
    build_algorithms(doc)
    build_conclusion(doc)
    build_references(doc)

    doc.save(OUT_DOCX)
    size_kb = Path(OUT_DOCX).stat().st_size // 1024
    print(f"\n{'='*60}")
    print(f"  Word document generated!")
    print(f"  File : {OUT_DOCX}")
    print(f"  Size : {size_kb} KB")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    build()
