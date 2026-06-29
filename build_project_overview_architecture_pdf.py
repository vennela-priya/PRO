from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    ListFlowable,
    ListItem,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


OUTPUT_PATH = "BrainTumorAI_Project_Overview_Schema_Architecture.pdf"


def styles():
    base = getSampleStyleSheet()
    base.add(ParagraphStyle(
        name="TitleCustom",
        parent=base["Title"],
        alignment=TA_CENTER,
        fontSize=22,
        leading=28,
        textColor=colors.HexColor("#17365d"),
        spaceAfter=12,
    ))
    base.add(ParagraphStyle(
        name="Sub",
        parent=base["BodyText"],
        alignment=TA_CENTER,
        fontSize=10.5,
        leading=15,
        textColor=colors.HexColor("#4b5563"),
        spaceAfter=16,
    ))
    base.add(ParagraphStyle(
        name="H",
        parent=base["Heading2"],
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#1f4e79"),
        spaceBefore=10,
        spaceAfter=6,
    ))
    base.add(ParagraphStyle(
        name="BodyCustom",
        parent=base["BodyText"],
        fontSize=9.6,
        leading=13.5,
        textColor=colors.HexColor("#263238"),
        spaceAfter=5,
    ))
    base.add(ParagraphStyle(
        name="Box",
        parent=base["BodyText"],
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#253044"),
        borderColor=colors.HexColor("#bfd0e4"),
        borderWidth=0.6,
        borderPadding=7,
        backColor=colors.HexColor("#f6f9fd"),
        spaceBefore=4,
        spaceAfter=8,
    ))
    return base


def bullets(items, st):
    return ListFlowable(
        [ListItem(Paragraph(item, st["BodyCustom"]), leftIndent=10) for item in items],
        bulletType="bullet",
        leftIndent=16,
        bulletFontSize=6,
        bulletColor=colors.HexColor("#1f4e79"),
    )


def header_footer(canvas, doc):
    canvas.saveState()
    width, _ = A4
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#6b7280"))
    canvas.drawString(doc.leftMargin, 0.35 * inch, "BrainTumorAI Project Overview")
    canvas.drawRightString(width - doc.rightMargin, 0.35 * inch, f"Page {doc.page}")
    canvas.restoreState()


def architecture_table(st):
    data = [
        ["Layer", "Role in Project"],
        ["Input Layer", "User uploads a brain MRI image through the Gradio web interface."],
        ["Preprocessing Layer", "Image is resized to 224 x 224, normalized, and converted into a tensor for PyTorch inference."],
        ["Model Layer", "EfficientNet-B3, ResNet50 + CBAM, and DenseNet121 extract MRI image features."],
        ["Ensemble Layer", "Weighted soft-voting combines model probabilities: EfficientNet 40%, ResNet-CBAM 30%, DenseNet 30%."],
        ["TTA Layer", "Test-time augmentation averages predictions from transformed views such as flips and rotations."],
        ["Output Layer", "System returns predicted class, confidence score, all-class probabilities, risk level, and clinical notes."],
        ["Explainability Layer", "Grad-CAM highlights image regions that influenced the prediction."],
        ["Segmentation Layer", "U-Net / Attention U-Net supports possible tumor-region mask generation."],
        ["Report Layer", "ReportLab creates a PDF report containing patient details, MRI image, prediction, confidence, and disclaimer."],
    ]
    table = Table(data, colWidths=[1.65 * inch, 5.0 * inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365d")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BACKGROUND", (0, 1), (0, -1), colors.HexColor("#eaf2fb")),
        ("BACKGROUND", (1, 1), (1, -1), colors.HexColor("#f8fbff")),
        ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.6),
        ("LEADING", (0, 0), (-1, -1), 11),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cbd5e1")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return table


def build_pdf():
    st = styles()
    doc = SimpleDocTemplate(
        OUTPUT_PATH,
        pagesize=A4,
        leftMargin=0.62 * inch,
        rightMargin=0.62 * inch,
        topMargin=0.62 * inch,
        bottomMargin=0.62 * inch,
        title="BrainTumorAI Project Overview, Schema and Architecture",
        author="Codex",
    )

    story = []
    story.append(Paragraph("BrainTumorAI Project Overview", st["TitleCustom"]))
    story.append(Paragraph("Schema and Architecture for Project / Thesis Review", st["Sub"]))

    story.append(Paragraph("Project Overview", st["H"]))
    story.append(Paragraph(
        "BrainTumorAI, also presented as NeuroScan AI, is a deep-learning-based brain MRI analysis system. It takes a brain MRI image as input, classifies it into one of four categories, shows confidence and probabilities, provides visual explanation using heatmaps and segmentation support, and generates a downloadable PDF patient report.",
        st["BodyCustom"],
    ))
    story.append(Paragraph(
        "The system is designed as an AI-assisted diagnostic support tool, not as a replacement for doctors.",
        st["Box"],
    ))

    story.append(Paragraph("Classes Predicted", st["H"]))
    story.append(bullets([
        "Glioma",
        "Meningioma",
        "Pituitary tumor",
        "No Tumor",
    ], st))

    story.append(Paragraph("System Schema", st["H"]))
    schema_steps = [
        "MRI Image Upload",
        "Image Preprocessing: resize to 224 x 224 and normalize",
        "Test-Time Augmentation: original, flips, and rotations",
        "Model Inference: EfficientNet-B3, ResNet50 + CBAM, DenseNet121",
        "Soft-Voting Ensemble: combines all model probabilities",
        "Final Outputs: class prediction, confidence, and all-class probabilities",
        "Explainability: Grad-CAM heatmap and U-Net segmentation support",
        "Clinical Notes and Risk Level",
        "PDF Report Generation",
        "Download Patient Report",
    ]
    story.append(bullets(schema_steps, st))

    story.append(Paragraph("Architecture", st["H"]))
    story.append(architecture_table(st))
    story.append(Spacer(1, 0.1 * inch))

    story.append(Paragraph("Model Architecture Details", st["H"]))
    model_data = [
        ["Model", "Purpose"],
        ["EfficientNet-B3", "Extracts efficient deep visual features from MRI images."],
        ["ResNet50 + CBAM", "Uses channel and spatial attention to focus on important image regions."],
        ["DenseNet121", "Reuses features across layers for stronger representation."],
    ]
    model_table = Table(model_data, colWidths=[1.8 * inch, 4.85 * inch])
    model_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365d")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f8fbff")),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cbd5e1")),
        ("FONTSIZE", (0, 0), (-1, -1), 8.8),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(model_table)

    story.append(Paragraph("Simple Review Explanation", st["H"]))
    story.append(Paragraph(
        "Our project follows a layered architecture. First, the MRI image is uploaded and preprocessed by resizing and normalization. Then it is passed through three deep learning models: EfficientNet-B3, ResNet50 with CBAM attention, and DenseNet121. Their predictions are combined using a weighted soft-voting ensemble with test-time augmentation. Finally, the system displays the result in a Gradio interface, provides heatmap and segmentation support, and generates a PDF report for the patient.",
        st["Box"],
    ))

    story.append(Paragraph("One-Line Summary", st["H"]))
    story.append(Paragraph(
        "BrainTumorAI is an end-to-end MRI brain tumor classification system that combines preprocessing, ensemble deep learning, explainability, segmentation support, and PDF report generation in a usable web application.",
        st["Box"],
    ))

    doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)


if __name__ == "__main__":
    build_pdf()
    print(OUTPUT_PATH)
