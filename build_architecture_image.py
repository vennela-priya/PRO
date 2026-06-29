from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


OUTPUT_PATH = Path("BrainTumorAI_Architecture_Diagram.png")


W, H = 1800, 1300
BG = "#f7fafc"
NAVY = "#17365d"
BLUE = "#2f6fed"
CYAN = "#0ea5e9"
GREEN = "#10b981"
PURPLE = "#7c3aed"
ORANGE = "#f59e0b"
RED = "#ef4444"
SLATE = "#334155"
LIGHT = "#ffffff"
BORDER = "#cbd5e1"
TEXT = "#1f2937"
MUTED = "#64748b"


def load_font(size, bold=False):
    candidates = [
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


TITLE = load_font(52, True)
SUBTITLE = load_font(25)
HEADER = load_font(27, True)
BODY = load_font(21)
SMALL = load_font(18)
TINY = load_font(15)


def rounded(draw, box, fill, outline=BORDER, width=3, radius=24):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def centered_text(draw, box, text, font, fill=TEXT, spacing=6):
    x1, y1, x2, y2 = box
    lines = text.split("\n")
    heights = []
    widths = []
    for line in lines:
        b = draw.textbbox((0, 0), line, font=font)
        widths.append(b[2] - b[0])
        heights.append(b[3] - b[1])
    total_h = sum(heights) + spacing * (len(lines) - 1)
    y = y1 + ((y2 - y1) - total_h) / 2
    for line, tw, th in zip(lines, widths, heights):
        draw.text((x1 + ((x2 - x1) - tw) / 2, y), line, font=font, fill=fill)
        y += th + spacing


def box(draw, x, y, w, h, title, body, color):
    rounded(draw, (x, y, x + w, y + h), LIGHT, outline=color, width=4, radius=22)
    draw.rounded_rectangle((x, y, x + w, y + 48), radius=22, fill=color)
    draw.rectangle((x, y + 24, x + w, y + 50), fill=color)
    centered_text(draw, (x + 12, y + 6, x + w - 12, y + 44), title, HEADER, fill="white")
    centered_text(draw, (x + 22, y + 58, x + w - 22, y + h - 14), body, BODY, fill=TEXT)


def arrow(draw, start, end, color=SLATE, width=5):
    draw.line((start, end), fill=color, width=width)
    sx, sy = start
    ex, ey = end
    if abs(ex - sx) >= abs(ey - sy):
        direction = 1 if ex > sx else -1
        pts = [(ex, ey), (ex - 18 * direction, ey - 10), (ex - 18 * direction, ey + 10)]
    else:
        direction = 1 if ey > sy else -1
        pts = [(ex, ey), (ex - 10, ey - 18 * direction), (ex + 10, ey - 18 * direction)]
    draw.polygon(pts, fill=color)


def build():
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)

    draw.text((90, 55), "BrainTumorAI / NeuroScan AI Architecture", font=TITLE, fill=NAVY)
    draw.text(
        (92, 122),
        "End-to-end MRI classification, explainability, segmentation support, and PDF report generation",
        font=SUBTITLE,
        fill=MUTED,
    )

    # Main pipeline
    box(draw, 90, 210, 285, 145, "1. MRI Upload", "Brain MRI image\nfrom Gradio UI", BLUE)
    box(draw, 460, 210, 305, 145, "2. Preprocessing", "Resize 224 x 224\nNormalize image\nTensor conversion", CYAN)
    box(draw, 850, 210, 295, 145, "3. TTA", "Original, flips,\nrotations\naveraged views", PURPLE)

    arrow(draw, (375, 282), (460, 282))
    arrow(draw, (765, 282), (850, 282))

    # Model ensemble
    draw.text((645, 440), "Three-Model Deep Learning Ensemble", font=HEADER, fill=NAVY)
    box(draw, 235, 510, 335, 150, "EfficientNet-B3", "Efficient feature\nextraction\nWeight: 40%", BLUE)
    box(draw, 725, 510, 335, 150, "ResNet50 + CBAM", "Channel + spatial\nattention\nWeight: 30%", PURPLE)
    box(draw, 1215, 510, 335, 150, "DenseNet121", "Dense feature reuse\nfor representation\nWeight: 30%", GREEN)

    arrow(draw, (998, 355), (402, 510), PURPLE)
    arrow(draw, (998, 355), (892, 510), PURPLE)
    arrow(draw, (998, 355), (1382, 510), PURPLE)

    box(draw, 670, 760, 445, 145, "Soft-Voting Ensemble", "Combines class probabilities\nFinal decision is not based\non one model alone", ORANGE)
    arrow(draw, (402, 660), (725, 792), SLATE)
    arrow(draw, (892, 660), (892, 760), SLATE)
    arrow(draw, (1382, 660), (1060, 792), SLATE)

    # Outputs
    box(draw, 145, 1010, 350, 145, "Prediction Output", "Class label\nConfidence score\nAll-class probabilities", GREEN)
    box(draw, 590, 1010, 350, 145, "Explainability", "Grad-CAM heatmap\nShows important\nMRI regions", RED)
    box(draw, 1035, 1010, 350, 145, "Segmentation", "U-Net / Attention U-Net\nPossible tumor\nregion mask", PURPLE)
    box(draw, 1480, 1010, 260, 145, "PDF Report", "Patient details\nMRI image\nDiagnosis summary", BLUE)

    arrow(draw, (892, 905), (320, 1010), SLATE)
    arrow(draw, (892, 905), (765, 1010), SLATE)
    arrow(draw, (892, 905), (1210, 1010), SLATE)
    arrow(draw, (1385, 1082), (1480, 1082), SLATE)

    # Sidebar facts
    rounded(draw, (1235, 190, 1710, 385), "#eef6ff", outline="#93c5fd", width=3, radius=24)
    draw.text((1265, 218), "Classes Predicted", font=HEADER, fill=NAVY)
    facts = ["Glioma", "Meningioma", "Pituitary tumor", "No Tumor"]
    y = 270
    for item in facts:
        draw.ellipse((1265, y + 7, 1277, y + 19), fill=GREEN)
        draw.text((1290, y), item, font=BODY, fill=TEXT)
        y += 34

    rounded(draw, (90, 1185, 1710, 1248), "#fff7ed", outline="#fed7aa", width=3, radius=20)
    centered_text(
        draw,
        (110, 1192, 1690, 1242),
        "Review explanation: MRI image -> preprocessing -> TTA -> three CNN models -> weighted soft-voting -> prediction, heatmap, segmentation, and downloadable patient PDF report.",
        SMALL,
        fill="#7c2d12",
    )

    img.save(OUTPUT_PATH, "PNG")
    print(OUTPUT_PATH.resolve())


if __name__ == "__main__":
    build()
