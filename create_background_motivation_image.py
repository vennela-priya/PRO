from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "BrainTumorAI_Background_and_Motivation.png"


def font(size, bold=False):
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf",
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def rounded(draw, xy, radius, fill, outline=None, width=1):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def wrap(draw, text, fnt, max_width):
    words = text.split()
    lines, line = [], ""
    for word in words:
        test = f"{line} {word}".strip()
        if draw.textbbox((0, 0), test, font=fnt)[2] <= max_width:
            line = test
        else:
            if line:
                lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines


def add_wrapped(draw, text, xy, fnt, fill, max_width, spacing=6):
    x, y = xy
    for line in wrap(draw, text, fnt, max_width):
        draw.text((x, y), line, font=fnt, fill=fill)
        y += fnt.size + spacing
    return y


def load_mri():
    candidates = [
        ROOT / "USE-Me Test" / "glioma" / "Te-glTr_0000.jpg",
        ROOT / "Project" / "Testing" / "glioma" / "Te-glTr_0000.jpg",
        ROOT / "Project" / "Testing" / "pituitary" / "Te-pi_0021.jpg",
    ]
    for path in candidates:
        if path.exists():
            img = Image.open(path).convert("RGB")
            img = img.resize((330, 330), Image.LANCZOS)
            return img
    return Image.new("RGB", (330, 330), "#172033")


def draw_arrow(draw, start, end, color):
    x1, y1 = start
    x2, y2 = end
    draw.line((x1, y1, x2, y2), fill=color, width=6)
    draw.polygon([(x2, y2), (x2 - 18, y2 - 11), (x2 - 18, y2 + 11)], fill=color)


def main():
    W, H = 1600, 900
    img = Image.new("RGB", (W, H), "#f6f9fc")
    draw = ImageDraw.Draw(img)

    # Soft background bands
    for y in range(H):
        shade = int(246 - (y / H) * 10)
        draw.line((0, y, W, y), fill=(shade, min(252, shade + 4), 255))
    draw.rectangle((0, 0, W, 150), fill="#12395b")
    draw.rectangle((0, 150, W, 158), fill="#2aa7c9")

    title_f = font(48, True)
    sub_f = font(22, False)
    h_f = font(25, True)
    body_f = font(19, False)
    small_f = font(16, False)

    draw.text((70, 38), "Background and Motivation", font=title_f, fill="white")
    draw.text(
        (72, 98),
        "AI-assisted brain tumor MRI classification with explainable diagnostic support",
        font=sub_f,
        fill="#d8eef7",
    )

    # MRI visual card
    rounded(draw, (70, 205, 465, 660), 26, "#ffffff", outline="#d6e2ee", width=2)
    mri = load_mri()
    mri = mri.filter(ImageFilter.UnsharpMask(radius=1.2, percent=130))
    img.paste(mri, (102, 238))
    draw.text((120, 590), "MRI scan input", font=h_f, fill="#12395b")
    draw.text((120, 626), "Clinical imaging data", font=small_f, fill="#58708a")

    # Motivation cards
    cards = [
        (
            515,
            205,
            "Clinical Challenge",
            "Manual MRI interpretation can be time-consuming and depends on expert availability. Early and accurate tumor identification supports better clinical planning.",
            "#e9f5fb",
            "#1f6f8b",
        ),
        (
            890,
            205,
            "AI Opportunity",
            "Deep learning can learn discriminative tumor patterns from MRI images and provide consistent classification across glioma, meningioma, no tumor, and pituitary cases.",
            "#eef8f0",
            "#267346",
        ),
        (
            1265,
            205,
            "Project Motivation",
            "BrainTumorAI combines ensemble classification, segmentation, and explainable heatmaps to build a practical decision-support system for academic and clinical study.",
            "#fff5e8",
            "#a26116",
        ),
    ]
    for x, y, heading, text, fill, accent in cards:
        rounded(draw, (x, y, x + 285, y + 330), 22, fill, outline="#d4dde8", width=2)
        draw.rectangle((x, y, x + 285, y + 8), fill=accent)
        draw.text((x + 24, y + 28), heading, font=h_f, fill=accent)
        add_wrapped(draw, text, (x + 24, y + 78), body_f, "#24384c", 235, spacing=8)

    # Workflow strip
    rounded(draw, (195, 700, 1405, 805), 24, "#ffffff", outline="#cddbe8", width=2)
    steps = [
        ("MRI Image", "#12395b"),
        ("Preprocessing", "#1f6f8b"),
        ("Ensemble AI", "#267346"),
        ("XAI + Segmentation", "#a26116"),
        ("Decision Support", "#7a3fa0"),
    ]
    x = 245
    for i, (label, color) in enumerate(steps):
        rounded(draw, (x, 726, x + 185, 779), 18, color)
        bbox = draw.textbbox((0, 0), label, font=small_f)
        draw.text((x + (185 - (bbox[2] - bbox[0])) / 2, 744), label, font=small_f, fill="white")
        if i < len(steps) - 1:
            draw_arrow(draw, (x + 195, 752), (x + 245, 752), "#8aa0b5")
        x += 245

    # Footer
    draw.text(
        (70, 850),
        "Purpose: reduce diagnostic workload, improve consistency, and make AI predictions interpretable through visual evidence.",
        font=small_f,
        fill="#496174",
    )

    img.save(OUT, quality=95)
    print(OUT)


if __name__ == "__main__":
    main()
