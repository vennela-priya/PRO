"""
Generate a consolidated PDF representation of the project folder.

Output:
    Final_Project_Output.pdf

The script scans recursively, includes code/text/notebook content, creates
captioned image contact sheets, and appends existing PDFs using PyMuPDF when
available.
"""

from __future__ import annotations

import html
import io
import json
import os
import textwrap
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image as RLImage,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "Final_Project_Output.pdf"
BASE_OUTPUT = ROOT / "_Final_Project_Output_base.pdf"
TEMP_DIR = ROOT / "_final_pdf_assets"

CODE_EXTS = {".py", ".js", ".html", ".css", ".ipynb"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
PDF_EXTS = {".pdf"}
TEXT_EXTS = {".txt", ".md"}
TARGET_EXTS = CODE_EXTS | IMAGE_EXTS | PDF_EXTS | TEXT_EXTS

MAX_CODE_CHARS = 45_000
MAX_TEXT_CHARS = 35_000
WRAP_WIDTH = 92
IMAGES_PER_SHEET = 12

EXCLUDE_DIRS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    "_final_pdf_assets",
}
EXCLUDE_FILES = {
    OUTPUT.name.lower(),
    BASE_OUTPUT.name.lower(),
}


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def is_target(path: Path) -> bool:
    if path.name.lower() in EXCLUDE_FILES:
        return False
    if any(part in EXCLUDE_DIRS for part in path.parts):
        return False
    return path.suffix.lower() in TARGET_EXTS


def scan_files() -> dict[str, list[Path]]:
    files = defaultdict(list)
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if not is_target(path):
            continue
        ext = path.suffix.lower()
        if ext in CODE_EXTS:
            files["code"].append(path)
        elif ext in IMAGE_EXTS:
            files["images"].append(path)
        elif ext in PDF_EXTS:
            files["pdfs"].append(path)
        elif ext in TEXT_EXTS:
            files["text"].append(path)
    for key in files:
        files[key].sort(key=lambda p: rel(p).lower())
    return files


def read_text(path: Path, max_chars: int | None = None) -> tuple[str, bool]:
    data = path.read_bytes()
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            text = data.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        text = data.decode("utf-8", errors="replace")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    truncated = False
    if max_chars and len(text) > max_chars:
        text = text[:max_chars]
        truncated = True
    return text, truncated


def notebook_to_text(path: Path) -> tuple[str, bool]:
    text, truncated_file = read_text(path, None)
    try:
        nb = json.loads(text)
    except Exception:
        return text[:MAX_CODE_CHARS], len(text) > MAX_CODE_CHARS
    parts = []
    for idx, cell in enumerate(nb.get("cells", []), start=1):
        ctype = cell.get("cell_type", "cell")
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)
        parts.append(f"# Cell {idx} [{ctype}]\n{source}".rstrip())
    joined = "\n\n".join(parts)
    truncated = truncated_file or len(joined) > MAX_CODE_CHARS
    return joined[:MAX_CODE_CHARS], truncated


def wrap_code(text: str) -> str:
    lines = []
    for line in text.splitlines():
        if not line:
            lines.append("")
            continue
        chunks = textwrap.wrap(
            line,
            width=WRAP_WIDTH,
            replace_whitespace=False,
            drop_whitespace=False,
            break_long_words=True,
            break_on_hyphens=False,
        )
        lines.extend(chunks or [""])
    return "\n".join(lines)


def make_styles():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="TitleCustom",
            parent=styles["Title"],
            alignment=TA_CENTER,
            fontSize=23,
            leading=29,
            textColor=colors.HexColor("#17365d"),
            spaceAfter=14,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Sub",
            parent=styles["BodyText"],
            alignment=TA_CENTER,
            fontSize=10,
            leading=14,
            textColor=colors.HexColor("#4b5563"),
            spaceAfter=14,
        )
    )
    styles.add(
        ParagraphStyle(
            name="H1",
            parent=styles["Heading1"],
            fontSize=17,
            leading=22,
            textColor=colors.HexColor("#1f4e79"),
            spaceBefore=10,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="H2",
            parent=styles["Heading2"],
            fontSize=12,
            leading=15,
            textColor=colors.HexColor("#17365d"),
            spaceBefore=8,
            spaceAfter=5,
            keepWithNext=True,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BodyCustom",
            parent=styles["BodyText"],
            fontSize=8.8,
            leading=12,
            textColor=colors.HexColor("#263238"),
            spaceAfter=4,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Small",
            parent=styles["BodyText"],
            fontSize=7.5,
            leading=9.5,
            textColor=colors.HexColor("#475569"),
            spaceAfter=3,
        )
    )
    styles.add(
        ParagraphStyle(
            name="CodeBlock",
            parent=styles["Code"],
            fontName="Courier",
            fontSize=6.4,
            leading=8.0,
            textColor=colors.HexColor("#111827"),
            backColor=colors.HexColor("#f8fafc"),
            borderColor=colors.HexColor("#d7dee8"),
            borderWidth=0.4,
            borderPadding=5,
            spaceAfter=8,
        )
    )
    return styles


def header_footer(canvas, doc):
    canvas.saveState()
    width, _ = A4
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#64748b"))
    canvas.drawString(doc.leftMargin, 0.35 * inch, f"{ROOT.name} consolidated project PDF")
    canvas.drawRightString(width - doc.rightMargin, 0.35 * inch, f"Page {doc.page}")
    canvas.restoreState()


def grouped_counts(files: dict[str, list[Path]]) -> list[list[str]]:
    return [
        ["Category", "Count"],
        ["Code files (.py, .js, .html, .css, .ipynb)", str(len(files["code"]))],
        ["Images (.png, .jpg, .jpeg)", str(len(files["images"]))],
        ["Existing PDFs", str(len(files["pdfs"]))],
        ["Text / Markdown files", str(len(files["text"]))],
    ]


def add_file_inventory(story, title: str, paths: Iterable[Path], styles):
    paths = list(paths)
    story.append(Paragraph(title, styles["H2"]))
    if not paths:
        story.append(Paragraph("None found.", styles["BodyCustom"]))
        return
    items = [f"{html.escape(rel(p))} ({p.stat().st_size:,} bytes)" for p in paths[:120]]
    if len(paths) > 120:
        items.append(f"... and {len(paths) - 120:,} more files in this category.")
    story.append(
        ListFlowable(
            [ListItem(Paragraph(item, styles["Small"]), leftIndent=8) for item in items],
            bulletType="bullet",
            leftIndent=14,
            bulletFontSize=5,
        )
    )


def add_title_and_toc(story, files, styles):
    story.append(Paragraph(f"{ROOT.name}", styles["TitleCustom"]))
    story.append(
        Paragraph(
            "Single consolidated PDF generated from the complete project folder scan.",
            styles["Sub"],
        )
    )
    story.append(Paragraph(f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}", styles["Sub"]))
    table = Table(grouped_counts(files), colWidths=[4.5 * inch, 1.4 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365d")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f8fbff")),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cbd5e1")),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph("Table of Contents", styles["H1"]))
    story.append(
        ListFlowable(
            [
                ListItem(Paragraph("Section 1: Code Files", styles["BodyCustom"])),
                ListItem(Paragraph("Section 2: Images / Dataset Samples", styles["BodyCustom"])),
                ListItem(Paragraph("Section 3: Existing PDFs", styles["BodyCustom"])),
                ListItem(Paragraph("Section 4: Additional Text / Notes", styles["BodyCustom"])),
            ],
            bulletType="1",
            leftIndent=18,
        )
    )
    story.append(Spacer(1, 0.1 * inch))
    add_file_inventory(story, "Scanned Code Inventory", files["code"], styles)
    add_file_inventory(story, "Scanned Image Inventory Preview", files["images"], styles)
    add_file_inventory(story, "Scanned PDF Inventory", files["pdfs"], styles)
    add_file_inventory(story, "Scanned Text Inventory", files["text"], styles)


def add_code_section(story, code_files, styles):
    story.append(PageBreak())
    story.append(Paragraph("Section 1: Code Files", styles["H1"]))
    if not code_files:
        story.append(Paragraph("No code files found.", styles["BodyCustom"]))
        return
    for path in code_files:
        story.append(Paragraph(html.escape(rel(path)), styles["H2"]))
        try:
            if path.suffix.lower() == ".ipynb":
                content, truncated = notebook_to_text(path)
            else:
                content, truncated = read_text(path, MAX_CODE_CHARS)
            if truncated:
                story.append(
                    Paragraph(
                        "Large file summarized: the first key portion is included to keep the PDF safe to build.",
                        styles["Small"],
                    )
                )
            content = wrap_code(content)
            story.append(Preformatted(content or "[empty file]", styles["CodeBlock"]))
        except Exception as exc:
            story.append(Paragraph(f"Unreadable file: {html.escape(str(exc))}", styles["BodyCustom"]))


def load_sheet_font(size=18):
    for candidate in (
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ):
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def make_contact_sheet(paths: list[Path], sheet_index: int) -> Path:
    TEMP_DIR.mkdir(exist_ok=True)
    sheet_w, sheet_h = 1600, 2100
    margin = 50
    gap = 30
    cols, rows = 3, 4
    cell_w = (sheet_w - 2 * margin - gap * (cols - 1)) // cols
    cell_h = (sheet_h - 2 * margin - gap * (rows - 1)) // rows
    img_h = cell_h - 58
    font = load_sheet_font(18)
    small = load_sheet_font(14)
    sheet = Image.new("RGB", (sheet_w, sheet_h), "white")
    draw = ImageDraw.Draw(sheet)
    for i, path in enumerate(paths):
        col = i % cols
        row = i // cols
        x = margin + col * (cell_w + gap)
        y = margin + row * (cell_h + gap)
        draw.rounded_rectangle((x, y, x + cell_w, y + cell_h), radius=18, outline="#cbd5e1", width=3, fill="#f8fafc")
        try:
            with Image.open(path) as im:
                im = im.convert("RGB")
                im.thumbnail((cell_w - 24, img_h - 18), Image.Resampling.LANCZOS)
                ix = x + (cell_w - im.width) // 2
                iy = y + 12 + (img_h - im.height) // 2
                sheet.paste(im, (ix, iy))
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            draw.text((x + 14, y + 40), "Unreadable image", fill="#b91c1c", font=font)
            draw.text((x + 14, y + 68), str(exc)[:45], fill="#64748b", font=small)
        caption = rel(path)
        caption = caption if len(caption) <= 46 else "..." + caption[-43:]
        draw.text((x + 14, y + img_h + 18), caption, fill="#17365d", font=small)
        draw.text((x + 14, y + img_h + 39), f"{path.stat().st_size:,} bytes", fill="#64748b", font=small)
    out = TEMP_DIR / f"image_sheet_{sheet_index:04d}.jpg"
    sheet.save(out, "JPEG", quality=82, optimize=True)
    return out


def add_image_section(story, image_files, styles):
    story.append(PageBreak())
    story.append(Paragraph("Section 2: Images / Dataset Samples", styles["H1"]))
    if not image_files:
        story.append(Paragraph("No image files found.", styles["BodyCustom"]))
        return
    story.append(
        Paragraph(
            f"{len(image_files):,} images were found. They are represented below as captioned contact sheets, grouped in scan order, to keep the PDF readable while still covering every image file.",
            styles["BodyCustom"],
        )
    )
    for index in range(0, len(image_files), IMAGES_PER_SHEET):
        batch = image_files[index : index + IMAGES_PER_SHEET]
        sheet_no = index // IMAGES_PER_SHEET + 1
        sheet_path = make_contact_sheet(batch, sheet_no)
        story.append(Paragraph(f"Image Contact Sheet {sheet_no} ({index + 1}-{index + len(batch)})", styles["H2"]))
        story.append(RLImage(str(sheet_path), width=6.6 * inch, height=8.65 * inch))
        if index + IMAGES_PER_SHEET < len(image_files):
            story.append(PageBreak())


def add_pdf_section(story, pdf_files, styles):
    story.append(PageBreak())
    story.append(Paragraph("Section 3: Existing PDFs", styles["H1"]))
    if not pdf_files:
        story.append(Paragraph("No existing PDFs found.", styles["BodyCustom"]))
        return
    story.append(
        Paragraph(
            "The files listed below are appended after the generated project sections using PyMuPDF, preserving their original PDF pages.",
            styles["BodyCustom"],
        )
    )
    rows = [["PDF File", "Size"]]
    for path in pdf_files:
        rows.append([rel(path), f"{path.stat().st_size:,} bytes"])
    table = Table(rows, colWidths=[5.1 * inch, 1.25 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365d")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cbd5e1")),
                ("FONTSIZE", (0, 0), (-1, -1), 7.2),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    story.append(table)


def add_text_section(story, text_files, styles):
    story.append(PageBreak())
    story.append(Paragraph("Section 4: Additional Text / Notes", styles["H1"]))
    if not text_files:
        story.append(Paragraph("No text or markdown files found.", styles["BodyCustom"]))
        return
    for path in text_files:
        story.append(Paragraph(html.escape(rel(path)), styles["H2"]))
        try:
            content, truncated = read_text(path, MAX_TEXT_CHARS)
            if truncated:
                story.append(
                    Paragraph(
                        "Large text file summarized: first portion included.",
                        styles["Small"],
                    )
                )
            for para in content.split("\n\n"):
                clean = html.escape(para.strip())
                if clean:
                    story.append(Paragraph(clean.replace("\n", "<br/>"), styles["BodyCustom"]))
        except Exception as exc:
            story.append(Paragraph(f"Unreadable file: {html.escape(str(exc))}", styles["BodyCustom"]))


def build_base_pdf(files):
    if TEMP_DIR.exists():
        for child in TEMP_DIR.glob("*"):
            try:
                child.unlink()
            except OSError:
                pass
    TEMP_DIR.mkdir(exist_ok=True)
    styles = make_styles()
    doc = SimpleDocTemplate(
        str(BASE_OUTPUT),
        pagesize=A4,
        leftMargin=0.55 * inch,
        rightMargin=0.55 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.6 * inch,
        title=f"{ROOT.name} Final Project Output",
        author="Codex",
    )
    story = []
    add_title_and_toc(story, files, styles)
    add_code_section(story, files["code"], styles)
    add_image_section(story, files["images"], styles)
    add_pdf_section(story, files["pdfs"], styles)
    add_text_section(story, files["text"], styles)
    doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)


def merge_existing_pdfs(pdf_files: list[Path]):
    try:
        import fitz
    except Exception:
        BASE_OUTPUT.replace(OUTPUT)
        return

    final = fitz.open(str(BASE_OUTPUT))
    for path in pdf_files:
        try:
            with fitz.open(str(path)) as src:
                final.insert_pdf(src)
        except Exception as exc:
            page = final.new_page()
            page.insert_text(
                (72, 72),
                f"Could not merge PDF: {rel(path)}\nReason: {exc}",
                fontsize=11,
                color=(0.7, 0, 0),
            )
    if OUTPUT.exists():
        OUTPUT.unlink()
    final.save(str(OUTPUT))
    final.close()
    try:
        BASE_OUTPUT.unlink()
    except OSError:
        pass


def main():
    files = scan_files()
    print(f"Root: {ROOT}")
    print(
        "Found:",
        f"{len(files['code'])} code,",
        f"{len(files['images'])} images,",
        f"{len(files['pdfs'])} pdfs,",
        f"{len(files['text'])} text files",
    )
    build_base_pdf(files)
    merge_existing_pdfs(files["pdfs"])
    print(f"Created: {OUTPUT}")


if __name__ == "__main__":
    main()
