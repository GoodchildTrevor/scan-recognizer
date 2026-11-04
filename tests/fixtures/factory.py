"""Fixture generators: synthetic typewriter scans and PDFs, created at test time.

Nothing binary is committed except the fonts. Every ``make_*`` function writes into the given
directory (usually ``tmp_path``) and returns the created path(s).
"""

from __future__ import annotations

import io
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pymupdf
from PIL import Image, ImageDraw, ImageFont

FIXTURES_DIR = Path(__file__).resolve().parent
MONO_FONT = FIXTURES_DIR / "fonts" / "DejaVuSansMono.ttf"
SANS_FONT = FIXTURES_DIR.parents[1] / "ocr_archive" / "assets" / "fonts" / "DejaVuSans.ttf"

A4_PT = (595.0, 842.0)
PAPER = 236
INK = 32

_WORDS_TEXT = (
    "отчёт о работе лаборатории за истекший год представлен комиссии при этом "
    "отмечено что план выполнен на 104 процента однако ряд показателей требует "
    "уточнения в частности расход материалов и сроки поставки оборудования "
    "заводу необходимо обеспечить своевременную подготовку технической "
    "документации и согласовать её с главным инженером треста до 15 марта "
    "по данным бухгалтерии затраты составили 12 345 рублей при смете 11 900"
)
_WORDS = _WORDS_TEXT.split()


@dataclass(frozen=True)
class SyntheticPage:
    """A rendered typewriter page with ground truth.

    :ivar image: RGB page image.
    :ivar lines: text of each drawn line, top to bottom; for two columns the left column first.
    :ivar boxes_pt: ink box of each line in PDF points, before skew.
    :ivar dpi: resolution of ``image``.
    """

    image: Image.Image
    lines: list[str]
    boxes_pt: list[tuple[float, float, float, float]]
    dpi: int


def typewriter_lines(
    count: int, *, seed: int = 0, min_chars: int = 45, max_chars: int = 64
) -> list[str]:
    """Return ``count`` Russian-looking lines of random length."""
    rng = random.Random(seed)
    lines: list[str] = []
    for _ in range(count):
        target = rng.randint(min_chars, max_chars)
        words: list[str] = []
        while len(" ".join(words)) < target:
            words.append(rng.choice(_WORDS))
        line = " ".join(words)[:max_chars].rstrip()
        lines.append(line[0].upper() + line[1:])
    return lines


def render_typewriter(
    lines: list[str],
    *,
    dpi: int = 200,
    font_pt: float = 11.0,
    line_spacing: float = 1.8,
    margin_pt: float = 60.0,
    columns: int = 1,
    column_gap_pt: float = 40.0,
    skew_deg: float = 0.0,
    noise: float = 6.0,
    seed: int = 0,
) -> SyntheticPage:
    """Render lines in DejaVu Sans Mono on an A4 page with noise and skew.

    :param columns: 1 or 2; with 2 the lines are split evenly between the columns.
    """
    width_px = round(A4_PT[0] / 72 * dpi)
    height_px = round(A4_PT[1] / 72 * dpi)
    px_per_pt = dpi / 72
    font = ImageFont.truetype(str(MONO_FONT), round(font_pt * px_per_pt))
    image = Image.new("L", (width_px, height_px), PAPER)
    draw = ImageDraw.Draw(image)
    step = font_pt * line_spacing * px_per_pt
    margin = margin_pt * px_per_pt
    column_width = (width_px - 2 * margin - (columns - 1) * column_gap_pt * px_per_pt) / columns
    per_column = (len(lines) + columns - 1) // columns
    boxes: list[tuple[float, float, float, float]] = []
    for index, text in enumerate(lines):
        column, row = divmod(index, per_column)
        x = margin + column * (column_width + column_gap_pt * px_per_pt)
        y = margin + row * step
        draw.text((x, y), text, font=font, fill=INK)
        left, top, right, bottom = draw.textbbox((x, y), text, font=font)
        boxes.append((left / px_per_pt, top / px_per_pt, right / px_per_pt, bottom / px_per_pt))
    if skew_deg:
        image = image.rotate(
            skew_deg, resample=Image.Resampling.BICUBIC, fillcolor=PAPER, expand=False
        )
    if noise > 0:
        rng = np.random.default_rng(seed)
        pixels = np.asarray(image, dtype=np.float64)
        pixels = pixels + rng.normal(0.0, noise, pixels.shape)
        image = Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))
    return SyntheticPage(image=image.convert("RGB"), lines=list(lines), boxes_pt=boxes, dpi=dpi)


def _image_stream(image: Image.Image, fmt: str = "PNG") -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return buffer.getvalue()


def images_pdf(path: Path, images: list[Image.Image], *, dpi: int = 200) -> Path:
    """Write each image as a full page of a PDF, sized by ``dpi``."""
    doc = pymupdf.open()
    for image in images:
        page = doc.new_page(width=image.width * 72 / dpi, height=image.height * 72 / dpi)
        page.insert_image(page.rect, stream=_image_stream(image))
    doc.save(path)
    doc.close()
    return path


def make_typewriter_pdf(
    directory: Path,
    *,
    n_lines: int = 30,
    skew_deg: float = 1.0,
    seed: int = 0,
    name: str = "typewriter.pdf",
    **render_kwargs: float,
) -> tuple[Path, SyntheticPage]:
    """A one-page scan of a typewritten page with known lines."""
    page = render_typewriter(
        typewriter_lines(n_lines, seed=seed), skew_deg=skew_deg, seed=seed, **render_kwargs
    )
    return images_pdf(directory / name, [page.image], dpi=page.dpi), page


def make_two_column_pdf(directory: Path, *, n_lines: int = 40) -> tuple[Path, SyntheticPage]:
    """A scan with two text columns (newspaper layout)."""
    lines = typewriter_lines(n_lines, seed=7, min_chars=26, max_chars=30)
    page = render_typewriter(lines, columns=2, skew_deg=0.5, font_pt=10.0)
    return images_pdf(directory / "two_columns.pdf", [page.image], dpi=page.dpi), page


def make_split_page_pdf(directory: Path, *, n_lines: int = 44) -> tuple[Path, SyntheticPage]:
    """A long straight page that can be cut between lines in the middle."""
    page = render_typewriter(
        typewriter_lines(n_lines, seed=3), font_pt=9.5, line_spacing=1.75, noise=4.0
    )
    return images_pdf(directory / "split.pdf", [page.image], dpi=page.dpi), page


def make_no_gap_pdf(directory: Path, *, n_lines: int = 44) -> tuple[Path, SyntheticPage]:
    """Like :func:`make_split_page_pdf`, but a diagonal stroke crosses every gap in the middle."""
    page = render_typewriter(
        typewriter_lines(n_lines, seed=3), font_pt=9.5, line_spacing=1.75, noise=4.0
    )
    image = page.image.copy()
    draw = ImageDraw.Draw(image)
    width, height = image.size
    draw.line(
        [(int(width * 0.15), int(height * 0.2)), (int(width * 0.85), int(height * 0.8))],
        fill=(INK, INK, INK),
        width=6,
    )
    return images_pdf(directory / "no_gap.pdf", [image], dpi=page.dpi), page


@dataclass(frozen=True)
class MissingSplitCase:
    """Page lines vs a transcription that lost one line and split another one in two.

    The counts match, so only the plausibility check can notice the shift.
    """

    pdf_path: Path
    page: SyntheticPage
    text_lines: list[str]


def make_missing_split_pair(
    directory: Path, *, n_lines: int = 20, split_at: int = 5
) -> MissingSplitCase:
    """Line ``split_at`` is split in two halves and line ``split_at + 1`` is missing."""
    lines = typewriter_lines(n_lines, seed=11, min_chars=50, max_chars=64)
    page = render_typewriter(lines, skew_deg=0.0, noise=4.0)
    target = lines[split_at]
    cut = target.rfind(" ", 0, len(target) // 2 + 4)
    cut = cut if cut > 0 else len(target) // 2
    text_lines = [
        *lines[:split_at],
        target[:cut].rstrip(),
        target[cut:].lstrip(),
        *lines[split_at + 2 :],
    ]
    pdf_path = images_pdf(directory / "missing_split.pdf", [page.image], dpi=page.dpi)
    return MissingSplitCase(pdf_path=pdf_path, page=page, text_lines=text_lines)


def _insert_sans(
    page: pymupdf.Page,
    point: tuple[float, float],
    text: str,
    *,
    fontsize: float = 11.0,
    color: tuple[float, float, float] = (0.0, 0.0, 0.0),
    render_mode: int = 0,
) -> None:
    page.insert_text(
        point,
        text,
        fontname="dejavu",
        fontfile=str(SANS_FONT),
        fontsize=fontsize,
        color=color,
        render_mode=render_mode,
    )


#: Paragraphs of :func:`make_text_pdf_with_table`, line by line.
TEXT_PDF_PARAGRAPHS = (
    ("Протокол заседания технического совета", "завода от 12 марта 1983 года."),
    ("Слушали доклад главного инженера о выполнении", "плана по выпуску продукции."),
)
#: Table rows of :func:`make_text_pdf_with_table`; the first row is the header.
TEXT_PDF_TABLE = (
    ("Цех", "План т", "Факт | т"),
    ("Литейный", "1 200", "1 248"),
    ("Механический", "860", "851"),
)
TEXT_PDF_CAPTION = "Таблица 1. Выпуск продукции по цехам"
TEXT_PDF_AFTER = "Постановили: работу цехов признать удовлетворительной."


def make_text_pdf_with_table(directory: Path) -> Path:
    """A born-digital page: two paragraphs, a ruled table and a caption right under it."""
    doc = pymupdf.open()
    page = doc.new_page(width=A4_PT[0], height=A4_PT[1])
    y = 80.0
    for paragraph in TEXT_PDF_PARAGRAPHS:
        for line in paragraph:
            _insert_sans(page, (72, y), line)
            y += 15.0
        y += 22.0
    top = y
    col_widths = (150.0, 100.0, 100.0)
    row_height = 22.0
    for r, row in enumerate(TEXT_PDF_TABLE):
        x = 72.0
        for c, cell in enumerate(row):
            rect = pymupdf.Rect(
                x, top + r * row_height, x + col_widths[c], top + (r + 1) * row_height
            )
            page.draw_rect(rect, color=(0, 0, 0), width=0.8)
            _insert_sans(page, (rect.x0 + 4, rect.y1 - 7), cell, fontsize=10)
            x += col_widths[c]
    bottom = top + len(TEXT_PDF_TABLE) * row_height
    _insert_sans(page, (72, bottom + 12), TEXT_PDF_CAPTION, fontsize=10)
    _insert_sans(page, (72, bottom + 50), TEXT_PDF_AFTER)
    path = directory / "text_table.pdf"
    doc.save(path)
    doc.close()
    return path


#: Table rows of :func:`make_table_pdf_with_positioned_words`; the first row is the header.
POSITIONED_TABLE = (
    ("Имя", "Статус", "Дата"),
    ("Тата", "Редактор по участникам", "06.04.2023"),
    ("Ильдус", "Режиссёр", "Выслал вопросы 11.04.2023"),
)


def make_table_pdf_with_positioned_words(directory: Path) -> Path:
    """A ruled table set the way some producers write text.

    Every 9 pt glyph is positioned on its own and no space glyph is drawn; the full stops of
    the dates are drawn after the digits, in another font and a larger size.
    """
    sans = pymupdf.Font(fontfile=str(SANS_FONT))
    mono = pymupdf.Font(fontfile=str(MONO_FONT))
    size, dot_size = 9.0, 14.0
    doc = pymupdf.open()
    page = doc.new_page(width=A4_PT[0], height=A4_PT[1])
    col_widths = (110.0, 170.0, 200.0)
    row_height = 20.0
    for r, row in enumerate(POSITIONED_TABLE):
        x = 60.0
        for c, cell in enumerate(row):
            rect = pymupdf.Rect(
                x, 100.0 + r * row_height, x + col_widths[c], 100.0 + (r + 1) * row_height
            )
            page.draw_rect(rect, color=(0, 0, 0), width=0.7)
            pen, baseline = rect.x0 + 3.0, rect.y1 - 6.0
            dots: list[float] = []
            for word in cell.split(" "):
                for char in word:
                    if char == ".":
                        dots.append(pen)
                        pen += mono.text_length(char, fontsize=dot_size)
                        continue
                    _insert_sans(page, (pen, baseline), char, fontsize=size)
                    pen += sans.text_length(char, fontsize=size)
                pen += sans.text_length(" ", fontsize=size)
            for dot_x in dots:
                page.insert_text(
                    (dot_x, baseline),
                    ".",
                    fontname="mono",
                    fontfile=str(MONO_FONT),
                    fontsize=dot_size,
                )
            x += col_widths[c]
    path = directory / "positioned_table.pdf"
    doc.save(path)
    doc.close()
    return path


def make_mojibake_pdf(directory: Path) -> Path:
    """Visible text whose Cyrillic was decoded as Latin-1 ("Ïðèâåò")."""
    source = "Протокол заседания технического совета завода, приказ номер 15."
    broken = source.encode("cp1251").decode("latin-1")
    doc = pymupdf.open()
    page = doc.new_page(width=A4_PT[0], height=A4_PT[1])
    page.insert_text((72, 100), broken, fontname="helv", fontsize=11)
    page.insert_text((72, 120), broken, fontname="helv", fontsize=11)
    path = directory / "mojibake.pdf"
    doc.save(path)
    doc.close()
    return path


def make_formula_pdf(directory: Path) -> Path:
    """Visible text with a formula set in a font whose name is Cambria Math."""
    doc = pymupdf.open()
    page = doc.new_page(width=A4_PT[0], height=A4_PT[1])
    _insert_sans(page, (72, 100), "Расчёт нагрузки на балку выполнен по формуле:")
    page.insert_text(
        (72, 130), "σ = M / W ≤ R", fontname="cmath", fontfile=str(SANS_FONT), fontsize=12
    )
    _insert_sans(page, (72, 160), "где M — изгибающий момент, W — момент сопротивления.")
    doc.save(directory / "formula_tmp.pdf")
    doc.close()
    path = directory / "formula.pdf"
    with pymupdf.open(directory / "formula_tmp.pdf") as tmp:
        for xref, _ext, _type, _basefont, refname, *_rest in tmp[0].get_fonts(full=False):
            if refname == "cmath":
                tmp.xref_set_key(xref, "BaseFont", "/ABCDEF+CambriaMath")
        tmp.save(path)
    (directory / "formula_tmp.pdf").unlink()
    return path


def make_short_text_pdf(directory: Path) -> Path:
    """Visible text with too few characters for the fast track."""
    doc = pymupdf.open()
    page = doc.new_page(width=A4_PT[0], height=A4_PT[1])
    _insert_sans(page, (72, 100), "Лист 5")
    path = directory / "short.pdf"
    doc.save(path)
    doc.close()
    return path


STAMP_TEXT = "ВХ. № 1234 от 12.03.1983"
PAGE_NUMBER_TEXT = "5"


def _scan_page(doc: pymupdf.Document, page: SyntheticPage) -> pymupdf.Page:
    pdf_page = doc.new_page(
        width=page.image.width * 72 / page.dpi, height=page.image.height * 72 / page.dpi
    )
    pdf_page.insert_image(pdf_page.rect, stream=_image_stream(page.image, "JPEG"))
    return pdf_page


def _insert_stamp(pdf_page: pymupdf.Page) -> None:
    _insert_sans(pdf_page, (380, 40), STAMP_TEXT, fontsize=10, color=(0.75, 0.1, 0.1))
    _insert_sans(pdf_page, (290, 820), PAGE_NUMBER_TEXT, fontsize=10)


def _insert_invisible_ocr(pdf_page: pymupdf.Page, page: SyntheticPage) -> None:
    for text, (x0, _y0, _x1, y1) in zip(page.lines, page.boxes_pt, strict=True):
        _insert_sans(pdf_page, (x0, y1), text, fontsize=9, render_mode=3)


def make_scan_with_stamp_pdf(directory: Path) -> Path:
    """A scan with a visible stamp and a visible page number over the image."""
    page = render_typewriter(typewriter_lines(25, seed=21))
    doc = pymupdf.open()
    _insert_stamp(_scan_page(doc, page))
    path = directory / "scan_stamp.pdf"
    doc.save(path)
    doc.close()
    return path


def make_scan_with_stamp_and_old_ocr_pdf(directory: Path) -> Path:
    """A scan with a visible stamp and an old invisible OCR layer at the same time."""
    page = render_typewriter(typewriter_lines(25, seed=22))
    doc = pymupdf.open()
    pdf_page = _scan_page(doc, page)
    _insert_invisible_ocr(pdf_page, page)
    # Old OCR also "read" the stamp: an invisible span overlapping the visible one.
    _insert_sans(pdf_page, (380, 40), STAMP_TEXT, fontsize=10, render_mode=3)
    _insert_stamp(pdf_page)
    path = directory / "scan_stamp_old_ocr.pdf"
    doc.save(path)
    doc.close()
    return path


def make_scan_with_invisible_ocr_pdf(directory: Path) -> Path:
    """A scan with an invisible OCR layer (render mode 3), as produced by OCR tools."""
    page = render_typewriter(typewriter_lines(25, seed=23))
    doc = pymupdf.open()
    _insert_invisible_ocr(_scan_page(doc, page), page)
    path = directory / "scan_invisible_ocr.pdf"
    doc.save(path)
    doc.close()
    return path


def make_multi_image_scan_pdf(directory: Path) -> Path:
    """A scan made of two images (top and bottom halves) with visible-mode text under them."""
    page = render_typewriter(typewriter_lines(25, seed=24))
    width, height = page.image.size
    top = page.image.crop((0, 0, width, height // 2))
    bottom = page.image.crop((0, height // 2, width, height))
    doc = pymupdf.open()
    pdf_page = doc.new_page(width=width * 72 / page.dpi, height=height * 72 / page.dpi)
    for text, (x0, _y0, _x1, y1) in zip(page.lines, page.boxes_pt, strict=True):
        _insert_sans(pdf_page, (x0, y1), text, fontsize=9)
    rect = pdf_page.rect
    middle = rect.y0 + rect.height / 2
    pdf_page.insert_image(
        pymupdf.Rect(rect.x0, rect.y0, rect.x1, middle), stream=_image_stream(top)
    )
    pdf_page.insert_image(
        pymupdf.Rect(rect.x0, middle, rect.x1, rect.y1), stream=_image_stream(bottom)
    )
    path = directory / "multi_image_scan.pdf"
    doc.save(path)
    doc.close()
    return path


def make_white_text_pdf(directory: Path) -> Path:
    """White text on a white page: present in the text layer, invisible on the render."""
    doc = pymupdf.open()
    page = doc.new_page(width=A4_PT[0], height=A4_PT[1])
    _insert_sans(page, (72, 100), "Скрытый белый текст на белом фоне страницы.", color=(1, 1, 1))
    path = directory / "white_text.pdf"
    doc.save(path)
    doc.close()
    return path


def make_jpeg_set(directory: Path, *, count: int = 3, dpi: int = 300) -> list[Path]:
    """A set of JPEG page images with DPI metadata, named 1.jpg, 2.jpg, ... (natural order)."""
    paths: list[Path] = []
    for index in range(count):
        page = render_typewriter(typewriter_lines(12, seed=40 + index), dpi=150, noise=3.0)
        path = directory / f"{index + 1}.jpg"
        page.image.save(path, format="JPEG", quality=85, dpi=(dpi, dpi))
        paths.append(path)
    return paths


def jpeg_set_pdf(directory: Path, jpegs: list[Path]) -> Path:
    """Put JPEG files into a PDF without re-encoding, as the ingest of an image set does."""
    doc = pymupdf.open()
    for jpeg in jpegs:
        with Image.open(jpeg) as image:
            width, height = image.size
            dpi = image.info.get("dpi", (300, 300))[0] or 300
        page = doc.new_page(width=width * 72 / dpi, height=height * 72 / dpi)
        page.insert_image(page.rect, stream=jpeg.read_bytes())
    path = directory / "images.pdf"
    doc.save(path)
    doc.close()
    return path
