"""Searchable PDF: a copy of the source with an invisible text layer (section 9.3).

Images are never re-encoded; text is only removed with fill-less redactions that leave images and
vector graphics alone, and our layer is appended at the end of the page content stream.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

from ocr_archive.build.markers import is_marker_only
from ocr_archive.exceptions import BuildError
from ocr_archive.lines import line_detection_from_json, line_plausible
from ocr_archive.models import LineBox, PageTrack, TextLayer
from ocr_archive.pdfpages import strip_invisible_spans, strip_text
from ocr_archive.textnorm import content_text, is_pipe_row, is_table_separator

logger = logging.getLogger(__name__)

#: Share of plausible lines required to bind the layer to detected lines (section 9.3).
LAYER_MIN_SHARE = 0.9
FONT_HEIGHT_SHARE = 0.85
PAGE_BLOCK_START_SIZE = 11.0
PAGE_BLOCK_MIN_SIZE = 0.2
PAGE_BLOCK_STEP = 0.85
REPLACEMENT_CHAR = "?"
FONT_NAME = "ocrlayer"

_HTML_TABLE_RE = re.compile(r"<table\b.*?</table\s*>", re.IGNORECASE | re.DOTALL)
_ROW_RE = re.compile(r"<tr\b[^>]*>(.*?)(?=<tr\b|</table\s*>|$)", re.IGNORECASE | re.DOTALL)
_CELL_RE = re.compile(r"<t[dh]\b[^>]*>(.*?)(?=<t[dh]\b|</tr\s*>|$)", re.IGNORECASE | re.DOTALL)
_UNESCAPED_PIPE_RE = re.compile(r"(?<!\\)\|")
_SPACES_RE = re.compile(r"[ \t ]+")


class PdfLayer:
    """How the layer of a page was written (manifest field ``pdf_layer``)."""

    LINES = "lines"
    PAGE_BLOCK = "page_block"
    ORIGINAL_KEPT = "original_kept"
    NONE = "none"


@dataclass(frozen=True)
class PdfPageInput:
    page_index: int
    track: str
    text_layer: str | None
    text: str | None
    line_boxes_json: str | None


@dataclass
class PdfPageResult:
    page_index: int
    pdf_layer: str
    old_text_kept: bool = False
    char_replacements: dict[str, int] = field(default_factory=dict)
    expected_text: str = ""  # text the page must hold after the build, for verify_pdf


def _html_table_lines(match: re.Match[str]) -> str:
    rows = []
    for row in _ROW_RE.finditer(match.group(0)):
        cells = [content_text(cell.group(1)).strip() for cell in _CELL_RE.finditer(row.group(1))]
        rows.append(" ".join(cell for cell in cells if cell))
    return "\n".join(rows)


def layer_lines(text: str) -> list[str]:
    """Lines of the text layer: verbatim lines with markers; pipe-table separators dropped, cells
    of pipe and HTML tables joined with spaces; LaTeX as it is."""
    text = _HTML_TABLE_RE.sub(_html_table_lines, text)
    lines: list[str] = []
    for raw in text.split("\n"):
        if is_table_separator(raw):
            continue
        if is_pipe_row(raw):
            cells = [c.strip().replace("\\|", "|") for c in _UNESCAPED_PIPE_RE.split(raw.strip())]
            line = " ".join(cell for cell in cells if cell)
        else:
            line = raw
        line = _SPACES_RE.sub(" ", line).strip()
        if line:
            lines.append(line)
    return lines


def _replace_missing(lines: list[str], font: pymupdf.Font) -> tuple[list[str], dict[str, int]]:
    replaced: Counter[str] = Counter()
    result = []
    for line in lines:
        chars = []
        for char in line:
            if char == " " or font.has_glyph(ord(char)):
                chars.append(char)
            else:
                replaced[char] += 1
                chars.append(REPLACEMENT_CHAR)
        result.append("".join(chars))
    return result, dict(replaced)


def _bind_lines(lines: list[str], boxes: list[LineBox]) -> list[tuple[str, LineBox]] | None:
    """Pair counted lines with boxes; marker-only lines join a neighbour (section 9.3)."""
    counted = [i for i, line in enumerate(lines) if not is_marker_only(line)]
    if len(counted) != len(boxes) or not counted:
        return None
    if not line_plausible([lines[i] for i in counted], boxes, min_share=LAYER_MIN_SHARE):
        return None
    texts = [lines[i] for i in counted]
    pending_prefix: list[str] = []
    slot = -1
    for i, line in enumerate(lines):
        if i in counted:
            slot += 1
            if pending_prefix:
                texts[slot] = " ".join([*pending_prefix, texts[slot]])
                pending_prefix = []
        elif slot >= 0:
            texts[slot] = f"{texts[slot]} {line}"
        else:
            pending_prefix.append(line)
    return list(zip(texts, boxes, strict=True))


def _write_lines(page: pymupdf.Page, pairs: list[tuple[str, LineBox]], font: pymupdf.Font) -> None:
    for text, box in pairs:
        height = box.y1 - box.y0
        width = box.x1 - box.x0
        if height <= 0 or width <= 0:
            continue
        size = FONT_HEIGHT_SHARE * height
        text_width = font.text_length(text, fontsize=size)
        if text_width <= 0:
            continue
        glyph_height = size * (font.ascender - font.descender)
        baseline = box.y0 + (height - glyph_height) / 2 + size * font.ascender
        pivot = pymupdf.Point(box.x0, baseline)
        writer = pymupdf.TextWriter(page.rect)
        writer.append(pivot, text, font=font, fontsize=size)
        writer.write_text(
            page,
            render_mode=3,
            overlay=True,
            morph=(pivot, pymupdf.Matrix(width / text_width, 1)),
        )


def _write_page_block(page: pymupdf.Page, text: str, font_path: Path) -> None:
    size = PAGE_BLOCK_START_SIZE
    rect = page.rect
    while size >= PAGE_BLOCK_MIN_SIZE:
        result = page.insert_textbox(
            rect,
            text,
            fontsize=size,
            fontname=FONT_NAME,
            fontfile=str(font_path),
            render_mode=3,
            overlay=True,
        )
        if result >= 0:
            return
        size *= PAGE_BLOCK_STEP
    raise BuildError(f"page {page.number + 1}: text does not fit the page block")


def _add_layer(
    page: pymupdf.Page,
    text: str,
    line_boxes_json: str | None,
    font: pymupdf.Font,
    font_path: Path,
    result: PdfPageResult,
) -> str:
    """Write our layer on the page; return the layer text as written."""
    lines, replacements = _replace_missing(layer_lines(text), font)
    result.char_replacements = replacements
    if not lines:
        result.pdf_layer = PdfLayer.NONE
        return ""
    pairs = None
    if line_boxes_json and page.rotation == 0:
        detection = line_detection_from_json(line_boxes_json)
        if detection.reliable:
            pairs = _bind_lines(lines, detection.boxes)
    if pairs is not None:
        _write_lines(page, pairs, font)
        result.pdf_layer = PdfLayer.LINES
        return "\n".join(text for text, _ in pairs)
    block = "\n".join(lines)
    _write_page_block(page, block, font_path)
    result.pdf_layer = PdfLayer.PAGE_BLOCK
    return block


def build_searchable_pdf(
    source_pdf: Path, out_pdf: Path, pages: list[PdfPageInput], font_path: Path
) -> list[PdfPageResult]:
    """Write the searchable PDF and return per-page results with the expected text.

    :param source_pdf: source PDF (for an image set, the PDF assembled at upload).
    :param out_pdf: file to write.
    :param pages: snapshot data of every page.
    :param font_path: font with Cyrillic glyphs for the layer.
    :raises BuildError: when the PDF cannot be processed.
    """
    results: list[PdfPageResult] = []
    try:
        font = pymupdf.Font(fontfile=str(font_path))
        with pymupdf.open(source_pdf) as doc:
            for item in pages:
                page = doc[item.page_index]
                result = PdfPageResult(item.page_index, PdfLayer.ORIGINAL_KEPT)
                if item.track == PageTrack.PDF_TEXT or item.text is None:
                    result.expected_text = page.get_text()
                    results.append(result)
                    continue
                if item.text_layer == TextLayer.HIDDEN:
                    strip_text(doc, item.page_index)
                elif item.text_layer == TextLayer.VISIBLE:
                    result.old_text_kept = not strip_invisible_spans(doc, item.page_index)
                page = doc[item.page_index]
                remaining = page.get_text()
                layer = _add_layer(page, item.text, item.line_boxes_json, font, font_path, result)
                result.expected_text = remaining + layer
                results.append(result)
            doc.save(out_pdf, garbage=3, deflate=True)
    except (RuntimeError, ValueError) as err:
        raise BuildError(f"cannot build PDF: {err}") from err
    return results
