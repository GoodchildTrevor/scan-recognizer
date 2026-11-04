"""Fast track: pages with their own visible text are taken from the PDF without the model (6.1)."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pymupdf

from ocr_archive.models import FastTrackCheck, PageTrack, TextLayer
from ocr_archive.pdfpages import image_area_share
from ocr_archive.settings import FastTrackConfig

logger = logging.getLogger(__name__)

_FONT_NAME_NOISE = re.compile(r"[\s_\-]+")


def _is_bad_char(char: str) -> bool:
    """Characters typical of a broken encoding: U+FFFD, private use area, Latin-1 letters.

    Cyrillic decoded with the wrong code page looks like "Ïðèâåò" (U+00C0–U+00FF).
    """
    code = ord(char)
    return code == 0xFFFD or 0xE000 <= code <= 0xF8FF or 0x00C0 <= code <= 0x00FF


def _normalize_font_name(name: str) -> str:
    return _FONT_NAME_NOISE.sub("", name).casefold()


def _has_math_fonts(page: pymupdf.Page, hints: list[str]) -> bool:
    normalized_hints = [_normalize_font_name(hint) for hint in hints if hint.strip()]
    for font in page.get_fonts(full=False):
        # Entry: (xref, ext, type, basefont, refname, encoding, ...).
        names = (str(font[3]), str(font[4]))
        for name in names:
            normalized = _normalize_font_name(name)
            if any(hint in normalized for hint in normalized_hints):
                return True
    return False


def fast_track_check(
    pdf_path: Path, page_index: int, layer: TextLayer, cfg: FastTrackConfig
) -> FastTrackCheck:
    """Decide whether a page goes the fast track (text from the PDF) or through the model.

    :param pdf_path: source PDF.
    :param page_index: 0-based page index.
    :param layer: result of :func:`ocr_archive.pdfpages.text_layer_kind`.
    :param cfg: ``fast_track`` block.
    :return: ``pdf_text`` with no reason, or ``vlm`` with the first failed condition:
        ``disabled``, ``no_visible_text``, ``images``, ``too_short``, ``bad_chars``,
        ``math_fonts``.
    """
    if not cfg.enabled:
        return FastTrackCheck(track=PageTrack.VLM, reason="disabled")
    if layer is not TextLayer.VISIBLE:
        return FastTrackCheck(track=PageTrack.VLM, reason="no_visible_text")
    with pymupdf.open(pdf_path) as doc:
        page = doc[page_index]
        if image_area_share(page) > cfg.max_image_area:
            return FastTrackCheck(track=PageTrack.VLM, reason="images")
        chars = [char for char in page.get_text("text") if not char.isspace()]
        if len(chars) < cfg.min_chars:
            return FastTrackCheck(track=PageTrack.VLM, reason="too_short")
        bad = sum(1 for char in chars if _is_bad_char(char))
        if bad / len(chars) > cfg.max_bad_char_ratio:
            return FastTrackCheck(track=PageTrack.VLM, reason="bad_chars")
        if _has_math_fonts(page, cfg.math_font_hints):
            return FastTrackCheck(track=PageTrack.VLM, reason="math_fonts")
    return FastTrackCheck(track=PageTrack.PDF_TEXT, reason=None)


def _cell_text(value: str | None) -> str:
    if value is None:
        return ""
    return " ".join(value.split("\n")).strip().replace("|", "\\|")


def _pipe_table(rows: list[list[str | None]]) -> str:
    """Write a table as a pipe table; the first row is the header."""
    width = max((len(row) for row in rows), default=0)
    if width == 0:
        return ""
    lines: list[str] = []
    for index, row in enumerate(rows):
        cells = [_cell_text(cell) for cell in row] + [""] * (width - len(row))
        lines.append("| " + " | ".join(cells) + " |")
        if index == 0:
            lines.append("|" + "|".join([" --- "] * width) + "|")
    return "\n".join(lines)


def _center_inside(bbox: tuple[float, float, float, float], rects: list[pymupdf.Rect]) -> bool:
    x = (bbox[0] + bbox[2]) / 2.0
    y = (bbox[1] + bbox[3]) / 2.0
    return any(rect.x0 <= x <= rect.x1 and rect.y0 <= y <= rect.y1 for rect in rects)


def pdf_text_markdown(pdf_path: Path, page_index: int) -> str:
    """Extract the visible text of a page in the model answer format (section 7).

    Tables found by ``find_tables`` become pipe tables; the remaining text comes from text
    blocks. Only lines whose box center lies inside a table box are dropped, so a caption
    right under a table survives. Blocks are separated by an empty line.

    :param pdf_path: source PDF.
    :param page_index: 0-based page index.
    :return: Markdown text of the page.
    """
    items: list[tuple[float, float, str]] = []
    with pymupdf.open(pdf_path) as doc:
        page = doc[page_index]
        table_rects: list[pymupdf.Rect] = []
        for table in page.find_tables().tables:
            rect = pymupdf.Rect(table.bbox)
            markdown = _pipe_table(table.extract())
            if markdown:
                table_rects.append(rect)
                items.append((rect.y0, rect.x0, markdown))
        for block in page.get_text("dict", sort=True)["blocks"]:
            if block.get("type") != 0:
                continue
            kept_lines: list[tuple[tuple[float, float, float, float], str]] = []
            for line in block["lines"]:
                bbox = tuple(line["bbox"])
                if _center_inside(bbox, table_rects):
                    continue
                text = "".join(span["text"] for span in line["spans"]).rstrip()
                if text.strip():
                    kept_lines.append((bbox, text))
            if not kept_lines:
                continue
            y0 = min(bbox[1] for bbox, _ in kept_lines)
            x0 = min(bbox[0] for bbox, _ in kept_lines)
            items.append((y0, x0, "\n".join(text for _, text in kept_lines)))
    items.sort(key=lambda item: (item[0], item[1]))
    return "\n\n".join(text for _, _, text in items)
