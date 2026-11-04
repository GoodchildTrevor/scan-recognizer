"""Fast track: pages with their own visible text are taken from the PDF without the model (6.1)."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import pymupdf
from pymupdf.table import Table

from ocr_archive.models import FastTrackCheck, PageTrack, TextLayer
from ocr_archive.pdfpages import image_area_share
from ocr_archive.settings import FastTrackConfig

logger = logging.getLogger(__name__)

_FONT_NAME_NOISE = re.compile(r"[\s_\-]+")
# A gap between two glyphs wider than this share of the font size is a word space.
_SPACE_GAP = 0.15
# Two glyphs whose baselines differ by less than this share of the font size are on one line.
_BASELINE_TOLERANCE = 0.4


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


@dataclass(frozen=True, slots=True)
class _Char:
    """A visible character of a page: its box, its baseline and the font size it is set in."""

    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    baseline: float
    size: float
    upright: bool

    def inside(self, rect: tuple[float, float, float, float]) -> bool:
        """Whether the center of the box lies in ``rect``, as ``Table.extract`` decides it."""
        x = (self.x0 + self.x1) / 2.0
        y = (self.y0 + self.y1) / 2.0
        return rect[0] <= x < rect[2] and rect[1] <= y < rect[3]


def _page_chars(page: pymupdf.Page) -> list[_Char]:
    chars: list[_Char] = []
    for block in page.get_text("rawdict")["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block["lines"]:
            upright = abs(line["dir"][1]) < 0.1
            for span in line["spans"]:
                size = float(span["size"])
                for char in span["chars"]:
                    if char["c"].isspace():
                        continue  # spaces, real or inserted by MuPDF, follow from the gaps
                    x0, y0, x1, y1 = char["bbox"]
                    baseline = float(char["origin"][1])
                    chars.append(_Char(char["c"], x0, y0, x1, y1, baseline, size, upright))
    return chars


def _text_lines(chars: list[_Char]) -> list[str]:
    """Join characters into lines by their baselines and the gaps between them.

    ``Table.extract`` groups characters by the top of their boxes with a fixed tolerance of
    3 pt and inserts a space only for a gap wider than 3 pt. A full stop set in another font
    then lands on a line of its own ("06042023 . ."), and 9-11 pt text whose words are
    positioned one by one, without space glyphs, loses its spaces. Here the baseline decides
    the line and the gap is measured against the font size. Space characters are not taken
    from the page: MuPDF inserts them for its own lines, which it builds in drawing order, so
    digits drawn before their full stops get a space where the full stop belongs.
    """
    lines: list[list[_Char]] = []
    for char in sorted(chars, key=lambda item: (item.baseline, item.x0)):
        if lines:
            first = lines[-1][0]
            tolerance = _BASELINE_TOLERANCE * max(char.size, first.size)
            if abs(char.baseline - first.baseline) <= tolerance:
                lines[-1].append(char)
                continue
        lines.append([char])
    texts: list[str] = []
    for line in lines:
        line.sort(key=lambda item: item.x0)
        parts: list[str] = []
        previous: _Char | None = None
        for char in line:
            if previous is not None and char.x0 - previous.x1 > _SPACE_GAP * max(
                char.size, previous.size
            ):
                parts.append(" ")
            parts.append(char.text)
            previous = char
        text = " ".join("".join(parts).split())
        if text:
            texts.append(text)
    return texts


def _table_rows(table: Table, chars: list[_Char]) -> list[list[str | None]]:
    """Cell texts of a table, ``None`` for a cell merged into its neighbour.

    Characters are assigned to cells the way ``Table.extract`` does it, by the center of their
    box; the text of a cell is then rebuilt by :func:`_text_lines`. A cell with rotated text
    keeps the text of ``Table.extract``, which handles the rotation.
    """
    extracted: list[list[str | None]] | None = None
    rows: list[list[str | None]] = []
    for row_index, row in enumerate(table.rows):
        row_chars = [char for char in chars if char.inside(row.bbox)]
        cells: list[str | None] = []
        for cell_index, cell in enumerate(row.cells):
            if cell is None:
                cells.append(None)
                continue
            cell_chars = [char for char in row_chars if char.inside(cell)]
            if all(char.upright for char in cell_chars):
                cells.append("\n".join(_text_lines(cell_chars)))
                continue
            if extracted is None:
                extracted = table.extract()
            cells.append(extracted[row_index][cell_index])
        rows.append(cells)
    return rows


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
        tables = page.find_tables().tables
        chars = _page_chars(page) if tables else []
        for table in tables:
            rect = pymupdf.Rect(table.bbox)
            markdown = _pipe_table(_table_rows(table, chars))
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
