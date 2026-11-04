"""Text normalization used to compare passes, extract numbers and describe tables (section 6.8)."""

from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass

from ocr_archive.build.markers import is_marker_only

QUOTES = "«»„“”‟″"
DASHES = "‐‑‒–—―−"
_QUOTE_DASH_TABLE = str.maketrans({**dict.fromkeys(QUOTES, '"'), **dict.fromkeys(DASHES, "-")})

_TAG_RE = re.compile(r"<[^>]*>")
_HYPHEN_BREAK_RE = re.compile(r"([^\W\d_])-[ \t]*\n[ \t]*([^\W\d_])")
_WS_RE = re.compile(r"\s+")
_UNESCAPED_PIPE_RE = re.compile(r"(?<!\\)\|")
_HTML_TABLE_RE = re.compile(r"<table\b.*?</table\s*>", re.IGNORECASE | re.DOTALL)
_TR_RE = re.compile(r"<tr\b[^>]*>(.*?)(?=<tr\b|</table\s*>|$)", re.IGNORECASE | re.DOTALL)
_CELL_RE = re.compile(r"<t[dh]\b([^>]*)>", re.IGNORECASE)
_SPAN_ATTR_RE = re.compile(r"\b(colspan|rowspan)\s*=\s*[\"']?(\d+)", re.IGNORECASE)
_FORMULA_RE = re.compile(r"\\\((.*?)\\\)|\\\[(.*?)\\\]", re.DOTALL)

_NUMBER_RE = re.compile(
    r"(?P<sign>(?<![^\s(=/])[-−+])?"
    r"(?<!\d)(?P<int>\d{1,3}(?:[   ]\d{3}(?!\d))+|\d+)"
    r"(?P<frac>[.,]\d+)?"
    r"(?:[   ]?(?P<pct>[%‰]))?"
)
_GROUP_SPACES = str.maketrans("", "", "   ")


def unify_quotes_dashes(text: str) -> str:
    """Replace typographic quotes with ``"`` and dashes with ``-``."""
    return text.translate(_QUOTE_DASH_TABLE)


def is_table_separator(line: str) -> bool:
    """Tell whether a line is a separator row of a Markdown pipe table."""
    stripped = line.strip()
    return "|" in stripped and "-" in stripped and set(stripped) <= set("|-: ")


def is_pipe_row(line: str) -> bool:
    """Tell whether a line belongs to a Markdown pipe table."""
    return line.strip().startswith("|")


def soft_normalize(text: str) -> str:
    """Normalize text for pass comparison: NFC, no markup, unified quotes and dashes, no hyphen
    breaks, collapsed whitespace."""
    text = unicodedata.normalize("NFC", text)
    text = _TAG_RE.sub(" ", text)
    text = "\n".join(line for line in text.split("\n") if not is_table_separator(line))
    text = text.replace("|", " ")
    text = unify_quotes_dashes(text)
    text = _HYPHEN_BREAK_RE.sub(r"\1\2", text)
    return _WS_RE.sub(" ", text).strip()


def content_text(text: str) -> str:
    """Return the content of a text without markup: tags with attributes removed, entities expanded,
    pipe-table separators removed, ``|`` replaced with a space. Formulas are kept: they are content.
    """
    text = unicodedata.normalize("NFC", text)
    text = _TAG_RE.sub(" ", text)
    text = html.unescape(text)
    text = "\n".join(line for line in text.split("\n") if not is_table_separator(line))
    return text.replace("\\|", " ").replace("|", " ")


def numbers(text: str) -> list[str]:
    """Return normalized number tokens in order of appearance (section 6.8).

    Signs ``-``, ``−`` and ``+`` count only after a line start, a space, ``(``, ``=`` or ``/``;
    groups of three digits separated by spaces form one number; the decimal separator is kept;
    a ``%`` or ``‰`` right after the number (with or without a space) belongs to it.
    """
    result = []
    for match in _NUMBER_RE.finditer(content_text(text)):
        sign = "-" if match.group("sign") in ("-", "−") else ""
        integer = match.group("int").translate(_GROUP_SPACES)
        frac = match.group("frac") or ""
        pct = match.group("pct") or ""
        result.append(f"{sign}{integer}{frac}{pct}")
    return result


def has_tables(text: str) -> bool:
    """Tell whether the text contains a pipe table row or an HTML table."""
    return "<table" in text.lower() or any(is_pipe_row(line) for line in text.split("\n"))


def has_formulas(text: str) -> bool:
    """Tell whether the text contains a LaTeX formula ``\\(`` or ``\\[``."""
    return "\\(" in text or "\\[" in text


def formula_spans(text: str) -> list[tuple[int, int]]:
    """Return ``[start, end)`` ranges of formulas ``\\(…\\)`` and ``\\[…\\]``."""
    return [(m.start(), m.end()) for m in _FORMULA_RE.finditer(text)]


def html_table_spans(text: str) -> list[tuple[int, int]]:
    """Return ``[start, end)`` ranges of HTML tables."""
    return [(m.start(), m.end()) for m in _HTML_TABLE_RE.finditer(text)]


def strip_formulas(text: str) -> str:
    """Remove formulas, keeping the rest of the text."""
    return _FORMULA_RE.sub(" ", text)


def _pipe_cells(line: str) -> int:
    parts = _UNESCAPED_PIPE_RE.split(line.strip())
    if parts and parts[0].strip() == "":
        parts = parts[1:]
    if parts and parts[-1].strip() == "":
        parts = parts[:-1]
    return len(parts)


def _html_shape(table_html: str) -> list[int]:
    rows: list[int] = []
    carry: dict[int, int] = {}  # column -> rows still covered by a rowspan from above
    for row_match in _TR_RE.finditer(table_html):
        occupied = {col for col, left in carry.items() if left > 0}
        next_carry = {col: left - 1 for col, left in carry.items() if left > 1}
        col = 0
        for cell in _CELL_RE.finditer(row_match.group(1)):
            attrs = {k.lower(): int(v) for k, v in _SPAN_ATTR_RE.findall(cell.group(1))}
            colspan = max(1, attrs.get("colspan", 1))
            rowspan = max(1, attrs.get("rowspan", 1))
            while col in occupied:
                col += 1
            for c in range(col, col + colspan):
                occupied.add(c)
                if rowspan > 1:
                    next_carry[c] = rowspan - 1
            col += colspan
        rows.append(len(occupied))
        carry = next_carry
    return rows


@dataclass(frozen=True)
class _TableBlock:
    start: int
    shape: list[int]


def table_shape(text: str) -> list[list[int]]:
    """Return the logical grid of every table: for each row the number of occupied columns.

    Pipe tables count cells by unescaped ``|``; HTML tables honour ``colspan`` and ``rowspan``,
    so the same table written in either markup gives the same shape.
    """
    blocks: list[_TableBlock] = []
    html_spans = html_table_spans(text)
    for start, end in html_spans:
        blocks.append(_TableBlock(start, _html_shape(text[start:end])))
    offset = 0
    current: list[int] | None = None
    current_start = 0
    for line in text.split("\n"):
        inside_html = any(s <= offset < e for s, e in html_spans)
        if is_pipe_row(line) and not inside_html:
            if current is None:
                current, current_start = [], offset
            if not is_table_separator(line):
                current.append(_pipe_cells(line))
        elif current is not None:
            blocks.append(_TableBlock(current_start, current))
            current = None
        offset += len(line) + 1
    if current is not None:
        blocks.append(_TableBlock(current_start, current))
    return [block.shape for block in sorted(blocks, key=lambda b: b.start)]


def counted_lines(text: str) -> list[str]:
    """Return lines that count against the line detector (sections 6.9, 6.13, 9.3): non-empty,
    not a pipe-table separator and not made of markers only."""
    return [
        line
        for line in text.split("\n")
        if line.strip() and not is_table_separator(line) and not is_marker_only(line)
    ]
