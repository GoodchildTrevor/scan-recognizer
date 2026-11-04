"""Pandoc Markdown for the DOCX build (section 9.2).

Ordinary text is escaped so that pandoc never reads structure into it: backslash escapes for
Markdown syntax characters, HTML entities for square brackets (``\\[`` starts a formula under
``tex_math_single_backslash``), and escapes for list markers at the start of a paragraph.
Formulas pass through unchanged, markers get the "Uncertain" character style, HTML tables are
passed as one raw HTML block for ``html_tables.lua``.

Along with the Markdown, every page gets a :class:`PageReference`: the content that was handed to
pandoc, used by the DOCX verification (section 9.5).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from urllib.parse import quote

from ocr_archive.build.markers import MARKER_RE
from ocr_archive.build.normalize import HINT_CLOSE, HINT_OPEN, Block, BlockKind
from ocr_archive.textnorm import content_text, is_table_separator, table_shape

UNCERTAIN_STYLE = "Uncertain"
UNVERIFIED_STYLE = "Unverified"
SOURCE_PAGE_STYLE = "Source Page"
DRAFT_BANNER_STYLE = "Draft Banner"
DRAFT_BANNER_TEXT = "ЧЕРНОВИК — не проверено"
NOTE_UNVERIFIED = " · не проверено"
NOTE_UNRECOGNIZED = " · не распознано"

_BACKSLASH_ESCAPED = frozenset("\\*_#<>|~^`&@")
_ENTITIES = {"[": "&#91;", "]": "&#93;"}
_LIST_MARKER_RE = re.compile(r"\(?(?:[0-9]{1,9}|[A-Za-z]|[IVXLCDMivxlcdm]+)([.)])(?=\s|$)")
_TABLE_CAPTION_RE = re.compile(r"(?i)table(:)")
_FORMULA_RE = re.compile(r"\\\((.*?)\\\)|\\\[(.*?)\\\]", re.DOTALL)
_ESCAPED_PIPE = "\\|"
_TAG_RE = re.compile(r"<[^>]*>")
_ENTITY_RE = re.compile(r"&#?\w+;")
_UNESCAPED_PIPE_RE = re.compile(r"(?<!\\)\|")
_BACKTICKS_RE = re.compile(r"`+")
_FIRST_ROW_RE = re.compile(r"<tr\b[^>]*>(.*?)(?=<tr\b|</table\s*>|$)", re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True)
class TableRef:
    """Logical grid of a table: occupied columns per row, and whether the first row is empty.

    An empty first row is reported because pandoc may drop an all-empty header row.
    """

    shape: list[int]
    first_row_empty: bool

    def canonical(self) -> list[int]:
        """Shape without an all-empty first row."""
        return self.shape[1:] if self.first_row_empty else self.shape


@dataclass
class PageReference:
    """What was handed to pandoc for one page, in a form comparable with the DOCX read back.

    :ivar texts: literal text units outside formulas: one per paragraph and per table row.
    :ivar formulas: TeX bodies of the formulas rendered as math.
    :ivar tables: logical grid of every table, in order.
    """

    texts: list[str] = field(default_factory=list)
    formulas: list[str] = field(default_factory=list)
    tables: list[TableRef] = field(default_factory=list)


@dataclass(frozen=True)
class MdPage:
    page_index: int
    printed_label: str | None
    blocks: list[Block]
    note: str | None = None  # draft annotation appended to the "Стр. N" line


def escape_markdown(text: str, *, line_start: bool = True) -> str:
    """Escape ordinary text for pandoc Markdown (section 9.2).

    ``\\ * _ # < > | ~ ^ ` & @`` get a backslash, square brackets become ``&#91;`` and ``&#93;``.
    At the start of a paragraph a list marker (``1.``, ``a)``, ``IV.``), a leading ``-``, ``+``
    or ``:`` and a ``Table:`` caption prefix are escaped as well. ``$`` stays a plain character.

    :param text: text without formulas.
    :param line_start: the text starts a paragraph or a line.
    """
    special: dict[int, str] = {}
    if line_start and text:
        if text[0] in "-+:":
            special[0] = "\\" + text[0]
        marker = _LIST_MARKER_RE.match(text)
        if marker is not None:
            position = marker.start(1)
            special[position] = "\\." if marker.group(1) == "." else "&#41;"
        caption = _TABLE_CAPTION_RE.match(text)
        if caption is not None:
            special[caption.start(1)] = "\\:"
    out: list[str] = []
    for position, char in enumerate(text):
        if position in special:
            out.append(special[position])
        elif char in _ENTITIES:
            out.append(_ENTITIES[char])
        elif char in _BACKSLASH_ESCAPED:
            out.append("\\" + char)
        elif char in "\t\r\n":
            out.append(" ")
        else:
            out.append(char)
    return "".join(out)


def parse_hints(text: str, open_hint: bool) -> tuple[str, list[tuple[int, int]], bool]:
    """Remove draft hint marks and return the ranges they delimit.

    :param text: text with :data:`HINT_OPEN` / :data:`HINT_CLOSE` marks.
    :param open_hint: a hint started in an earlier block and is still open.
    :return: clean text, hint ranges in clean coordinates, whether a hint stays open.
    """
    clean: list[str] = []
    ranges: list[tuple[int, int]] = []
    start: int | None = 0 if open_hint else None
    for char in text:
        if char == HINT_OPEN:
            if start is None:
                start = len(clean)
        elif char == HINT_CLOSE:
            if start is not None:
                ranges.append((start, len(clean)))
                start = None
        else:
            clean.append(char)
    still_open = start is not None
    if start is not None:
        ranges.append((start, len(clean)))
    return "".join(clean), [r for r in ranges if r[1] > r[0]], still_open


def _align_ranges(
    ranges: list[tuple[int, int]], atoms: list[tuple[int, int]], length: int
) -> list[tuple[int, int]]:
    """Expand ranges so that they never cut an atom (formula, marker, entity); merge overlaps."""
    aligned: list[tuple[int, int]] = []
    for start, end in ranges:
        start, end = max(0, start), min(length, end)
        for atom_start, atom_end in atoms:
            if atom_start < start < atom_end:
                start = atom_start
            if atom_start < end < atom_end:
                end = atom_end
        if end > start:
            aligned.append((start, end))
    aligned.sort()
    merged: list[tuple[int, int]] = []
    for start, end in aligned:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


@dataclass(frozen=True)
class _Inline:
    markdown: str
    literal: str  # text outside formulas, formulas replaced with a space
    formulas: list[str]


def _render_inline(
    clean: str, ranges: list[tuple[int, int]], *, line_start: bool, raw_atoms: dict[int, int]
) -> _Inline:
    """Render one paragraph or table cell.

    :param raw_atoms: extra ``start -> end`` ranges written verbatim (escaped pipes in cells).
    """
    formulas = {m.start(): m for m in _FORMULA_RE.finditer(clean)}
    markers = {
        m.start(): m
        for m in MARKER_RE.finditer(clean)
        if not any(f.start() <= m.start() < f.end() for f in formulas.values())
    }
    atoms = sorted(
        [(m.start(), m.end()) for m in formulas.values()]
        + [(m.start(), m.end()) for m in markers.values()]
        + list(raw_atoms.items())
    )
    hint_ranges = _align_ranges(ranges, atoms, len(clean))
    opens = {start for start, _ in hint_ranges}
    closes = {end for _, end in hint_ranges}

    out: list[str] = []
    literal: list[str] = []
    found_formulas: list[str] = []
    run: list[str] = []
    run_start = 0

    def flush_run() -> None:
        if run:
            out.append(escape_markdown("".join(run), line_start=line_start and run_start == 0))
            literal.append("".join(run))
            run.clear()

    position = 0
    while position <= len(clean):
        if position in closes:
            flush_run()
            out.append(f']{{custom-style="{UNVERIFIED_STYLE}"}}')
        if position == len(clean):
            break
        if position in opens:
            flush_run()
            out.append("[")
        if position in formulas:
            flush_run()
            match = formulas[position]
            out.append(match.group(0))
            literal.append(" ")
            found_formulas.append(match.group(1) if match.group(1) is not None else match.group(2))
            position = match.end()
            continue
        if position in markers:
            flush_run()
            match = markers[position]
            inner = escape_markdown(match.group(0)[1:-1], line_start=False)
            out.append(f'[&#91;{inner}&#93;]{{custom-style="{UNCERTAIN_STYLE}"}}')
            literal.append(match.group(0))
            position = match.end()
            continue
        if position in raw_atoms:
            flush_run()
            end = raw_atoms[position]
            out.append(clean[position:end])
            literal.append(clean[position:end].replace(_ESCAPED_PIPE, "|"))
            position = end
            continue
        if not run:
            run_start = position
        run.append(clean[position])
        position += 1
    flush_run()
    return _Inline("".join(out), "".join(literal), found_formulas)


def _cell_spans(line: str) -> list[tuple[int, int]]:
    """Content ranges of the cells of a pipe row, edge pipes excluded."""
    bounds = [m.start() for m in _UNESCAPED_PIPE_RE.finditer(line)]
    edges = [-1, *bounds, len(line)]
    spans = [(edges[i] + 1, edges[i + 1]) for i in range(len(edges) - 1)]
    if spans and line[: spans[0][1]].strip() == "":
        spans = spans[1:]
    if spans and line[spans[-1][0] :].strip() == "":
        spans = spans[:-1]
    return spans


def _clip(ranges: list[tuple[int, int]], start: int, end: int) -> list[tuple[int, int]]:
    return [
        (max(s, start) - start, min(e, end) - start) for s, e in ranges if s < end and e > start
    ]


def _render_pipe_table(text: str, open_hint: bool, ref: PageReference) -> tuple[str, bool]:
    rows: list[list[_Inline]] = []
    for raw_line in text.split("\n"):
        clean, ranges, open_hint = parse_hints(raw_line, open_hint)
        if is_table_separator(clean) or not clean.strip():
            continue
        cells: list[_Inline] = []
        for start, end in _cell_spans(clean):
            raw_cell = clean[start:end]
            lead = len(raw_cell) - len(raw_cell.lstrip())
            cell_text = raw_cell.strip()
            cell_ranges = [
                (s - lead, e - lead) for s, e in _clip(ranges, start, end) if e - lead > 0
            ]
            escaped_pipes = {
                m.start(): m.end() for m in re.finditer(re.escape(_ESCAPED_PIPE), cell_text)
            }
            cells.append(
                _render_inline(cell_text, cell_ranges, line_start=False, raw_atoms=escaped_pipes)
            )
        rows.append(cells)
    if not rows:
        return "", open_hint
    width = max(len(row) for row in rows) or 1
    empty = _Inline("", "", [])
    rows = [row + [empty] * (width - len(row)) for row in rows]
    lines = ["| " + " | ".join(cell.markdown for cell in row) + " |" for row in rows]
    lines.insert(1, "|" + "|".join(["---"] * width) + "|")
    first_empty = all(not cell.literal.strip() and not cell.formulas for cell in rows[0])
    ref.tables.append(TableRef([width] * len(rows), first_empty))
    for row in rows:
        ref.texts.append(" ".join(cell.literal for cell in row))
        for cell in row:
            ref.formulas.extend(cell.formulas)
    return "\n".join(lines), open_hint


def _wrap_html_text(segment: str, ranges: list[tuple[int, int]]) -> str:
    """Wrap markers and hint ranges of a text segment between HTML tags in styled spans."""
    markers = list(MARKER_RE.finditer(segment))
    atoms = sorted(
        [(m.start(), m.end()) for m in markers]
        + [(m.start(), m.end()) for m in _ENTITY_RE.finditer(segment)]
    )
    hint_ranges = _align_ranges(ranges, atoms, len(segment))
    marker_at = {m.start(): m.end() for m in markers}
    out: list[str] = []
    position = 0
    opens = {s for s, _ in hint_ranges}
    closes = {e for _, e in hint_ranges}
    while position <= len(segment):
        if position in closes:
            out.append("</span>")
        if position == len(segment):
            break
        if position in opens:
            out.append(f'<span data-custom-style="{UNVERIFIED_STYLE}">')
        if position in marker_at:
            end = marker_at[position]
            out.append(
                f'<span data-custom-style="{UNCERTAIN_STYLE}">{segment[position:end]}</span>'
            )
            position = end
            continue
        out.append(segment[position])
        position += 1
    return "".join(out)


def _render_html_table(text: str, open_hint: bool, ref: PageReference) -> tuple[str, bool]:
    clean, ranges, open_hint = parse_hints(text, open_hint)
    out: list[str] = []
    cursor = 0
    for tag in [*_TAG_RE.finditer(clean), None]:
        end = tag.start() if tag is not None else len(clean)
        segment = clean[cursor:end]
        out.append(_wrap_html_text(segment, _clip(ranges, cursor, end)))
        if tag is not None:
            out.append(tag.group(0))
            cursor = tag.end()
    html = "".join(out)
    first_row = _FIRST_ROW_RE.search(clean)
    first_empty = first_row is not None and not content_text(first_row.group(1)).strip()
    ref.tables.extend(TableRef(shape, first_empty) for shape in table_shape(clean))
    ref.texts.append(content_text(clean))
    longest = max((len(m.group(0)) for m in _BACKTICKS_RE.finditer(html)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}{{=html}}\n{html}\n{fence}", open_hint


def _render_formula(text: str, open_hint: bool, ref: PageReference) -> tuple[str, bool]:
    clean, ranges, open_hint = parse_hints(text, open_hint)
    for match in _FORMULA_RE.finditer(clean):
        ref.formulas.append(match.group(1) if match.group(1) is not None else match.group(2))
    if ranges:
        return f'[{clean}]{{custom-style="{UNVERIFIED_STYLE}"}}', open_hint
    return clean, open_hint


def render_blocks(blocks: list[Block]) -> tuple[list[str], PageReference]:
    """Render the blocks of one page.

    :return: Markdown blocks in order and the reference content of the page.
    """
    ref = PageReference()
    rendered: list[str] = []
    open_hint = False
    for block in blocks:
        if block.kind == BlockKind.PARAGRAPH:
            clean, ranges, open_hint = parse_hints(block.text, open_hint)
            if not clean.strip():
                continue
            inline = _render_inline(clean, ranges, line_start=True, raw_atoms={})
            rendered.append(inline.markdown)
            ref.texts.append(inline.literal)
            ref.formulas.extend(inline.formulas)
        elif block.kind == BlockKind.PIPE_TABLE:
            markdown, open_hint = _render_pipe_table(block.text, open_hint, ref)
            if markdown:
                rendered.append(markdown)
        elif block.kind == BlockKind.HTML_TABLE:
            markdown, open_hint = _render_html_table(block.text, open_hint, ref)
            rendered.append(markdown)
        else:
            markdown, open_hint = _render_formula(block.text, open_hint, ref)
            rendered.append(markdown)
    return rendered, ref


def page_link(pdf_name: str, page_index: int) -> str:
    """Relative link to a page of the PDF saved next to the DOCX."""
    return f"{quote(pdf_name, safe='')}#page={page_index + 1}"


def _fenced_div(style: str, content: str) -> str:
    return f'::: {{custom-style="{style}"}}\n{content}\n:::'


def to_pandoc_markdown(
    title: str, pages: list[MdPage], pdf_name: str, *, draft: bool = False
) -> tuple[str, list[PageReference]]:
    """Build the pandoc Markdown of a document.

    :param title: document title, the first heading.
    :param pages: pages in order.
    :param pdf_name: file name of the PDF the "Стр. N" links point to.
    :param draft: start with the draft banner (section 9.6).
    :return: Markdown text and the reference content of every page.
    """
    parts: list[str] = []
    if draft:
        parts.append(_fenced_div(DRAFT_BANNER_STYLE, escape_markdown(DRAFT_BANNER_TEXT)))
    parts.append("# " + escape_markdown(title, line_start=False))
    references: list[PageReference] = []
    for page in pages:
        label = f"Стр. {page.page_index + 1}"
        line = f"[{label}]({page_link(pdf_name, page.page_index)})"
        if page.printed_label:
            line += " (" + escape_markdown(page.printed_label, line_start=False) + ")"
        if page.note:
            line += escape_markdown(page.note, line_start=False)
        parts.append(_fenced_div(SOURCE_PAGE_STYLE, line))
        rendered, reference = render_blocks(page.blocks)
        parts.extend(rendered)
        references.append(reference)
    return "\n\n".join(parts) + "\n", references
