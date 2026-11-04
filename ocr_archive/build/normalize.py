"""Normalization of the verbatim page text into blocks for DOCX and chunks (section 9.1).

The verbatim text stays canonical in the database; this module only prepares a reading form:
paragraphs joined from lines, word breaks resolved by the document vocabulary, tables and display
formulas kept as they are.

Draft builds mark hint ranges with the private-use characters :data:`HINT_OPEN` and
:data:`HINT_CLOSE` before normalization (section 9.6); every step here keeps them in place and
never uses them to decide the structure of the text.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

from ocr_archive.textnorm import is_pipe_row

HINT_OPEN = ""
HINT_CLOSE = ""
HINT_CHARS = HINT_OPEN + HINT_CLOSE

_HINT_RE = re.compile(f"[{HINT_CHARS}]")
_WORD_RE = re.compile(r"[^\W\d_]+")
# A line ending with "letters + hyphen", hint marks allowed around the hyphen.
_BREAK_END_RE = re.compile(
    rf"((?:[^\W\d_]|[{HINT_CHARS}])*[^\W\d_][{HINT_CHARS}]*)-([{HINT_CHARS}]*)$"
)
_BREAK_START_RE = re.compile(rf"^([{HINT_CHARS}]*)([^\W\d_](?:[^\W\d_]|[{HINT_CHARS}])*)")
_HTML_TABLE_RE = re.compile(r"<table\b.*?</table\s*>", re.IGNORECASE | re.DOTALL)
_DISPLAY_FORMULA_RE = re.compile(r"\\\[.*?\\\]", re.DOTALL)


class BlockKind(StrEnum):
    PARAGRAPH = "paragraph"
    PIPE_TABLE = "pipe_table"
    HTML_TABLE = "html_table"
    FORMULA = "formula"


@dataclass(frozen=True)
class Block:
    kind: BlockKind
    text: str  # paragraph text, table lines joined by "\n", or the display formula as written


def strip_hints(text: str) -> str:
    """Remove the draft hint marks."""
    return _HINT_RE.sub("", text)


def _is_break(line: str, next_line: str) -> bool:
    stripped = line.rstrip()
    return (
        len(stripped) >= 2
        and stripped.endswith("-")
        and _WORD_RE.fullmatch(stripped[-2]) is not None
        and _WORD_RE.match(next_line.lstrip()) is not None
    )


def _vocabulary_words(text: str) -> Iterable[str]:
    lines = strip_hints(text).split("\n")
    breaks = [_is_break(lines[i], lines[i + 1]) for i in range(len(lines) - 1)] + [False]
    for number, line in enumerate(lines):
        words = _WORD_RE.findall(line)
        # Parts torn by a word break are not words: skip the left part of a break and the
        # right part at the start of the next line.
        if words and breaks[number]:
            words = words[:-1]
        if words and number > 0 and breaks[number - 1]:
            words = words[1:]
        yield from (word.lower() for word in words)


def document_vocabulary(texts: Iterable[str]) -> set[str]:
    """Lower-case words of all page texts, without parts torn by word breaks.

    :param texts: verbatim texts of the pages of one document.
    """
    vocabulary: set[str] = set()
    for text in texts:
        vocabulary.update(_vocabulary_words(text))
    return vocabulary


def _join_lines(lines: list[str], vocabulary: set[str]) -> str:
    """Join paragraph lines with spaces, resolving word breaks by the document vocabulary."""
    result = ""
    for raw in lines:
        line = raw.strip()
        if not result:
            result = line
            continue
        end = _BREAK_END_RE.search(result)
        start = _BREAK_START_RE.match(line)
        if end is None or start is None:
            result = f"{result} {line}"
            continue
        left = strip_hints(end.group(1))
        right = strip_hints(start.group(2))
        hyphen = "" if (left + right).lower() in vocabulary else "-"
        result = result[: end.start()] + end.group(1) + hyphen + end.group(2) + line
    return result


def _protected_spans(text: str) -> list[tuple[int, int, BlockKind]]:
    """HTML tables and display formulas, found on the text without hint marks and mapped back."""
    clean_chars: list[str] = []
    index_map: list[int] = []
    for position, char in enumerate(text):
        if char in HINT_CHARS:
            continue
        clean_chars.append(char)
        index_map.append(position)
    clean = "".join(clean_chars)
    found: list[tuple[int, int, BlockKind]] = []
    for regex, kind in (
        (_HTML_TABLE_RE, BlockKind.HTML_TABLE),
        (_DISPLAY_FORMULA_RE, BlockKind.FORMULA),
    ):
        for match in regex.finditer(clean):
            start = index_map[match.start()]
            end = index_map[match.end() - 1] + 1
            while start > 0 and text[start - 1] in HINT_CHARS:
                start -= 1
            while end < len(text) and text[end] in HINT_CHARS:
                end += 1
            found.append((start, end, kind))
    found.sort()
    spans: list[tuple[int, int, BlockKind]] = []
    for span in found:
        # A formula inside an HTML table belongs to the table.
        if spans and span[0] < spans[-1][1]:
            continue
        spans.append(span)
    return spans


def _flow_blocks(text: str, vocabulary: set[str]) -> list[Block]:
    blocks: list[Block] = []
    paragraph: list[str] = []
    table: list[str] = []

    def flush_paragraph() -> None:
        if paragraph:
            joined = _join_lines(paragraph, vocabulary)
            if joined.strip():
                blocks.append(Block(BlockKind.PARAGRAPH, joined))
            paragraph.clear()

    def flush_table() -> None:
        if table:
            blocks.append(Block(BlockKind.PIPE_TABLE, "\n".join(line.strip() for line in table)))
            table.clear()

    for line in text.split("\n"):
        clean = strip_hints(line)
        if is_pipe_row(clean):
            flush_paragraph()
            table.append(line)
        elif not clean.strip():
            flush_paragraph()
            flush_table()
            if line.strip():
                # A line of hint marks only: keep the marks so that ranges stay balanced.
                blocks.append(Block(BlockKind.PARAGRAPH, line.strip()))
        else:
            flush_table()
            paragraph.append(line)
    flush_paragraph()
    flush_table()
    return blocks


def normalize_page(text: str, vocabulary: set[str]) -> list[Block]:
    """Split a verbatim page text into blocks (section 9.1).

    Paragraphs are separated by empty lines; lines inside a paragraph are joined with a space,
    a word break ("letter + -" at the end of a line, a letter at the start of the next one)
    loses its hyphen only if the joined word occurs in the document vocabulary. Pipe tables,
    HTML tables and display formulas are kept unchanged. Markers stay everywhere.

    :param text: verbatim page text, possibly with draft hint marks.
    :param vocabulary: :func:`document_vocabulary` of the document.
    """
    blocks: list[Block] = []
    cursor = 0
    for start, end, kind in _protected_spans(text):
        blocks.extend(_flow_blocks(text[cursor:start], vocabulary))
        blocks.append(Block(kind, text[start:end].strip()))
        cursor = end
    blocks.extend(_flow_blocks(text[cursor:], vocabulary))
    return blocks
