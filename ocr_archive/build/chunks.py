"""Chunks for the ingester (section 9.4): one JSON object per line, never crossing a page."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ocr_archive.build.normalize import Block, BlockKind, strip_hints
from ocr_archive.textnorm import is_table_separator

_HTML_ROW_RE = re.compile(r"<tr\b.*?</tr\s*>", re.IGNORECASE | re.DOTALL)
_WS_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class ChunkPage:
    page_index: int
    printed_label: str | None
    blocks: list[Block]
    reviewed: bool


def chunk_id(document_id: str, page_index: int, chunk_no: int) -> str:
    """Stable chunk id, independent of the revision: sha1 of ``document_id:page_index:chunk_no``."""
    return hashlib.sha1(f"{document_id}:{page_index}:{chunk_no}".encode()).hexdigest()


def _split_long(text: str, max_chars: int) -> list[str]:
    """Split a paragraph longer than the limit at spaces, or hard when a word is too long."""
    pieces: list[str] = []
    current = ""
    for word in text.split(" "):
        while len(word) > max_chars:
            if current:
                pieces.append(current)
                current = ""
            pieces.append(word[:max_chars])
            word = word[max_chars:]
        candidate = f"{current} {word}" if current else word
        if len(candidate) > max_chars:
            pieces.append(current)
            current = word
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces


def _rows_chunks(
    header: list[str], rows: list[str], max_chars: int, wrap: tuple[str, str]
) -> list[str]:
    prefix, suffix = wrap
    chunks: list[str] = []
    current: list[str] = []
    for row in rows:
        candidate = prefix + "\n".join([*header, *current, row]) + suffix
        if current and len(candidate) > max_chars:
            chunks.append(prefix + "\n".join([*header, *current]) + suffix)
            current = [row]
        else:
            current.append(row)
    if current or not chunks:
        chunks.append(prefix + "\n".join([*header, *current]) + suffix)
    return chunks


def split_table(block: Block, max_chars: int) -> list[str]:
    """Split a table into chunks by rows, repeating the header row in every chunk."""
    text = strip_hints(block.text)
    if len(text) <= max_chars:
        return [text]
    if block.kind == BlockKind.HTML_TABLE:
        rows = [m.group(0) for m in _HTML_ROW_RE.finditer(text)]
        if len(rows) < 2:
            return [text]
        return _rows_chunks([rows[0]], rows[1:], max_chars, ("<table>", "</table>"))
    lines = text.split("\n")
    header = lines[:1]
    body = lines[1:]
    if body and is_table_separator(body[0]):
        header.append(body.pop(0))
    if not body:
        return [text]
    return _rows_chunks(header, body, max_chars, ("", ""))


def _page_chunk_texts(blocks: list[Block], max_chars: int) -> list[str]:
    texts: list[str] = []
    current = ""
    for block in blocks:
        if block.kind in (BlockKind.PIPE_TABLE, BlockKind.HTML_TABLE):
            if current:
                texts.append(current)
                current = ""
            texts.extend(split_table(block, max_chars))
            continue
        paragraph = _WS_RE.sub(" ", strip_hints(block.text)).strip()
        if not paragraph:
            continue
        for piece in (
            _split_long(paragraph, max_chars) if len(paragraph) > max_chars else [paragraph]
        ):
            candidate = f"{current}\n\n{piece}" if current else piece
            if len(candidate) > max_chars:
                texts.append(current)
                current = piece
            else:
                current = candidate
    if current:
        texts.append(current)
    return texts


def build_chunks(
    *,
    document_id: str,
    title: str,
    build_version: int,
    pages: list[ChunkPage],
    max_chars: int,
) -> list[dict[str, Any]]:
    """Cut the normalized text into chunks (section 9.4).

    A chunk stays inside one page, is cut at paragraphs and holds at most ``max_chars``
    characters; a table is a chunk of its own, a long table is cut by rows with the header
    repeated. Markers stay in the content.
    """
    chunks: list[dict[str, Any]] = []
    for page in pages:
        for number, content in enumerate(_page_chunk_texts(page.blocks, max_chars)):
            chunks.append(
                {
                    "chunk_id": chunk_id(document_id, page.page_index, number),
                    "content": content,
                    "document_id": document_id,
                    "title": title,
                    "page_index": page.page_index,
                    "page_number": page.page_index + 1,
                    "printed_label": page.printed_label,
                    "build_version": build_version,
                    "review_status": "reviewed" if page.reviewed else "auto",
                }
            )
    return chunks


def write_chunks(path: Path, chunks: list[dict[str, Any]]) -> None:
    """Write chunks as JSON Lines in UTF-8."""
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")
