"""Chunks for the ingester (section 9.4)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ocr_archive.build.chunks import ChunkPage, build_chunks, chunk_id, write_chunks
from ocr_archive.build.normalize import Block, BlockKind

pytestmark = pytest.mark.p4


def para(text: str) -> Block:
    return Block(BlockKind.PARAGRAPH, text)


def chunks_of(pages: list[ChunkPage], max_chars: int = 100) -> list[dict[str, object]]:
    return build_chunks(
        document_id="01DOC", title="Отчёт", build_version=3, pages=pages, max_chars=max_chars
    )


def test_chunks_stay_inside_a_page_and_respect_the_limit() -> None:
    pages = [
        ChunkPage(0, "9 об.", [para("а" * 40), para("б" * 40), para("в" * 40)], reviewed=True),
        ChunkPage(1, None, [para("короткий [неразб.] текст")], reviewed=False),
    ]
    chunks = chunks_of(pages)
    assert [c["content"] for c in chunks] == [
        "а" * 40 + "\n\n" + "б" * 40,
        "в" * 40,
        "короткий [неразб.] текст",
    ]
    assert all(len(str(c["content"])) <= 100 for c in chunks)
    assert [c["page_number"] for c in chunks] == [1, 1, 2]
    assert chunks[0]["printed_label"] == "9 об."
    assert [c["review_status"] for c in chunks] == ["reviewed", "reviewed", "auto"]
    assert chunks[0]["build_version"] == 3


def test_long_paragraph_is_split_at_spaces() -> None:
    words = " ".join(["слово"] * 50)
    chunks = chunks_of([ChunkPage(0, None, [para(words)], reviewed=False)], max_chars=60)
    assert all(len(str(c["content"])) <= 60 for c in chunks)
    assert " ".join(str(c["content"]) for c in chunks) == words


def test_table_is_a_chunk_of_its_own_and_long_tables_repeat_the_header() -> None:
    rows = "\n".join(f"| {i} | значение {i} |" for i in range(20))
    table = Block(BlockKind.PIPE_TABLE, "| № | Значение |\n|---|---|\n" + rows)
    pages = [ChunkPage(0, None, [para("До"), table, para("После")], reviewed=False)]
    chunks = chunks_of(pages, max_chars=120)
    contents = [str(c["content"]) for c in chunks]
    assert contents[0] == "До"
    assert contents[-1] == "После"
    table_chunks = contents[1:-1]
    assert len(table_chunks) > 1
    for content in table_chunks:
        assert content.startswith("| № | Значение |\n|---|---|\n")
    body = [line for c in table_chunks for line in c.split("\n")[2:]]
    assert body == rows.split("\n")


def test_long_html_table_is_split_by_rows() -> None:
    rows = "".join(f"<tr><td>{i}</td><td>строка {i}</td></tr>" for i in range(30))
    table = Block(BlockKind.HTML_TABLE, f"<table><tr><th>№</th><th>Текст</th></tr>{rows}</table>")
    chunks = chunks_of([ChunkPage(0, None, [table], reviewed=False)], max_chars=200)
    assert len(chunks) > 1
    for chunk in chunks:
        content = str(chunk["content"])
        assert content.startswith("<table><tr><th>№</th><th>Текст</th></tr>")
        assert content.endswith("</table>")


def test_chunk_id_is_stable_and_independent_of_the_text() -> None:
    first = chunks_of([ChunkPage(0, None, [para("версия один")], reviewed=False)])
    second = chunks_of([ChunkPage(0, None, [para("исправленная версия")], reviewed=True)])
    assert first[0]["chunk_id"] == second[0]["chunk_id"] == chunk_id("01DOC", 0, 0)
    assert chunk_id("01DOC", 0, 1) != chunk_id("01DOC", 1, 0)


def test_page_without_text_has_no_chunks(tmp_path: Path) -> None:
    chunks = chunks_of([ChunkPage(0, None, [], reviewed=False)])
    assert chunks == []
    path = tmp_path / "chunks.jsonl"
    write_chunks(path, chunks_of([ChunkPage(0, None, [para("текст")], reviewed=False)]))
    lines = path.read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0])["content"] == "текст"
