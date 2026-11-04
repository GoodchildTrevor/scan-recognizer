"""Normalization of page texts for DOCX and chunks (section 9.1)."""

from __future__ import annotations

import pytest

from ocr_archive.build.markers import MARKER_RE, find_markers
from ocr_archive.build.normalize import (
    HINT_CLOSE,
    HINT_OPEN,
    Block,
    BlockKind,
    document_vocabulary,
    normalize_page,
)

pytestmark = pytest.mark.p4


def test_compound_word_keeps_its_hyphen() -> None:
    text = "Развитие социально-\nэкономический процесс."
    blocks = normalize_page(text, document_vocabulary([text]))
    assert blocks == [Block(BlockKind.PARAGRAPH, "Развитие социально-экономический процесс.")]


def test_word_break_resolved_by_document_vocabulary() -> None:
    page = "Стоит пере-\nнос в строке."
    other = "Другой перенос без разрыва."
    vocabulary = document_vocabulary([page, other])
    assert normalize_page(page, vocabulary)[0].text == "Стоит перенос в строке."
    # Without the joined word in the vocabulary the hyphen stays, parts joined without a space.
    assert normalize_page(page, document_vocabulary([page]))[0].text == "Стоит пере-нос в строке."


def test_vocabulary_skips_parts_torn_by_a_break() -> None:
    vocabulary = document_vocabulary(["Слово пере-\nнос и ещё"])
    assert "пере" not in vocabulary
    assert "нос" not in vocabulary
    assert {"слово", "и", "ещё"} <= vocabulary


def test_paragraphs_are_split_by_empty_lines_and_lines_joined() -> None:
    text = "Первая строка\nвторая строка\n\n\nНовый абзац\n  с отступом  "
    blocks = normalize_page(text, set())
    assert [b.text for b in blocks] == ["Первая строка вторая строка", "Новый абзац с отступом"]
    assert all(b.kind == BlockKind.PARAGRAPH for b in blocks)


def test_tables_and_display_formulas_are_kept_unchanged() -> None:
    pipe = "| Год | Сумма |\n|---|---|\n| 1983 | 12 345 |"
    html = '<table>\n<tr><td colspan="2">Итого</td></tr>\n</table>'
    formula = "\\[\n x^2 + y^2\n\\]"
    text = f"До таблицы\n{pipe}\nпосле\n\n{html}\n\n{formula}\nхвост"
    blocks = normalize_page(text, set())
    assert blocks == [
        Block(BlockKind.PARAGRAPH, "До таблицы"),
        Block(BlockKind.PIPE_TABLE, pipe),
        Block(BlockKind.PARAGRAPH, "после"),
        Block(BlockKind.HTML_TABLE, html),
        Block(BlockKind.FORMULA, formula),
        Block(BlockKind.PARAGRAPH, "хвост"),
    ]


def test_markers_are_kept() -> None:
    text = "Подпись [подпись] и [неразб.: Иванов?]\nконец [рук.: неразб.]"
    blocks = normalize_page(text, set())
    assert [m.inner for m in find_markers(blocks[0].text)] == [
        "подпись",
        "неразб.: Иванов?",
        "рук.: неразб.",
    ]


def test_hint_marks_survive_a_word_break() -> None:
    page = f"Текст {HINT_OPEN}пере-\nнос{HINT_CLOSE} дальше"
    blocks = normalize_page(page, {"перенос"})
    assert blocks[0].text == f"Текст {HINT_OPEN}перенос{HINT_CLOSE} дальше"
    page = f"Текст {HINT_OPEN}пере-{HINT_CLOSE}\nнос дальше"
    assert (
        normalize_page(page, {"перенос"})[0].text == f"Текст {HINT_OPEN}пере{HINT_CLOSE}нос дальше"
    )


def test_hint_marks_do_not_hide_table_rows() -> None:
    text = f"{HINT_OPEN}| a | b |\n|---|---|\n| 1 | 2 |{HINT_CLOSE}"
    blocks = normalize_page(text, set())
    assert [b.kind for b in blocks] == [BlockKind.PIPE_TABLE]


def test_marker_regex_contract() -> None:
    found = [
        m.group(0) for m in MARKER_RE.finditer("[неразб.] [рук.: текст] [таблица: проверить] [x]")
    ]
    assert found == ["[неразб.]", "[рук.: текст]", "[таблица: проверить]"]
