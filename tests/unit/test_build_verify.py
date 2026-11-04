"""DOCX verification against the reference content (section 9.5), on hand-made pandoc ASTs.

The ASTs follow the pandoc JSON format that ``pandoc -f docx+styles -t json`` produces, so the
comparison logic is tested without pandoc; the round trip through pandoc is covered by the
integration tests that run when pandoc is installed.
"""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import pytest

from ocr_archive.build.docx import set_core_properties
from ocr_archive.build.markdown import MdPage, PageReference, TableRef, to_pandoc_markdown
from ocr_archive.build.normalize import Block, BlockKind
from ocr_archive.build.verify import compare_page, docx_pages, verify_docx
from ocr_archive.exceptions import BuildError
from ocr_archive.settings import PACKAGE_DIR

pytestmark = pytest.mark.p4

REFERENCE_DOCX = PACKAGE_DIR / "assets" / "reference.docx"
W_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def inlines(text: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for number, word in enumerate(text.split(" ")):
        if number:
            out.append({"t": "Space"})
        if word:
            out.append({"t": "Str", "c": word})
    return out


def para(text: str, *extra: dict[str, Any]) -> dict[str, Any]:
    return {"t": "Para", "c": inlines(text) + list(extra)}


def math(tex: str) -> dict[str, Any]:
    return {"t": "Math", "c": [{"t": "InlineMath"}, tex]}


def div(style: str, blocks: list[dict[str, Any]]) -> dict[str, Any]:
    return {"t": "Div", "c": [["", [], [["custom-style", style]]], blocks]}


def table(rows: list[list[str]]) -> dict[str, Any]:
    def cell(text: str) -> list[Any]:
        return [["", [], []], {"t": "AlignDefault"}, 1, 1, [{"t": "Plain", "c": inlines(text)}]]

    def row(cells: list[str]) -> list[Any]:
        return [["", [], []], [cell(c) for c in cells]]

    head = [["", [], []], [row(rows[0])]]
    body = [["", [], []], 0, [], [row(r) for r in rows[1:]]]
    return {
        "t": "Table",
        "c": [["", [], []], [None, []], [], head, [body], [["", [], []], []]],
    }


def page_header(number: int) -> dict[str, Any]:
    link = {
        "t": "Link",
        "c": [["", [], []], inlines(f"Стр. {number}"), [f"x.pdf#page={number}", ""]],
    }
    return div("Source Page", [{"t": "Para", "c": [link]}])


def ast(blocks: list[dict[str, Any]]) -> dict[str, Any]:
    return {"pandoc-api-version": [1, 23, 1], "meta": {}, "blocks": blocks}


def reference(
    *texts: str, formulas: list[str] | None = None, tables: list[TableRef] | None = None
) -> PageReference:
    return PageReference(list(texts), formulas or [], tables or [])


def test_pages_split_by_source_page_divs_not_by_text() -> None:
    doc = ast(
        [
            div("Draft Banner", [para("ЧЕРНОВИК — не проверено")]),
            div("Heading 1", [{"t": "Header", "c": [1, ["", [], []], inlines("Отчёт")]}]),
            page_header(1),
            div("First Paragraph", [para("Стр. 5 колонтитул")]),
            div("Body Text", [para("Текст 12")]),
            # Two empty pages in a row end up in one div.
            div("Source Page", [para("Стр. 2"), para("Стр. 3")]),
            page_header(4),
            table([["Год", "Сумма"], ["1983", "12 345"]]),
        ]
    )
    pages = docx_pages(doc)
    assert len(pages) == 4
    assert pages[0].texts == ["Стр. 5 колонтитул", "Текст 12"]
    assert pages[1].texts == [] and pages[2].texts == []
    assert pages[3].texts == ["Год Сумма", "1983 12 345"]
    assert [t.shape for t in pages[3].tables] == [[2, 2]]


def test_identical_page_passes() -> None:
    ref = reference(
        "Расход 12 345 руб, t = −5, рост 15 %.",
        "Год Сумма",
        "1983 3,5",
        formulas=["x^{2} + 10"],
        tables=[TableRef([2, 2], False)],
    )
    page = docx_pages(
        ast(
            [
                page_header(1),
                para("Расход 12 345 руб, t = −5, рост 15 %."),
                table([["Год", "Сумма"], ["1983", "3,5"]]),
            ]
        )
    )[0]
    page.formulas.append("x^2 + 10")
    assert compare_page(1, ref, page, 0.99) == []


@pytest.mark.parametrize(
    ("docx_text", "problem"),
    [
        ("Расход 12 34 руб", "numbers differ"),
        ("Расход 12 345 рублей", "words differ"),
        ("Расход 12 345", "words differ"),
    ],
)
def test_lost_number_or_word_is_caught(docx_text: str, problem: str) -> None:
    ref = reference("Расход 12 345 руб")
    page = docx_pages(ast([page_header(1), para(docx_text)]))[0]
    problems = compare_page(1, ref, page, 0.99)
    assert any(problem in p for p in problems)


def test_removed_table_cell_is_caught() -> None:
    ref = reference("a b", "1 2", tables=[TableRef([2, 2], False)])
    page = docx_pages(ast([page_header(1), table([["a", "b"], ["1"]])]))[0]
    page.texts[-1] = "1 2"
    assert any("tables differ" in p for p in compare_page(1, ref, page, 0.99))


def test_empty_first_row_may_be_dropped_by_pandoc() -> None:
    ref = reference(" ", "1 2", tables=[TableRef([2, 2], True)])
    page = docx_pages(ast([page_header(1), table([["1", "2"]])]))[0]
    page.tables[0] = TableRef([2], False)
    assert compare_page(1, ref, page, 0.0) == []


def test_swapped_paragraphs_fail_the_order_check() -> None:
    sentence = (
        "отчёт о работе лаборатории за истекший год представлен комиссии при этом отмечено "
        "что план выполнен однако ряд показателей требует уточнения в частности расход "
        "материалов и сроки поставки оборудования заводу необходимо обеспечить подготовку"
    )
    words = sentence.split(" ")
    paragraphs = [" ".join(words[i * 4 : i * 4 + 4]) for i in range(8)]
    ref = reference(*paragraphs)
    swapped = paragraphs[:]
    swapped[2], swapped[5] = swapped[5], swapped[2]
    page = docx_pages(ast([page_header(1), *[para(p) for p in swapped]]))[0]
    problems = compare_page(1, ref, page, 0.99)
    assert problems and all("similarity" in p for p in problems)


def test_formulas_are_excluded_from_words_but_their_digits_count() -> None:
    ref = reference("Если   то", formulas=["x_{1} = 2"])
    page = docx_pages(ast([page_header(1), para("Если ", math("x_1 = 2"), *inlines(" то"))]))[0]
    assert page.formulas == ["x_1 = 2"]
    assert compare_page(1, ref, page, 0.99) == []
    page.formulas[0] = "x_1 = 3"
    assert any("formula digits" in p for p in compare_page(1, ref, page, 0.99))


def test_formula_written_back_as_tex_text_is_absorbed() -> None:
    ref = reference("Если   то", formulas=["\\frac{a}{b"])
    page = docx_pages(ast([page_header(1), para("Если $\\frac{a}{b$ то")]))[0]
    assert compare_page(1, ref, page, 0.9) == []


def test_literal_angle_brackets_and_ampersands_are_compared_as_text() -> None:
    ref = reference("при t < 5 и p > 3, A&B &amp; 7")
    page = docx_pages(ast([page_header(1), para("при t < 5 и p > 3, A&B &amp; 7")]))[0]
    assert compare_page(1, ref, page, 0.99) == []


def test_verify_docx_checks_page_count() -> None:
    with pytest.raises(BuildError, match="2 pages, expected 1"):
        verify_docx(ast([page_header(1), page_header(2)]), [reference()], 0.99)


def test_markdown_reference_matches_its_own_docx_shape() -> None:
    blocks = [
        Block(BlockKind.PARAGRAPH, "Текст [неразб.] 12 345"),
        Block(BlockKind.PIPE_TABLE, "| a | b |\n|---|---|\n| 1 | 2 |"),
    ]
    _, refs = to_pandoc_markdown("T", [MdPage(0, None, blocks)], "t.pdf")
    page = docx_pages(
        ast([page_header(1), para("Текст [неразб.] 12 345"), table([["a", "b"], ["1", "2"]])])
    )[0]
    assert compare_page(1, refs[0], page, 0.99) == []


def test_reference_docx_defines_the_archive_styles() -> None:
    with zipfile.ZipFile(REFERENCE_DOCX) as archive:
        styles = ElementTree.fromstring(archive.read("word/styles.xml"))
    names = {
        (style.get(f"{W_NS}type"), style.find(f"{W_NS}name").get(f"{W_NS}val"))  # type: ignore[union-attr]
        for style in styles.iter(f"{W_NS}style")
    }
    assert {
        ("paragraph", "Source Page"),
        ("character", "Uncertain"),
        ("character", "Unverified"),
        ("paragraph", "Draft Banner"),
        ("paragraph", "Body Text"),
        ("table", "Table"),
    } <= names


def test_core_properties_are_set_and_other_parts_kept(tmp_path: Path) -> None:
    docx = tmp_path / "draft.docx"
    shutil.copyfile(REFERENCE_DOCX, docx)
    set_core_properties(docx, title="ЧЕРНОВИК — Отчёт", subject="не проверено")
    with zipfile.ZipFile(docx) as archive, zipfile.ZipFile(REFERENCE_DOCX) as original:
        assert sorted(archive.namelist()) == sorted(original.namelist())
        core = ElementTree.fromstring(archive.read("docProps/core.xml"))
    dc = "{http://purl.org/dc/elements/1.1/}"
    assert core.find(f"{dc}title").text == "ЧЕРНОВИК — Отчёт"  # type: ignore[union-attr]
    assert core.find(f"{dc}subject").text == "не проверено"  # type: ignore[union-attr]
