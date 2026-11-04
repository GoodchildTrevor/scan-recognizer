"""Pandoc Markdown for the DOCX build (section 9.2) and draft marking (section 9.6)."""

from __future__ import annotations

import pytest

from ocr_archive.build.markdown import (
    DRAFT_BANNER_TEXT,
    MdPage,
    TableRef,
    escape_markdown,
    render_blocks,
    to_pandoc_markdown,
)
from ocr_archive.build.normalize import HINT_CLOSE, HINT_OPEN, Block, BlockKind, normalize_page

pytestmark = pytest.mark.p4


def para(text: str) -> Block:
    return Block(BlockKind.PARAGRAPH, text)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Цена $5 и $10", "Цена $5 и $10"),
        ("a | b", "a \\| b"),
        ("см. [1]", "см. &#91;1&#93;"),
        ("путь C:\\dir", "путь C:\\\\dir"),
        ("1. Пункт", "1\\. Пункт"),
        ("12) Пункт", "12&#41; Пункт"),
        ("IV. Раздел", "IV\\. Раздел"),
        ("- тире в начале", "\\- тире в начале"),
        ("+ плюс", "\\+ плюс"),
        ("::: div", "\\::: div"),
        ("Table: подпись", "Table\\: подпись"),
        ("*жирный* _курсив_", "\\*жирный\\* \\_курсив\\_"),
        ("<b>тег</b> & @cite ~x^", "\\<b\\>тег\\</b\\> \\& \\@cite \\~x\\^"),
        ("Текст 1. не пункт", "Текст 1. не пункт"),
        ("`код` # заголовок", "\\`код\\` \\# заголовок"),
    ],
)
def test_escape_markdown(text: str, expected: str) -> None:
    assert escape_markdown(text) == expected


def test_list_marker_escape_only_at_paragraph_start() -> None:
    assert escape_markdown("1. Пункт", line_start=False) == "1. Пункт"


def test_formulas_pass_through_and_are_collected() -> None:
    rendered, ref = render_blocks([para("Если \\(x_1 = [a]\\) то *b*")])
    assert rendered == ["Если \\(x_1 = [a]\\) то \\*b\\*"]
    assert ref.formulas == ["x_1 = [a]"]
    assert ref.texts == ["Если   то *b*"]


def test_markers_get_the_uncertain_style_with_entity_brackets() -> None:
    rendered, ref = render_blocks([para("Подпись [неразб.: Иванов?] здесь")])
    assert rendered == ['Подпись [&#91;неразб.: Иванов?&#93;]{custom-style="Uncertain"} здесь']
    assert ref.texts == ["Подпись [неразб.: Иванов?] здесь"]


def test_pipe_table_is_regularized_and_cells_escaped() -> None:
    table = "| Год | Сумма |\n| 1983 | 12 345 \\| 6 |\n| только |"
    rendered, ref = render_blocks([Block(BlockKind.PIPE_TABLE, table)])
    assert rendered == ["| Год | Сумма |\n|---|---|\n| 1983 | 12 345 \\| 6 |\n| только |  |"]
    assert ref.tables == [TableRef([2, 2, 2], first_row_empty=False)]
    assert ref.texts == ["Год Сумма", "1983 12 345 | 6", "только "]


def test_pipe_table_marker_and_formula_in_cells() -> None:
    table = "| [неразб.] | \\(x^2\\) |\n|---|---|\n| a | b |"
    rendered, ref = render_blocks([Block(BlockKind.PIPE_TABLE, table)])
    assert '[&#91;неразб.&#93;]{custom-style="Uncertain"}' in rendered[0]
    assert "\\(x^2\\)" in rendered[0]
    assert ref.formulas == ["x^2"]


def test_html_table_is_one_raw_block_with_styled_marker() -> None:
    html = (
        '<table><tr><td colspan="2">[неразб.] &amp; 5</td></tr>'
        "<tr><td>1</td><td>2</td></tr></table>"
    )
    rendered, ref = render_blocks([Block(BlockKind.HTML_TABLE, html)])
    assert rendered[0].startswith("```{=html}\n<table>")
    assert rendered[0].endswith("</table>\n```")
    assert '<span data-custom-style="Uncertain">[неразб.]</span>' in rendered[0]
    assert ref.tables == [TableRef([2, 2], first_row_empty=False)]
    assert "& 5" in ref.texts[0]


def test_html_table_hint_gets_the_unverified_span_between_tags() -> None:
    html = f"<table><tr><td>{HINT_OPEN}12</td><td>34{HINT_CLOSE}</td></tr></table>"
    rendered, _ = render_blocks([Block(BlockKind.HTML_TABLE, html)])
    assert (
        '<td><span data-custom-style="Unverified">12</span></td>'
        '<td><span data-custom-style="Unverified">34</span></td>' in rendered[0]
    )


def test_hint_wraps_text_and_is_expanded_over_a_cut_marker() -> None:
    text = f"Слово {HINT_OPEN}[неразб.:{HINT_CLOSE} вариант?] конец"
    rendered, ref = render_blocks([para(text)])
    assert rendered == [
        'Слово [[&#91;неразб.: вариант?&#93;]{custom-style="Uncertain"}]'
        '{custom-style="Unverified"} конец'
    ]
    assert ref.texts == ["Слово [неразб.: вариант?] конец"]


def test_hint_open_across_paragraphs_is_closed_and_reopened() -> None:
    blocks = [para(f"начало {HINT_OPEN}один"), para(f"два{HINT_CLOSE} конец")]
    rendered, _ = render_blocks(blocks)
    assert rendered == [
        'начало [один]{custom-style="Unverified"}',
        '[два]{custom-style="Unverified"} конец',
    ]


def test_draft_hints_survive_normalization_and_word_breaks() -> None:
    text = f"Первая {HINT_OPEN}пере-\nнос{HINT_CLOSE} и\n\n1. пункт"
    blocks = normalize_page(text, {"перенос"})
    rendered, ref = render_blocks(blocks)
    assert rendered == ['Первая [перенос]{custom-style="Unverified"} и', "1\\. пункт"]
    assert ref.texts == ["Первая перенос и", "1. пункт"]


def test_document_markdown_has_banner_title_and_page_headers() -> None:
    pages = [
        MdPage(0, "9 об.", [para("Текст")], note=" · не проверено"),
        MdPage(1, None, [para("Стр. 5 колонтитул")]),
    ]
    markdown, refs = to_pandoc_markdown("Отчёт [1983]", pages, "Отчёт 1983 (копия).pdf", draft=True)
    parts = markdown.rstrip("\n").split("\n\n")
    assert parts[0] == f'::: {{custom-style="Draft Banner"}}\n{DRAFT_BANNER_TEXT}\n:::'
    assert parts[1] == "# Отчёт &#91;1983&#93;"
    assert parts[2] == (
        '::: {custom-style="Source Page"}\n'
        "[Стр. 1](%D0%9E%D1%82%D1%87%D1%91%D1%82%201983%20%28%D0%BA%D0%BE%D0%BF%D0%B8%D1%8F%29.pdf"
        "#page=1) (9 об.) · не проверено\n:::"
    )
    assert parts[3] == "Текст"
    assert parts[4].startswith('::: {custom-style="Source Page"}\n[Стр. 2](')
    # A "Стр. N" line inside the page text is plain text, not a page header.
    assert parts[5] == "Стр. 5 колонтитул"
    assert markdown.count("Source Page") == 2
    assert len(refs) == 2


def test_final_markdown_has_no_banner() -> None:
    markdown, _ = to_pandoc_markdown("T", [MdPage(0, None, [])], "t.pdf")
    assert "Draft Banner" not in markdown
    assert markdown.startswith("# T\n\n")
