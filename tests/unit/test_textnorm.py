"""Text normalization, numbers, table shapes, markers (sections 6.8, 7)."""

from __future__ import annotations

import pytest

from ocr_archive.build.markers import (
    find_markers,
    handwriting_markers,
    hard_markers_in,
    is_marker_only,
)
from ocr_archive.settings import DecideConfig
from ocr_archive.textnorm import (
    content_text,
    counted_lines,
    numbers,
    soft_normalize,
    table_shape,
)

pytestmark = pytest.mark.p2


def test_soft_normalize_quotes_dashes_hyphens_and_tags() -> None:
    text = "«Слово» — тире\nпере-\nнос <b>жирный</b> | ячейка"
    assert soft_normalize(text) == '"Слово" - тире перенос жирный ячейка'


def test_signs() -> None:
    assert numbers("t = −5") == ["-5"]
    assert numbers("t = 5") == ["5"]
    assert numbers("t = −5") != numbers("t = 5")
    assert numbers("(-3) и /+2") == ["-3", "2"]


def test_ranges_are_two_unsigned_numbers() -> None:
    assert numbers("1941–1945") == ["1941", "1945"]
    assert numbers("1941-1945") == ["1941", "1945"]


@pytest.mark.parametrize("space", [" ", " ", " "])
def test_thousand_groups(space: str) -> None:
    assert numbers(f"итого 12{space}345 руб.") == ["12345"]


def test_decimal_separator_kept() -> None:
    assert numbers("3,5") == ["3,5"]
    assert numbers("3.5") == ["3.5"]
    assert numbers("3,5") != numbers("3.5")


def test_percent_with_or_without_space() -> None:
    assert numbers("15 %") == numbers("15%") == ["15%"]
    assert numbers("2‰") == ["2‰"]


def test_colspan_attribute_gives_no_number() -> None:
    html = '<table><tr><td colspan="2">7</td></tr></table>'
    assert numbers(html) == ["7"]
    assert "colspan" not in content_text(html)


def test_same_table_html_and_pipe_same_numbers_and_shape() -> None:
    pipe = "| Год | План |\n|---|---|\n| 1983 | 125 |"
    html = "<table><tr><th>Год</th><th>План</th></tr><tr><td>1983</td><td>125</td></tr></table>"
    assert numbers(pipe) == numbers(html) == ["1983", "125"]
    assert table_shape(pipe) == table_shape(html) == [[2, 2]]


def test_table_shape_pipe_with_escaped_pipe() -> None:
    text = "| a | b \\| c | d |\n|---|---|---|\n| 1 | 2 | 3 |"
    assert table_shape(text) == [[3, 3]]


def test_table_shape_html_colspan_rowspan() -> None:
    html = (
        "<table>"
        "<tr><td colspan=2>A</td><td rowspan=2>B</td></tr>"
        "<tr><td>c</td><td>d</td></tr>"
        "<tr><td>e</td><td>f</td><td>g</td></tr>"
        "</table>"
    )
    assert table_shape(html) == [[3, 3, 3]]


def test_counted_lines_skip_separators_and_marker_lines() -> None:
    text = "строка один\n\n[подпись]\n|---|---|\n| a | b |\nстрока [неразб.] два"
    assert counted_lines(text) == ["строка один", "| a | b |", "строка [неразб.] два"]


def test_hard_markers() -> None:
    hard = DecideConfig().hard_markers
    text = "a [неразб.] b [неразб.: вариант?] c [рук.: неразб.] d [рук.: текст] [печать]"
    inners = [m.inner for m in hard_markers_in(text, hard)]
    assert inners == ["неразб.", "неразб.: вариант?", "рук.: неразб."]
    assert [m.inner for m in handwriting_markers(text, hard)] == ["рук.: текст"]
    assert len(find_markers(text)) == 5


def test_marker_only_line() -> None:
    assert is_marker_only("  [подпись] [печать: ООО]  ")
    assert not is_marker_only("Директор [подпись]")
    assert not is_marker_only("")
