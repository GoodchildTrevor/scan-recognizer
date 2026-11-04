"""Safe file and folder names derived from titles (section 10.2)."""

from __future__ import annotations

import pytest

from ocr_archive.build.export import safe_name

pytestmark = pytest.mark.p3

DOC_ID = "01JABCDEFGHJKMNPQRSTVWXYZ9"
FALLBACK = "документ-" + DOC_ID[-8:]


@pytest.mark.parametrize(
    "title",
    ["", "   ", ".", "..", "...", "CON", "con", "Con.txt", "nul.pdf", "COM1", "lpt9", '/\\:*?"<>|'],
)
def test_unusable_titles_fall_back_to_the_id(title: str) -> None:
    assert safe_name(title, DOC_ID) == FALLBACK


def test_forbidden_and_control_characters_are_removed() -> None:
    assert safe_name('Отчёт: "план"/факт\t1983?\x7f', DOC_ID) == "Отчёт планфакт1983"


def test_whitespace_is_collapsed_and_edges_trimmed() -> None:
    assert safe_name("  .Отчёт   за   1983 год.  ", DOC_ID) == "Отчёт за 1983 год"


def test_long_title_is_cut_to_100_characters() -> None:
    name = safe_name("Я" * 300, DOC_ID)
    assert name == "Я" * 100
    assert len(name.encode("utf-8")) <= 255


def test_reserved_name_only_as_whole_stem() -> None:
    assert safe_name("CONSOLE", DOC_ID) == "CONSOLE"
    assert safe_name("COM10", DOC_ID) == "COM10"
