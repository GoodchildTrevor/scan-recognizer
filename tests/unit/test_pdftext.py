"""Fast track check and text extraction from born-digital pages (section 6.1)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from ocr_archive.models import PageTrack, TextLayer
from ocr_archive.pdfpages import text_layer_kind
from ocr_archive.pdftext import fast_track_check, pdf_text_markdown
from ocr_archive.settings import FastTrackConfig
from tests.fixtures import factory

pytestmark = pytest.mark.p1


@pytest.mark.parametrize(
    ("make", "track", "reason"),
    [
        (factory.make_text_pdf_with_table, PageTrack.PDF_TEXT, None),
        (factory.make_scan_with_stamp_pdf, PageTrack.VLM, "images"),
        (factory.make_mojibake_pdf, PageTrack.VLM, "bad_chars"),
        (factory.make_formula_pdf, PageTrack.VLM, "math_fonts"),
        (factory.make_short_text_pdf, PageTrack.VLM, "too_short"),
        (factory.make_scan_with_invisible_ocr_pdf, PageTrack.VLM, "no_visible_text"),
    ],
)
def test_fast_track_check(
    tmp_path: Path, make: Callable[[Path], Path], track: PageTrack, reason: str | None
) -> None:
    pdf = make(tmp_path)

    check = fast_track_check(pdf, 0, text_layer_kind(pdf, 0), FastTrackConfig())

    assert check.track is track
    assert check.reason == reason


def test_fast_track_can_be_disabled(tmp_path: Path) -> None:
    pdf = factory.make_text_pdf_with_table(tmp_path)

    check = fast_track_check(pdf, 0, TextLayer.VISIBLE, FastTrackConfig(enabled=False))

    assert check.track is PageTrack.VLM
    assert check.reason == "disabled"


@pytest.fixture
def markdown(tmp_path: Path) -> str:
    return pdf_text_markdown(factory.make_text_pdf_with_table(tmp_path), 0)


def test_paragraphs_are_separated_by_an_empty_line(markdown: str) -> None:
    first, second = factory.TEXT_PDF_PARAGRAPHS

    assert "\n".join(first) + "\n\n" + "\n".join(second) in markdown


def test_table_is_a_pipe_table_with_escaped_pipe(markdown: str) -> None:
    lines = markdown.splitlines()
    header = next(i for i, line in enumerate(lines) if line.startswith("| Цех"))

    assert lines[header] == "| Цех | План т | Факт \\| т |"
    assert lines[header + 1] == "| --- | --- | --- |"
    assert lines[header + 2] == "| Литейный | 1 200 | 1 248 |"
    assert lines[header + 3] == "| Механический | 860 | 851 |"


def test_table_text_is_not_repeated_in_paragraphs(markdown: str) -> None:
    assert markdown.count("Литейный") == 1
    assert markdown.count("1 248") == 1


def test_caption_under_the_table_is_kept(markdown: str) -> None:
    blocks = markdown.split("\n\n")
    table_index = next(i for i, block in enumerate(blocks) if block.startswith("| Цех"))

    assert blocks[table_index + 1] == factory.TEXT_PDF_CAPTION
    assert blocks[-1] == factory.TEXT_PDF_AFTER
