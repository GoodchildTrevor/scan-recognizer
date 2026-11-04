"""Searchable PDF (section 9.3) and its verification (section 9.5) on generated scans."""

from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

from ocr_archive.build import pdf as pdf_build
from ocr_archive.build.pdf import (
    PdfLayer,
    PdfPageInput,
    build_searchable_pdf,
    layer_lines,
)
from ocr_archive.build.verify import PdfPageExpectation, comparable_text, verify_pdf
from ocr_archive.exceptions import BuildError
from ocr_archive.lines import detect_lines, line_detection_to_json
from ocr_archive.models import TextLayer
from ocr_archive.pdfpages import text_layer_kind
from ocr_archive.settings import PACKAGE_DIR
from tests.fixtures import factory

pytestmark = pytest.mark.p4

FONT = PACKAGE_DIR / "assets" / "fonts" / "DejaVuSans.ttf"
OUR_TEXT = "Наш распознанный текст страницы\n[подпись]\nвторая строка 1983"


def build_and_verify(source: Path, out: Path, item: PdfPageInput) -> pdf_build.PdfPageResult:
    results = build_searchable_pdf(source, out, [item], FONT)
    verify_pdf(source, out, [PdfPageExpectation(r.page_index, r.expected_text) for r in results])
    return results[0]


def page_text(path: Path, index: int = 0) -> str:
    with pymupdf.open(path) as doc:
        return doc[index].get_text()


def test_scan_without_text_gets_a_layer_bound_to_lines(tmp_path: Path) -> None:
    source, page = factory.make_typewriter_pdf(tmp_path, n_lines=24, skew_deg=0.3)
    detection = detect_lines(source, 0)
    assert detection.reliable and len(detection.boxes) == len(page.lines)
    text = "\n".join(page.lines)
    out = tmp_path / "out.pdf"
    result = build_and_verify(
        source, out, PdfPageInput(0, "vlm", TextLayer.NONE, text, line_detection_to_json(detection))
    )
    assert result.pdf_layer == PdfLayer.LINES
    assert comparable_text(page_text(out)) == comparable_text(text)
    with pymupdf.open(source) as a, pymupdf.open(out) as b:
        assert a.page_count == b.page_count == 1
        assert a[0].rect == b[0].rect
        # The pixels are the same (an uncompressed stream may only get lossless Flate).
        xref_a = a[0].get_images()[0][0]
        xref_b = b[0].get_images()[0][0]
        assert a.xref_stream(xref_a) == b.xref_stream(xref_b)


def test_jpeg_scan_image_is_copied_byte_for_byte(tmp_path: Path) -> None:
    source = factory.make_scan_with_stamp_pdf(tmp_path)
    out = tmp_path / "out.pdf"
    build_and_verify(source, out, PdfPageInput(0, "vlm", TextLayer.VISIBLE, OUR_TEXT, None))
    with pymupdf.open(source) as a, pymupdf.open(out) as b:
        xref_a = a[0].get_images()[0][0]
        xref_b = b[0].get_images()[0][0]
        assert a.xref_get_key(xref_a, "Filter")[1] == "/DCTDecode"
        assert a.xref_stream_raw(xref_a) == b.xref_stream_raw(xref_b)


def test_unbindable_text_goes_into_one_page_block(tmp_path: Path) -> None:
    source, _ = factory.make_typewriter_pdf(tmp_path, n_lines=24)
    detection = line_detection_to_json(detect_lines(source, 0))
    out = tmp_path / "out.pdf"
    result = build_and_verify(
        source, out, PdfPageInput(0, "vlm", TextLayer.NONE, OUR_TEXT, detection)
    )
    assert result.pdf_layer == PdfLayer.PAGE_BLOCK
    assert comparable_text(page_text(out)) == comparable_text(OUR_TEXT)


def test_invisible_old_ocr_is_replaced_without_a_double_layer(tmp_path: Path) -> None:
    source = factory.make_scan_with_invisible_ocr_pdf(tmp_path)
    assert text_layer_kind(source, 0) == TextLayer.HIDDEN
    old = page_text(source)
    out = tmp_path / "out.pdf"
    build_and_verify(source, out, PdfPageInput(0, "vlm", TextLayer.HIDDEN, OUR_TEXT, None))
    text = comparable_text(page_text(out))
    assert text == comparable_text(OUR_TEXT)
    assert comparable_text(old) not in text


def test_stamp_stays_and_old_ocr_goes_on_a_visible_page(tmp_path: Path) -> None:
    source = factory.make_scan_with_stamp_and_old_ocr_pdf(tmp_path)
    assert text_layer_kind(source, 0) == TextLayer.VISIBLE
    out = tmp_path / "out.pdf"
    result = build_and_verify(
        source, out, PdfPageInput(0, "vlm", TextLayer.VISIBLE, OUR_TEXT, None)
    )
    assert result.old_text_kept is False
    text = comparable_text(page_text(out))
    assert comparable_text(factory.STAMP_TEXT) in text
    assert comparable_text(factory.PAGE_NUMBER_TEXT) in text
    assert text.endswith(comparable_text(OUR_TEXT))
    assert comparable_text(page_text(source)) != text


def test_old_text_is_kept_when_its_removal_would_change_the_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = factory.make_scan_with_stamp_and_old_ocr_pdf(tmp_path)
    monkeypatch.setattr(pdf_build, "strip_invisible_spans", lambda doc, index: False)
    out = tmp_path / "out.pdf"
    result = build_and_verify(
        source, out, PdfPageInput(0, "vlm", TextLayer.VISIBLE, OUR_TEXT, None)
    )
    assert result.old_text_kept is True
    assert comparable_text(page_text(out)) == comparable_text(page_text(source) + OUR_TEXT)


def test_pdf_text_page_is_unchanged_even_after_a_manual_edit(tmp_path: Path) -> None:
    source = factory.make_text_pdf_with_table(tmp_path)
    out = tmp_path / "out.pdf"
    result = build_and_verify(
        source, out, PdfPageInput(0, "pdf_text", TextLayer.VISIBLE, "исправленный текст", None)
    )
    assert result.pdf_layer == PdfLayer.ORIGINAL_KEPT
    assert page_text(out) == page_text(source)


def test_verification_catches_a_wrong_layer(tmp_path: Path) -> None:
    source, _ = factory.make_typewriter_pdf(tmp_path, n_lines=10)
    out = tmp_path / "out.pdf"
    build_searchable_pdf(
        source, out, [PdfPageInput(0, "vlm", TextLayer.NONE, OUR_TEXT, None)], FONT
    )
    with pytest.raises(BuildError, match="page 1: text differs"):
        verify_pdf(source, out, [PdfPageExpectation(0, OUR_TEXT + " лишнее")])
    with pytest.raises(BuildError, match="page 1: text differs"):
        verify_pdf(source, out, [PdfPageExpectation(0, "вторая строка 1983" + OUR_TEXT)])


def test_verification_catches_a_changed_render(tmp_path: Path) -> None:
    source, _ = factory.make_typewriter_pdf(tmp_path, n_lines=10)
    out = tmp_path / "out.pdf"
    with pymupdf.open(source) as doc:
        doc[0].insert_text((100, 100), "видимый", fontfile=str(FONT), fontname="dv")
        doc.save(out)
    with pytest.raises(BuildError, match="render changed"):
        verify_pdf(source, out, [PdfPageExpectation(0, "видимый")])


def test_layer_lines_flatten_tables() -> None:
    text = (
        "Заголовок  таблицы\n| A | B \\| C |\n|---|---|\n| 1 | 2 |\n"
        '<table><tr><td colspan="2">Итого &amp; всё</td></tr><tr><td>3</td><td>4</td></tr></table>'
        "\n\\(x^2\\) [неразб.]"
    )
    assert layer_lines(text) == [
        "Заголовок таблицы",
        "A B | C",
        "1 2",
        "Итого & всё",
        "3 4",
        "\\(x^2\\) [неразб.]",
    ]


def test_missing_glyphs_are_replaced_and_counted(tmp_path: Path) -> None:
    source, _ = factory.make_typewriter_pdf(tmp_path, n_lines=10)
    out = tmp_path / "out.pdf"
    result = build_and_verify(
        source, out, PdfPageInput(0, "vlm", TextLayer.NONE, "формула 𝑥 + 𝑥 = 2𝑥", None)
    )
    assert result.char_replacements == {"𝑥": 3}
    assert comparable_text(page_text(out)) == comparable_text("формула ? + ? = 2?")


def test_marker_only_line_joins_the_previous_box() -> None:
    from ocr_archive.models import LineBox

    boxes = [LineBox(50, 100 + 20 * i, 500, 112 + 20 * i) for i in range(3)]
    lines = [
        "[подпись]",
        "первая строка текста отчёта",
        "[печать]",
        "вторая строка текста отч",
        "третья строка текста отчёта",
    ]
    pairs = pdf_build._bind_lines(lines, boxes)
    assert pairs is not None
    assert [text for text, _ in pairs] == [
        "[подпись] первая строка текста отчёта [печать]",
        "вторая строка текста отч",
        "третья строка текста отчёта",
    ]
