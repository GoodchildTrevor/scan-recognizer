"""Text layer kind, rendering and text removal (sections 6.1, 6.2, 9.3)."""

from __future__ import annotations

import io
from collections.abc import Callable
from pathlib import Path

import pymupdf
import pytest
from PIL import Image

from ocr_archive import pdfpages
from ocr_archive.models import TextLayer
from tests.fixtures import factory

pytestmark = pytest.mark.p1


@pytest.mark.parametrize(
    ("make", "expected"),
    [
        (factory.make_scan_with_invisible_ocr_pdf, TextLayer.HIDDEN),
        (factory.make_multi_image_scan_pdf, TextLayer.HIDDEN),
        (factory.make_white_text_pdf, TextLayer.HIDDEN),
        (factory.make_text_pdf_with_table, TextLayer.VISIBLE),
        (factory.make_scan_with_stamp_pdf, TextLayer.VISIBLE),
        (factory.make_scan_with_stamp_and_old_ocr_pdf, TextLayer.VISIBLE),
    ],
)
def test_text_layer_kind(tmp_path: Path, make: Callable[[Path], Path], expected: TextLayer) -> None:
    assert pdfpages.text_layer_kind(make(tmp_path), 0) is expected


def test_image_set_has_no_text_layer(tmp_path: Path) -> None:
    pdf = factory.jpeg_set_pdf(tmp_path, factory.make_jpeg_set(tmp_path))

    assert [pdfpages.text_layer_kind(pdf, i) for i in range(3)] == [TextLayer.NONE] * 3


def test_strip_text_keeps_images_and_render(tmp_path: Path) -> None:
    pdf = factory.make_scan_with_invisible_ocr_pdf(tmp_path)
    with pymupdf.open(pdf) as original, pymupdf.open(pdf) as stripped:
        pdfpages.strip_text(stripped, 0)

        assert stripped[0].get_texttrace() == []
        assert len(stripped[0].get_image_info()) == len(original[0].get_image_info())
        assert pdfpages.pages_look_equal(original, 0, stripped, 0)


@pytest.mark.p4
def test_strip_invisible_spans_keeps_visible_stamp(tmp_path: Path) -> None:
    pdf = factory.make_scan_with_stamp_and_old_ocr_pdf(tmp_path)
    with pymupdf.open(pdf) as original, pymupdf.open(pdf) as doc:
        assert pdfpages.strip_invisible_spans(doc, 0)

        spans = doc[0].get_texttrace()
        assert spans
        assert all(span["type"] != 3 and span["opacity"] > 0 for span in spans)
        text = doc[0].get_text()
        assert factory.STAMP_TEXT in text
        assert factory.PAGE_NUMBER_TEXT in text
        assert pdfpages.pages_look_equal(original, 0, doc, 0)


@pytest.mark.p4
def test_strip_invisible_spans_rolls_back_when_render_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf = factory.make_scan_with_stamp_and_old_ocr_pdf(tmp_path)
    monkeypatch.setattr(pdfpages, "pages_look_equal", lambda *args, **kwargs: False)
    with pymupdf.open(pdf) as doc:
        before = len(doc[0].get_texttrace())

        assert not pdfpages.strip_invisible_spans(doc, 0)

        assert len(doc[0].get_texttrace()) == before


def test_pages_look_equal_detects_difference(tmp_path: Path) -> None:
    pdf = factory.make_text_pdf_with_table(tmp_path)
    with pymupdf.open(pdf) as doc, pymupdf.open(pdf) as other:
        other[0].draw_rect(pymupdf.Rect(10, 10, 20, 20), color=(0, 0, 0), fill=(0, 0, 0))

        assert pdfpages.pages_look_equal(doc, 0, doc, 0)
        assert not pdfpages.pages_look_equal(doc, 0, other, 0)


def test_image_area_share(tmp_path: Path) -> None:
    scan = factory.make_scan_with_stamp_pdf(tmp_path)
    text = factory.make_text_pdf_with_table(tmp_path)
    with pymupdf.open(scan) as scan_doc, pymupdf.open(text) as text_doc:
        assert pdfpages.image_area_share(scan_doc[0]) == pytest.approx(1.0, abs=0.01)
        assert pdfpages.image_area_share(text_doc[0]) == 0.0


def _size(data: bytes) -> tuple[int, int]:
    with Image.open(io.BytesIO(data)) as image:
        return image.size


def test_render_page_long_side_and_clip(tmp_path: Path) -> None:
    pdf = factory.make_text_pdf_with_table(tmp_path)

    width, height = _size(pdfpages.render_page(pdf, 0, 1000))
    assert max(width, height) == pytest.approx(1000, abs=1)

    clip = pymupdf.Rect(0, 0, 300, 100)
    width, height = _size(pdfpages.render_page(pdf, 0, 600, clip=clip))
    assert width == pytest.approx(600, abs=1)
    assert height == pytest.approx(200, abs=1)


def test_render_view_clip_crops_the_whole_page_render(tmp_path: Path) -> None:
    pdf = factory.make_text_pdf_with_table(tmp_path)

    full = pdfpages.render_view(pdf, 0, 1000, clip_fractions=None)
    half = pdfpages.render_view(pdf, 0, 1000, clip_fractions=(0.0, 0.0, 0.5, 0.5), fmt="jpeg")

    assert full[:2] == b"\xff\xd8"
    full_w, full_h = _size(full)
    half_w, half_h = _size(half)
    assert half_w == pytest.approx(full_w / 2, abs=2)
    assert half_h == pytest.approx(full_h / 2, abs=2)


def test_render_cached_reuses_and_trims(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pdf = factory.make_text_pdf_with_table(tmp_path)
    cache = tmp_path / "cache"
    monkeypatch.setattr(pdfpages, "_cache_writes", 1)

    first = pdfpages.render_cached(cache, 7, pdf, 0, 400, max_gb=5)
    cached = cache / "render" / "7" / "400.png"
    assert cached.exists()
    assert pdfpages.render_cached(cache, 7, pdf, 0, 400, max_gb=5) == first

    clip = pymupdf.Rect(0, 0, 100, 100)
    pdfpages.render_cached(cache, 7, pdf, 0, 400, clip=clip, max_gb=5)
    assert len(list((cache / "render" / "7").glob("400_*.png"))) == 1

    # The next write runs the cleanup; a tiny limit removes the oldest files.
    monkeypatch.setattr(pdfpages, "_cache_writes", 0)
    pdfpages.render_cached(cache, 8, pdf, 0, 300, max_gb=1e-9)
    assert not cached.exists()
