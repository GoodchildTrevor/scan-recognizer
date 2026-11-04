"""Line detector, split gap and line plausibility (sections 6.3, 6.7, 6.13)."""

from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

from ocr_archive.lines import (
    detect_lines,
    find_split_gap,
    line_detection_from_json,
    line_detection_to_json,
    line_plausible,
    lines_version,
)
from ocr_archive.models import LineBox
from ocr_archive.settings import ChecksConfig
from tests.fixtures import factory

pytestmark = pytest.mark.p2


@pytest.mark.parametrize("skew", [-2.0, 0.0, 1.0, 2.0])
def test_detect_lines_counts_typewriter_lines(tmp_path: Path, skew: float) -> None:
    pdf, page = factory.make_typewriter_pdf(tmp_path, n_lines=30, skew_deg=skew, seed=5)

    detection = detect_lines(pdf, 0)

    assert abs(len(detection.boxes) - len(page.lines)) <= 1
    assert detection.reliable
    assert not detection.columns_detected
    assert detection.skew_deg == pytest.approx(skew, abs=0.5)


def test_boxes_match_ground_truth_without_skew(tmp_path: Path) -> None:
    pdf, page = factory.make_typewriter_pdf(tmp_path, n_lines=20, skew_deg=0.0)

    detection = detect_lines(pdf, 0)

    assert len(detection.boxes) == len(page.boxes_pt)
    for box, (x0, y0, x1, y1) in zip(detection.boxes, page.boxes_pt, strict=True):
        # x uses the 1st/99th percentile of ink columns, so it is a little inside the glyphs.
        assert box.x0 == pytest.approx(x0, abs=6)
        assert box.x1 == pytest.approx(x1, abs=6)
        assert box.y0 == pytest.approx(y0, abs=2)
        assert box.y1 == pytest.approx(y1, abs=2)


def test_two_column_page_is_not_reliable(tmp_path: Path) -> None:
    pdf, _page = factory.make_two_column_pdf(tmp_path)

    detection = detect_lines(pdf, 0)

    assert detection.columns_detected
    assert not detection.reliable


def test_blank_page_has_no_lines(tmp_path: Path) -> None:
    pdf = tmp_path / "blank.pdf"
    with pymupdf.open() as doc:
        doc.new_page()
        doc.save(pdf)

    detection = detect_lines(pdf, 0)

    assert detection.boxes == []
    assert not detection.reliable


def test_split_gap_is_found_between_middle_lines(tmp_path: Path) -> None:
    pdf, page = factory.make_split_page_pdf(tmp_path)
    detection = detect_lines(pdf, 0)
    assert detection.reliable

    cut = find_split_gap(pdf, 0, detection, ChecksConfig())

    assert cut is not None
    middle = len(page.boxes_pt) // 2
    gaps = [
        (page.boxes_pt[k][3], page.boxes_pt[k + 1][1])
        for k in range(middle - 3, min(len(page.boxes_pt) - 1, middle + 4))
    ]
    assert any(bottom < cut < top for bottom, top in gaps)


def test_split_gap_is_not_found_when_every_gap_has_ink(tmp_path: Path) -> None:
    pdf, _page = factory.make_no_gap_pdf(tmp_path)
    detection = detect_lines(pdf, 0)

    assert find_split_gap(pdf, 0, detection, ChecksConfig()) is None


@pytest.mark.p3
def test_missing_plus_split_line_fails_strict_plausibility(tmp_path: Path) -> None:
    case = factory.make_missing_split_pair(tmp_path)
    detection = detect_lines(case.pdf_path, 0)
    assert len(detection.boxes) == len(case.text_lines)

    # Crops need every line to fit; the PDF layer tolerates 10% of outliers (sections 6.13, 9.3).
    assert not line_plausible(case.text_lines, detection.boxes, min_share=1.0)
    assert line_plausible(case.text_lines, detection.boxes, min_share=0.9)
    assert line_plausible(case.page.lines, detection.boxes, min_share=1.0)


@pytest.mark.p3
def test_line_plausible_ignores_short_lines() -> None:
    boxes = [LineBox(0, 0, 100, 10), LineBox(0, 20, 400, 30)]

    assert line_plausible(["[подпись]", "short"], boxes, min_share=1.0)
    assert line_plausible([], [], min_share=1.0)


def test_detection_json_roundtrip(tmp_path: Path) -> None:
    pdf, _page = factory.make_typewriter_pdf(tmp_path, n_lines=10)
    detection = detect_lines(pdf, 0)

    restored = line_detection_from_json(line_detection_to_json(detection))

    assert restored.reliable == detection.reliable
    assert restored.columns_detected == detection.columns_detected
    assert len(restored.boxes) == len(detection.boxes)
    for a, b in zip(restored.boxes, detection.boxes, strict=True):
        assert a.x0 == pytest.approx(b.x0, abs=0.01)
        assert a.y1 == pytest.approx(b.y1, abs=0.01)


def test_lines_version_is_stable() -> None:
    assert lines_version() == lines_version()
    assert lines_version().startswith("1:")
