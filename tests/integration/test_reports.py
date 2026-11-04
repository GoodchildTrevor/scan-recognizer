"""Pilot and spot-check reports, bench metrics, doctor storage checks (section 15, 17; p5)."""

from __future__ import annotations

from pathlib import Path

import pytest

from ocr_archive import operations
from ocr_archive.bench import cer, export_gold, import_gold, line_recall
from ocr_archive.db import repo
from ocr_archive.doctor import Level, check_storage
from ocr_archive.reports import audit_report, pilot_report
from tests.conftest import FakeVlm
from tests.integration.helpers import drain, make_document, make_worker, page_text, synthetic_pages

pytestmark = pytest.mark.p5


async def test_pilot_and_audit_reports(conn, settings, tmp_path: Path, fake_vlm: FakeVlm) -> None:
    synth = synthetic_pages(3)
    doc_id, ids = make_document(conn, settings, tmp_path, synth)
    for number, page in enumerate(synth, start=1):
        fake_vlm.texts[(number, 0)] = page_text(page)
    fake_vlm.texts[(2, 0)] = page_text(synth[1]) + " [неразб.]"
    worker = make_worker(conn, settings, fake_vlm)
    await drain(worker)
    view = operations.build_item_view(conn, ids[1], settings, worker.ctx.rules_version)
    operations.submit_review(
        conn,
        ids[1],
        operations.ReviewInput(
            request_id="request-r1",
            item_version=view.item_version,
            base_attempt_id=view.base_attempt_id,
            base_revision_id=None,
            draft_seq=0,
            text=view.text.replace(" [неразб.]", " слово"),
            duration_s=30,
        ),
    )
    sampled = operations.audit_sample(conn, doc_id, 1)
    sample_view = operations.build_item_view(conn, sampled[0], settings, worker.ctx.rules_version)
    operations.submit_review(
        conn,
        sampled[0],
        operations.ReviewInput(
            request_id="request-r2",
            item_version=sample_view.item_version,
            base_attempt_id=sample_view.base_attempt_id,
            base_revision_id=None,
            draft_seq=0,
            text=sample_view.text,
            duration_s=90,
        ),
    )
    report = pilot_report(conn, doc_id, settings)
    assert "| markers | 1 | 1 |" in report
    assert "Pages with flags: 1 of 3" in report
    assert "Pages without flags read in full: 1, corrected 0" in report
    assert "statistical guarantee" in report
    audit = audit_report(conn, doc_id)
    assert "Corrected: 0 of 1" in audit


def test_bench_metrics() -> None:
    assert cer("абв", "абв") == 0
    assert cer("абвг", "абвд") == pytest.approx(0.25)
    assert line_recall("строка один\nстрока два", "строка один\nдругое") == 0.5


async def test_gold_roundtrip(conn, settings, tmp_path: Path, fake_vlm: FakeVlm) -> None:
    synth = synthetic_pages(2)
    doc_id, _ = make_document(conn, settings, tmp_path, synth)
    fake_vlm.texts[(1, 0)] = page_text(synth[0])
    fake_vlm.texts[(2, 0)] = page_text(synth[1])
    await drain(make_worker(conn, settings, fake_vlm))
    paths = export_gold(conn, settings, doc_id, [2])
    assert paths[0].read_text(encoding="utf-8") == page_text(synth[1])
    paths[0].write_text("исправленный эталон", encoding="utf-8")
    assert import_gold(conn, settings) == 1
    assert repo.gold_pages(conn)[0]["text"] == "исправленный эталон"


def test_doctor_storage(settings) -> None:
    checks = {c.name: c for c in check_storage(settings)}
    assert checks["data"].level == Level.OK
    assert checks["export"].level == Level.OK
