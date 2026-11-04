"""Recognition → review → final build → export → delivery, with real pandoc (p4)."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ocr_archive import operations
from ocr_archive.db import repo
from ocr_archive.db.connection import immediate
from ocr_archive.models import DocumentStatus
from ocr_archive.worker import recover
from tests.conftest import API_KEY, FakeVlm
from tests.integration.helpers import (
    doc_status,
    drain,
    make_document,
    make_worker,
    page_text,
    synthetic_pages,
)

pytestmark = [
    pytest.mark.p4,
    pytest.mark.skipif(shutil.which("pandoc") is None, reason="pandoc is not installed"),
]
HEADERS = {"X-API-Key": API_KEY}


async def _reviewed_document(conn, settings, tmp_path: Path, fake: FakeVlm) -> str:
    synth = synthetic_pages(2)
    doc_id, ids = make_document(conn, settings, tmp_path, synth)
    fake.texts[(1, 0)] = page_text(synth[0])
    fake.texts[(2, 0)] = page_text(synth[1]) + "\nИтого 1 240 руб. [неразб.]"
    worker = make_worker(conn, settings, fake)
    await drain(worker)
    assert doc_status(conn, doc_id) == DocumentStatus.NEEDS_REVIEW
    view = operations.build_item_view(conn, ids[1], settings, worker.ctx.rules_version)
    result = operations.submit_review(
        conn,
        ids[1],
        operations.ReviewInput(
            request_id="request-e2e-1",
            item_version=view.item_version,
            base_attempt_id=view.base_attempt_id,
            base_revision_id=None,
            draft_seq=0,
            text=view.text.replace("[неразб.]", "(подпись)"),
            printed_label="2 об.",
        ),
    )
    assert result.document_status == DocumentStatus.READY
    return doc_id


async def test_review_build_export_delivery(conn, settings, tmp_path: Path, fake_vlm: FakeVlm):
    doc_id = await _reviewed_document(conn, settings, tmp_path, fake_vlm)
    worker = make_worker(conn, settings, fake_vlm)
    assert await worker.round()
    assert doc_status(conn, doc_id) == DocumentStatus.DONE
    build = repo.latest_done_build(conn, doc_id)
    assert build is not None and build["version"] == 1
    out = settings.data_dir / build["output_dir"]
    assert {p.suffix for p in out.iterdir()} >= {".pdf", ".docx", ".jsonl", ".json"}
    export = settings.export_dir / "Отчёт 1983"
    assert (export / "Отчёт 1983.pdf").exists() and (export / "Отчёт 1983.docx").exists()
    index = (settings.export_dir / "список.csv").read_bytes()
    assert index.startswith(b"\xef\xbb\xbf")
    assert repo.get_document(conn, doc_id)["exported_version"] == 1

    from ocr_archive.api.app import create_app

    with TestClient(create_app(settings, vlm=worker.ctx.vlm)) as client:
        deliveries = client.get("/api/deliveries", headers=HEADERS).json()
        assert deliveries == [{"document_id": doc_id, "title": "Отчёт 1983", "version": 1}]
        pdf = client.get(f"/api/documents/{doc_id}/files/pdf", headers=HEADERS)
        assert pdf.status_code == 200 and pdf.content.startswith(b"%PDF")
        assert "filename*=UTF-8''" in pdf.headers["content-disposition"]
        assert (
            client.post(
                f"/api/documents/{doc_id}/delivered", json={"version": 1}, headers=HEADERS
            ).status_code
            == 204
        )
        client.post(f"/api/documents/{doc_id}/delivered", json={"version": 1}, headers=HEADERS)
        assert client.get("/api/deliveries", headers=HEADERS).json() == []


async def test_worker_killed_during_build_rebuilds(conn, settings, tmp_path, fake_vlm):
    doc_id = await _reviewed_document(conn, settings, tmp_path, fake_vlm)
    with immediate(conn):
        repo.transition_document(conn, doc_id, "ready", "building")
        _, version = repo.insert_build(conn, doc_id, "final", "running")
    tmp_dir = settings.outputs_dir / doc_id / f".tmp-v{version}"
    tmp_dir.mkdir(parents=True)
    (tmp_dir / "partial.pdf").write_bytes(b"%PDF-partial")
    recover(conn, settings)
    builds = repo.builds_of_document(conn, doc_id)
    assert builds[0]["status"] == "failed" and builds[0]["error"] == "interrupted"
    assert not tmp_dir.exists()
    assert doc_status(conn, doc_id) == DocumentStatus.READY
    worker = make_worker(conn, settings, fake_vlm)
    assert await worker.round()
    assert repo.latest_done_build(conn, doc_id)["version"] == version + 1
    assert doc_status(conn, doc_id) == DocumentStatus.DONE
