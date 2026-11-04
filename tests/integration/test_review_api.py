"""REST API of the review: queue, drafts, "Done", retry, statuses (section 16, p3)."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from ocr_archive import operations
from ocr_archive.api.app import create_app
from ocr_archive.db import repo
from ocr_archive.db.connection import immediate
from ocr_archive.models import DocumentStatus, PageStatus
from ocr_archive.settings import Settings
from ocr_archive.status import recompute_status
from ocr_archive.vlm.client import VlmClient
from ocr_archive.worker import Worker, recompute_all
from tests.conftest import API_KEY, FakeVlm
from tests.integration.helpers import (
    doc_status,
    drain,
    make_document,
    make_worker,
    mark_built,
    page_row,
    page_text,
    synthetic_pages,
)

pytestmark = pytest.mark.p3
HEADERS = {"X-API-Key": API_KEY}


@pytest.fixture
def client(settings: Settings, fake_vlm: FakeVlm) -> Iterator[TestClient]:
    vlm = VlmClient(settings.vlm, "", transport=httpx.MockTransport(fake_vlm.handler))
    with TestClient(create_app(settings, vlm=vlm)) as test_client:
        yield test_client


async def _flagged_document(
    conn: sqlite3.Connection, settings: Settings, tmp_path: Path, fake: FakeVlm, *, pages: int = 3
) -> tuple[str, list[int], Worker]:
    """Document whose page 2 is flagged (numbers differ) and the others ok."""
    synth = synthetic_pages(pages)
    doc_id, ids = make_document(conn, settings, tmp_path, synth)
    for number, page in enumerate(synth, start=1):
        fake.texts[(number, 0)] = page_text(page)
    text = page_text(synth[1])
    fake.texts[(2, 1)] = text.replace(synth[1].lines[0], synth[1].lines[0] + " 17")
    fake.texts[(2, 2)] = text.replace(synth[1].lines[0], synth[1].lines[0] + " 71")
    worker = make_worker(conn, settings, fake)
    await drain(worker)
    return doc_id, ids, worker


def _item(client: TestClient, page_id: int) -> dict[str, Any]:
    response = client.get(f"/api/review/items/{page_id}", headers=HEADERS)
    assert response.status_code == 200, response.text
    return response.json()


def _put(
    client: TestClient, page_id: int, item: dict[str, Any], text: str, seq: int, **extra: Any
) -> httpx.Response:
    body = {
        "expected_seq": seq,
        "item_version": item["item_version"],
        "base_attempt_id": item["base_attempt_id"],
        "base_revision_id": item["base_revision_id"],
        "text": text,
        "printed_label": None,
        "choices": {},
        "hints": [],
        "cursor": 0,
    } | extra
    return client.put(f"/api/pages/{page_id}/draft", json=body, headers=HEADERS)


def _done(
    client: TestClient,
    page_id: int,
    item: dict[str, Any],
    text: str,
    *,
    request_id: str = "request-0001",
    draft_seq: int = 0,
    **extra: Any,
) -> httpx.Response:
    body = {
        "request_id": request_id,
        "item_version": item["item_version"],
        "base_attempt_id": item["base_attempt_id"],
        "base_revision_id": item["base_revision_id"],
        "draft_seq": draft_seq,
        "text": text,
    } | extra
    return client.post(f"/api/pages/{page_id}/review", json=body, headers=HEADERS)


def test_auth_required(client: TestClient) -> None:
    assert client.get("/api/review/queue").status_code == 401
    assert client.get("/api/review/queue", headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.get("/health").status_code == 200


async def test_queue_order_drafts_first(conn, settings, tmp_path, fake_vlm, client) -> None:
    doc_id, ids, _ = await _flagged_document(conn, settings, tmp_path, fake_vlm)
    queue = client.get("/api/review/queue", headers=HEADERS).json()
    assert [q["page_id"] for q in queue] == [ids[1]]
    assert queue[0]["scope"] == "hints" and "digits_disagreement" in queue[0]["reasons"]
    fix = client.get(f"/api/documents/{doc_id}/pages/3/item", headers=HEADERS).json()
    assert fix["scope"] == "full"
    assert _put(client, ids[2], fix, fix["text"] + " правка", 0).status_code == 200
    queue = client.get("/api/review/queue", headers=HEADERS).json()
    assert [q["page_id"] for q in queue] == [ids[2], ids[1]]
    assert queue[0]["has_draft"] and queue[0]["scope"] == "full"
    skipped = client.get(f"/api/review/queue?exclude={ids[2]}", headers=HEADERS).json()
    assert [q["page_id"] for q in skipped] == [ids[1]]


async def test_processing_document_pages_appear(conn, settings, tmp_path, fake_vlm, client) -> None:
    synth = synthetic_pages(2)
    doc_id, ids = make_document(conn, settings, tmp_path, synth)
    fake_vlm.texts[(1, 1)] = page_text(synth[0]) + " [неразб.]"
    fake_vlm.texts[(1, 2)] = page_text(synth[0]) + " [неразб.]"
    worker = make_worker(conn, settings, fake_vlm)
    await worker._document_step()  # classification
    await worker._document_step()  # page 1
    assert doc_status(conn, doc_id) == DocumentStatus.PROCESSING
    queue = client.get("/api/review/queue", headers=HEADERS).json()
    assert [q["page_id"] for q in queue] == [ids[0]]
    assert client.get(f"/api/review/items/{ids[1]}", headers=HEADERS).status_code == 409
    assert (
        client.get(f"/api/review/items/{ids[1]}", headers=HEADERS).json()["detail"]
        == "invalid_transition"
    )


async def test_draft_roundtrip_and_conflicts(conn, settings, tmp_path, fake_vlm, client) -> None:
    _, ids, _ = await _flagged_document(conn, settings, tmp_path, fake_vlm)
    page_id = ids[1]
    item = _item(client, page_id)
    saved = _put(client, page_id, item, "текст A", 0, cursor=3, printed_label="9 об.")
    assert saved.json()["seq"] == 1
    again = _item(client, page_id)["draft"]
    assert again["text"] == "текст A" and again["cursor"] == 3 and again["printed_label"] == "9 об."
    assert not again["stale"]
    assert _put(client, page_id, item, "текст B", 1).json()["seq"] == 2
    old_tab = _put(client, page_id, item, "старая вкладка", 1)
    assert old_tab.status_code == 409 and old_tab.json()["detail"] == "draft_conflict"
    assert _item(client, page_id)["draft"]["text"] == "текст B"
    assert _done(client, page_id, item, "x", draft_seq=1).json()["detail"] == "draft_conflict"
    assert (
        client.delete(f"/api/pages/{page_id}/draft?draft_seq=1", headers=HEADERS).json()["detail"]
        == "draft_conflict"
    )
    assert _item(client, page_id)["draft"]["text"] == "текст B"
    assert (
        client.delete(f"/api/pages/{page_id}/draft?draft_seq=2", headers=HEADERS).status_code == 204
    )
    assert (
        client.delete(f"/api/pages/{page_id}/draft?draft_seq=2", headers=HEADERS).status_code == 204
    )
    assert _item(client, page_id)["draft"] is None


async def test_stale_draft_and_keep_stale(conn, settings, tmp_path, fake_vlm, client) -> None:
    _, ids, _ = await _flagged_document(conn, settings, tmp_path, fake_vlm)
    item = _item(client, ids[1])
    stale_item = item | {"base_attempt_id": 999999}
    assert _put(client, ids[1], stale_item, "x", 0).json()["detail"] == "stale"
    assert _put(client, ids[1], stale_item, "x", 0, keep_stale=True).status_code == 200
    draft = _item(client, ids[1])["draft"]
    assert draft["stale"] and draft["stale_reason"] == "keep_stale"


async def test_done_flow(conn, settings, tmp_path, fake_vlm, client) -> None:
    doc_id, ids, _ = await _flagged_document(conn, settings, tmp_path, fake_vlm)
    share = operations.document_view(conn, doc_id, settings)["flagged_share"]
    item = _item(client, ids[1])
    stale = _done(client, ids[1], item | {"base_attempt_id": 1}, "x")
    assert stale.status_code == 409 and stale.json()["detail"] == "stale"
    with immediate(conn):
        repo.set_document_status(conn, doc_id, DocumentStatus.BUILDING)
    assert _done(client, ids[1], item, "x").json()["detail"] == "document_building"
    with immediate(conn):
        repo.set_document_status(conn, doc_id, DocumentStatus.NEEDS_REVIEW)
    response = _done(
        client,
        ids[1],
        item | {"item_version": "other"},
        item["text"] + " правка",
        printed_label="9 об.",
        duration_s=40,
    )
    assert response.status_code == 201
    assert response.json()["document_status"] == DocumentStatus.READY
    repeat = _done(client, ids[1], item, "что угодно")
    assert repeat.status_code == 200
    assert repeat.json()["revision_id"] == response.json()["revision_id"]
    row = page_row(conn, ids[1])
    assert row["status"] == PageStatus.REVIEWED and row["printed_label"] == "9 об."
    assert "digits_disagreement" in row["flags_json"]
    assert operations.document_view(conn, doc_id, settings)["flagged_share"] == share


async def test_draft_blocks_built_document(conn, settings, tmp_path, fake_vlm, client) -> None:
    doc_id, ids, _ = await _flagged_document(conn, settings, tmp_path, fake_vlm)
    item = _item(client, ids[1])
    _done(client, ids[1], item, item["text"])
    mark_built(conn, doc_id)
    fix = client.get(f"/api/documents/{doc_id}/pages/1/item", headers=HEADERS).json()
    _put(client, ids[0], fix, "правка", 0)
    assert doc_status(conn, doc_id) == DocumentStatus.NEEDS_REVIEW
    client.delete(f"/api/pages/{ids[0]}/draft?draft_seq=1", headers=HEADERS)
    assert doc_status(conn, doc_id) == DocumentStatus.DONE
    with immediate(conn):
        repo.set_document_status(conn, doc_id, DocumentStatus.BUILDING)
    assert _put(client, ids[0], fix, "во время сборки", 0).status_code == 200
    with immediate(conn):
        repo.set_document_status(conn, doc_id, DocumentStatus.DONE)
        recompute_status(conn, doc_id)
    assert doc_status(conn, doc_id) == DocumentStatus.NEEDS_REVIEW


async def test_retry_keeps_draft(conn, settings, tmp_path, fake_vlm, client) -> None:
    synth = synthetic_pages(1)
    doc_id, ids = make_document(conn, settings, tmp_path, synth)
    fake_vlm.texts[(1, 0)] = page_text(synth[0])
    fake_vlm.failures = [(422, "bad"), (422, "bad")]
    worker = make_worker(conn, settings, fake_vlm)
    await drain(worker)
    assert page_row(conn, ids[0])["status"] == PageStatus.FAILED
    item = _item(client, ids[0])
    assert item["no_candidate"] and item["text"] == ""
    assert _put(client, ids[0], item, "набрал руками", 0).status_code == 200
    assert client.post(f"/api/pages/{ids[0]}/retry", headers=HEADERS).json()["status"] == "pending"
    frozen = _put(client, ids[0], item, "ещё", 1)
    assert frozen.json()["detail"] == "invalid_transition"
    await drain(worker)
    assert page_row(conn, ids[0])["status"] == PageStatus.OK
    assert doc_status(conn, doc_id) == DocumentStatus.NEEDS_REVIEW
    draft = _item(client, ids[0])["draft"]
    assert draft["text"] == "набрал руками" and draft["stale"]
    assert draft["stale_reason"] == "recognized_again"
    queue = client.get("/api/review/queue", headers=HEADERS).json()
    assert queue[0]["page_id"] == ids[0] and queue[0]["has_draft"]


async def test_retry_failing_again_keeps_fresh_draft(
    conn, settings, tmp_path, fake_vlm, client
) -> None:
    synth = synthetic_pages(1)
    make_document(conn, settings, tmp_path, synth)
    ids = [int(r["id"]) for r in conn.execute("SELECT id FROM pages").fetchall()]
    fake_vlm.failures = [(422, "bad")] * 4
    worker = make_worker(conn, settings, fake_vlm)
    await drain(worker)
    item = _item(client, ids[0])
    _put(client, ids[0], item, "черновик", 0)
    client.post(f"/api/pages/{ids[0]}/retry", headers=HEADERS)
    await drain(worker)
    assert page_row(conn, ids[0])["status"] == PageStatus.FAILED
    assert not _item(client, ids[0])["draft"]["stale"]


async def test_rebuild_survives_restart(conn, settings, tmp_path, fake_vlm, client) -> None:
    doc_id, ids, worker = await _flagged_document(conn, settings, tmp_path, fake_vlm)
    item = _item(client, ids[1])
    _done(client, ids[1], item, item["text"])
    mark_built(conn, doc_id)
    with immediate(conn):
        recompute_status(conn, doc_id)
    assert doc_status(conn, doc_id) == DocumentStatus.DONE
    assert client.post(f"/api/documents/{doc_id}/rebuild", headers=HEADERS).status_code == 202
    assert doc_status(conn, doc_id) == DocumentStatus.READY
    recompute_all(conn, worker.ctx.rules_version)
    assert doc_status(conn, doc_id) == DocumentStatus.READY
    assert repo.get_document(conn, doc_id)["rebuild_requested"] == 1


async def test_spot_check_without_changes_returns_to_done(
    conn, settings, tmp_path, fake_vlm, client
) -> None:
    doc_id, ids, _ = await _flagged_document(conn, settings, tmp_path, fake_vlm)
    item = _item(client, ids[1])
    _done(client, ids[1], item, item["text"])
    build_id = mark_built(conn, doc_id)
    with immediate(conn):
        recompute_status(conn, doc_id)
    chosen = operations.audit_sample(conn, doc_id, 1)
    assert doc_status(conn, doc_id) == DocumentStatus.NEEDS_REVIEW
    queue = client.get("/api/review/queue", headers=HEADERS).json()
    assert queue[0]["scope"] == "full" and queue[0]["reasons"] == ["spot_check"]
    sample_item = _item(client, chosen[0])
    _done(client, chosen[0], sample_item, sample_item["text"], request_id="request-0002")
    assert doc_status(conn, doc_id) == DocumentStatus.DONE
    assert len(repo.builds_of_document(conn, doc_id)) == 1
    assert repo.get_ingest_for_build(conn, build_id) is None


async def test_redecide_waits_for_draft(conn, settings, tmp_path, fake_vlm, client) -> None:
    doc_id, ids, _ = await _flagged_document(conn, settings, tmp_path, fake_vlm)
    item = _item(client, ids[1])
    _done(client, ids[1], item, item["text"])
    mark_built(conn, doc_id)
    fix = client.get(f"/api/documents/{doc_id}/pages/1/item", headers=HEADERS).json()
    _put(client, ids[0], fix, "правка", 0)
    from tests.conftest import make_settings

    strict = make_settings(tmp_path, signals={"min_diff_words": 1})
    worker2 = make_worker(conn, strict, fake_vlm)
    operations.request_redecide(conn, doc_id)
    await drain(worker2)
    assert repo.get_document(conn, doc_id)["redecide_requested"] == 1
    summary = operations.review_summary(conn, strict, worker2.ctx.rules_version)
    assert summary["waiting_redecide"][0]["draft_pages"] == 1
    operations.delete_draft(conn, ids[0], 1)
    assert doc_status(conn, doc_id) == DocumentStatus.PROCESSING
    await drain(worker2)
    assert repo.get_document(conn, doc_id)["redecide_requested"] == 0
    assert page_row(conn, ids[0])["rules_version"] == worker2.ctx.rules_version


async def test_summary_and_documents(conn, settings, tmp_path, fake_vlm, client) -> None:
    doc_id, _, _ = await _flagged_document(conn, settings, tmp_path, fake_vlm)
    summary = client.get("/api/review/summary", headers=HEADERS).json()
    assert summary["queue_total"] == 1
    doc = next(d for d in summary["documents"] if d["id"] == doc_id)
    assert doc["over_guideline"] is True and doc["to_review"] == 1
    detail = client.get(f"/api/documents/{doc_id}", headers=HEADERS).json()
    assert detail["progress"]["flagged"] == 1


async def test_images_and_region(conn, settings, tmp_path, fake_vlm, client) -> None:
    _, ids, _ = await _flagged_document(conn, settings, tmp_path, fake_vlm, pages=2)
    image = client.get(f"/api/pages/{ids[0]}/image?long_side=800&clip=0,0,0.5,0.5", headers=HEADERS)
    assert image.status_code == 200 and image.content[:2] == b"\xff\xd8"
    fake_vlm.texts[(1, 0)] = "𝑥 слово 42"
    fake_vlm.low = {"42": 0.5}
    region = client.post(
        f"/api/pages/{ids[0]}/ocr-region",
        json={"x0": 0.1, "y0": 0.1, "x1": 0.6, "y1": 0.3},
        headers=HEADERS,
    )
    assert region.status_code == 200
    body = region.json()
    span = body["low_conf"][0]
    assert (
        body["text"].encode("utf-16-le")[span["start"] * 2 : span["end"] * 2].decode("utf-16-le")
        == "42"
    )
    fake_vlm.failures = [503]
    down = client.post(
        f"/api/pages/{ids[0]}/ocr-region",
        json={"x0": 0.1, "y0": 0.1, "x1": 0.6, "y1": 0.3},
        headers=HEADERS,
    )
    assert down.status_code == 503


def test_upload_duplicate_and_ingest_disabled(client: TestClient, tmp_path: Path) -> None:
    from tests.fixtures.factory import make_typewriter_pdf

    pdf, _ = make_typewriter_pdf(tmp_path, n_lines=5)
    files = [("files[]", ("скан.pdf", pdf.read_bytes(), "application/pdf"))]
    first = client.post("/api/documents", files=files, headers=HEADERS)
    assert first.status_code == 201 and first.json()["title"] == "скан"
    second = client.post("/api/documents", files=files, headers=HEADERS)
    assert second.json()["duplicate"] is True and second.json()["id"] == first.json()["id"]
    refused = client.post("/api/documents", files=files, data={"ingest": "true"}, headers=HEADERS)
    assert refused.status_code == 409 and refused.json()["detail"] == "ingester_disabled"
    doc_id = first.json()["id"]
    assert (
        client.post(f"/api/documents/{doc_id}/ingest", headers=HEADERS).json()["detail"]
        == "ingester_disabled"
    )
    assert (
        client.post(f"/api/documents/{doc_id}/draft", headers=HEADERS).json()["detail"]
        == "invalid_transition"
    )


async def test_draft_keeps_the_scope_the_review_started_with(
    conn, settings, tmp_path, fake_vlm, client
) -> None:
    doc_id, ids, _ = await _flagged_document(conn, settings, tmp_path, fake_vlm)
    fix = client.get(f"/api/documents/{doc_id}/pages/2/item", headers=HEADERS).json()
    assert fix["scope"] == "full"
    assert _put(client, ids[1], fix, fix["text"] + " правка", 0, scope="full").status_code == 200
    assert repo.get_draft(conn, ids[1])["scope"] == "full"
    queue = client.get("/api/review/queue", headers=HEADERS).json()
    assert queue[0]["page_id"] == ids[1] and queue[0]["scope"] == "full"
    assert _item(client, ids[1])["scope"] == "full"
    # Without the field the service falls back to the default scope of the page.
    client.delete(f"/api/pages/{ids[1]}/draft?draft_seq=1", headers=HEADERS)
    assert _put(client, ids[1], fix, "ещё", 0).status_code == 200
    assert repo.get_draft(conn, ids[1])["scope"] == "hints"
