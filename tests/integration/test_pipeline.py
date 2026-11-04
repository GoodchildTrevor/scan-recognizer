"""Pipeline with a fake llama-server (section 16, integration p2)."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest

from ocr_archive import operations, pipeline
from ocr_archive.db import repo
from ocr_archive.db.connection import immediate
from ocr_archive.exceptions import StatusTransitionError
from ocr_archive.ingest import SourceFile, create_document
from ocr_archive.models import DocumentStatus, PageStatus
from ocr_archive.pipeline import process_page, redecide_page, reprocess_document
from ocr_archive.status import RULES_VERSION_KEY
from ocr_archive.worker import Heartbeat, heartbeat_age_s, recompute_all
from tests.conftest import FakeVlm, make_ctx, make_settings
from tests.fixtures.factory import make_text_pdf_with_table
from tests.integration.helpers import (
    doc_status,
    drain,
    flags,
    make_document,
    make_worker,
    mark_built,
    page_row,
    page_text,
    synthetic_pages,
)

pytestmark = pytest.mark.p2


class SimulatedKill(BaseException):
    """Stands for the process being killed in the middle of a request."""


def _script(fake: FakeVlm, pages: list) -> None:
    for number, page in enumerate(pages, start=1):
        fake.texts[(number, 0)] = page_text(page)


async def test_three_pages_reach_needs_review(
    conn: sqlite3.Connection, settings, tmp_path: Path, fake_vlm: FakeVlm
) -> None:
    pages = synthetic_pages(3)
    doc_id, ids = make_document(conn, settings, tmp_path, pages)
    _script(fake_vlm, pages)
    second = page_text(pages[1])
    fake_vlm.texts[(2, 2)] = second.replace(pages[1].lines[3], pages[1].lines[3] + " 125")
    fake_vlm.texts[(2, 1)] = second.replace(pages[1].lines[3], pages[1].lines[3] + " 126")
    third = page_text(pages[2]) + " [неразб.]"
    fake_vlm.texts[(3, 0)] = third
    worker = make_worker(conn, settings, fake_vlm)
    await drain(worker)
    assert doc_status(conn, doc_id) == DocumentStatus.NEEDS_REVIEW
    assert page_row(conn, ids[0])["status"] == PageStatus.OK
    assert "digits_disagreement" in flags(conn, ids[1])
    assert "markers" in flags(conn, ids[2])
    view = operations.document_view(conn, doc_id, settings)
    assert view["no_logprobs_pages"] == 0


async def test_without_logprobs(conn, settings, tmp_path: Path, fake_vlm: FakeVlm) -> None:
    pages = synthetic_pages(3)
    doc_id, _ = make_document(conn, settings, tmp_path, pages)
    _script(fake_vlm, pages)
    fake_vlm.logprobs = False
    await drain(make_worker(conn, settings, fake_vlm))
    assert doc_status(conn, doc_id) == DocumentStatus.READY
    assert operations.document_view(conn, doc_id, settings)["no_logprobs_pages"] == 3


async def test_pdf_text_only_document_without_model(
    conn, settings, tmp_path: Path, fake_vlm: FakeVlm
) -> None:
    pdf = make_text_pdf_with_table(tmp_path)
    created = create_document(conn, settings, [SourceFile("text.pdf", pdf.read_bytes())])
    await drain(make_worker(conn, settings, fake_vlm))
    assert doc_status(conn, created.document_id) == DocumentStatus.READY
    assert fake_vlm.calls == []
    page = repo.pages_of_document(conn, created.document_id)[0]
    assert page["track"] == "pdf_text" and page["status"] == PageStatus.OK


async def test_resume_after_kill_reuses_first_pass(
    conn, settings, tmp_path: Path, fake_vlm: FakeVlm
) -> None:
    pages = synthetic_pages(2)
    _, ids = make_document(conn, settings, tmp_path, pages)
    _script(fake_vlm, pages)
    ctx = make_ctx(conn, settings, fake_vlm)
    await process_page(ids[0], ctx)

    def kill_on_page2_pass2(body: dict) -> None:
        if fake_vlm.key_for(body) == (2, 2):
            raise SimulatedKill

    fake_vlm.hook = kill_on_page2_pass2
    with pytest.raises(SimulatedKill):
        await process_page(ids[1], ctx)
    fake_vlm.hook = None
    fake_vlm.calls.clear()
    status = await process_page(ids[1], make_ctx(conn, settings, fake_vlm))
    assert status == PageStatus.OK
    assert fake_vlm.keys_called() == [(2, 2)]
    attempts = conn.execute("SELECT COUNT(*) FROM attempts WHERE page_id = ?", (ids[1],)).fetchone()
    assert attempts[0] == 2


async def test_unavailable_model_keeps_page_pending(
    conn, settings, tmp_path: Path, fake_vlm: FakeVlm
) -> None:
    pages = synthetic_pages(1)
    _, ids = make_document(conn, settings, tmp_path, pages)
    _script(fake_vlm, pages)
    fake_vlm.failures = [503, "timeout"]
    ctx = make_ctx(conn, settings, fake_vlm)
    assert await process_page(ids[0], ctx) == PageStatus.PENDING
    row = page_row(conn, ids[0])
    assert row["status"] == PageStatus.PENDING and row["retry_at"] is not None
    assert await process_page(ids[0], ctx) == PageStatus.PENDING
    assert page_row(conn, ids[0])["retry_count"] == 2
    assert await process_page(ids[0], ctx) == PageStatus.OK
    assert page_row(conn, ids[0])["retry_count"] == 0


async def test_permanent_error_fails_page_and_retry_restarts(
    conn, settings, tmp_path: Path, fake_vlm: FakeVlm
) -> None:
    pages = synthetic_pages(2)
    doc_id, ids = make_document(conn, settings, tmp_path, pages)
    _script(fake_vlm, pages)
    fake_vlm.failures = [(422, "bad image")]
    worker = make_worker(conn, settings, fake_vlm)
    await drain(worker)
    assert page_row(conn, ids[0])["status"] == PageStatus.FAILED
    assert page_row(conn, ids[0])["current_attempt_id"] is not None  # pass 2 still worked
    assert page_row(conn, ids[1])["status"] == PageStatus.OK
    assert doc_status(conn, doc_id) == DocumentStatus.NEEDS_REVIEW
    assert operations.retry_page(conn, ids[0]) == PageStatus.PENDING
    assert doc_status(conn, doc_id) == DocumentStatus.PROCESSING
    assert page_row(conn, ids[0])["ocr_generation"] == 2
    await drain(worker)
    assert page_row(conn, ids[0])["status"] == PageStatus.OK
    assert doc_status(conn, doc_id) == DocumentStatus.READY


async def test_unexpected_exception_fails_only_that_page(
    conn, settings, tmp_path: Path, fake_vlm: FakeVlm, monkeypatch: pytest.MonkeyPatch
) -> None:
    pages = synthetic_pages(2)
    doc_id, ids = make_document(conn, settings, tmp_path, pages)
    _script(fake_vlm, pages)
    original = pipeline.decide

    def broken_decide(page, *args, **kwargs):  # type: ignore[no-untyped-def]
        if page.page_id == ids[0]:
            raise ZeroDivisionError("boom")
        return original(page, *args, **kwargs)

    monkeypatch.setattr(pipeline, "decide", broken_decide)
    await drain(make_worker(conn, settings, fake_vlm))
    assert page_row(conn, ids[0])["status"] == PageStatus.FAILED
    assert page_row(conn, ids[1])["status"] == PageStatus.OK
    assert doc_status(conn, doc_id) == DocumentStatus.NEEDS_REVIEW


async def test_rules_change_redecides_unbuilt_only(
    conn, settings, tmp_path: Path, fake_vlm: FakeVlm
) -> None:
    pages = synthetic_pages(2)
    doc_a, ids_a = make_document(conn, settings, tmp_path, pages, name="a.pdf")
    pdf = make_text_pdf_with_table(tmp_path)
    doc_b = create_document(conn, settings, [SourceFile("b.pdf", pdf.read_bytes())]).document_id
    doc_c, ids_c = make_document(conn, settings, tmp_path, pages[:1], name="c.pdf")
    _script(fake_vlm, pages)
    first = page_text(pages[0])
    fake_vlm.texts[(1, 1)] = first.replace("а", "о", 1)  # one word differs: 2 < 4
    worker = make_worker(conn, settings, fake_vlm)
    await drain(worker)
    assert doc_status(conn, doc_a) == DocumentStatus.READY
    mark_built(conn, doc_c)
    old_version = page_row(conn, ids_a[0])["rules_version"]

    strict = make_settings(tmp_path, signals={"min_diff_words": 1})
    worker2 = make_worker(conn, strict, fake_vlm)
    recompute_all(conn, worker2.ctx.rules_version)
    assert doc_status(conn, doc_a) == DocumentStatus.PROCESSING
    assert doc_status(conn, doc_b) == DocumentStatus.READY  # pdf_text pages never re-decided
    assert doc_status(conn, doc_c) == DocumentStatus.DONE
    fake_vlm.calls.clear()
    await drain(worker2)
    assert fake_vlm.calls == []
    assert "pass_diff" in flags(conn, ids_a[0])
    assert page_row(conn, ids_a[1])["rules_version"] != old_version
    assert doc_status(conn, doc_a) == DocumentStatus.NEEDS_REVIEW
    assert page_row(conn, ids_c[0])["rules_version"] == old_version
    operations.request_redecide(conn, doc_c)
    await drain(worker2)
    assert page_row(conn, ids_c[0])["rules_version"] == worker2.ctx.rules_version
    assert not repo.get_document(conn, doc_c)["redecide_requested"]


async def test_redecide_skips_page_reviewed_meanwhile(
    conn, settings, tmp_path: Path, fake_vlm: FakeVlm
) -> None:
    pages = synthetic_pages(1)
    doc_id, ids = make_document(conn, settings, tmp_path, pages)
    _script(fake_vlm, pages)
    worker = make_worker(conn, settings, fake_vlm)
    await drain(worker)
    stale_row = page_row(conn, ids[0])
    view = operations.build_item_view(conn, ids[0], settings, worker.ctx.rules_version)
    operations.submit_review(
        conn,
        ids[0],
        operations.ReviewInput(
            request_id="req-00000001",
            item_version=view.item_version,
            base_attempt_id=view.base_attempt_id,
            base_revision_id=None,
            draft_seq=0,
            text=view.text,
        ),
    )
    written = redecide_page(
        conn,
        stale_row,
        worker.ctx,
        settings.data_dir / str(repo.get_document(conn, doc_id)["source_path"]),
    )
    assert not written
    assert page_row(conn, ids[0])["status"] == PageStatus.REVIEWED


async def test_reprocess_during_recognition_drops_result(
    conn, settings, tmp_path: Path, fake_vlm: FakeVlm
) -> None:
    pages = synthetic_pages(1)
    doc_id, ids = make_document(conn, settings, tmp_path, pages)
    _script(fake_vlm, pages)
    ctx = make_ctx(conn, settings, fake_vlm)
    pipeline.classify_pages(conn, doc_id, settings)

    def bump_generation(body: dict) -> None:
        if fake_vlm.key_for(body) == (1, 2):
            with immediate(conn):
                conn.execute(
                    "UPDATE pages SET ocr_generation = ocr_generation + 1 WHERE id = ?", (ids[0],)
                )

    fake_vlm.hook = bump_generation
    assert await process_page(ids[0], ctx) == PageStatus.PENDING
    assert page_row(conn, ids[0])["status"] == PageStatus.PENDING
    fake_vlm.hook = None
    fake_vlm.calls.clear()
    assert await process_page(ids[0], ctx) == PageStatus.OK
    assert len(fake_vlm.calls) == 2  # a new request for both passes


async def test_reprocess_document(conn, settings, tmp_path: Path, fake_vlm: FakeVlm) -> None:
    pages = synthetic_pages(3)
    doc_id, ids = make_document(conn, settings, tmp_path, pages)
    _script(fake_vlm, pages)
    fake_vlm.texts[(3, 0)] = page_text(pages[2]) + " [неразб.]"
    worker = make_worker(conn, settings, fake_vlm)
    await drain(worker)
    view = operations.build_item_view(conn, ids[0], settings, worker.ctx.rules_version)
    operations.submit_review(
        conn,
        ids[0],
        operations.ReviewInput(
            request_id="req-00000002",
            item_version=view.item_version,
            base_attempt_id=view.base_attempt_id,
            base_revision_id=None,
            draft_seq=0,
            text=view.text + " правка",
        ),
    )
    operations.save_draft(
        conn,
        ids[2],
        operations.DraftInput(
            expected_seq=0,
            item_version="x",
            base_attempt_id=page_row(conn, ids[2])["current_attempt_id"],
            base_revision_id=None,
            text="черновик",
        ),
    )
    status = reprocess_document(doc_id, worker.ctx)
    assert status == DocumentStatus.PROCESSING
    assert page_row(conn, ids[0])["status"] == PageStatus.REVIEWED
    assert page_row(conn, ids[1])["status"] == PageStatus.PENDING
    assert repo.get_draft(conn, ids[2]) is None
    reprocess_document(doc_id, worker.ctx, include_reviewed=True)
    row = page_row(conn, ids[0])
    assert row["status"] == PageStatus.PENDING and row["current_revision_id"] is None
    fake_vlm.calls.clear()
    await drain(worker)
    assert len(fake_vlm.calls) == 6
    with immediate(conn):
        repo.set_document_status(conn, doc_id, DocumentStatus.BUILDING)
    with pytest.raises(StatusTransitionError):
        reprocess_document(doc_id, worker.ctx)


def test_heartbeat_during_blocking_work(conn, settings) -> None:
    beat = Heartbeat(settings.database_path, interval=0.2)
    beat.start()
    try:
        deadline = time.monotonic() + 1.5
        while time.monotonic() < deadline:
            sum(range(10000))  # synchronous work in the main thread
        age = heartbeat_age_s(conn)
        assert age is not None and age < 2
    finally:
        beat.stop()


def test_rules_version_recorded(conn, settings, fake_vlm: FakeVlm) -> None:
    ctx = make_ctx(conn, settings, fake_vlm)
    assert repo.get_control(conn, RULES_VERSION_KEY) == ctx.rules_version


async def test_context_overflow_retries_pass2_split(
    conn, settings, tmp_path: Path, fake_vlm: FakeVlm
) -> None:
    pages = synthetic_pages(1, n_lines=14)
    _, ids = make_document(conn, settings, tmp_path, pages)
    lines = pages[0].lines
    fake_vlm.texts[(1, 0)] = page_text(pages[0])
    fake_vlm.texts[(1, ("part", 1))] = "\n".join(lines[:7])
    fake_vlm.texts[(1, ("part", 2))] = "\n".join(lines[7:])
    ctx = make_ctx(conn, settings, fake_vlm)
    fake_vlm.hook = None

    def overflow_once(body: dict) -> None:
        if fake_vlm.key_for(body) == (1, 2) and not fake_vlm.failures and len(fake_vlm.calls) == 2:
            fake_vlm.failures.append((400, "the request exceeds the available context size"))

    fake_vlm.hook = overflow_once
    status = await process_page(ids[0], ctx)
    keys = fake_vlm.keys_called()
    assert (1, ("part", 1)) in keys and (1, ("part", 2)) in keys
    assert status in (PageStatus.OK, PageStatus.FLAGGED)
    roles = [
        r["role"]
        for r in conn.execute("SELECT role FROM attempts WHERE page_id = ? ORDER BY id", (ids[0],))
    ]
    assert roles == ["full", "part", "part", "combined"]


async def test_repeated_timeouts_fail_the_page(
    conn, settings, tmp_path: Path, fake_vlm: FakeVlm
) -> None:
    pages = synthetic_pages(1)
    _, ids = make_document(conn, settings, tmp_path, pages)
    _script(fake_vlm, pages)
    ctx = make_ctx(conn, settings, fake_vlm)
    fake_vlm.failures = ["timeout"] * (pipeline.MAX_TIMEOUT_RETRIES + 1)
    for _ in range(pipeline.MAX_TIMEOUT_RETRIES):
        assert await process_page(ids[0], ctx) == PageStatus.PENDING
    assert page_row(conn, ids[0])["retry_count"] == pipeline.MAX_TIMEOUT_RETRIES
    assert await process_page(ids[0], ctx) == PageStatus.FAILED
    row = page_row(conn, ids[0])
    assert row["status"] == PageStatus.FAILED and "timeouts" in str(row["error"])
    assert operations.retry_page(conn, ids[0]) == PageStatus.PENDING
    assert await process_page(ids[0], ctx) == PageStatus.OK


async def test_connection_errors_are_retried_without_limit(
    conn, settings, tmp_path: Path, fake_vlm: FakeVlm
) -> None:
    pages = synthetic_pages(1)
    _, ids = make_document(conn, settings, tmp_path, pages)
    _script(fake_vlm, pages)
    ctx = make_ctx(conn, settings, fake_vlm)
    attempts = pipeline.MAX_TIMEOUT_RETRIES + 2
    fake_vlm.failures = [503] * attempts
    for _ in range(attempts):
        assert await process_page(ids[0], ctx) == PageStatus.PENDING
    row = page_row(conn, ids[0])
    assert row["status"] == PageStatus.PENDING and row["retry_count"] == attempts


async def test_reprocess_takes_pdf_text_pages_only_on_request(
    conn, settings, tmp_path: Path, fake_vlm: FakeVlm
) -> None:
    pdf = make_text_pdf_with_table(tmp_path)
    created = create_document(conn, settings, [SourceFile("text.pdf", pdf.read_bytes())])
    worker = make_worker(conn, settings, fake_vlm)
    await drain(worker)
    page_id = int(repo.pages_of_document(conn, created.document_id)[0]["id"])
    first_attempt = page_row(conn, page_id)["current_attempt_id"]

    reprocess_document(created.document_id, worker.ctx)
    assert page_row(conn, page_id)["status"] == PageStatus.OK

    status = reprocess_document(created.document_id, worker.ctx, include_pdf_text=True)
    assert status == DocumentStatus.PROCESSING
    assert page_row(conn, page_id)["status"] == PageStatus.PENDING
    await drain(worker)
    row = page_row(conn, page_id)
    assert row["status"] == PageStatus.OK
    assert row["current_attempt_id"] != first_attempt
    assert fake_vlm.calls == []
    assert doc_status(conn, created.document_id) == DocumentStatus.READY
