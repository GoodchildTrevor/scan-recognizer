"""Status recomputation, the five rules of section 5.1 (p2)."""

from __future__ import annotations

import sqlite3

import pytest

from ocr_archive.db import repo
from ocr_archive.db.connection import immediate, utc_now
from ocr_archive.models import DocumentStatus, VlmResult
from ocr_archive.status import recompute_status, snapshot_key

pytestmark = pytest.mark.p2
RV = "rules-1"


def _doc(conn: sqlite3.Connection, statuses: list[str], doc_status: str = "processing") -> str:
    doc_id = f"doc{len(repo.list_documents(conn))}"
    with immediate(conn):
        repo.insert_document(
            conn,
            document_id=doc_id,
            title="t",
            source_sha256=doc_id,
            source_path="x.pdf",
            original_name="x.pdf",
            page_count=len(statuses),
            priority=0,
            ingest_requested=False,
        )
        repo.set_document_status(conn, doc_id, doc_status)
        conn.execute("INSERT OR IGNORE INTO profiles VALUES ('p', 'pass2', '{}', ?)", (utc_now(),))
        for row, status in zip(repo.pages_of_document(conn, doc_id), statuses, strict=True):
            attempt = repo.insert_attempt(
                conn,
                page_id=int(row["id"]),
                profile_id="p",
                generation=1,
                pass_no=2,
                role="full",
                region=None,
                result=VlmResult("текст", "stop", 1, 1, 1, None),
            )
            conn.execute(
                "UPDATE pages SET status = ?, track = 'vlm', current_attempt_id = ?, "
                "rules_version = ? WHERE id = ?",
                (status, attempt if status != "pending" else None, RV, row["id"]),
            )
    return doc_id


def _recompute(conn: sqlite3.Connection, doc_id: str, rv: str = RV) -> DocumentStatus:
    with immediate(conn):
        return recompute_status(conn, doc_id, rv)


def test_rule1_pending_processing(conn) -> None:
    assert _recompute(conn, _doc(conn, ["ok", "pending"])) == DocumentStatus.PROCESSING


def test_rule3_flagged_needs_review(conn) -> None:
    assert _recompute(conn, _doc(conn, ["ok", "flagged"])) == DocumentStatus.NEEDS_REVIEW
    assert _recompute(conn, _doc(conn, ["ok", "failed"])) == DocumentStatus.NEEDS_REVIEW


def test_reviewed_and_ok_ready(conn) -> None:
    assert _recompute(conn, _doc(conn, ["ok", "reviewed"])) == DocumentStatus.READY


def test_rule2_old_rules(conn) -> None:
    doc_id = _doc(conn, ["ok", "ok"])
    assert _recompute(conn, doc_id, "rules-2") == DocumentStatus.PROCESSING
    other = _doc(conn, ["ok", "ok"])
    page = repo.pages_of_document(conn, other)[0]
    with immediate(conn):
        repo.insert_draft(
            conn,
            int(page["id"]),
            {
                "item_version": "v",
                "base_attempt_id": None,
                "base_revision_id": None,
                "text": "x",
                "printed_label": None,
            },
        )
        conn.execute("UPDATE pages SET rules_version = 'old' WHERE document_id = ?", (other,))
        conn.execute(
            "UPDATE pages SET status = 'reviewed' WHERE id != ? AND document_id = ?",
            (page["id"], other),
        )
    assert _recompute(conn, other, "rules-2") == DocumentStatus.NEEDS_REVIEW


def test_rule4_done_when_snapshot_matches(conn) -> None:
    doc_id = _doc(conn, ["ok", "ok"], "ready")
    with immediate(conn):
        build_id, _ = repo.insert_build(conn, doc_id, "final", "running")
        repo.finish_build(
            conn,
            build_id,
            status="done",
            output_dir="o",
            manifest={"snapshot": snapshot_key(repo.page_texts(conn, doc_id))},
        )
    assert _recompute(conn, doc_id) == DocumentStatus.DONE
    with immediate(conn):
        conn.execute(
            "UPDATE pages SET printed_label = '9 об.' WHERE page_index = 0 AND document_id = ?",
            (doc_id,),
        )
    assert _recompute(conn, doc_id) == DocumentStatus.READY
    with immediate(conn):
        conn.execute("UPDATE pages SET printed_label = NULL WHERE document_id = ?", (doc_id,))
        repo.set_document_flag(conn, doc_id, "rebuild_requested", 1)
    assert _recompute(conn, doc_id) == DocumentStatus.READY
    with immediate(conn):
        repo.set_document_flag(conn, doc_id, "rebuild_requested", 0)
    assert _recompute(conn, doc_id) == DocumentStatus.DONE
    assert _recompute(conn, doc_id, "rules-2") == DocumentStatus.DONE  # built: no auto re-decide


def test_queued_building_failed_untouched(conn) -> None:
    for status in ("queued", "building", "failed"):
        doc_id = _doc(conn, ["ok"], status)
        assert _recompute(conn, doc_id) == status
    with immediate(conn):
        assert recompute_status(conn, doc_id, RV, force=True) == DocumentStatus.READY


def test_open_sample_needs_review(conn) -> None:
    doc_id = _doc(conn, ["ok", "ok"])
    with immediate(conn):
        repo.insert_sample(conn, int(repo.pages_of_document(conn, doc_id)[0]["id"]))
    assert _recompute(conn, doc_id) == DocumentStatus.NEEDS_REVIEW
