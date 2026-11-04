"""Document status recomputation (section 5.1) and the page text snapshot it relies on."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from typing import Any

from ocr_archive.db import repo
from ocr_archive.models import DocumentStatus

RULES_VERSION_KEY = "rules_version"

#: Statuses the recomputation never touches.
_FROZEN = (DocumentStatus.QUEUED, DocumentStatus.BUILDING)


def page_text(row: sqlite3.Row) -> str | None:
    """Text of a page for the build: revision, else candidate attempt, else none (section 5.2).

    :param row: a row of :func:`repo.page_texts`.
    """
    if row["current_revision_id"] is not None:
        return str(row["revision_text"])
    if row["current_attempt_id"] is not None:
        return str(row["attempt_text"])
    return None


def text_sha256(text: str | None) -> str:
    """sha256 of the UTF-8 text; a page without text hashes as the empty string."""
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def snapshot_key(rows: list[sqlite3.Row]) -> list[dict[str, Any]]:
    """Per-page identity used by rule 4: page index, text hash and printed label."""
    return [
        {
            "page_index": int(row["page_index"]),
            "text_sha256": text_sha256(page_text(row)),
            "printed_label": row["printed_label"],
        }
        for row in rows
    ]


def snapshot_matches(manifest: dict[str, Any], rows: list[sqlite3.Row]) -> bool:
    """Tell whether the current texts and labels equal the snapshot of a build manifest.

    :param manifest: parsed ``builds.manifest_json``; its ``snapshot`` list holds per-page
        ``page_index``, ``text_sha256`` and ``printed_label``.
    """
    stored = [
        {
            "page_index": int(entry["page_index"]),
            "text_sha256": entry["text_sha256"],
            "printed_label": entry.get("printed_label"),
        }
        for entry in manifest.get("snapshot", [])
    ]
    return stored == snapshot_key(rows)


def current_rules_version(conn: sqlite3.Connection) -> str | None:
    """Rules version recorded by the worker at start (section 6.11)."""
    return repo.get_control(conn, RULES_VERSION_KEY)


def recompute_status(
    conn: sqlite3.Connection,
    document_id: str,
    rules_version: str | None = None,
    *,
    force: bool = False,
) -> DocumentStatus:
    """Recompute the document status by the five rules of section 5.1.

    Must run inside ``BEGIN IMMEDIATE``. Documents in ``queued`` and ``building`` are never
    touched; a ``failed`` document only when ``force`` is set (rebuild).

    :param rules_version: current rules version; read from ``control`` when omitted.
    :param force: also recompute a failed document.
    :return: the resulting status.
    """
    doc = repo.get_document(conn, document_id)
    if doc is None:
        raise KeyError(document_id)
    status = DocumentStatus(doc["status"])
    if status in _FROZEN or (status == DocumentStatus.FAILED and not force):
        return status

    version = rules_version or current_rules_version(conn)
    inputs = repo.status_inputs(conn, document_id)
    built_row = repo.latest_final_snapshot_build(conn, document_id)
    to_redecide = repo.count_pages_to_redecide(conn, document_id, version) if version else 0
    redecide_requested = bool(doc["redecide_requested"])
    if redecide_requested and to_redecide == 0:
        waiting = (
            repo.count_draft_pages_waiting_redecide(conn, document_id, version) if version else 0
        )
        if waiting == 0:
            repo.set_document_flag(conn, document_id, "redecide_requested", 0)
            redecide_requested = False

    if inputs["pending"]:
        new = DocumentStatus.PROCESSING
    elif (built_row is None or redecide_requested) and to_redecide:
        new = DocumentStatus.PROCESSING
    elif inputs["open_flags"] or inputs["drafts"] or inputs["samples_open"]:
        new = DocumentStatus.NEEDS_REVIEW
    elif (
        built_row is not None
        and not doc["rebuild_requested"]
        and snapshot_matches(
            json.loads(built_row["manifest_json"] or "{}"), repo.page_texts(conn, document_id)
        )
    ):
        new = DocumentStatus.DONE
    else:
        new = DocumentStatus.READY

    if new != status:
        repo.set_document_status(conn, document_id, new)
    return new
