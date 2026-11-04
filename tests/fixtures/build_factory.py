"""Database helpers for build tests: documents with pages in given states, finished builds."""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ulid import ULID

from ocr_archive.db import repo
from ocr_archive.db.connection import immediate, utc_now
from ocr_archive.models import BuildContext, VlmResult
from ocr_archive.settings import Settings
from ocr_archive.status import snapshot_key

PROFILE_ID = "test-profile"


@dataclass
class PageSpec:
    """State of one page of a test document.

    :ivar text: candidate text (an attempt), or None for a page without a candidate.
    :ivar revision_text: text of a "Done" revision; the page becomes ``reviewed``.
    """

    text: str | None
    status: str = "ok"
    track: str = "vlm"
    text_layer: str = "none"
    revision_text: str | None = None
    printed_label: str | None = None
    flags: list[str] = field(default_factory=list)
    review: dict[str, Any] | None = None
    line_boxes_json: str | None = None


def build_context(settings: Settings) -> BuildContext:
    return BuildContext(
        settings=settings,
        db_path=settings.database_path,
        outputs_dir=settings.outputs_dir,
        export_dir=settings.export_dir,
    )


def create_document(
    conn: sqlite3.Connection,
    settings: Settings,
    source_pdf: Path,
    pages: list[PageSpec],
    *,
    title: str = "Отчёт 1983",
    status: str = "ready",
    ingest: bool = False,
) -> tuple[str, list[int]]:
    """Insert a document with attempts and revisions as described; return its id and page ids."""
    sha = hashlib.sha256(source_pdf.read_bytes() + uuid.uuid4().bytes).hexdigest()
    target = settings.data_dir / "sources" / sha / "source.pdf"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_pdf, target)
    document_id = str(ULID())
    with immediate(conn):
        repo.ensure_profile(conn, PROFILE_ID, "pass2", "{}")
        repo.insert_document(
            conn,
            document_id=document_id,
            title=title,
            source_sha256=sha,
            source_path=f"sources/{sha}/source.pdf",
            original_name=source_pdf.name,
            page_count=len(pages),
            priority=0,
            ingest_requested=ingest,
        )
        repo.set_document_status(conn, document_id, status)
        page_ids = [int(row["id"]) for row in repo.pages_of_document(conn, document_id)]
        for page_id, spec in zip(page_ids, pages, strict=True):
            attempt_id = None
            if spec.text is not None:
                pdf_text = spec.track == "pdf_text"
                attempt_id = repo.insert_attempt(
                    conn,
                    page_id=page_id,
                    profile_id=PROFILE_ID,
                    generation=1,
                    pass_no=0 if pdf_text else 2,
                    role="pdf_text" if pdf_text else "full",
                    region=None,
                    result=VlmResult(spec.text, "stop", 10, 10, 5, None),
                )
            conn.execute(
                "UPDATE pages SET status = ?, track = ?, text_layer = ?, current_attempt_id = ?, "
                "flags_json = ?, review_json = ?, line_boxes_json = ?, printed_label = ?, "
                "rules_version = 'rules-test' WHERE id = ?",
                (
                    spec.status,
                    spec.track,
                    spec.text_layer,
                    attempt_id,
                    json.dumps(spec.flags),
                    None if spec.review is None else json.dumps(spec.review),
                    spec.line_boxes_json,
                    spec.printed_label,
                    page_id,
                ),
            )
            if spec.revision_text is not None:
                revision_id = repo.insert_revision(
                    conn,
                    page_id=page_id,
                    request_id=str(uuid.uuid4()),
                    base_attempt_id=attempt_id,
                    base_revision_id=None,
                    action="edit" if spec.revision_text != spec.text else "accept",
                    scope="hints",
                    text=spec.revision_text,
                    printed_label=spec.printed_label,
                    choices=None,
                    duration_s=30,
                )
                repo.mark_page_reviewed(conn, page_id, revision_id, spec.printed_label)
    return document_id, page_ids


def add_final_build(
    conn: sqlite3.Connection, settings: Settings, document_id: str, files: dict[str, bytes]
) -> int:
    """Record a successful final build with the given files (role → content); return its version.

    File names follow the build: ``<title>.pdf`` and ``<title>.docx``.
    """
    from ocr_archive.build.export import safe_name

    with immediate(conn):
        document = repo.get_document(conn, document_id)
        assert document is not None
        build_id, version = repo.insert_build(conn, document_id, "final", "running")
        base = safe_name(str(document["title"]), document_id)
        names = {"pdf": f"{base}.pdf", "docx": f"{base}.docx", "chunks": "chunks.jsonl"}
        out_dir = settings.outputs_dir / document_id / f"v{version}"
        out_dir.mkdir(parents=True, exist_ok=True)
        for role, content in files.items():
            (out_dir / names[role]).write_bytes(content)
        rows = repo.page_texts(conn, document_id)
        snapshot = [
            {
                **entry,
                "flags": json.loads(row["flags_json"]),
                "track": row["track"],
                "edited": row["revision_text"] is not None
                and row["revision_text"] != row["attempt_text"],
            }
            for entry, row in zip(snapshot_key(rows), rows, strict=True)
        ]
        manifest = {
            "files": {role: names[role] for role in files},
            "snapshot": snapshot,
            "created_at": utc_now(),
        }
        repo.finish_build(
            conn,
            build_id,
            status="done",
            output_dir=f"outputs/{document_id}/v{version}",
            manifest=manifest,
        )
        repo.set_document_status(conn, document_id, "done")
    return version
