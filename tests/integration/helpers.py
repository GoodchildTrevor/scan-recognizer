"""Helpers for integration tests: documents from synthetic scans, a worker on the fake model."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from ocr_archive.db import repo
from ocr_archive.db.connection import immediate
from ocr_archive.ingest import SourceFile, create_document
from ocr_archive.models import BuildContext, DocumentStatus
from ocr_archive.settings import Settings
from ocr_archive.status import snapshot_key
from ocr_archive.worker import Worker
from tests.conftest import FakeVlm, make_ctx
from tests.fixtures.factory import (
    SyntheticPage,
    images_pdf,
    render_typewriter,
    typewriter_lines,
)


def synthetic_pages(count: int, *, n_lines: int = 12) -> list[SyntheticPage]:
    return [
        render_typewriter(typewriter_lines(n_lines, seed=100 + i), seed=i, skew_deg=0.0)
        for i in range(count)
    ]


def make_document(
    conn: sqlite3.Connection,
    settings: Settings,
    tmp_path: Path,
    pages: list[SyntheticPage],
    *,
    name: str = "doc.pdf",
    title: str = "Отчёт 1983",
) -> tuple[str, list[int]]:
    """Queue a scanned document made of synthetic pages; return its id and page ids."""
    pdf = images_pdf(tmp_path / name, [p.image for p in pages], dpi=pages[0].dpi)
    created = create_document(conn, settings, [SourceFile(name, pdf.read_bytes())], title=title)
    page_ids = [int(r["id"]) for r in repo.pages_of_document(conn, created.document_id)]
    return created.document_id, page_ids


def page_text(page: SyntheticPage) -> str:
    return "\n".join(page.lines)


def make_worker(conn: sqlite3.Connection, settings: Settings, fake: FakeVlm) -> Worker:
    ctx = make_ctx(conn, settings, fake)
    build_ctx = BuildContext(
        settings=settings,
        db_path=settings.database_path,
        outputs_dir=settings.outputs_dir,
        export_dir=settings.export_dir,
    )
    return Worker(settings=settings, ctx=ctx, build_ctx=build_ctx, conn=conn)


async def drain(worker: Worker, *, limit: int = 200) -> int:
    """Run document steps (recognition, re-deciding) until nothing is left; no builds."""
    steps = 0
    while steps < limit and await worker._document_step():
        steps += 1
    return steps


def doc_status(conn: sqlite3.Connection, document_id: str) -> str:
    row = repo.get_document(conn, document_id)
    assert row is not None
    return str(row["status"])


def page_row(conn: sqlite3.Connection, page_id: int) -> sqlite3.Row:
    row = repo.get_page(conn, page_id)
    assert row is not None
    return row


def mark_built(conn: sqlite3.Connection, document_id: str) -> int:
    """Record a successful final build of the current texts without running the build."""
    with immediate(conn):
        build_id, version = repo.insert_build(conn, document_id, "final", "running")
        manifest = {"snapshot": snapshot_key(repo.page_texts(conn, document_id)), "files": {}}
        repo.finish_build(
            conn,
            build_id,
            status="done",
            output_dir=f"outputs/{document_id}/v{version}",
            manifest=manifest,
        )
        repo.set_document_status(conn, document_id, DocumentStatus.DONE)
    return build_id


def flags(conn: sqlite3.Connection, page_id: int) -> list[str]:
    return list(json.loads(page_row(conn, page_id)["flags_json"]))
