"""Build runner (section 9): snapshot, files, verification, atomic publication, pruning."""

from __future__ import annotations

import logging
import os
import shutil
import sqlite3
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ocr_archive.build.chunks import ChunkPage, build_chunks, write_chunks
from ocr_archive.build.docx import build_docx, pandoc_version, read_docx_ast, set_core_properties
from ocr_archive.build.export import output_path, safe_name
from ocr_archive.build.manifest import build_manifest, tool_versions, write_manifest
from ocr_archive.build.markdown import (
    NOTE_UNRECOGNIZED,
    NOTE_UNVERIFIED,
    MdPage,
    to_pandoc_markdown,
)
from ocr_archive.build.normalize import (
    HINT_CLOSE,
    HINT_OPEN,
    document_vocabulary,
    normalize_page,
    strip_hints,
)
from ocr_archive.build.pdf import PdfPageInput, PdfPageResult, build_searchable_pdf
from ocr_archive.build.verify import PdfPageExpectation, verify_docx, verify_pdf
from ocr_archive.db import repo
from ocr_archive.db.connection import connect, immediate, read_snapshot
from ocr_archive.exceptions import BuildError, StatusTransitionError
from ocr_archive.models import (
    BuildContext,
    BuildKind,
    BuildStatus,
    DocumentStatus,
    IngestStatus,
    PageStatus,
    PageTrack,
)
from ocr_archive.settings import resolve_asset
from ocr_archive.status import current_rules_version, page_text, recompute_status, text_sha256

logger = logging.getLogger(__name__)

DRAFT_SUFFIX = " (черновик)"
DRAFT_TITLE_PREFIX = "ЧЕРНОВИК — "
DRAFT_SUBJECT = "не проверено"
CHUNKS_NAME = "chunks.jsonl"
MANIFEST_NAME = "manifest.json"
MARKDOWN_NAME = "document.md"

#: Errors of the build steps that are reported as a failed build.
_STEP_ERRORS = (
    OSError,
    RuntimeError,
    ValueError,
    subprocess.SubprocessError,
    sqlite3.Error,
    zipfile.BadZipFile,
)


@dataclass(frozen=True)
class SnapshotPage:
    page_id: int
    page_index: int
    text: str | None
    source: str | None  # revision | attempt | None
    source_id: int | None
    attempt_text: str | None
    printed_label: str | None
    status: str
    track: str | None
    text_layer: str | None
    profile_id: str | None
    rules_version: str | None
    flags: list[str]
    signals: dict[str, Any] | None
    review: dict[str, Any] | None
    review_scope: str | None
    line_boxes_json: str | None
    open_item: bool

    @property
    def edited(self) -> bool:
        """The page has a revision whose text differs from the candidate."""
        return self.source == "revision" and self.text != self.attempt_text


@dataclass(frozen=True)
class Snapshot:
    document_id: str
    title: str
    source_pdf: Path
    rules_version: str | None
    pages: list[SnapshotPage]


def _read_snapshot(conn: sqlite3.Connection, ctx: BuildContext, document_id: str) -> Snapshot:
    """Read every page text in one read transaction (section 9, "Снимок")."""
    with read_snapshot(conn):
        document = repo.get_document(conn, document_id)
        if document is None:
            raise BuildError(f"document {document_id} not found")
        pages: list[SnapshotPage] = []
        for row in repo.page_texts(conn, document_id):
            text = page_text(row)
            if row["current_revision_id"] is not None:
                source, source_id = "revision", int(row["current_revision_id"])
            elif row["current_attempt_id"] is not None:
                source, source_id = "attempt", int(row["current_attempt_id"])
            else:
                source, source_id = None, None
            status = str(row["status"])
            open_item = status in (PageStatus.FLAGGED, PageStatus.FAILED) or repo.open_sample(
                conn, int(row["page_id"])
            )
            pages.append(
                SnapshotPage(
                    page_id=int(row["page_id"]),
                    page_index=int(row["page_index"]),
                    text=text,
                    source=source,
                    source_id=source_id,
                    attempt_text=row["attempt_text"],
                    printed_label=row["printed_label"],
                    status=status,
                    track=row["track"],
                    text_layer=row["text_layer"],
                    profile_id=row["attempt_profile_id"],
                    rules_version=row["rules_version"],
                    flags=repo.loads(row["flags_json"], []),
                    signals=repo.loads(row["signals_json"]),
                    review=repo.loads(row["review_json"]),
                    review_scope=row["revision_scope"],
                    line_boxes_json=row["line_boxes_json"],
                    open_item=open_item,
                )
            )
        return Snapshot(
            document_id=document_id,
            title=str(document["title"]),
            source_pdf=ctx.settings.data_dir / str(document["source_path"]),
            rules_version=current_rules_version(conn),
            pages=pages,
        )


def _snapshot_entries(
    snapshot: Snapshot, pdf_results: dict[int, PdfPageResult] | None
) -> list[dict[str, Any]]:
    entries = []
    for page in snapshot.pages:
        result = pdf_results.get(page.page_index) if pdf_results else None
        entries.append(
            {
                "page_index": page.page_index,
                "page_id": page.page_id,
                "source": page.source,
                "source_id": page.source_id,
                "text_sha256": text_sha256(page.text),
                "printed_label": page.printed_label,
                "track": page.track,
                "text_layer": page.text_layer,
                "profile_id": page.profile_id,
                "rules_version": page.rules_version,
                "flags": page.flags,
                "signals": page.signals,
                "review_scope": page.review_scope,
                "edited": page.edited,
                "pdf_layer": result.pdf_layer if result else None,
                "old_text_kept": result.old_text_kept if result else False,
                "char_replacements": result.char_replacements if result else {},
                "pdf_text_edited": page.track == PageTrack.PDF_TEXT and page.source == "revision",
            }
        )
    return entries


def _mark_hints(text: str, review: dict[str, Any] | None) -> str:
    """Wrap the hint ranges of the review item in private-use marks (section 9.6)."""
    if not review:
        return text
    ranges = sorted(
        (int(hint["start"]), int(hint["end"]))
        for hint in review.get("hints", [])
        if 0 <= int(hint["start"]) < int(hint["end"]) <= len(text)
    )
    merged: list[tuple[int, int]] = []
    for start, end in ranges:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    for start, end in reversed(merged):
        text = text[:start] + HINT_OPEN + text[start:end] + HINT_CLOSE + text[end:]
    return text


def _docx(
    snapshot: Snapshot,
    work_dir: Path,
    docx_name: str,
    pdf_name: str,
    ctx: BuildContext,
    *,
    draft: bool,
) -> None:
    """Normalize, write Markdown, run pandoc and verify the DOCX (sections 9.1, 9.2, 9.5)."""
    texts: list[str] = []
    notes: list[str | None] = []
    for page in snapshot.pages:
        text = page.text or ""
        note = None
        if draft and page.open_item:
            note = NOTE_UNVERIFIED if page.text is not None else NOTE_UNRECOGNIZED
            if page.source == "attempt":
                text = _mark_hints(text, page.review)
        texts.append(text)
        notes.append(note)
    vocabulary = document_vocabulary(strip_hints(text) for text in texts)
    md_pages = [
        MdPage(page.page_index, page.printed_label, normalize_page(text, vocabulary), note)
        for page, text, note in zip(snapshot.pages, texts, notes, strict=True)
    ]
    markdown, references = to_pandoc_markdown(snapshot.title, md_pages, pdf_name, draft=draft)
    markdown_path = work_dir / MARKDOWN_NAME
    markdown_path.write_text(markdown, encoding="utf-8")
    docx_path = work_dir / docx_name
    build_docx(markdown_path, docx_path, ctx.settings.build)
    markdown_path.unlink()
    if draft:
        set_core_properties(
            docx_path, title=DRAFT_TITLE_PREFIX + snapshot.title, subject=DRAFT_SUBJECT
        )
    ast = read_docx_ast(docx_path, ctx.settings.build)
    verify_docx(ast, references, ctx.settings.build.docx_min_similarity)


def _final_files(
    snapshot: Snapshot, work_dir: Path, version: int, ctx: BuildContext
) -> dict[str, Any]:
    base = safe_name(snapshot.title, snapshot.document_id)
    files = {"pdf": f"{base}.pdf", "docx": f"{base}.docx", "chunks": CHUNKS_NAME}
    _docx(snapshot, work_dir, files["docx"], files["pdf"], ctx, draft=False)

    pdf_inputs = [
        PdfPageInput(
            page.page_index, page.track or "", page.text_layer, page.text, page.line_boxes_json
        )
        for page in snapshot.pages
    ]
    pdf_path = work_dir / files["pdf"]
    results = build_searchable_pdf(
        snapshot.source_pdf, pdf_path, pdf_inputs, resolve_asset(ctx.settings.build.font_path)
    )
    verify_pdf(
        snapshot.source_pdf,
        pdf_path,
        [PdfPageExpectation(r.page_index, r.expected_text) for r in results],
    )

    vocabulary = document_vocabulary(page.text or "" for page in snapshot.pages)
    chunk_pages = [
        ChunkPage(
            page.page_index,
            page.printed_label,
            normalize_page(page.text or "", vocabulary),
            reviewed=page.source == "revision",
        )
        for page in snapshot.pages
    ]
    chunks = build_chunks(
        document_id=snapshot.document_id,
        title=snapshot.title,
        build_version=version,
        pages=chunk_pages,
        max_chars=ctx.settings.chunks.max_chars,
    )
    write_chunks(work_dir / files["chunks"], chunks)

    return build_manifest(
        document_id=snapshot.document_id,
        title=snapshot.title,
        kind=BuildKind.FINAL,
        version=version,
        rules_version=snapshot.rules_version,
        snapshot=_snapshot_entries(snapshot, {r.page_index: r for r in results}),
        files=files,
        output_dir=work_dir,
        versions=tool_versions(ctx.settings, pandoc_version(ctx.settings.build)),
    )


def _draft_files(
    snapshot: Snapshot, work_dir: Path, version: int, ctx: BuildContext
) -> dict[str, Any]:
    base = safe_name(snapshot.title, snapshot.document_id)
    files = {"docx": f"{base}{DRAFT_SUFFIX}.docx"}
    _docx(snapshot, work_dir, files["docx"], f"{base}.pdf", ctx, draft=True)
    return build_manifest(
        document_id=snapshot.document_id,
        title=snapshot.title,
        kind=BuildKind.DRAFT,
        version=version,
        rules_version=snapshot.rules_version,
        snapshot=_snapshot_entries(snapshot, None),
        files=files,
        output_dir=work_dir,
        versions=tool_versions(ctx.settings, pandoc_version(ctx.settings.build)),
    )


def _start(conn: sqlite3.Connection, document_id: str, kind: BuildKind) -> tuple[int, int]:
    with immediate(conn):
        if kind == BuildKind.FINAL:
            if not repo.transition_document(
                conn, document_id, DocumentStatus.READY, DocumentStatus.BUILDING
            ):
                raise StatusTransitionError(
                    "invalid_transition", f"document {document_id} is not ready"
                )
            return repo.insert_build(conn, document_id, BuildKind.FINAL, BuildStatus.RUNNING)
        row = repo.active_draft_build(conn, document_id)
        if (
            row is None
            or row["status"] != BuildStatus.PENDING
            or not repo.set_build_running(conn, int(row["id"]))
        ):
            raise StatusTransitionError(
                "invalid_transition", f"document {document_id} has no pending draft build"
            )
        return int(row["id"]), int(row["version"])


def _relative(ctx: BuildContext, path: Path) -> str:
    try:
        return path.relative_to(ctx.settings.data_dir).as_posix()
    except ValueError:
        return str(path)


def _finish(
    conn: sqlite3.Connection,
    document_id: str,
    build_id: int,
    kind: BuildKind,
    output_dir: str,
    manifest: dict[str, Any],
) -> None:
    with immediate(conn):
        repo.finish_build(
            conn, build_id, status=BuildStatus.DONE, output_dir=output_dir, manifest=manifest
        )
        if kind == BuildKind.DRAFT:
            return
        repo.set_document_status(conn, document_id, DocumentStatus.DONE)
        repo.set_document_flag(conn, document_id, "rebuild_requested", 0)
        document = repo.get_document(conn, document_id)
        if document is not None and document["ingest_requested"]:
            repo.create_ingest(conn, build_id)
            repo.set_ingest_status(conn, document_id, IngestStatus.PENDING)
        # A draft started during the build sends the document back to needs_review at once.
        recompute_status(conn, document_id)


def _record_failure(
    conn: sqlite3.Connection, document_id: str, build_id: int, kind: BuildKind, error: str
) -> None:
    try:
        with immediate(conn):
            repo.finish_build(conn, build_id, status=BuildStatus.FAILED, error=error)
            if kind == BuildKind.FINAL:
                repo.set_document_status(conn, document_id, DocumentStatus.FAILED, error=error)
    except sqlite3.Error:
        logger.exception("Cannot record the failure of build %d", build_id)


def _prune(conn: sqlite3.Connection, ctx: BuildContext, document_id: str) -> None:
    """Remove output directories beyond build.keep_versions / build.keep_drafts."""
    build_cfg = ctx.settings.build
    with immediate(conn):
        rows = repo.builds_to_prune(
            conn, document_id, build_cfg.keep_versions, build_cfg.keep_drafts
        )
        for row in rows:
            repo.mark_pruned(conn, int(row["id"]))
    for row in rows:
        if row["output_dir"]:
            shutil.rmtree(output_path(ctx, str(row["output_dir"])), ignore_errors=True)
            logger.info("Pruned build v%d of %s", row["version"], document_id)


def run_build(document_id: str, ctx: BuildContext, *, kind: BuildKind) -> int:
    """Build a document (section 9) and publish it atomically.

    A final build moves the document ``ready → building → done``; a draft build takes the
    pending draft build of the document and never changes the document status (section 9.6).
    Any failure marks the build failed (and a final build's document failed) and removes the
    temporary directory; an interruption of the process (SIGTERM) is left to the worker
    recovery, which retries the build.

    :return: build version.
    :raises StatusTransitionError: when there is nothing to build.
    :raises BuildError: when a step or a verification fails.
    """
    conn = connect(ctx.db_path)
    try:
        build_id, version = _start(conn, document_id, kind)
        document_dir = ctx.outputs_dir / document_id
        work_dir = document_dir / f".tmp-v{version}"
        final_dir = document_dir / f"v{version}"
        succeeded = False
        try:
            try:
                shutil.rmtree(work_dir, ignore_errors=True)
                work_dir.mkdir(parents=True)
                snapshot = _read_snapshot(conn, ctx, document_id)
                if kind == BuildKind.FINAL:
                    manifest = _final_files(snapshot, work_dir, version, ctx)
                else:
                    manifest = _draft_files(snapshot, work_dir, version, ctx)
                write_manifest(work_dir / MANIFEST_NAME, manifest)
                # A directory left by a crashed run would make the rename fail.
                shutil.rmtree(final_dir, ignore_errors=True)
                os.replace(work_dir, final_dir)
                _finish(conn, document_id, build_id, kind, _relative(ctx, final_dir), manifest)
            except _STEP_ERRORS as err:
                raise BuildError(f"{type(err).__name__}: {err}") from err
            succeeded = True
        finally:
            if not succeeded:
                error = sys.exc_info()[1]
                # Only real errors fail the build: an interruption (SIGTERM, KeyboardInterrupt)
                # leaves it running, and the worker recovery rebuilds it after the restart.
                if isinstance(error, Exception):
                    _record_failure(conn, document_id, build_id, kind, str(error))
                    shutil.rmtree(work_dir, ignore_errors=True)
                    logger.warning("Build v%d of %s failed", version, document_id)
        logger.info("Built %s v%d (%s)", document_id, version, kind.value)
        try:
            _prune(conn, ctx, document_id)
        except sqlite3.Error:
            # The build is published; old versions are pruned after the next build.
            logger.exception("Cannot prune old builds of %s", document_id)
        return version
    finally:
        conn.close()
