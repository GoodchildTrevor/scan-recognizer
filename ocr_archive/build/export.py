"""Export folder (section 9.7): final files with human names, plus the index список.csv."""

from __future__ import annotations

import csv
import json
import logging
import os
import re
import shutil
import sqlite3
from pathlib import Path
from typing import Any

from ocr_archive.db import repo
from ocr_archive.db.connection import connect, immediate
from ocr_archive.exceptions import ExportError
from ocr_archive.models import BuildContext, BuildKind, BuildStatus, PageTrack

logger = logging.getLogger(__name__)

INDEX_NAME = "список.csv"
INDEX_HEADER = [
    "название",
    "папка",
    "ID документа",
    "версия",
    "дата сборки (UTC)",
    "всего страниц",
    "страниц с флагами",
    "страниц исправлено",
    "страниц pdf_text",
]
FALLBACK_PREFIX = "документ-"
MAX_NAME_CHARS = 100
ID_SUFFIX_CHARS = 8

_FORBIDDEN_CHARS = frozenset('/\\:*?"<>|') | {chr(c) for c in range(0x20)} | {"\x7f"}
_WS_RE = re.compile(r"\s+")
_RESERVED_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{i}" for i in range(1, 10)}
    | {f"LPT{i}" for i in range(1, 10)}
)


def safe_name(title: str, document_id: str) -> str:
    """File and folder name derived from a title (section 10.2).

    Forbidden characters ``/\\:*?"<>|`` and control characters are removed, whitespace is
    collapsed, spaces and dots at the edges are trimmed, the length is cut to 100 characters.
    An empty name, ``.``, ``..`` or a reserved Windows name falls back to
    ``документ-<last 8 characters of the ID>``. Cyrillic is kept.
    """
    name = "".join(char for char in title if char not in _FORBIDDEN_CHARS)
    name = _WS_RE.sub(" ", name).strip(" .")
    name = name[:MAX_NAME_CHARS].strip(" .")
    stem = name.split(".", 1)[0].strip().upper()
    if name in ("", ".", "..") or stem in _RESERVED_NAMES:
        return f"{FALLBACK_PREFIX}{document_id[-ID_SUFFIX_CHARS:]}"
    return name


def output_path(ctx: BuildContext, output_dir: str) -> Path:
    """Absolute path of a build output directory stored relative to the data directory."""
    path = Path(output_dir)
    return path if path.is_absolute() else ctx.settings.data_dir / path


def _allocate_export_name(conn: sqlite3.Connection, document: sqlite3.Row) -> str:
    if document["export_name"]:
        return str(document["export_name"])
    name = safe_name(str(document["title"]), str(document["id"]))
    if repo.export_name_taken(conn, name, str(document["id"])):
        # The ULID tail is its random part; the head is a timestamp shared by documents
        # created in the same second.
        name = f"{name} ({str(document['id'])[-ID_SUFFIX_CHARS:]})"
    repo.set_export_name(conn, str(document["id"]), name)
    return name


def _atomic_copy(source: Path, target: Path) -> None:
    tmp = target.with_name(f".{target.name}.tmp")
    try:
        shutil.copyfile(source, tmp)
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)


def _index_row(document: sqlite3.Row | dict[str, Any], build: sqlite3.Row) -> list[str]:
    manifest = json.loads(build["manifest_json"] or "{}")
    snapshot = manifest.get("snapshot", [])
    flagged = sum(1 for entry in snapshot if entry.get("flags"))
    edited = sum(1 for entry in snapshot if entry.get("edited"))
    pdf_text = sum(1 for entry in snapshot if entry.get("track") == PageTrack.PDF_TEXT)
    return [
        str(document["title"]),
        str(document["export_name"] or ""),
        str(document["id"]),
        str(build["version"]),
        str(manifest.get("created_at") or build["created_at"]),
        str(document["page_count"]),
        str(flagged),
        str(edited),
        str(pdf_text),
    ]


def write_index_csv(
    conn: sqlite3.Connection, export_dir: Path, *, overrides: dict[str, int] | None = None
) -> None:
    """Rewrite список.csv atomically: UTF-8 with BOM so that Excel opens Cyrillic.

    :param overrides: document id to the version just copied, not yet recorded in the database.
    """
    rows: dict[str, list[str]] = {}
    for document in repo.list_documents(conn):
        version = (overrides or {}).get(str(document["id"]), document["exported_version"])
        if version is None:
            continue
        build = repo.build_by_version(conn, str(document["id"]), int(version))
        if build is None:
            continue
        rows[str(document["id"])] = _index_row(document, build)
    ordered = sorted(rows.values(), key=lambda row: (row[1], row[2]))
    export_dir.mkdir(parents=True, exist_ok=True)
    target = export_dir / INDEX_NAME
    tmp = export_dir / f".{INDEX_NAME}.tmp"
    try:
        with tmp.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(INDEX_HEADER)
            writer.writerows(ordered)
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)


def export_document(document_id: str, build_version: int, ctx: BuildContext) -> Path:
    """Copy the final PDF and DOCX of a build into the export folder (section 9.7).

    Files are written under temporary names and renamed over the old ones, PDF first, then
    DOCX; список.csv is rewritten last, so a version listed there has both files in place.

    :return: the document folder in the export directory.
    :raises ExportError: when the build is missing or copying fails; the error is recorded in
        ``documents.export_error`` and the worker retries later.
    """
    conn = connect(ctx.db_path)
    try:
        with immediate(conn):
            document = repo.get_document(conn, document_id)
            build = repo.build_by_version(conn, document_id, build_version)
            if (
                document is None
                or build is None
                or build["kind"] != BuildKind.FINAL
                or build["status"] != BuildStatus.DONE
                or build["pruned_at"] is not None
            ):
                raise ExportError(f"no final build v{build_version} of {document_id} to export")
            folder_name = _allocate_export_name(conn, document)
        manifest = json.loads(build["manifest_json"] or "{}")
        files = manifest.get("files", {})
        source_dir = output_path(ctx, str(build["output_dir"]))
        base = safe_name(str(document["title"]), document_id)
        target = ctx.export_dir / folder_name
        try:
            target.mkdir(parents=True, exist_ok=True)
            _atomic_copy(source_dir / files["pdf"], target / f"{base}.pdf")
            _atomic_copy(source_dir / files["docx"], target / f"{base}.docx")
            write_index_csv(conn, ctx.export_dir, overrides={document_id: build_version})
        except (OSError, KeyError) as err:
            with immediate(conn):
                repo.set_export_result(conn, document_id, version=None, error=str(err))
            logger.warning("Export of %s v%d failed: %s", document_id, build_version, err)
            raise ExportError(f"export of {document_id} v{build_version} failed: {err}") from err
        with immediate(conn):
            repo.set_export_result(conn, document_id, version=build_version, error=None)
        logger.info("Exported %s v%d", document_id, build_version)
        return target
    finally:
        conn.close()
