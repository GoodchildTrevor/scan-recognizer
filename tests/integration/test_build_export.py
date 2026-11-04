"""Export folder and список.csv (section 9.7)."""

from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

import pytest

from ocr_archive.build.export import INDEX_HEADER, INDEX_NAME, export_document
from ocr_archive.db import repo
from ocr_archive.db.connection import immediate
from ocr_archive.exceptions import ExportError
from ocr_archive.settings import Settings
from tests.conftest import make_settings
from tests.fixtures import factory
from tests.fixtures.build_factory import PageSpec, add_final_build, build_context, create_document

pytestmark = pytest.mark.p4


def make_built(
    conn: sqlite3.Connection, settings: Settings, tmp_path: Path, title: str, content: bytes = b"v1"
) -> tuple[str, int]:
    source, _ = factory.make_typewriter_pdf(tmp_path, n_lines=5, name=f"{len(title)}.pdf")
    document_id, _ = create_document(
        conn,
        settings,
        source,
        [PageSpec("текст", flags=["pass_diff"]), PageSpec("x", revision_text="y")],
        title=title,
        status="ready",
    )
    version = add_final_build(
        conn, settings, document_id, {"pdf": b"%PDF " + content, "docx": b"PK " + content}
    )
    return document_id, version


def read_index(settings: Settings) -> list[list[str]]:
    raw = (settings.export_dir / INDEX_NAME).read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")
    with (settings.export_dir / INDEX_NAME).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.reader(handle))


def test_export_copies_files_with_human_names(
    conn: sqlite3.Connection, settings: Settings, tmp_path: Path
) -> None:
    document_id, version = make_built(conn, settings, tmp_path, 'Отчёт: "план" 1983')
    folder = export_document(document_id, version, build_context(settings))
    assert folder == settings.export_dir / "Отчёт план 1983"
    assert sorted(p.name for p in folder.iterdir()) == [
        "Отчёт план 1983.docx",
        "Отчёт план 1983.pdf",
    ]
    assert (folder / "Отчёт план 1983.pdf").read_bytes() == b"%PDF v1"
    # No temporary files are left anywhere in the export folder.
    assert not [p for p in settings.export_dir.rglob(".*.tmp")]
    rows = read_index(settings)
    assert rows[0] == INDEX_HEADER
    assert rows[1][:4] == ['Отчёт: "план" 1983', "Отчёт план 1983", document_id, str(version)]
    assert rows[1][5:] == ["2", "1", "1", "0"]
    document = repo.get_document(conn, document_id)
    assert document is not None
    assert document["exported_version"] == version
    assert document["export_error"] is None
    assert document["export_name"] == "Отчёт план 1983"


def test_taken_folder_name_gets_the_id_suffix(
    conn: sqlite3.Connection, settings: Settings, tmp_path: Path
) -> None:
    first, v1 = make_built(conn, settings, tmp_path, "Отчёт")
    second, v2 = make_built(conn, settings, tmp_path / "..", "Отчёт", b"other")
    export_document(first, v1, build_context(settings))
    folder = export_document(second, v2, build_context(settings))
    assert folder.name == f"Отчёт ({second[-8:]})"
    assert (folder / "Отчёт.pdf").read_bytes() == b"%PDF other"
    assert len(read_index(settings)) == 3


def test_new_version_replaces_files_and_keeps_the_folder(
    conn: sqlite3.Connection, settings: Settings, tmp_path: Path
) -> None:
    document_id, v1 = make_built(conn, settings, tmp_path, "Отчёт")
    export_document(document_id, v1, build_context(settings))
    v2 = add_final_build(conn, settings, document_id, {"pdf": b"%PDF v2", "docx": b"PK v2"})
    folder = export_document(document_id, v2, build_context(settings))
    assert folder.name == "Отчёт"
    assert (folder / "Отчёт.pdf").read_bytes() == b"%PDF v2"
    assert (folder / "Отчёт.docx").read_bytes() == b"PK v2"
    assert read_index(settings)[1][3] == str(v2)


def test_failed_export_is_recorded_and_can_be_retried(
    conn: sqlite3.Connection, tmp_path: Path
) -> None:
    blocked = tmp_path / "blocked"
    blocked.write_text("a file where the export folder should be", encoding="utf-8")
    settings = make_settings(tmp_path, export_dir=blocked)
    document_id, version = make_built(conn, settings, tmp_path, "Отчёт")
    with pytest.raises(ExportError):
        export_document(document_id, version, build_context(settings))
    document = repo.get_document(conn, document_id)
    assert document is not None
    assert document["status"] == "done"
    assert document["export_error"]
    assert document["exported_version"] is None
    assert repo.documents_needing_export(conn, "9999-12-31")[0]["id"] == document_id

    blocked.unlink()
    export_document(document_id, version, build_context(settings))
    document = repo.get_document(conn, document_id)
    assert document is not None
    assert document["exported_version"] == version
    assert document["export_error"] is None


def test_export_requires_a_final_done_build(
    conn: sqlite3.Connection, settings: Settings, tmp_path: Path
) -> None:
    document_id, version = make_built(conn, settings, tmp_path, "Отчёт")
    with immediate(conn):
        repo.mark_pruned(conn, int(repo.build_by_version(conn, document_id, version)["id"]))
    with pytest.raises(ExportError):
        export_document(document_id, version, build_context(settings))
