"""Build runner (section 9): final and draft builds, failures, verification through pandoc.

Tests marked ``needs_pandoc`` run the real pandoc and are skipped when it is not installed;
the Docker image always has it.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import zipfile
from pathlib import Path
from typing import Any

import pytest

from ocr_archive.build import document as build_runner
from ocr_archive.build.docx import build_docx, read_docx_ast
from ocr_archive.build.markdown import MdPage, to_pandoc_markdown
from ocr_archive.build.normalize import document_vocabulary, normalize_page
from ocr_archive.build.verify import docx_pages, verify_docx
from ocr_archive.db import repo
from ocr_archive.db.connection import immediate
from ocr_archive.exceptions import BuildError, StatusTransitionError
from ocr_archive.models import BuildKind
from ocr_archive.settings import Settings
from tests.conftest import make_settings
from tests.fixtures import factory
from tests.fixtures.build_factory import PageSpec, build_context, create_document

pytestmark = pytest.mark.p4

needs_pandoc = pytest.mark.skipif(shutil.which("pandoc") is None, reason="pandoc is not installed")

PAGE_ONE = (
    "Отчёт о работе за 1983 год.\n"
    "Расход составил 12 345 руб., t = −5, рост 15 %.\n"
    "\n"
    "| Год | Сумма |\n|---|---|\n| 1983 | 3,5 |\n"
    "\n"
    "Формула \\(x^2 + 1\\) и подпись [подпись], место [неразб.: Иванов?]."
)
PAGE_TWO = (
    "Стр. 5\n"
    '<table><tr><td colspan="2">Итого [неразб.]</td></tr><tr><td>10</td><td>20</td></tr></table>\n'
    "\n"
    "Социально-\nэкономический раздел."
)


def scan_pdf(tmp_path: Path, pages: int) -> Path:
    images = [
        factory.render_typewriter(factory.typewriter_lines(8, seed=i), dpi=100, noise=2.0).image
        for i in range(pages)
    ]
    return factory.images_pdf(tmp_path / "scan.pdf", images, dpi=100)


def build_row(conn: sqlite3.Connection, document_id: str, version: int) -> sqlite3.Row:
    row = repo.build_by_version(conn, document_id, version)
    assert row is not None
    return row


def document_row(conn: sqlite3.Connection, document_id: str) -> sqlite3.Row:
    row = repo.get_document(conn, document_id)
    assert row is not None
    return row


def missing_pandoc_settings(tmp_path: Path) -> Settings:
    return make_settings(tmp_path, build={"pandoc_path": "pandoc-is-not-installed-here"})


# --- without pandoc ------------------------------------------------------------------------------


def test_failed_final_build_marks_build_and_document(tmp_path: Path) -> None:
    settings = missing_pandoc_settings(tmp_path)
    from ocr_archive.db.connection import open_db

    conn = open_db(settings.database_path)
    document_id, _ = create_document(conn, settings, scan_pdf(tmp_path, 1), [PageSpec(PAGE_ONE)])
    with pytest.raises(BuildError, match="pandoc"):
        build_runner.run_build(document_id, build_context(settings), kind=BuildKind.FINAL)
    build = build_row(conn, document_id, 1)
    assert build["status"] == "failed"
    assert "pandoc" in build["error"]
    document = document_row(conn, document_id)
    assert document["status"] == "failed"
    assert "pandoc" in document["error"]
    document_dir = settings.outputs_dir / document_id
    assert not list(document_dir.glob(".tmp-v*"))
    assert not (document_dir / "v1").exists()
    conn.close()


def test_final_build_needs_a_ready_document(
    conn: sqlite3.Connection, settings: Settings, tmp_path: Path
) -> None:
    document_id, _ = create_document(
        conn,
        settings,
        scan_pdf(tmp_path, 1),
        [PageSpec(PAGE_ONE, status="flagged")],
        status="needs_review",
    )
    with pytest.raises(StatusTransitionError) as caught:
        build_runner.run_build(document_id, build_context(settings), kind=BuildKind.FINAL)
    assert caught.value.code == "invalid_transition"
    assert repo.builds_of_document(conn, document_id) == []
    assert document_row(conn, document_id)["status"] == "needs_review"


def test_failed_draft_build_leaves_the_document_status(tmp_path: Path) -> None:
    settings = missing_pandoc_settings(tmp_path)
    from ocr_archive.db.connection import open_db

    conn = open_db(settings.database_path)
    document_id, _ = create_document(
        conn,
        settings,
        scan_pdf(tmp_path, 1),
        [PageSpec(PAGE_ONE, status="flagged")],
        status="needs_review",
    )
    with immediate(conn):
        repo.insert_build(conn, document_id, "draft", "pending")
    with pytest.raises(BuildError):
        build_runner.run_build(document_id, build_context(settings), kind=BuildKind.DRAFT)
    assert build_row(conn, document_id, 1)["status"] == "failed"
    document = document_row(conn, document_id)
    assert document["status"] == "needs_review"
    assert document["error"] is None
    conn.close()


def test_interrupted_build_is_left_for_the_worker_recovery(
    conn: sqlite3.Connection, settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    document_id, _ = create_document(conn, settings, scan_pdf(tmp_path, 1), [PageSpec(PAGE_ONE)])

    def interrupted(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise KeyboardInterrupt

    monkeypatch.setattr(build_runner, "_final_files", interrupted)
    with pytest.raises(KeyboardInterrupt):
        build_runner.run_build(document_id, build_context(settings), kind=BuildKind.FINAL)
    assert build_row(conn, document_id, 1)["status"] == "running"
    assert document_row(conn, document_id)["status"] == "building"


def test_one_ingest_per_final_build(
    conn: sqlite3.Connection, settings: Settings, tmp_path: Path
) -> None:
    document_id, _ = create_document(conn, settings, scan_pdf(tmp_path, 1), [PageSpec("x")])
    with immediate(conn):
        build_id, _ = repo.insert_build(conn, document_id, "final", "done")
        assert repo.create_ingest(conn, build_id) is True
        assert repo.create_ingest(conn, build_id) is False
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO ingests (build_id, status, created_at) VALUES (?, 'pending', 'now')",
                (build_id,),
            )
    rows = conn.execute("SELECT COUNT(*) FROM ingests WHERE build_id = ?", (build_id,)).fetchone()
    assert rows[0] == 1


def test_draft_hints_are_marked_with_private_use_characters() -> None:
    review = {"hints": [{"start": 0, "end": 5}, {"start": 3, "end": 8}, {"start": 50, "end": 99}]}
    assert build_runner._mark_hints("Текст страницы", review) == "Текст страницы"


# --- with pandoc ---------------------------------------------------------------------------------


def two_page_document(
    conn: sqlite3.Connection,
    settings: Settings,
    tmp_path: Path,
    *,
    status: str = "ready",
    ingest: bool = False,
    flagged: bool = False,
) -> str:
    review = {"hints": [{"id": 1, "start": 0, "end": 5, "reasons": ["diff"]}]}
    pages = [
        PageSpec(PAGE_ONE, printed_label="9 об.", revision_text=PAGE_ONE + " Исправлено."),
        PageSpec(
            PAGE_TWO,
            status="flagged" if flagged else "ok",
            review=review,
            flags=["pass_diff"] if flagged else [],
        ),
    ]
    document_id, _ = create_document(
        conn, settings, scan_pdf(tmp_path, 2), pages, status=status, ingest=ingest
    )
    return document_id


@needs_pandoc
def test_final_build_publishes_verified_files(
    conn: sqlite3.Connection, settings: Settings, tmp_path: Path
) -> None:
    document_id = two_page_document(conn, settings, tmp_path, ingest=True)
    with immediate(conn):
        repo.set_document_flag(conn, document_id, "rebuild_requested", 1)
    version = build_runner.run_build(document_id, build_context(settings), kind=BuildKind.FINAL)
    assert version == 1
    build = build_row(conn, document_id, version)
    assert build["status"] == "done"
    assert build["output_dir"] == f"outputs/{document_id}/v1"
    out_dir = settings.data_dir / build["output_dir"]
    assert sorted(p.name for p in out_dir.iterdir()) == [
        "chunks.jsonl",
        "manifest.json",
        "Отчёт 1983.docx",
        "Отчёт 1983.pdf",
    ]
    manifest = json.loads(build["manifest_json"])
    assert manifest == json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert [e["source"] for e in manifest["snapshot"]] == ["revision", "attempt"]
    assert manifest["snapshot"][0]["edited"] is True
    assert set(manifest["sha256"]) == {"pdf", "docx", "chunks"}
    assert manifest["versions"]["pandoc"].startswith("pandoc")
    document = document_row(conn, document_id)
    assert document["status"] == "done"
    assert document["rebuild_requested"] == 0
    assert document["ingest_status"] == "pending"
    assert repo.get_ingest_for_build(conn, int(build["id"])) is not None


@needs_pandoc
def test_old_versions_are_pruned(
    conn: sqlite3.Connection, settings: Settings, tmp_path: Path
) -> None:
    document_id = two_page_document(conn, settings, tmp_path)
    ctx = build_context(settings)
    for _ in range(3):
        with immediate(conn):
            repo.set_document_status(conn, document_id, "ready")
        build_runner.run_build(document_id, ctx, kind=BuildKind.FINAL)
    document_dir = settings.outputs_dir / document_id
    assert sorted(p.name for p in document_dir.iterdir()) == ["v2", "v3"]
    assert build_row(conn, document_id, 1)["pruned_at"] is not None


@needs_pandoc
def test_draft_build_is_docx_only_and_keeps_the_status(
    conn: sqlite3.Connection, settings: Settings, tmp_path: Path
) -> None:
    document_id = two_page_document(conn, settings, tmp_path, status="needs_review", flagged=True)
    with immediate(conn):
        repo.insert_build(conn, document_id, "draft", "pending")
    version = build_runner.run_build(document_id, build_context(settings), kind=BuildKind.DRAFT)
    out_dir = settings.outputs_dir / document_id / f"v{version}"
    assert sorted(p.name for p in out_dir.iterdir()) == [
        "manifest.json",
        "Отчёт 1983 (черновик).docx",
    ]
    document = document_row(conn, document_id)
    assert document["status"] == "needs_review"
    assert document["ingest_status"] == "none"
    assert document["exported_version"] is None
    docx = out_dir / "Отчёт 1983 (черновик).docx"
    ast = read_docx_ast(docx, settings.build)
    first = ast["blocks"][0]
    assert first["t"] == "Div"
    assert ["custom-style", "Draft Banner"] in first["c"][0][2]
    text = json.dumps(ast, ensure_ascii=False)
    page_notes = [
        " ".join(_strings(block))
        for block in ast["blocks"]
        if block["t"] == "Div" and ["custom-style", "Source Page"] in block["c"][0][2]
    ]
    assert any("· не проверено" in note for note in page_notes)
    assert '"Unverified"' in text
    with zipfile.ZipFile(docx) as archive:
        core = archive.read("docProps/core.xml").decode("utf-8")
    assert "ЧЕРНОВИК — Отчёт 1983" in core
    assert "не проверено" in core


def _strings(node: Any) -> list[str]:
    """Words of a pandoc AST node; pandoc splits text into Str nodes separated by Space."""
    if isinstance(node, dict):
        if node.get("t") == "Str":
            return [str(node["c"])]
        return _strings(node.get("c"))
    if isinstance(node, list):
        return [word for item in node for word in _strings(item)]
    return []


def docx_for(blocks_text: list[str], work: Path, settings: Settings) -> tuple[Path, list[Any]]:
    vocabulary = document_vocabulary(blocks_text)
    pages = [MdPage(i, None, normalize_page(t, vocabulary)) for i, t in enumerate(blocks_text)]
    markdown, refs = to_pandoc_markdown("Отчёт", pages, "Отчёт.pdf")
    md = work / "in.md"
    md.write_text(markdown, encoding="utf-8")
    docx = work / "out.docx"
    build_docx(md, docx, settings.build)
    return docx, refs


def rewrite_document_xml(docx: Path, old: str, new: str) -> None:
    with zipfile.ZipFile(docx) as archive:
        entries = [(info, archive.read(info.filename)) for info in archive.infolist()]
    with zipfile.ZipFile(docx, "w", zipfile.ZIP_DEFLATED) as archive:
        for info, data in entries:
            if info.filename == "word/document.xml":
                xml = data.decode("utf-8")
                assert old in xml
                data = xml.replace(old, new, 1).encode("utf-8")
            archive.writestr(info, data)


@needs_pandoc
def test_round_trip_passes_and_page_header_lines_in_text_do_not_split(
    settings: Settings, tmp_path: Path
) -> None:
    docx, refs = docx_for([PAGE_ONE, PAGE_TWO], tmp_path, settings)
    ast = read_docx_ast(docx, settings.build)
    assert len(docx_pages(ast)) == 2
    verify_docx(ast, refs, settings.build.docx_min_similarity)
    text = json.dumps(ast, ensure_ascii=False)
    assert '"Uncertain"' in text


@needs_pandoc
@pytest.mark.parametrize(
    ("old", "new", "problem"),
    [
        ("12 345", "12 34", "numbers differ"),
        ("Отчёт о работе", "Отчёт о раб", "words differ"),
    ],
)
def test_verification_catches_changes_in_the_docx(
    settings: Settings, tmp_path: Path, old: str, new: str, problem: str
) -> None:
    docx, refs = docx_for([PAGE_ONE, PAGE_TWO], tmp_path, settings)
    rewrite_document_xml(docx, old, new)
    with pytest.raises(BuildError, match=problem):
        verify_docx(read_docx_ast(docx, settings.build), refs, settings.build.docx_min_similarity)


@needs_pandoc
def test_verification_catches_a_removed_table_cell(settings: Settings, tmp_path: Path) -> None:
    docx, refs = docx_for([PAGE_ONE], tmp_path, settings)
    with zipfile.ZipFile(docx) as archive:
        xml = archive.read("word/document.xml").decode("utf-8")
    start = xml.index("<w:tc>", xml.index("3,5") - 600)
    end = xml.index("</w:tc>", start) + len("</w:tc>")
    rewrite_document_xml(docx, xml[start:end], "")
    with pytest.raises(BuildError, match="differ"):
        verify_docx(read_docx_ast(docx, settings.build), refs, settings.build.docx_min_similarity)


@needs_pandoc
def test_verification_catches_swapped_paragraphs(settings: Settings, tmp_path: Path) -> None:
    paragraphs = [
        "Первый абзац рассказывает о плане работ лаборатории на год.",
        "Второй абзац описывает расход материалов и сроки поставки.",
        "Третий абзац перечисляет замечания комиссии по оборудованию.",
        "Четвёртый абзац подводит итоги и называет ответственных лиц.",
    ]
    docx, refs = docx_for(["\n\n".join(paragraphs)], tmp_path, settings)
    with zipfile.ZipFile(docx) as archive:
        xml = archive.read("word/document.xml").decode("utf-8")
    first = xml[
        xml.rindex("<w:p>", 0, xml.index("Первый")) : xml.index("</w:p>", xml.index("Первый")) + 6
    ]
    last = xml[
        xml.rindex("<w:p>", 0, xml.index("Четвёртый")) : xml.index("</w:p>", xml.index("Четвёртый"))
        + 6
    ]
    swapped = xml.replace(first, "@@FIRST@@").replace(last, first).replace("@@FIRST@@", last)
    rewrite_document_xml(docx, xml, swapped)
    with pytest.raises(BuildError, match="similarity"):
        verify_docx(read_docx_ast(docx, settings.build), refs, settings.build.docx_min_similarity)
