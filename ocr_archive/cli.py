"""Command line interface ``ocr`` (section 13). Output is in English."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path, PurePath
from typing import Annotated

import typer

from ocr_archive import operations
from ocr_archive.db import repo
from ocr_archive.db.connection import open_db
from ocr_archive.exceptions import ConfigError, OcrArchiveError, StatusTransitionError
from ocr_archive.ingest import (
    IMAGE_SUFFIXES,
    PDF_SUFFIXES,
    TIFF_SUFFIXES,
    create_document,
    read_source_files,
)
from ocr_archive.logging_setup import configure_logging
from ocr_archive.models import PipelineContext
from ocr_archive.settings import Settings, load_settings

app = typer.Typer(no_args_is_help=True, add_completion=False, help="OCR archive service.")
audit_app = typer.Typer(no_args_is_help=True, help="Spot checks (section 15.4).")
pilot_app = typer.Typer(no_args_is_help=True, help="Pilot report (section 15.2).")
gold_app = typer.Typer(no_args_is_help=True, help="Gold set (section 15.1).")
app.add_typer(audit_app, name="audit")
app.add_typer(pilot_app, name="pilot")
app.add_typer(gold_app, name="gold")


def _settings() -> Settings:
    try:
        return load_settings()
    except ConfigError as err:
        typer.echo(f"config error: {err}", err=True)
        raise typer.Exit(2) from err


def _pipeline_ctx(settings: Settings) -> PipelineContext:
    from ocr_archive.versions import rules_version
    from ocr_archive.vlm.client import VlmClient

    return PipelineContext(
        settings=settings,
        db_path=settings.database_path,
        vlm=VlmClient(settings.vlm, settings.llama_api_key.get_secret_value()),
        rules_version=rules_version(settings),
    )


def _fail(message: str, code: int = 1) -> typer.Exit:
    typer.echo(message, err=True)
    return typer.Exit(code)


@app.command()
def serve() -> None:
    """Run the REST API (uvicorn)."""
    import uvicorn

    from ocr_archive.api.app import create_app

    settings = _settings()
    uvicorn.run(
        create_app(settings), host=settings.api.host, port=settings.api.port, log_config=None
    )


@app.command()
def worker() -> None:
    """Run the single worker."""
    from ocr_archive.worker import run_worker

    raise typer.Exit(run_worker(_settings()))


def _group_files(paths: list[Path]) -> list[list[Path]]:
    """Each PDF or TIFF is its own document; all other images form one document."""
    groups: list[list[Path]] = []
    images: list[Path] = []
    for path in paths:
        suffix = path.suffix.lower()
        if suffix in PDF_SUFFIXES or suffix in TIFF_SUFFIXES:
            groups.append([path])
        elif suffix in IMAGE_SUFFIXES:
            images.append(path)
        else:
            raise _fail(f"unsupported file: {path}")
    if images:
        groups.append(images)
    return groups


@app.command()
def add(
    paths: Annotated[
        list[Path], typer.Argument(help="PDF, TIFF or image files (in Docker: /inbox).")
    ],
    title: Annotated[str | None, typer.Option(help="Document title.")] = None,
    ingest: Annotated[
        bool, typer.Option(help="Load into the knowledge base after the build.")
    ] = False,
    priority: Annotated[int, typer.Option(help="Higher is processed first.")] = 0,
) -> None:
    """Queue documents without HTTP."""
    settings = _settings()
    if ingest and not settings.ingester.enabled:
        raise _fail("ingester is disabled")
    conn = open_db(settings.database_path)
    try:
        for group in _group_files(
            [p if p.is_absolute() or p.exists() else Path("/inbox") / p for p in paths]
        ):
            created = create_document(
                conn,
                settings,
                read_source_files(group),
                title=title,
                ingest=ingest,
                priority=priority,
            )
            note = " (duplicate)" if created.duplicate else ""
            typer.echo(
                f"{created.document_id} {created.status} {created.page_count} pages "
                f"{created.title}{note}"
            )
    except OcrArchiveError as err:
        raise _fail(str(err)) from err
    finally:
        conn.close()


@app.command()
def status(document_id: Annotated[str | None, typer.Argument()] = None) -> None:
    """Show documents, or one document in detail."""
    settings = _settings()
    conn = open_db(settings.database_path)
    try:
        if document_id:
            typer.echo(
                json.dumps(
                    operations.document_view(conn, document_id, settings),
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return
        for doc in operations.list_documents(conn, None, settings):
            progress = doc["progress"]
            typer.echo(
                f"{doc['id']}  {doc['status']:<12} {progress['total']:>4} pages  "
                f"ok {progress['ok']} flagged {progress['flagged']} failed {progress['failed']} "
                f"reviewed {progress['reviewed']} pending {progress['pending']}  {doc['title']}"
            )
    except KeyError as err:
        raise _fail(f"unknown document {document_id}") from err
    finally:
        conn.close()


@app.command()
def rebuild(document_id: str) -> None:
    """Force a new final build of a document."""
    settings = _settings()
    conn = open_db(settings.database_path)
    try:
        typer.echo(operations.request_rebuild(conn, document_id))
    except StatusTransitionError as err:
        raise _fail(err.code) from err
    except KeyError as err:
        raise _fail(f"unknown document {document_id}") from err
    finally:
        conn.close()


@app.command()
def draft(document_id: str) -> None:
    """Request a draft DOCX build (section 9.6)."""
    settings = _settings()
    conn = open_db(settings.database_path)
    try:
        typer.echo(f"draft build {operations.request_draft_build(conn, document_id)}")
    except StatusTransitionError as err:
        raise _fail(err.code) from err
    except KeyError as err:
        raise _fail(f"unknown document {document_id}") from err
    finally:
        conn.close()


@app.command("ingest")
def ingest_cmd(document_id: str) -> None:
    """Load a built document into the knowledge base."""
    settings = _settings()
    conn = open_db(settings.database_path)
    try:
        typer.echo(operations.request_ingest(conn, document_id, settings))
    except StatusTransitionError as err:
        raise _fail(err.code) from err
    except KeyError as err:
        raise _fail(f"unknown document {document_id}") from err
    finally:
        conn.close()


@app.command()
def redecide(
    document_id: Annotated[str | None, typer.Argument()] = None,
    all_documents: Annotated[bool, typer.Option("--all", help="All documents.")] = False,
) -> None:
    """Ask the worker to re-decide built documents by the current rules (section 6.11)."""
    if not document_id and not all_documents:
        raise _fail("give a document id or --all")
    settings = _settings()
    conn = open_db(settings.database_path)
    try:
        ids = [str(d["id"]) for d in repo.list_documents(conn)] if all_documents else [document_id]
        for doc_id in ids:
            typer.echo(f"{doc_id} {operations.request_redecide(conn, str(doc_id))}")
    except KeyError as err:
        raise _fail(f"unknown document {document_id}") from err
    finally:
        conn.close()


@app.command()
def reprocess(
    document_id: str,
    include_reviewed: Annotated[bool, typer.Option(help="Also drop manual revisions.")] = False,
) -> None:
    """Recognize a document again with a new request to the model (section 6.11)."""
    from ocr_archive.pipeline import reprocess_document

    settings = _settings()
    open_db(settings.database_path).close()
    try:
        status_value = reprocess_document(
            document_id, _pipeline_ctx(settings), include_reviewed=include_reviewed
        )
    except StatusTransitionError as err:
        raise _fail(err.code) from err
    except KeyError as err:
        raise _fail(f"unknown document {document_id}") from err
    typer.echo(status_value)


@audit_app.command("sample")
def audit_sample(
    document_id: str,
    n: Annotated[int, typer.Option("--n", help="Number of pages.")] = 10,
    track: Annotated[str | None, typer.Option(help="vlm or pdf_text.")] = None,
) -> None:
    """Put random ok pages without flags into the review queue with scope full."""
    settings = _settings()
    conn = open_db(settings.database_path)
    try:
        pages = operations.audit_sample(conn, document_id, n, track=track)
    except KeyError as err:
        raise _fail(f"unknown document {document_id}") from err
    finally:
        conn.close()
    typer.echo(f"sampled {len(pages)} pages")


@audit_app.command("report")
def audit_report(document_id: Annotated[str | None, typer.Argument()] = None) -> None:
    """Spot-check report."""
    from ocr_archive.reports import audit_report as build_report
    from ocr_archive.reports import write_report

    settings = _settings()
    conn = open_db(settings.database_path)
    try:
        content = build_report(conn, document_id)
    finally:
        conn.close()
    typer.echo(content)
    typer.echo(f"written to {write_report(settings, 'audit', content)}")


@pilot_app.command("report")
def pilot_report(document_id: str) -> None:
    """Pilot report."""
    from ocr_archive.reports import pilot_report as build_report
    from ocr_archive.reports import write_report

    settings = _settings()
    conn = open_db(settings.database_path)
    try:
        content = build_report(conn, document_id, settings)
    except KeyError as err:
        raise _fail(f"unknown document {document_id}") from err
    finally:
        conn.close()
    typer.echo(content)
    typer.echo(f"written to {write_report(settings, 'pilot', content)}")


@app.command()
def doctor(
    record_fingerprint: Annotated[bool, typer.Option(help="Store the model fingerprint.")] = False,
) -> None:
    """Check the environment (section 17)."""
    from ocr_archive.doctor import Level, run_doctor

    checks = run_doctor(_settings(), record_fingerprint=record_fingerprint)
    for check in checks:
        typer.echo(f"{check.level.value:<8} {check.name}: {check.message}")
    if any(check.level == Level.FAIL for check in checks):
        raise typer.Exit(1)


@gold_app.command("export")
def gold_export(
    document_id: str,
    pages: Annotated[str, typer.Option(help="Comma-separated page numbers, e.g. 3,17,42.")],
) -> None:
    """Write first-pass texts of pages for proofreading."""
    from ocr_archive.bench import export_gold

    settings = _settings()
    conn = open_db(settings.database_path)
    try:
        numbers = [int(part) for part in pages.split(",") if part.strip()]
        for path in export_gold(conn, settings, document_id, numbers):
            typer.echo(str(path))
    except (KeyError, ValueError) as err:
        raise _fail(f"bad page list or document: {err}") from err
    finally:
        conn.close()


@gold_app.command("import")
def gold_import() -> None:
    """Load proofread gold texts."""
    from ocr_archive.bench import import_gold

    settings = _settings()
    conn = open_db(settings.database_path)
    try:
        typer.echo(f"imported {import_gold(conn, settings)} pages")
    finally:
        conn.close()


@app.command()
def bench(
    profile: Annotated[list[str], typer.Option(help="Profile to measure; repeatable.")],
) -> None:
    """Measure profiles on the gold pages (section 15.1)."""
    from ocr_archive.bench import run_bench
    from ocr_archive.reports import write_report

    settings = _settings()
    conn = open_db(settings.database_path)
    try:
        content = asyncio.run(run_bench(conn, settings, profile))
    finally:
        conn.close()
    typer.echo(content)
    typer.echo(f"written to {write_report(settings, 'bench', content)}")


@app.command()
def page(file: Path, n: int) -> None:
    """Debug one page: text, uncertain spans, signals and flags; nothing is stored."""
    from ocr_archive.debug import debug_page

    settings = _settings()
    if PurePath(file).suffix.lower() not in PDF_SUFFIXES:
        raise _fail("give a PDF file")
    typer.echo(asyncio.run(debug_page(settings, file, n - 1)))


def main() -> None:
    """Entry point of the ``ocr`` script."""
    configure_logging()
    try:
        app()
    except OcrArchiveError as err:
        typer.echo(f"error: {err}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
