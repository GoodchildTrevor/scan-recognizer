"""REST routes of section 11."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import quote

import pymupdf
from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from fastapi.responses import FileResponse, JSONResponse
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile

from ocr_archive import operations, pdfpages
from ocr_archive.api.auth import require_api_key
from ocr_archive.api.schemas import (
    BuildInfo,
    DeliveredRequest,
    Delivery,
    DocumentCreated,
    DraftBuildCreated,
    DraftRequest,
    DraftSaved,
    Health,
    IngestRequested,
    LowConfSpan,
    QueueEntry,
    RegionRequest,
    RegionResponse,
    ReviewRequest,
    ReviewResponse,
    StatusResponse,
)
from ocr_archive.build.export import safe_name
from ocr_archive.db import repo
from ocr_archive.db.connection import connect, immediate
from ocr_archive.exceptions import StatusTransitionError, VlmRequestError, VlmTransportError
from ocr_archive.ingest import CreatedDocument, InvalidSourceError, SourceFile, create_document
from ocr_archive.models import BuildKind
from ocr_archive.review import Utf16Map
from ocr_archive.settings import Settings
from ocr_archive.vlm.logprobs import low_confidence_spans
from ocr_archive.worker import heartbeat_age_s

HEARTBEAT_MAX_AGE_S = 30.0
FILE_KINDS = {
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "chunks": "application/x-ndjson",
}

router = APIRouter()
api = APIRouter(prefix="/api", dependencies=[Depends(require_api_key)])


def get_settings(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    return settings


def get_conn(request: Request) -> Iterator[sqlite3.Connection]:
    conn = connect(request.app.state.settings.database_path)
    try:
        yield conn
    finally:
        conn.close()


Conn = Annotated[sqlite3.Connection, Depends(get_conn)]
Config = Annotated[Settings, Depends(get_settings)]


def _rules_version(request: Request) -> str:
    return str(request.app.state.rules_version)


def _not_found(what: str) -> HTTPException:
    return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"{what}_not_found")


# --- health ------------------------------------------------------------------------------------


@router.get("/health", response_model=Health)
async def health(request: Request) -> Health:
    settings: Settings = request.app.state.settings
    try:
        conn = connect(settings.database_path)
        try:
            conn.execute("SELECT 1").fetchone()
            age = heartbeat_age_s(conn)
        finally:
            conn.close()
        db_state = "ok"
    except sqlite3.Error:
        db_state, age = "error", None
    vlm_ok = await request.app.state.vlm.is_alive()
    return Health(
        db=db_state,
        vlm="ok" if vlm_ok else "down",
        worker_alive=age is not None and age < HEARTBEAT_MAX_AGE_S,
    )


# --- documents ---------------------------------------------------------------------------------


@api.post("/documents", status_code=status.HTTP_201_CREATED, response_model=DocumentCreated)
async def upload_document(request: Request, response: Response) -> DocumentCreated:
    settings: Settings = request.app.state.settings
    limit = settings.max_upload_mb * 1024 * 1024
    length = request.headers.get("content-length")
    if length is not None and int(length) > limit:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="too_large")
    form = await request.form()
    uploads = [
        f for f in form.getlist("files[]") + form.getlist("files") if isinstance(f, UploadFile)
    ]
    files: list[SourceFile] = []
    total = 0
    for upload in uploads:
        data = await upload.read()
        total += len(data)
        if total > limit:
            raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="too_large")
        files.append(SourceFile(upload.filename or "upload", data))
    ingest = str(form.get("ingest", "false")).lower() in ("1", "true", "yes", "on")
    if ingest and not settings.ingester.enabled:
        raise HTTPException(status.HTTP_409_CONFLICT, detail="ingester_disabled")
    order_raw = form.get("order")
    order: list[str] | None = None
    if isinstance(order_raw, str) and order_raw.strip():
        try:
            parsed = json.loads(order_raw)
        except ValueError as err:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="invalid_order") from err
        if not isinstance(parsed, list):
            raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="invalid_order")
        order = [str(item) for item in parsed]
    title = form.get("title")
    try:
        priority = int(str(form.get("priority", "0")))
    except ValueError as err:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="invalid_priority") from err

    def _create() -> CreatedDocument:
        conn = connect(settings.database_path)
        try:
            return create_document(
                conn,
                settings,
                files,
                title=title if isinstance(title, str) else None,
                ingest=ingest,
                priority=priority,
                order=order,
            )
        finally:
            conn.close()

    try:
        created = await run_in_threadpool(_create)
    except InvalidSourceError as err:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(err)) from err
    if created.duplicate:
        response.status_code = status.HTTP_200_OK
    return DocumentCreated(
        id=created.document_id,
        title=created.title,
        status=created.status,
        page_count=created.page_count,
        duplicate=created.duplicate,
    )


@api.get("/documents")
def list_documents(
    conn: Conn, settings: Config, status_filter: Annotated[str | None, Query(alias="status")] = None
) -> list[dict[str, Any]]:
    return operations.list_documents(conn, status_filter, settings)


@api.get("/documents/{document_id}")
def get_document(document_id: str, conn: Conn, settings: Config) -> dict[str, Any]:
    try:
        return operations.document_view(conn, document_id, settings)
    except KeyError as err:
        raise _not_found("document") from err


@api.post(
    "/documents/{document_id}/rebuild",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=StatusResponse,
)
def rebuild(document_id: str, conn: Conn) -> StatusResponse:
    try:
        return StatusResponse(status=operations.request_rebuild(conn, document_id))
    except KeyError as err:
        raise _not_found("document") from err


@api.post(
    "/documents/{document_id}/draft",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=DraftBuildCreated,
)
def draft_build(document_id: str, conn: Conn) -> DraftBuildCreated:
    try:
        return DraftBuildCreated(build_id=operations.request_draft_build(conn, document_id))
    except KeyError as err:
        raise _not_found("document") from err


@api.post(
    "/documents/{document_id}/ingest",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=IngestRequested,
)
def ingest(document_id: str, conn: Conn, settings: Config) -> IngestRequested:
    try:
        return IngestRequested(ingest_status=operations.request_ingest(conn, document_id, settings))
    except KeyError as err:
        raise _not_found("document") from err


def _content_disposition(filename: str) -> str:
    fallback = filename.encode("ascii", "ignore").decode() or "file"
    fallback = fallback.replace('"', "")
    return f"attachment; filename=\"{fallback}\"; filename*=UTF-8''{quote(filename)}"


@api.get("/documents/{document_id}/files/{kind}")
def document_file(
    document_id: str,
    kind: str,
    conn: Conn,
    settings: Config,
    version: int | None = None,
    draft: bool = False,
) -> FileResponse:
    if kind not in FILE_KINDS:
        raise _not_found("file")
    doc = repo.get_document(conn, document_id)
    if doc is None:
        raise _not_found("document")
    if draft:
        if kind != "docx":
            raise _not_found("file")
        build = repo.latest_done_build(conn, document_id, BuildKind.DRAFT)
    elif version is not None:
        build = repo.build_by_version(conn, document_id, version)
        if build is not None and (
            build["kind"] != BuildKind.FINAL
            or build["status"] != "done"
            or build["pruned_at"] is not None
        ):
            build = None
    else:
        build = repo.latest_done_build(conn, document_id, BuildKind.FINAL)
    if build is None or not build["output_dir"]:
        raise _not_found("build")
    manifest = json.loads(build["manifest_json"] or "{}")
    name = manifest.get("files", {}).get(kind)
    if not name:
        base = safe_name(str(doc["title"]), document_id)
        name = {"pdf": f"{base}.pdf", "docx": f"{base}.docx", "chunks": "chunks.jsonl"}[kind]
    path = settings.data_dir / str(build["output_dir"]) / str(name)
    if not path.exists():
        raise _not_found("file")
    return FileResponse(
        path,
        media_type=FILE_KINDS[kind],
        headers={"Content-Disposition": _content_disposition(str(name))},
    )


@api.get("/builds/{build_id}", response_model=BuildInfo)
def get_build(build_id: int, conn: Conn) -> BuildInfo:
    build = repo.get_build(conn, build_id)
    if build is None:
        raise _not_found("build")
    return BuildInfo(
        id=int(build["id"]),
        document_id=str(build["document_id"]),
        kind=str(build["kind"]),
        version=int(build["version"]),
        status=str(build["status"]),
        error=build["error"],
    )


@api.get("/deliveries", response_model=list[Delivery])
def deliveries(conn: Conn) -> list[Delivery]:
    return [
        Delivery(
            document_id=str(r["document_id"]), title=str(r["title"]), version=int(r["version"])
        )
        for r in repo.deliveries(conn)
    ]


@api.post("/documents/{document_id}/delivered", status_code=status.HTTP_204_NO_CONTENT)
def delivered(document_id: str, body: DeliveredRequest, conn: Conn) -> Response:
    if repo.get_document(conn, document_id) is None:
        raise _not_found("document")
    with immediate(conn):
        repo.set_delivered(conn, document_id, body.version)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# --- review ------------------------------------------------------------------------------------


@api.get("/review/summary")
def review_summary(request: Request, conn: Conn, settings: Config) -> dict[str, Any]:
    return operations.review_summary(conn, settings, _rules_version(request))


@api.get("/review/queue", response_model=list[QueueEntry])
def review_queue(
    conn: Conn, limit: Annotated[int, Query(ge=1, le=500)] = 50, exclude: str = ""
) -> list[dict[str, Any]]:
    try:
        skip = [int(part) for part in exclude.split(",") if part.strip()]
    except ValueError as err:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="invalid_exclude") from err
    return operations.review_queue(conn, limit=limit, exclude=skip)


@api.get("/review/items/{page_id}")
def review_item(page_id: int, request: Request, conn: Conn, settings: Config) -> dict[str, Any]:
    try:
        view = operations.build_item_view(conn, page_id, settings, _rules_version(request))
    except KeyError as err:
        raise _not_found("page") from err
    return view.to_api()


@api.get("/documents/{document_id}/pages/{page_number}/item")
def page_item(
    document_id: str, page_number: int, request: Request, conn: Conn, settings: Config
) -> dict[str, Any]:
    try:
        view = operations.item_for_page_number(
            conn, document_id, page_number, settings, _rules_version(request)
        )
    except KeyError as err:
        raise _not_found("page") from err
    return view.to_api()


def _page_source(conn: sqlite3.Connection, settings: Settings, page_id: int) -> tuple[Path, int]:
    row = conn.execute(
        "SELECT p.page_index, d.source_path FROM pages p JOIN documents d ON d.id = p.document_id "
        "WHERE p.id = ?",
        (page_id,),
    ).fetchone()
    if row is None:
        raise _not_found("page")
    return settings.data_dir / str(row["source_path"]), int(row["page_index"])


def _parse_clip(clip: str | None) -> tuple[float, float, float, float] | None:
    if not clip:
        return None
    try:
        values = [float(v) for v in clip.split(",")]
    except ValueError as err:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="invalid_clip") from err
    if (
        len(values) != 4
        or not all(0 <= v <= 1 for v in values)
        or values[0] >= values[2]
        or values[1] >= values[3]
    ):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="invalid_clip")
    return values[0], values[1], values[2], values[3]


@api.get("/pages/{page_id}/image")
def page_image(
    page_id: int,
    conn: Conn,
    settings: Config,
    long_side: Annotated[int, Query(ge=100, le=8000)] = 2000,
    clip: str | None = None,
    format: str = "jpeg",
) -> Response:
    if format not in ("jpeg", "png"):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="invalid_format")
    pdf_path, index = _page_source(conn, settings, page_id)
    data = pdfpages.render_view(
        pdf_path,
        index,
        long_side,
        clip_fractions=_parse_clip(clip),
        fmt=format,
        quality=settings.review.jpeg_quality,
    )
    return Response(content=data, media_type=f"image/{format}")


@api.put("/pages/{page_id}/draft", response_model=DraftSaved)
def put_draft(page_id: int, body: DraftRequest, conn: Conn) -> DraftSaved:
    data = operations.DraftInput(**body.model_dump())
    try:
        seq = operations.save_draft(conn, page_id, data)
    except KeyError as err:
        raise _not_found("page") from err
    return DraftSaved(seq=seq, text_sha256=operations.text_sha256(body.text))


@api.delete("/pages/{page_id}/draft", status_code=status.HTTP_204_NO_CONTENT)
def remove_draft(page_id: int, conn: Conn, draft_seq: int = 0) -> Response:
    try:
        operations.delete_draft(conn, page_id, draft_seq)
    except KeyError as err:
        raise _not_found("page") from err
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@api.post(
    "/pages/{page_id}/review", status_code=status.HTTP_201_CREATED, response_model=ReviewResponse
)
def post_review(
    page_id: int, body: ReviewRequest, conn: Conn, response: Response
) -> ReviewResponse:
    try:
        result = operations.submit_review(
            conn, page_id, operations.ReviewInput(**body.model_dump())
        )
    except KeyError as err:
        raise _not_found("page") from err
    if not result.created:
        response.status_code = status.HTTP_200_OK
    return ReviewResponse(revision_id=result.revision_id, document_status=result.document_status)


@api.post(
    "/pages/{page_id}/retry", status_code=status.HTTP_202_ACCEPTED, response_model=StatusResponse
)
def retry(page_id: int, conn: Conn) -> StatusResponse:
    try:
        return StatusResponse(status=operations.retry_page(conn, page_id))
    except KeyError as err:
        raise _not_found("page") from err


@api.post("/pages/{page_id}/ocr-region", response_model=RegionResponse)
async def ocr_region(page_id: int, body: RegionRequest, request: Request) -> RegionResponse:
    settings: Settings = request.app.state.settings
    if body.x0 >= body.x1 or body.y0 >= body.y1:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="invalid_region")
    conn = connect(settings.database_path)
    try:
        pdf_path, index = _page_source(conn, settings, page_id)
    finally:
        conn.close()
    profile = settings.profiles["region"]

    def _render() -> bytes:
        # Rendering is CPU-bound: it runs in a thread so the event loop stays free.
        with pymupdf.open(pdf_path) as doc:
            rect = doc[index].rect
        clip = pymupdf.Rect(
            rect.x0 + body.x0 * rect.width,
            rect.y0 + body.y0 * rect.height,
            rect.x0 + body.x1 * rect.width,
            rect.y0 + body.y1 * rect.height,
        )
        return pdfpages.render_page(pdf_path, index, profile.image_long_side, clip=clip)

    png = await run_in_threadpool(_render)
    try:
        result = await request.app.state.vlm.transcribe(
            png, profile, user_text=f"Фрагмент страницы {index + 1}."
        )
    except VlmTransportError as err:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail="vlm_unavailable") from err
    except VlmRequestError as err:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, detail="vlm_error") from err
    spans = (
        low_confidence_spans(result.token_probs, settings.checks)
        if result.token_probs is not None
        else []
    )
    mapping = Utf16Map(result.text)
    return RegionResponse(
        text=result.text,
        low_conf=[
            LowConfSpan(start=mapping.to_utf16(s.start), end=mapping.to_utf16(s.end), min_p=s.min_p)
            for s in spans
        ],
    )


def status_error_response(err: StatusTransitionError) -> JSONResponse:
    """409 with the machine-readable reason as detail."""
    return JSONResponse(status_code=status.HTTP_409_CONFLICT, content={"detail": err.code})
