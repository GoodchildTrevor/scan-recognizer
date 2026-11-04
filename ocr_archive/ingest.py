"""Acceptance of source files: storage by hash, image sets to PDF, document creation (10.2)."""

from __future__ import annotations

import hashlib
import io
import logging
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path, PurePath

import pymupdf
from PIL import Image, UnidentifiedImageError
from ulid import ULID

from ocr_archive.db import repo
from ocr_archive.db.connection import immediate
from ocr_archive.exceptions import OcrArchiveError
from ocr_archive.settings import Settings

logger = logging.getLogger(__name__)

DEFAULT_DPI = 300
PDF_SUFFIXES = {".pdf"}
TIFF_SUFFIXES = {".tif", ".tiff"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
JPEG_SUFFIXES = {".jpg", ".jpeg"}


class InvalidSourceError(OcrArchiveError):
    """The uploaded files are not a PDF or a set of images."""


@dataclass(frozen=True)
class SourceFile:
    name: str
    data: bytes


@dataclass(frozen=True)
class CreatedDocument:
    document_id: str
    title: str
    status: str
    page_count: int
    duplicate: bool


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def natural_key(name: str) -> list[object]:
    """Sort key that orders 1, 2, 10 numerically."""
    return [int(part) if part.isdigit() else part.casefold() for part in re.split(r"(\d+)", name)]


def _suffix(name: str) -> str:
    return PurePath(name).suffix.lower()


def order_images(files: list[SourceFile], order: list[str] | None) -> list[SourceFile]:
    """Order images by the explicit list of names, else by natural sort of names."""
    if order:
        by_name = {f.name: f for f in files}
        missing = [name for name in order if name not in by_name]
        if missing or len(order) != len(files):
            raise InvalidSourceError("order must list every uploaded file exactly once")
        return [by_name[name] for name in order]
    return sorted(files, key=lambda f: natural_key(f.name))


def image_set_sha256(images: list[SourceFile]) -> str:
    """Hash of the ordered list of image hashes; independent of the generated PDF's metadata."""
    return _sha256("\n".join(_sha256(img.data) for img in images).encode("ascii"))


def _frames(image: SourceFile) -> list[tuple[bytes, int, int, float, float]]:
    """Return (stream, width px, height px, dpi x, dpi y) per frame; JPEG streams stay as is."""
    try:
        with Image.open(io.BytesIO(image.data)) as img:
            frames = []
            count = getattr(img, "n_frames", 1)
            for index in range(count):
                img.seek(index)
                dpi = img.info.get("dpi") or (DEFAULT_DPI, DEFAULT_DPI)
                dpi_x = float(dpi[0]) or DEFAULT_DPI
                dpi_y = float(dpi[1]) or DEFAULT_DPI
                if _suffix(image.name) in JPEG_SUFFIXES and count == 1:
                    stream = image.data
                else:
                    buffer = io.BytesIO()
                    frame = img if img.mode in ("1", "L", "RGB") else img.convert("RGB")
                    frame.save(buffer, format="PNG")
                    stream = buffer.getvalue()
                frames.append((stream, img.width, img.height, dpi_x, dpi_y))
            return frames
    except (UnidentifiedImageError, OSError) as err:
        raise InvalidSourceError(f"{image.name} is not a readable image") from err


def images_to_pdf(images: list[SourceFile]) -> bytes:
    """Build a PDF with one page per image frame; JPEG is embedded without re-encoding.

    The page size follows the image DPI (300 when missing).
    """
    doc = pymupdf.open()
    try:
        for image in images:
            for stream, width, height, dpi_x, dpi_y in _frames(image):
                page_w = width / dpi_x * 72
                page_h = height / dpi_y * 72
                page = doc.new_page(width=page_w, height=page_h)
                page.insert_image(page.rect, stream=stream)
        return bytes(doc.tobytes(garbage=3, deflate=True))
    finally:
        doc.close()


def _pdf_page_count(data: bytes, name: str) -> int:
    try:
        with pymupdf.open(stream=data, filetype="pdf") as doc:
            if doc.needs_pass:
                raise InvalidSourceError(f"{name} is encrypted")
            return int(doc.page_count)
    except (RuntimeError, ValueError) as err:
        raise InvalidSourceError(f"{name} is not a readable PDF") from err


def store_source(
    settings: Settings, sha256: str, pdf: bytes, images: list[SourceFile] | None = None
) -> str:
    """Write the source PDF (and original images) under sources/<sha256>/.

    :return: path of the PDF relative to ``data_dir``.
    """
    folder = settings.sources_dir / sha256
    folder.mkdir(parents=True, exist_ok=True)
    pdf_path = folder / "source.pdf"
    if not pdf_path.exists():
        tmp = folder / ".source.pdf.tmp"
        tmp.write_bytes(pdf)
        tmp.replace(pdf_path)
    if images:
        image_dir = folder / "images"
        image_dir.mkdir(exist_ok=True)
        for index, image in enumerate(images, start=1):
            target = image_dir / f"{index:04d}{_suffix(image.name) or '.img'}"
            if not target.exists():
                target.write_bytes(image.data)
    return pdf_path.relative_to(settings.data_dir).as_posix()


def create_document(
    conn: sqlite3.Connection,
    settings: Settings,
    files: list[SourceFile],
    *,
    title: str | None = None,
    ingest: bool = False,
    priority: int = 0,
    order: list[str] | None = None,
) -> CreatedDocument:
    """Accept one PDF, one TIFF or a set of images as a queued document.

    A repeated upload of the same source returns the existing document with ``duplicate``.

    :raises InvalidSourceError: unsupported or unreadable files.
    """
    if not files:
        raise InvalidSourceError("no files")
    suffixes = {_suffix(f.name) for f in files}
    images: list[SourceFile] | None = None
    if len(files) == 1 and suffixes <= PDF_SUFFIXES:
        source = files[0]
        sha = _sha256(source.data)
        pdf = source.data
    elif suffixes <= IMAGE_SUFFIXES:
        images = order_images(files, order)
        sha = image_set_sha256(images)
        pdf = b""
    else:
        raise InvalidSourceError("upload one PDF or a set of images")

    existing = repo.find_document_by_sha(conn, sha)
    if existing is not None:
        return CreatedDocument(
            str(existing["id"]),
            str(existing["title"]),
            str(existing["status"]),
            int(existing["page_count"]),
            True,
        )
    if images is not None:
        pdf = images_to_pdf(images)
    page_count = _pdf_page_count(pdf, files[0].name)
    if page_count == 0:
        raise InvalidSourceError("the document has no pages")
    source_path = store_source(settings, sha, pdf, images)
    document_id = str(ULID())
    doc_title = (title or "").strip() or PurePath(files[0].name).stem or document_id
    original_name = files[0].name if len(files) == 1 else f"{len(files)} images"
    with immediate(conn):
        existing = repo.find_document_by_sha(conn, sha)
        if existing is not None:
            return CreatedDocument(
                str(existing["id"]),
                str(existing["title"]),
                str(existing["status"]),
                int(existing["page_count"]),
                True,
            )
        repo.insert_document(
            conn,
            document_id=document_id,
            title=doc_title,
            source_sha256=sha,
            source_path=source_path,
            original_name=original_name,
            page_count=page_count,
            priority=priority,
            ingest_requested=ingest,
        )
    logger.info("document %s queued: %d pages", document_id, page_count)
    return CreatedDocument(document_id, doc_title, "queued", page_count, False)


def read_source_files(paths: list[Path]) -> list[SourceFile]:
    """Read files given on the command line."""
    return [SourceFile(path.name, path.read_bytes()) for path in paths]
