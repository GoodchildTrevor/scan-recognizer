"""Source acceptance (10.2) and the ingester client (3.2, p6)."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pymupdf
import pytest

from ocr_archive.exceptions import ConfigError, IngestError
from ocr_archive.ingest import (
    SourceFile,
    create_document,
    image_set_sha256,
    images_to_pdf,
    natural_key,
)
from ocr_archive.ingester_client import upload_chunks
from ocr_archive.settings import IngesterConfig, load_settings
from tests.fixtures.factory import make_jpeg_set


@pytest.mark.p3
def test_image_set_hash_ignores_pdf_metadata(tmp_path: Path) -> None:
    images = [SourceFile(p.name, p.read_bytes()) for p in make_jpeg_set(tmp_path)]
    first = images_to_pdf(images)
    second = images_to_pdf(images)
    assert image_set_sha256(images) == image_set_sha256(list(images))
    assert image_set_sha256(images) != image_set_sha256(list(reversed(images)))
    assert len(first) > 0 and len(second) > 0


@pytest.mark.p3
def test_image_set_document(conn, settings, tmp_path: Path) -> None:
    paths = make_jpeg_set(tmp_path)
    files = [SourceFile(p.name, p.read_bytes()) for p in reversed(paths)]
    created = create_document(conn, settings, files)
    assert created.page_count == 3 and not created.duplicate
    pdf = (
        settings.data_dir
        / f"sources/{image_set_sha256(sorted(files, key=lambda f: natural_key(f.name)))}/source.pdf"
    )
    with pymupdf.open(pdf) as doc:
        xref = doc[0].get_images()[0][0]
        assert doc.extract_image(xref)["image"] == paths[0].read_bytes()
        assert abs(doc[0].rect.width - 1240 * 72 / 300) < 72  # page size from the 300 dpi metadata
    again = create_document(conn, settings, list(files))
    assert again.duplicate and again.document_id == created.document_id
    assert natural_key("2.jpg") < natural_key("10.jpg")


@pytest.mark.p6
async def test_upload_chunks_contract(tmp_path: Path) -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = request.content.decode("utf-8", errors="replace")
        seen["body"] = body
        seen["key"] = request.headers.get("X-API-Key")
        return httpx.Response(200, json={"status": "success", "chunks_failed": 0})

    path = tmp_path / "chunks.jsonl"
    path.write_text(json.dumps({"chunk_id": "x", "content": "текст"}) + "\n", encoding="utf-8")
    cfg = IngesterConfig(enabled=True, api_key="secret")
    result = await upload_chunks(path, "01DOC", cfg, transport=httpx.MockTransport(handler))
    assert result["status"] == "success"
    body = str(seen["body"])
    assert 'filename="01DOC.jsonl"' in body
    assert "inline_threshold" not in body
    assert 'name="prechunked"' in body and 'name="extra_payload"' in body
    assert seen["key"] == "secret"


@pytest.mark.p6
@pytest.mark.parametrize(
    "answer",
    [
        {"status": "partial", "chunks_failed": 2},
        {"status": "failed", "chunks_failed": 0},
        {"status": "success", "chunks_failed": 1},
    ],
)
async def test_partial_is_failure(tmp_path: Path, answer: dict[str, object]) -> None:
    path = tmp_path / "chunks.jsonl"
    path.write_text("{}\n", encoding="utf-8")
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=answer))
    with pytest.raises(IngestError):
        await upload_chunks(path, "01DOC", IngesterConfig(enabled=True), transport=transport)


@pytest.mark.p6
def test_reserved_payload_keys_rejected(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("ingester:\n  extra_payload: {name: x}\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_settings(config)
