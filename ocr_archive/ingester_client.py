"""Client of the existing qdrant-ingester (section 3.2)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import httpx

from ocr_archive.exceptions import IngestError
from ocr_archive.settings import IngesterConfig

logger = logging.getLogger(__name__)


def _headers(cfg: IngesterConfig) -> dict[str, str]:
    key = cfg.api_key.get_secret_value()
    return {"X-API-Key": key} if key else {}


async def upload_chunks(
    path: Path,
    document_id: str,
    cfg: IngesterConfig,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> dict[str, object]:
    """Upload ``chunks.jsonl`` as ``<document_id>.jsonl``; a re-upload overwrites the same points.

    ``inline_threshold`` is never sent: a small file would come back as text and nothing would
    be written to Qdrant. HTTP 200 is returned even on failure, so success means
    ``status == "success"`` and ``chunks_failed == 0``.

    :param transport: optional httpx transport, used by tests.
    :raises IngestError: network error, HTTP error, ``partial`` or ``failed``.
    """
    files = {"file": (f"{document_id}.jsonl", path.read_bytes(), "application/x-ndjson")}
    data = {
        "collection": cfg.collection,
        "prechunked": "true",
        "extra_payload": json.dumps(cfg.extra_payload, ensure_ascii=False),
    }
    try:
        async with httpx.AsyncClient(timeout=cfg.timeout_s, transport=transport) as client:
            response = await client.post(
                cfg.base_url.rstrip("/") + "/ingest_upload",
                files=files,
                data=data,
                headers=_headers(cfg),
            )
    except httpx.HTTPError as err:
        raise IngestError(f"ingester unreachable: {err.__class__.__name__}") from err
    if response.status_code >= 400:
        raise IngestError(f"ingester HTTP {response.status_code}: {response.text[:300]}")
    try:
        body: dict[str, Any] = response.json()
    except ValueError as err:
        raise IngestError("ingester response is not JSON") from err
    if body.get("status") != "success" or int(body.get("chunks_failed") or 0) != 0:
        raise IngestError(f"ingest {body.get('status')}: {body.get('chunks_failed')} chunks failed")
    logger.info("document %s ingested into %s", document_id, cfg.collection)
    return body


async def ingester_health(cfg: IngesterConfig) -> dict[str, Any]:
    """Return ``GET /health`` of the ingester.

    :raises IngestError: the ingester is not reachable or unhealthy.
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(cfg.base_url.rstrip("/") + "/health", headers=_headers(cfg))
            response.raise_for_status()
            data = response.json()
    except (httpx.HTTPError, ValueError) as err:
        raise IngestError(f"ingester health check failed: {err.__class__.__name__}") from err
    return data if isinstance(data, dict) else {"raw": data}
