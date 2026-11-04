"""FastAPI application factory."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from ocr_archive import __version__
from ocr_archive.api.routes import api, router, status_error_response
from ocr_archive.db.connection import open_db
from ocr_archive.exceptions import StatusTransitionError
from ocr_archive.settings import Settings, load_settings
from ocr_archive.versions import rules_version
from ocr_archive.vlm.client import VlmClient

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None, *, vlm: VlmClient | None = None) -> FastAPI:
    """Create the API application.

    :param settings: configuration; loaded from config.yaml and the environment when omitted.
    :param vlm: model client for region recognition; created from settings when omitted.
    """
    config = settings if settings is not None else load_settings()
    conn = open_db(config.database_path)
    conn.close()
    client = (
        vlm if vlm is not None else VlmClient(config.vlm, config.llama_api_key.get_secret_value())
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        yield
        await client.aclose()

    app = FastAPI(title="ocr-archive", version=__version__, lifespan=lifespan)
    app.state.settings = config
    app.state.rules_version = rules_version(config)
    app.state.vlm = client

    @app.exception_handler(StatusTransitionError)
    async def _conflict(request: Request, err: StatusTransitionError) -> JSONResponse:
        return status_error_response(err)

    @app.exception_handler(Exception)
    async def _internal(request: Request, err: Exception) -> JSONResponse:
        # Broad on purpose: any unexpected error becomes 500 without touching statuses.
        logger.exception("request %s %s failed", request.method, request.url.path)
        return JSONResponse(status_code=500, content={"detail": "internal_error"})

    app.include_router(router)
    app.include_router(api)
    return app
