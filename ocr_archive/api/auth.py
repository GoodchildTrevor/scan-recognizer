"""X-API-Key authentication: the clients are the OpenWebUI pipe and the CLI."""

from __future__ import annotations

import hmac

from fastapi import Header, HTTPException, Request, status


def require_api_key(request: Request, x_api_key: str | None = Header(default=None)) -> None:
    """Reject requests without the configured key; an empty configured key rejects everything.

    :raises HTTPException: 401 on a missing or wrong key.
    """
    expected: str = request.app.state.settings.ocr_api_key.get_secret_value()
    if not expected or x_api_key is None or not hmac.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="unauthorized")
