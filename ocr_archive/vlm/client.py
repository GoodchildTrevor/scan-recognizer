"""Client of llama-server's OpenAI-compatible API (section 3.1)."""

from __future__ import annotations

import base64
import logging
import time
from typing import Any

import httpx

from ocr_archive.exceptions import (
    VlmInvalidResponseError,
    VlmRequestError,
    VlmTimeoutError,
    VlmTransportError,
)
from ocr_archive.models import VlmResult
from ocr_archive.settings import ProfileConfig, VlmConfig
from ocr_archive.vlm import load_prompt
from ocr_archive.vlm.logprobs import parse_logprobs

logger = logging.getLogger(__name__)

_TRANSIENT_STATUS = frozenset({502, 503, 504})


def _is_context_error(status_code: int, body: str) -> bool:
    lowered = body.lower()
    return (
        status_code == 400
        and "context" in lowered
        and ("exceed" in lowered or "size" in lowered or "too long" in lowered)
    )


class VlmClient:
    """Asynchronous client; one instance per process and event loop.

    :param cfg: ``vlm`` block of the settings.
    :param api_key: value of ``LLAMA_API_KEY``.
    :param transport: optional httpx transport, used by tests.
    """

    def __init__(
        self,
        cfg: VlmConfig,
        api_key: str,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._cfg = cfg
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.AsyncClient(
            base_url=cfg.base_url.rstrip("/") + "/",
            headers=headers,
            timeout=httpx.Timeout(cfg.timeout_s, connect=10.0),
            transport=transport,
        )

    @property
    def model(self) -> str:
        return self._cfg.model

    async def aclose(self) -> None:
        """Close the HTTP connection pool."""
        await self._client.aclose()

    def request_body(
        self, png: bytes, profile: ProfileConfig, *, user_text: str, prompt: str | None = None
    ) -> dict[str, Any]:
        """Return the request body used everywhere (section 3.1)."""
        image_url = "data:image/png;base64," + base64.b64encode(png).decode("ascii")
        return {
            "model": self._cfg.model,
            "messages": [
                {"role": "system", "content": load_prompt(prompt or profile.prompt)},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text", "text": user_text},
                    ],
                },
            ],
            "temperature": profile.temperature,
            "top_p": profile.top_p,
            "max_tokens": profile.max_tokens,
            "logprobs": True,
            "top_logprobs": profile.top_logprobs,
            "chat_template_kwargs": {"enable_thinking": False},
        }

    async def chat(self, body: dict[str, Any]) -> dict[str, Any]:
        """Send a chat completion request and return the parsed JSON.

        :raises VlmTransportError: timeout, connection error, 502/503/504 or "loading model".
        :raises VlmRequestError: any other HTTP error; ``context_exceeded`` for a context overflow.
        :raises VlmInvalidResponseError: the body is not JSON.
        """
        try:
            response = await self._client.post("chat/completions", json=body)
        except httpx.TimeoutException as err:
            raise VlmTimeoutError(f"timeout: {err.__class__.__name__}") from err
        except httpx.TransportError as err:
            raise VlmTransportError(f"connection error: {err.__class__.__name__}") from err
        text = response.text
        if response.status_code in _TRANSIENT_STATUS or (
            response.status_code >= 400 and "loading model" in text.lower()
        ):
            raise VlmTransportError(f"server unavailable: HTTP {response.status_code}")
        if response.status_code >= 400:
            raise VlmRequestError(
                f"HTTP {response.status_code}: {text[:300]}",
                status_code=response.status_code,
                context_exceeded=_is_context_error(response.status_code, text),
            )
        try:
            data = response.json()
        except ValueError as err:
            raise VlmInvalidResponseError("response is not JSON") from err
        if not isinstance(data, dict):
            raise VlmInvalidResponseError("response is not a JSON object")
        return data

    async def transcribe(
        self, png: bytes, profile: ProfileConfig, *, user_text: str, prompt: str | None = None
    ) -> VlmResult:
        """Transcribe one image.

        :param png: PNG image.
        :param profile: rendering and sampling profile.
        :param user_text: user message, e.g. "Страница 12.".
        :param prompt: prompt name overriding the profile's, e.g. for split parts.
        :raises VlmInvalidResponseError: the response has no choices.
        """
        started = time.monotonic()
        data = await self.chat(self.request_body(png, profile, user_text=user_text, prompt=prompt))
        duration_ms = int((time.monotonic() - started) * 1000)
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise VlmInvalidResponseError("response has no choices")
        choice = choices[0]
        message = choice.get("message") or {}
        content = message.get("content")
        if not isinstance(content, str):
            raise VlmInvalidResponseError("response has no message content")
        logprobs = choice.get("logprobs") or {}
        entries = logprobs.get("content") if isinstance(logprobs, dict) else None
        usage = data.get("usage") or {}
        return VlmResult(
            text=content,
            finish_reason=choice.get("finish_reason"),
            tokens_in=usage.get("prompt_tokens"),
            tokens_out=usage.get("completion_tokens"),
            duration_ms=duration_ms,
            token_probs=parse_logprobs(content, entries),
        )

    async def models(self) -> list[str]:
        """Return model ids from ``GET /models``.

        :raises VlmTransportError: the server is not reachable.
        """
        try:
            response = await self._client.get("models")
            response.raise_for_status()
        except httpx.HTTPError as err:
            raise VlmTransportError(f"GET /models failed: {err.__class__.__name__}") from err
        data = response.json()
        return [str(item.get("id")) for item in data.get("data", []) if isinstance(item, dict)]

    async def props(self) -> dict[str, Any] | None:
        """Return llama-server ``GET /props`` (outside ``/v1``), or None if unavailable."""
        url = self._cfg.base_url.rstrip("/").removesuffix("/v1") + "/props"
        try:
            response = await self._client.get(url)
            response.raise_for_status()
            data = response.json()
        except (httpx.HTTPError, ValueError):
            return None
        return data if isinstance(data, dict) else None

    async def is_alive(self) -> bool:
        """Tell whether the server answers ``GET /models``."""
        try:
            await self.models()
        except VlmTransportError:
            return False
        return True
