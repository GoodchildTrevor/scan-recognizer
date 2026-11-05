"""The model client: request body and error classification (section 3.1)."""

from __future__ import annotations

import httpx
import pytest

from ocr_archive.exceptions import VlmRequestError, VlmTransportError
from ocr_archive.settings import ProfileConfig, VlmConfig
from ocr_archive.vlm.client import VlmClient

pytestmark = pytest.mark.p1


def test_request_carries_sampling_and_repetition_penalties() -> None:
    client = VlmClient(VlmConfig(), "")
    profile = ProfileConfig(prompt="transcribe_v1", image_long_side=2000)

    body = client.request_body(b"png", profile, user_text="страница 1")

    assert body["temperature"] == 0.0
    assert body["top_p"] == 1.0
    assert body["presence_penalty"] == 0.0
    assert body["frequency_penalty"] == 0.0
    assert body["repeat_penalty"] == 1.0
    assert body["logprobs"] is True
    assert body["top_logprobs"] == 1


async def _chat_with(response: httpx.Response) -> None:
    client = VlmClient(VlmConfig(), "", transport=httpx.MockTransport(lambda request: response))
    try:
        await client.chat({"model": "m", "messages": []})
    finally:
        await client.aclose()


async def test_a_model_pushed_out_by_the_router_is_a_transient_error() -> None:
    body = (
        '{"error":{"message":"litellm.InternalServerError: OpenAIException - '
        'model name=qwen3.6-35b failed to load"}}'
    )

    with pytest.raises(VlmTransportError):
        await _chat_with(httpx.Response(500, text=body))


async def test_another_server_error_is_permanent() -> None:
    with pytest.raises(VlmRequestError):
        await _chat_with(httpx.Response(500, text='{"error":"boom"}'))
