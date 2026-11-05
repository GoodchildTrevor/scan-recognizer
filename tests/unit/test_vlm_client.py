"""Request body of the model client (section 3.1)."""

from __future__ import annotations

import pytest

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
