"""Shared fixtures: settings in a temp dir, a database, and a fake llama-server."""

from __future__ import annotations

import base64
import json
import math
import re
import sqlite3
import struct
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import pytest

from ocr_archive.db.connection import open_db
from ocr_archive.models import PipelineContext
from ocr_archive.settings import Settings
from ocr_archive.versions import rules_version
from ocr_archive.vlm.client import VlmClient

API_KEY = "test-api-key"


def make_settings(tmp_path: Path, **overrides: Any) -> Settings:
    """Settings rooted in a temp directory."""
    data: dict[str, Any] = {
        "data_dir": tmp_path / "data",
        "export_dir": tmp_path / "export",
        "ocr_api_key": API_KEY,
        "vlm": {"base_url": "http://fake-llama/v1", "model": "qwen-test"},
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(data.get(key), dict):
            data[key] = {**data[key], **value}
        else:
            data[key] = value
    return Settings(**data)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return make_settings(tmp_path)


@pytest.fixture
def conn(settings: Settings) -> Iterator[sqlite3.Connection]:
    connection = open_db(settings.database_path)
    yield connection
    connection.close()


_PAGE_RE = re.compile(r"(?:Страница|Фрагмент страницы) (\d+)")
_PART_RE = re.compile(r"часть (\d+) из (\d+)")


def png_size(png: bytes) -> tuple[int, int]:
    width, height = struct.unpack(">II", png[16:24])
    return width, height


@dataclass
class FakeVlm:
    """Scripted llama-server: answers by page number and pass (detected from image size).

    ``texts`` maps (page number, pass) → text, pass being 1, 2 or ("part", k); pass 0 is the
    fallback for both passes.
    """

    texts: dict[Any, str] = field(default_factory=dict)
    default: str = "Текст страницы без особенностей, обычная машинопись."
    logprobs: bool = True
    low: dict[str, float] = field(default_factory=dict)
    finish: dict[Any, str] = field(default_factory=dict)
    failures: list[Any] = field(default_factory=list)  # status codes or "timeout", consumed FIFO
    calls: list[dict[str, Any]] = field(default_factory=list)
    hook: Callable[[dict[str, Any]], None] | None = None
    pass1_long_side: int = 2000

    def key_for(self, body: dict[str, Any]) -> tuple[int, Any]:
        content = body["messages"][1]["content"]
        user_text = next(c["text"] for c in content if c["type"] == "text")
        url = next(c["image_url"]["url"] for c in content if c["type"] == "image_url")
        png = base64.b64decode(url.split(",", 1)[1])
        number = int(_PAGE_RE.search(user_text).group(1))  # type: ignore[union-attr]
        part = _PART_RE.search(user_text)
        if part:
            return number, ("part", int(part.group(1)))
        width, height = png_size(png)
        return number, 1 if max(width, height) <= self.pass1_long_side else 2

    def text_for(self, key: tuple[int, Any]) -> str:
        if key in self.texts:
            return self.texts[key]
        return self.texts.get((key[0], 0), self.default)

    def completion(self, body: dict[str, Any]) -> dict[str, Any]:
        key = self.key_for(body)
        text = self.text_for(key)
        choice: dict[str, Any] = {
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": self.finish.get(key, "stop"),
        }
        if self.logprobs:
            tokens = re.findall(r"\s+|\S+", text)
            choice["logprobs"] = {
                "content": [
                    {
                        "token": t,
                        "logprob": math.log(self.low[t.strip()]) if t.strip() in self.low else 0.0,
                    }
                    for t in tokens
                ]
            }
        return {"choices": [choice], "usage": {"prompt_tokens": 100, "completion_tokens": 50}}

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "qwen-test"}]})
        body = json.loads(request.content)
        self.calls.append(body)
        if self.hook is not None:
            self.hook(body)
        if self.failures:
            failure = self.failures.pop(0)
            if failure == "timeout":
                raise httpx.ReadTimeout("timeout", request=request)
            if isinstance(failure, tuple):
                return httpx.Response(failure[0], text=failure[1])
            return httpx.Response(int(failure), text="error")
        return httpx.Response(200, json=self.completion(body))

    def keys_called(self) -> list[tuple[int, Any]]:
        return [self.key_for(body) for body in self.calls]


@pytest.fixture
def fake_vlm() -> FakeVlm:
    return FakeVlm()


def make_ctx(conn: sqlite3.Connection, settings: Settings, fake: FakeVlm) -> PipelineContext:
    """Pipeline context with registered profiles and a model client on the fake transport."""
    from ocr_archive.db import repo
    from ocr_archive.db.connection import immediate
    from ocr_archive.status import RULES_VERSION_KEY
    from ocr_archive.worker import register_profiles

    version = rules_version(settings)
    with immediate(conn):
        repo.set_control(conn, RULES_VERSION_KEY, version)
    client = VlmClient(settings.vlm, "", transport=httpx.MockTransport(fake.handler))
    return PipelineContext(
        settings=settings,
        db_path=settings.database_path,
        vlm=client,
        rules_version=version,
        profile_ids=register_profiles(conn, settings),
    )
