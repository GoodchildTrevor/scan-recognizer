"""The P0.5 prototype pipe shares the window code with the real pipe."""

from __future__ import annotations

from typing import Any

import pytest

from tests.owui.helpers import (
    PIPE_PATH,
    PROTOTYPE_PATH,
    Emitter,
    FakeWindow,
    done,
    leave,
    load_module,
)

pytestmark = pytest.mark.p3

proto = load_module("owui_window_prototype", PROTOTYPE_PATH)


def test_shared_block_matches_the_pipe() -> None:
    pipe_block = proto.shared_block(PIPE_PATH.read_text(encoding="utf-8"))
    proto_block = proto.shared_block(PROTOTYPE_PATH.read_text(encoding="utf-8"))
    assert pipe_block.strip(), "shared block is empty"
    assert proto_block == pipe_block, "run: python prototype/owui_window_pipe.py --sync"


async def run_prototype(
    text: str, window: FakeWindow | None = None, task: str | None = None
) -> str:
    pipe = proto.Pipe()
    pipe.valves.allowed_user_ids = ["u1"]
    pipe.valves.build_wait_s = 0
    body: dict[str, Any] = {"messages": [{"role": "user", "content": text}], "stream": True}
    result: str = await pipe.pipe(
        body,
        __user__={"id": "u1"},
        __task__=task,
        __event_emitter__=Emitter(),
        __event_call__=window,
    )
    return result


async def test_prototype_review_runs_on_fake_pages() -> None:
    await run_prototype("сброс")
    window = FakeWindow([done(), leave("later", "черновик прототипа"), leave("stop")])
    reply = await run_prototype("проверка", window)
    assert window.shown_pages == [9001, 9002, 9003]
    assert window.closed
    assert "Проверено 1 страница" in reply
    service = proto._service()
    assert service.drafts[9002]["text"] == "черновик прототипа"
    assert service.pages[9001]["status"] == "reviewed"
    hints = window.payloads[1]["page"]["hints"]
    assert any(h["variants"] for h in hints)
    assert window.payloads[1]["images"]["page"].startswith("data:image/svg+xml;base64,")


async def test_prototype_logs_tasks_and_keeps_drafts_between_calls() -> None:
    await run_prototype("сброс")
    window = FakeWindow([leave("stop", "правка до закрытия вкладки")])
    await run_prototype("проверка", window)
    assert await run_prototype("x", task="title_generation") == '{"title": "OCR-архив"}'
    log = await run_prototype("журнал")
    assert "task title_generation" in log
    second = FakeWindow([leave("stop")])
    await run_prototype("проверка", second)
    assert second.payloads[1]["page"]["draft"]["text"] == "правка до закрытия вкладки"
