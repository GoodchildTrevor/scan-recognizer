"""The review session of the pipe with a scripted window (sections 8 and 14)."""

from __future__ import annotations

import time
from typing import Any

import pytest

from tests.owui.helpers import (
    Emitter,
    FakeFiles,
    FakeService,
    FakeWindow,
    PipeForTest,
    discard,
    done,
    draft,
    keep_mine,
    leave,
    op,
    region,
    run_pipe,
)

pytestmark = pytest.mark.p3

PAGE_TEXT = {1: "Первая страница 47 опытов", 2: "Вторая страница 12,5 га", 3: "Третья 31 ц/га"}


def service_with_pages(*page_ids: int) -> FakeService:
    service = FakeService()
    service.add_document("d1", "Отчёт 1983")
    for page_id in page_ids:
        service.add_page(page_id, "d1", page_id, PAGE_TEXT[page_id])
    return service


async def review(
    service: FakeService,
    window: FakeWindow,
    *,
    files: FakeFiles | None = None,
    emitter: Emitter | None = None,
    build_wait_s: int = 0,
) -> str:
    pipe = PipeForTest(service, files)
    pipe.valves.build_wait_s = build_wait_s
    return await run_pipe(pipe, "Проверить накопленное", window=window, emitter=emitter)


def last_acks(window: FakeWindow, kind: str) -> list[dict[str, Any]]:
    return [a for a in window.acks if a["type"] == kind]


async def test_done_on_every_page_reviews_the_queue_and_closes_the_window() -> None:
    service = service_with_pages(1, 2)
    window = FakeWindow([done(), done("Вторая страница 12,5 га, исправлено")])
    reply = await review(service, window)
    assert window.shown_pages == [1, 2]
    assert window.closed and not window.exhausted
    assert service.pages[2]["text"] == "Вторая страница 12,5 га, исправлено"
    assert {p["status"] for p in service.pages.values()} == {"reviewed"}
    assert reply.endswith("Проверено 2 страницы, готово 0 документов, в очереди 0.")
    show = window.payloads[1]
    assert show["page"]["header"]["counter"] == "1 из 2"
    assert show["page"]["header"]["reasons"] == "3 расхождения проходов"
    assert show["intro"].startswith("К проверке 2 страницы")
    assert show["images"]["page"].startswith("data:image/jpeg;base64,")
    assert [a["status"] for a in last_acks(window, "done")] == ["ok", "ok"]


async def test_draft_events_save_with_increasing_save_numbers() -> None:
    service = service_with_pages(1)
    window = FakeWindow([draft("правка 1"), draft("правка 2"), done("правка 3")])
    await review(service, window)
    puts = service.calls_to("PUT", "/api/pages/1/draft")
    assert [c.body["expected_seq"] for c in puts] == [0, 1]
    assert [(a["status"], a["seq"]) for a in last_acks(window, "draft")] == [("ok", 1), ("ok", 2)]
    review_call = service.calls_to("POST", "/api/pages/1/review")[0]
    assert review_call.body["draft_seq"] == 2
    assert service.pages[1]["text"] == "правка 3"


async def test_later_keeps_the_draft_and_does_not_show_the_page_again() -> None:
    service = service_with_pages(1, 2)
    window = FakeWindow([leave("later", "начатая правка"), done()])
    await review(service, window)
    assert window.shown_pages == [1, 2]
    assert service.drafts[1]["text"] == "начатая правка"
    queue_calls = service.calls_to("GET", "/api/review/queue")
    assert queue_calls[-1].params["exclude"] == "1,2"


async def test_back_reopens_the_previous_page_and_a_new_done_makes_a_new_revision() -> None:
    service = service_with_pages(1, 2)
    window = FakeWindow([done(), leave("back"), done("снова"), done()])
    await review(service, window)
    assert window.shown_pages == [1, 2, 1, 2]
    assert window.payloads[1]["can_back"] is False
    assert window.payloads[2]["can_back"] is True
    reopened = window.payloads[3]["page"]
    assert reopened["text"] == PAGE_TEXT[1] and reopened["base_revision_id"] == 100
    assert len(service.revisions) == 3
    assert service.pages[1]["text"] == "снова"


async def test_region_result_goes_back_to_the_window() -> None:
    service = service_with_pages(1)
    service.region_text = "прочитано"
    rect = {"x0": 0.1, "y0": 0.2, "x1": 0.5, "y1": 0.3}
    window = FakeWindow([region(rect), done()])
    await review(service, window)
    assert service.calls_to("POST", "/api/pages/1/ocr-region")[0].body == rect
    assert last_acks(window, "region")[0] | {"event_id": None} == {
        "event_id": None,
        "page_id": 1,
        "type": "region",
        "status": "ok",
        "text": "прочитано",
    }


async def test_retry_sends_the_draft_first_and_skips_the_page() -> None:
    service = service_with_pages(1)
    service.pages[1]["status"] = "failed"
    window = FakeWindow([leave("retry", "набрал руками")])
    await review(service, window)
    methods = [(c.method, c.path) for c in service.calls if c.path.startswith("/api/pages/1/")]
    assert methods[-2:] == [("PUT", "/api/pages/1/draft"), ("POST", "/api/pages/1/retry")]
    assert service.pages[1]["status"] == "pending"
    assert service.drafts[1]["text"] == "набрал руками"
    assert window.closed


async def test_stale_done_then_keep_mine_finishes_with_on_conflict() -> None:
    service = service_with_pages(1)

    def recognized_again(w: FakeWindow, payload: dict[str, Any]) -> dict[str, Any]:
        service.pages[1]["base_attempt_id"] = 11  # recognized again meanwhile
        return done("мой текст")(w, payload)

    window = FakeWindow([recognized_again, keep_mine("мой текст", finish=True)])
    await review(service, window)
    stale = last_acks(window, "done")[0]
    assert stale["status"] == "stale"
    assert stale["item"]["base_attempt_id"] == 11
    final = service.calls_to("POST", "/api/pages/1/review")[-1]
    assert final.body["on_conflict"] == "keep_mine"
    assert final.body["base_attempt_id"] == 11
    assert service.pages[1]["text"] == "мой текст"
    assert last_acks(window, "keep_mine")[0]["finish"] is True


async def test_keep_mine_without_finish_saves_a_draft_on_the_new_base() -> None:
    service = service_with_pages(1)
    service.drafts[1] = {
        "seq": 3,
        "text": "старая правка",
        "base_attempt_id": 9,
        "base_revision_id": None,
        "item_version": "v0",
    }
    window = FakeWindow([keep_mine("старая правка", finish=False), done("старая правка")])
    await review(service, window)
    shown = window.payloads[1]["page"]
    assert shown["draft"]["stale"] is True
    put = service.calls_to("PUT", "/api/pages/1/draft")[0]
    assert put.body["expected_seq"] == 3 and put.body["base_attempt_id"] == 10
    assert service.revisions


async def test_discard_draft_deletes_it_and_reshows_the_page() -> None:
    service = service_with_pages(1)
    service.drafts[1] = {
        "seq": 2,
        "text": "брошенная правка",
        "base_attempt_id": 9,
        "base_revision_id": None,
        "item_version": "v0",
    }
    window = FakeWindow([discard(), done()])
    await review(service, window)
    delete = service.calls_to("DELETE", "/api/pages/1/draft")[0]
    assert delete.params == {"draft_seq": "2"}
    reshow = window.payloads[2]
    assert reshow["op"] == "show" and reshow["reuse_images"] is True
    assert reshow["page"]["draft"] is None and reshow["images"] is None
    assert service.pages[1]["status"] == "reviewed"


async def test_stop_saves_the_draft_and_closes() -> None:
    service = service_with_pages(1, 2)
    window = FakeWindow([leave("stop", "недописанное")])
    reply = await review(service, window)
    assert service.drafts[1]["text"] == "недописанное"
    assert window.closed and window.shown_pages == [1]
    assert "Проверено 0 страниц" in reply


async def test_window_error_pauses_the_review() -> None:
    service = service_with_pages(1, 2)
    window = FakeWindow([done()])
    window.steps.append({"error": "client disconnected"})
    reply = await review(service, window)
    assert reply.endswith("Проверка приостановлена, осталось 1 страница.")
    assert not window.closed


async def test_done_while_building_keeps_the_answer_and_skips_the_page() -> None:
    service = service_with_pages(1, 2)
    service.documents["d1"]["status"] = "building"
    window = FakeWindow([done("правка во время сборки"), done()])
    await review(service, window)
    building = last_acks(window, "done")[0]
    assert building["status"] == "building" and building["seq"] == 1
    assert service.drafts[1]["text"] == "правка во время сборки"
    assert window.shown_pages == [1, 2]


async def test_draft_conflict_answer_carries_the_server_draft() -> None:
    service = service_with_pages(1)

    def other_tab_saved(w: FakeWindow, payload: dict[str, Any]) -> dict[str, Any]:
        service.drafts[1] = {
            "seq": 4,
            "text": "из другой вкладки",
            "base_attempt_id": 10,
            "base_revision_id": None,
            "item_version": "v1-1",
        }
        return draft("из этой вкладки")(w, payload)

    window = FakeWindow([other_tab_saved, leave("stop")])
    await review(service, window)
    conflict = last_acks(window, "draft")[0]
    assert conflict["status"] == "draft_conflict"
    assert conflict["server_draft"]["text"] == "из другой вкладки"
    assert service.drafts[1]["text"] == "из другой вкладки"


async def test_outbox_answers_are_sent_first() -> None:
    service = service_with_pages(1, 2, 3)
    service.drafts[3] = {
        "seq": 5,
        "text": "другая вкладка",
        "base_attempt_id": 30,
        "base_revision_id": None,
    }

    def answer(page_id: int, text: str, **extra: Any) -> dict[str, Any]:
        review = {
            "request_id": f"r{page_id}",
            "item_version": f"v1-{page_id}",
            "base_attempt_id": page_id * 10,
            "base_revision_id": None,
            "draft_seq": 0,
            "text": text,
            "printed_label": None,
            "duration_s": 1,
            "choices": {},
            **extra,
        }
        return {"request_id": f"r{page_id}", "page_id": page_id, "review": review}

    outbox = [
        answer(1, "готово офлайн"),
        answer(2, "устарело", base_attempt_id=19),
        answer(3, "конфликт"),
    ]
    window = FakeWindow([], outbox=outbox)
    await review(service, window)
    by_request = {a["request_id"]: a for a in window.outbox_acks}
    assert by_request["r1"] == {"request_id": "r1", "action": "remove"}
    assert by_request["r2"] == {"request_id": "r2", "action": "remove", "seq": 1}
    assert by_request["r3"] == {"request_id": "r3", "action": "to_local"}
    assert service.pages[1]["text"] == "готово офлайн"
    stale_draft = service.drafts[2]
    assert stale_draft["text"] == "устарело" and stale_draft["keep_stale"] is True
    assert service.drafts[3]["text"] == "другая вкладка"


async def test_outbox_answer_for_a_building_document_stays() -> None:
    service = service_with_pages(1)
    service.documents["d1"]["status"] = "building"
    review_body = {
        "request_id": "r1",
        "item_version": "v1-1",
        "base_attempt_id": 10,
        "base_revision_id": None,
        "draft_seq": 0,
        "text": "x",
    }
    window = FakeWindow(
        [leave("stop")], outbox=[{"request_id": "r1", "page_id": 1, "review": review_body}]
    )
    await review(service, window)
    assert window.outbox_acks == [{"request_id": "r1", "action": "keep"}]


@pytest.mark.p4
async def test_build_wait_is_bounded() -> None:
    service = service_with_pages(1)
    service.documents["d1"]["status_sequence"] = ["building"] * 1000
    window = FakeWindow([done()])
    started = time.monotonic()
    reply = await review(service, window, build_wait_s=1)
    assert time.monotonic() - started < 5
    assert "готово 0 документов" in reply


@pytest.mark.p4
async def test_built_documents_are_delivered_and_confirmed() -> None:
    service = service_with_pages(1)
    service.documents["d1"]["status_sequence"] = ["building", "done"]

    def finish_and_build(w: FakeWindow, payload: dict[str, Any]) -> dict[str, Any]:
        service.pending_deliveries = [{"document_id": "d1", "title": "Отчёт 1983", "version": 1}]
        return done()(w, payload)

    files = FakeFiles()
    emitter = Emitter()
    reply = await review(
        service, FakeWindow([finish_and_build]), files=files, emitter=emitter, build_wait_s=2
    )
    assert [f.name for f in files.stored] == ["Отчёт 1983.pdf", "Отчёт 1983.docx"]
    file_events = emitter.of_type("files")
    assert [e["name"] for e in file_events[0]["data"]["files"]] == [
        "Отчёт 1983.pdf",
        "Отчёт 1983.docx",
    ]
    assert service.delivered == [("d1", 1)]
    assert "готово 1 документ" in reply
    assert "Готовые файлы в чате: «Отчёт 1983»." in reply


@pytest.mark.p4
async def test_delivery_is_not_confirmed_when_the_upload_fails() -> None:
    service = service_with_pages(1)
    service.pending_deliveries = [{"document_id": "d1", "title": "Отчёт 1983", "version": 2}]
    files = FakeFiles()
    files.fail_store = True
    reply = await review(service, FakeWindow([done()]), files=files)
    assert service.delivered == []
    assert "Не удалось прикрепить файлы «Отчёт 1983»" in reply


async def test_large_images_go_through_the_file_store_and_are_removed() -> None:
    service = service_with_pages(1)
    service.images[1] = b"\xff\xd8" + b"0" * 1_000_000
    files = FakeFiles()
    window = FakeWindow([done()])
    await review(service, window, files=files)
    show = window.payloads[1]
    assert show["images"]["page"] == "/api/v1/files/f1/content"
    assert files.deleted == ["f1"]


async def test_crops_are_fetched_with_the_pass2_long_side() -> None:
    service = service_with_pages(1)
    service.pages[1]["hints"] = [
        {
            "id": 7,
            "start": 0,
            "end": 6,
            "reasons": ["diff"],
            "variants": None,
            "crop": [0, 0.1, 1, 0.2],
        }
    ]
    window = FakeWindow([done()])
    await review(service, window)
    image_calls = service.calls_to("GET", "/api/pages/1/image")
    assert image_calls[0].params == {"long_side": "2000", "format": "jpeg"}
    assert image_calls[1].params["long_side"] == "2800"
    assert image_calls[1].params["clip"] == "0.000000,0.100000,1.000000,0.200000"
    assert set(window.payloads[1]["images"]["crops"]) == {"7"}


async def test_messages_never_contain_page_text() -> None:
    service = service_with_pages(1, 2)
    emitter = Emitter()
    reply = await review(
        service, FakeWindow([done(), leave("later", "черновой текст")]), emitter=emitter
    )
    chat = reply + " ".join(str(e) for e in emitter.events)
    for text in [*PAGE_TEXT.values(), "черновой текст"]:
        assert text not in chat


async def test_fix_command_opens_one_page_with_full_scope() -> None:
    service = service_with_pages(1, 2)
    service.pages[2]["status"] = "ok"
    window = FakeWindow([done("исправлено вручную")])
    reply = await run_pipe(PipeForTest(service), "исправить Отчёт 1983, стр. 2", window=window)
    assert window.shown_pages == [2]
    show = window.payloads[1]
    assert show["page"]["scope"] == "full" and show["can_back"] is False
    assert show["page"]["header"]["scope_note"] == op.msg("scope_full")
    assert service.pages[2]["text"] == "исправлено вручную"
    assert "Проверено 1 страница" in reply


async def test_fix_command_on_a_page_being_recognized() -> None:
    service = service_with_pages(1)
    service.pages[1]["status"] = "pending"
    reply = await run_pipe(PipeForTest(service), "исправить отчёт 1983 стр 1", window=FakeWindow())
    assert reply == op.msg("fix_pending", title="Отчёт 1983", page=1)


@pytest.mark.p4
async def test_draft_command_waits_for_the_build_and_attaches_the_docx() -> None:
    service = service_with_pages(1)
    files = FakeFiles()
    emitter = Emitter()
    reply = await run_pipe(PipeForTest(service, files), "черновик отчёт 1983", emitter=emitter)
    assert reply == op.msg("draft_delivered", title="Отчёт 1983")
    assert [f.name for f in files.stored] == ["Отчёт 1983 (черновик).docx"]
    download = service.calls_to("GET", "/api/documents/d1/files/docx")[0]
    assert download.params == {"draft": "true"}
    assert service.delivered == []


@pytest.mark.p4
async def test_files_command_sends_the_latest_final_build() -> None:
    service = service_with_pages(1)
    service.documents["d1"]["builds"] = [
        {"id": 1, "kind": "final", "version": 1, "status": "done"},
        {"id": 2, "kind": "draft", "version": 2, "status": "done"},
        {"id": 3, "kind": "final", "version": 3, "status": "failed"},
    ]
    reply = await run_pipe(PipeForTest(service), "файлы отчёт", emitter=Emitter())
    assert reply == op.msg("files_delivered", title="Отчёт 1983", version=1)
    assert service.calls_to("GET", "/api/documents/d1/files/pdf")[0].params == {"version": "1"}
    assert service.delivered == [("d1", 1)]
