"""Chat commands of the pipe (section 14)."""

from __future__ import annotations

import json

import pytest

from tests.owui.helpers import (
    USER_ID,
    Emitter,
    FakeService,
    PipeForTest,
    op,
    run_pipe,
)

pytestmark = pytest.mark.p3


def attachment(name: str, content_type: str, data: bytes = b"data") -> dict[str, object]:
    return {"type": "file", "id": name, "name": name, "_data": data, "_type": content_type}


def test_parse_command_normalizes_text() -> None:
    assert op.parse_command("  Проверить накопленное! ") == op.Command("review")
    assert op.parse_command("Что в очереди?") == op.Command("queue")
    assert op.parse_command("Исправить «Отчёт 1983», стр. 47") == op.Command(
        "fix", "отчет 1983", 47
    )
    assert op.parse_command("исправить отчет 1983 страница 5") == op.Command("fix", "отчет 1983", 5)
    assert op.parse_command("файлы Отчёт 1983") == op.Command("files", "отчет 1983")
    assert op.parse_command("черновик отчёт") == op.Command("draft", "отчет")
    assert op.parse_command("в базу отчёт") == op.Command("ingest", "отчет")
    assert op.parse_command("привет") == op.Command("help")
    assert op.parse_command("") == op.Command("help")


def test_match_documents_prefers_exact_title() -> None:
    docs = [{"id": "a", "title": "Отчёт 1983"}, {"id": "b", "title": "Отчёт 1983 приложение"}]
    assert [d["id"] for d in op.match_documents(docs, "отчет 1983")] == ["a"]
    assert [d["id"] for d in op.match_documents(docs, "отчет 19")] == ["a", "b"]
    assert op.match_documents(docs, "план") == []


async def test_task_is_answered_without_the_service() -> None:
    service = FakeService()
    pipe = PipeForTest(service)
    title = await run_pipe(pipe, "что угодно", task="title_generation")
    assert json.loads(title) == {"title": "OCR-архив"}
    assert json.loads(await run_pipe(pipe, "x", task="tags_generation")) == {"tags": []}
    assert json.loads(await run_pipe(pipe, "x", task="follow_up_generation")) == {"follow_ups": []}
    assert service.calls == []


async def test_unknown_user_is_refused_without_service_calls() -> None:
    service = FakeService()
    pipe = PipeForTest(service)
    reply = await run_pipe(pipe, "что в очереди", user_id="intruder")
    assert reply == op.msg("access_denied")
    pipe.valves.allowed_user_ids = []
    assert await run_pipe(pipe, "что в очереди") == op.msg("access_denied")
    assert service.calls == []


async def test_ambiguous_name_lists_documents_and_does_nothing() -> None:
    service = FakeService()
    service.add_document("d1", "Отчёт 1983")
    service.add_document("d2", "Отчёт 1984")
    reply = await run_pipe(PipeForTest(service), "файлы отчёт 198")
    assert "«Отчёт 1983»" in reply and "«Отчёт 1984»" in reply
    assert [(c.method, c.path) for c in service.calls] == [("GET", "/api/documents")]


async def test_picture_attachments_become_one_document_in_order() -> None:
    service = FakeService()
    files = [
        attachment("scan2.jpg", "image/jpeg", b"two"),
        attachment("scan1.jpg", "image/jpeg", b"one"),
        attachment("scan3.png", "image/png", b"three"),
    ]
    reply = await run_pipe(PipeForTest(service), "", files=files)
    assert len(service.uploads) == 1
    upload = service.uploads[0]
    names = [f["filename"] for f in upload["files"]]
    assert names == ["0001_scan2.jpg", "0002_scan1.jpg", "0003_scan3.png"]
    assert {f["field"] for f in upload["files"]} == {"files"}
    assert [f["data"] for f in upload["files"]] == [b"two", b"one", b"three"]
    assert json.loads(upload["order"]) == names
    assert upload["ingest"] == "false"
    assert reply.startswith("В очереди: «0001_scan2»")
    assert "214 страниц" in reply


async def test_pdf_and_tiff_are_queued_as_separate_documents() -> None:
    service = FakeService()
    files = [attachment("a.pdf", "application/pdf"), attachment("b.tif", "image/tiff")]
    reply = await run_pipe(PipeForTest(service), "вот", files=files)
    assert [len(u["files"]) for u in service.uploads] == [1, 1]
    assert all("order" not in u for u in service.uploads)
    assert reply.count("В очереди") == 2


async def test_unsupported_and_duplicate_files_are_reported() -> None:
    service = FakeService()
    service.add_document("d1", "Старый отчёт", status="done", source_name="old.pdf")
    files = [attachment("old.pdf", "application/pdf"), attachment("notes.txt", "text/plain")]
    reply = await run_pipe(PipeForTest(service), "", files=files)
    assert "Уже есть: «Старый отчёт», готово." in reply
    assert "«notes.txt» не поддерживается" in reply


async def test_command_text_wins_over_attachments() -> None:
    service = FakeService()
    files = [attachment("a.pdf", "application/pdf")]
    await run_pipe(PipeForTest(service), "что в очереди", files=files)
    assert service.uploads == []


async def test_unknown_text_gets_help() -> None:
    reply = await run_pipe(PipeForTest(FakeService()), "привет")
    assert reply == op.msg("help")


async def test_ingest_disabled_is_explained() -> None:
    service = FakeService()
    service.add_document("d1", "Отчёт 1983", status="done")
    service.ingest_enabled = False
    reply = await run_pipe(PipeForTest(service), "в базу отчёт 1983")
    assert reply == op.msg("ingest_disabled")
    service.ingest_enabled = True
    reply = await run_pipe(PipeForTest(service), "в базу отчёт 1983")
    assert reply == op.msg("ingest_requested", title="Отчёт 1983")


async def test_service_down_is_reported() -> None:
    class DownPipe(PipeForTest):
        def _make_client(self):  # type: ignore[no-untyped-def]
            import httpx

            def handler(request: httpx.Request) -> httpx.Response:
                raise httpx.ConnectError("refused")

            return httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://x")

    reply = await run_pipe(DownPipe(FakeService()), "что в очереди")
    assert reply == op.msg("service_down")


async def test_queue_summary_lists_everything_the_morning_needs() -> None:
    service = FakeService()
    service.add_document("d1", "Отчёт 1983")
    service.add_page(1, "d1", 1, "секретный текст страницы")
    service.summary_extra = {
        "stale_drafts": [
            {"document_id": "d1", "title": "Отчёт 1983", "page_number": 47, "age_days": 5}
        ],
        "suppressed_rules": [{"flag": "digits_disagreement", "pages": 37}],
        "waiting_redecide": [{"document_id": "d1", "title": "Отчёт 1983", "draft_pages": 1}],
    }
    service.documents["d1"]["status"] = "needs_review"
    emitter = Emitter()
    reply = await run_pipe(PipeForTest(service), "что в очереди", emitter=emitter)
    assert "К проверке 1 страница" in reply
    assert "стр. 47 в «Отчёт 1983», черновику 5 дней — доделай или выкинь" in reply
    assert "правило «числа разошлись» выключено, не проверено 37 страниц" in reply
    assert "перерешение «Отчёт 1983» ждёт: 1 страница с черновиком" in reply
    assert "секретный текст" not in reply


def test_summary_marks_documents_over_the_guideline() -> None:
    summary = {
        "queue_total": 70,
        "est_review_sec": 7200,
        "load_guideline": 0.1,
        "documents": [
            {
                "title": "Отчёт 1983",
                "status": "needs_review",
                "to_review": 66,
                "flagged_share": 0.31,
                "over_guideline": True,
                "est_review_sec": 7200,
                "no_logprobs_pages": 3,
                "export_error": "No space left on device",
            }
        ],
    }
    text = op.format_summary(summary, with_logprobs=True)
    assert "флагов 31% — выше ориентира 10%, проверка примерно 2 ч" in text
    assert "без вероятностей токенов 3 страницы" in text
    assert "ошибка экспорта: No space left on device" in text
    assert "без вероятностей" not in op.format_summary(summary, with_logprobs=False)


def test_reasons_are_described_with_counters() -> None:
    item = {
        "reasons": ["pass_diff", "low_conf_digit", "markers"],
        "signals": {"diff_words": 3, "low_digits": 2},
        "status": "flagged",
    }
    assert op.describe_reasons(item) == (
        "3 расхождения проходов, 2 неуверенные цифры, модель не прочитала часть текста"
    )
    assert op.describe_reasons({"reasons": [], "status": "failed"}) == "страница не распознана"
    assert op.describe_reasons({"reasons": ["spot_check"]}) == "выборочная проверка"


def test_plural_forms() -> None:
    assert [op.count_phrase(n, "forms_pages") for n in (1, 2, 5, 11, 21, 22, 112)] == [
        "1 страница",
        "2 страницы",
        "5 страниц",
        "11 страниц",
        "21 страница",
        "22 страницы",
        "112 страниц",
    ]


async def test_allowed_user_ids_tolerates_the_valves_form() -> None:
    pipe = PipeForTest(FakeService())
    pipe.valves.allowed_user_ids = [f' "{USER_ID}" , other-user']
    assert await run_pipe(pipe, "что в очереди") != op.msg("access_denied")
    pipe.valves.allowed_user_ids = ["other-user"]
    assert await run_pipe(pipe, "что в очереди") == op.msg("access_denied")
