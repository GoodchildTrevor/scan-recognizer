"""
title: OCR Archive
description: Queue scanned documents for the local OCR archive and review flagged pages.
requirements: httpx
"""

# The embedded window JS and CSS keep their own line lengths.
# ruff: noqa: E501

from __future__ import annotations

import asyncio
import base64
import hashlib
import inspect
import io
import json
import logging
import re
import time
import unicodedata
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import quote, unquote

import httpx
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---- shared-with-prototype: begin ----
# Everything between the two markers is copied verbatim into prototype/owui_window_pipe.py
# (run `python prototype/owui_window_pipe.py --sync`); tests/owui checks that both copies match.

EventCall = Callable[[dict[str, Any]], Awaitable[Any]]
EventEmitter = Callable[[dict[str, Any]], Awaitable[Any]]

DEFAULT_PAGE_LONG_SIDE = 2000
DEFAULT_CROP_LONG_SIDE = 2800
# One execute call is reliable up to about 1 MB (section 8.3); bigger image sets go through
# OpenWebUI's file store instead of data: URIs.
MAX_INLINE_IMAGES_BYTES = 900_000
QUEUE_LIMIT = 50
BUILD_POLL_S = 3.0

MESSAGES_RU: dict[str, str] = {
    "access_denied": "Нет доступа к OCR-архиву.",
    "service_down": "Сервис OCR-архива недоступен, попробуй позже.",
    "service_error": "Сервис OCR-архива ответил ошибкой {code}.",
    "help": (
        "Кинь в этот чат PDF, TIFF или сканы страниц — поставлю в очередь.\n"
        "Кнопки: «Проверить накопленное», «Что в очереди».\n"
        "Команды:\n"
        "• файлы <название> — прислать готовые PDF и DOCX;\n"
        "• черновик <название> — черновой DOCX без проверки;\n"
        "• исправить <название>, стр. N — открыть страницу для правки;\n"
        "• в базу <название> — загрузить документ в базу знаний."
    ),
    "cmd_review": "проверить накопленное",
    "cmd_queue": "что в очереди",
    "cmd_files": "файлы",
    "cmd_draft": "черновик",
    "cmd_fix": "исправить",
    "cmd_ingest": "в базу",
    "cmd_page_words": "стр|страница|страницу|с",
    "model_name": "OCR-архив",
    "queued": "В очереди: «{title}», {pages}, готово примерно к {eta}.",
    "queued_no_eta": "В очереди: «{title}», {pages}.",
    "duplicate": "Уже есть: «{title}», {status}.",
    "upload_failed": "Не удалось поставить в очередь «{name}»: {error}.",
    "unsupported_file": "Файл «{name}» не поддерживается: нужен PDF, TIFF, JPEG или PNG.",
    "read_failed": "Не удалось прочитать вложение «{name}».",
    "not_found": "Документ «{query}» не найден.",
    "ambiguous": "Подходит несколько документов, уточни название:\n{items}",
    "ambiguous_item": "• «{title}» — {status}",
    "summary_queue": "К проверке {pages}, примерно {time}.",
    "summary_queue_empty": "Очередь проверки пуста.",
    "summary_doc": "• «{title}» — {status}",
    "summary_doc_recognized": "распознано {done} из {total}",
    "summary_doc_eta": "готово примерно к {eta}",
    "summary_doc_to_review": "к проверке {pages}",
    "summary_doc_drafts": "начатых правок {count}",
    "summary_doc_over": "флагов {share}% — выше ориентира {guideline}%, проверка примерно {time}",
    "summary_doc_no_logprobs": "без вероятностей токенов {pages}",
    "summary_doc_export_error": "ошибка экспорта: {error}",
    "summary_stale_draft": ("• стр. {page} в «{title}», черновику {age} — доделай или выкинь"),
    "summary_suppressed": "• правило «{rule}» выключено, не проверено {pages}",
    "summary_waiting_redecide": "• перерешение «{title}» ждёт: {pages} с черновиком",
    "status_queued": "в очереди",
    "status_processing": "распознаётся",
    "status_needs_review": "ждёт проверки",
    "status_ready": "собирается",
    "status_building": "собирается",
    "status_done": "готово",
    "status_failed": "ошибка сборки",
    "rule_broken": "оба чтения сломаны",
    "rule_unverified": "одно чтение",
    "rule_digits_disagreement": "числа разошлись",
    "rule_table_mismatch": "таблицы разошлись",
    "rule_line_count_mismatch": "число строк не сходится",
    "rule_markers": "модель не прочитала",
    "rule_pass_diff": "расхождения проходов",
    "rule_low_conf_digit": "неуверенные цифры",
    "rule_low_conf_text": "неуверенный текст",
    "reason_broken": "оба чтения сломаны",
    "reason_unverified": "сверить не с чем: одно чтение сломано",
    "reason_digits_disagreement": "числа в проходах разошлись",
    "reason_digits_disagreement_n": (
        "расхождение в числах|расхождения в числах|расхождений в числах"
    ),
    "reason_table_mismatch": "таблицы в проходах разошлись",
    "reason_line_count_mismatch": "число строк не совпадает со сканом",
    "reason_line_count_mismatch_n": (
        "строка не совпадает со сканом|строки не совпадают со сканом|строк не совпадают со сканом"
    ),
    "reason_markers": "модель не прочитала часть текста",
    "reason_pass_diff": "расхождения проходов",
    "reason_pass_diff_n": "расхождение проходов|расхождения проходов|расхождений проходов",
    "reason_low_conf_digit": "неуверенные цифры",
    "reason_low_conf_digit_n": "неуверенная цифра|неуверенные цифры|неуверенных цифр",
    "reason_low_conf_text": "неуверенный текст",
    "reason_low_conf_text_n": "неуверенное место|неуверенных места|неуверенных мест",
    "reason_failed": "страница не распознана",
    "reason_spot_check": "выборочная проверка",
    "reason_none": "правка по запросу",
    "scope_full": "вычитай всю страницу по скану",
    "counter": "{pos} из {total}",
    "page_label": "Стр. {n}",
    "status_review": "проверка: {pos} из {total}",
    "status_building_docs": "собираю: {docs}",
    "status_draft": "собираю черновик «{title}»",
    "status_upload_wait": "жду «{title}»: {status}",
    "review_paused": "Проверка приостановлена, осталось {left}.",
    "review_paused_unknown": "Проверка приостановлена.",
    "review_done": "Проверено {pages}, готово {docs}, в очереди {left}.",
    "review_empty": "Проверять нечего.",
    "delivered": "Готовые файлы в чате: {titles}.",
    "delivery_failed": "Не удалось прикрепить файлы «{title}», пришлю в следующий раз.",
    "files_none": "У «{title}» ещё нет готовых файлов.",
    "files_delivered": "Файлы «{title}», версия {version}.",
    "draft_not_allowed": (
        "Черновик «{title}» сейчас не собрать: документ ещё распознаётся или итоговые файлы "
        "уже есть."
    ),
    "draft_failed": "Черновик «{title}» не собрался: {error}.",
    "draft_timeout": "Черновик «{title}» ещё собирается, повтори команду позже.",
    "draft_delivered": "Черновик «{title}» в чате.",
    "fix_pending": "Страница {page} документа «{title}» ещё распознаётся.",
    "fix_no_page": "В «{title}» нет страницы {page}.",
    "ingest_requested": "«{title}»: загрузка в базу запрошена.",
    "ingest_disabled": "Загрузка в базу выключена.",
    "duration_less_minute": "меньше минуты",
    "duration_minutes": "{m} мин",
    "duration_hours": "{h} ч",
    "duration_hours_minutes": "{h} ч {m} мин",
    "forms_pages": "страница|страницы|страниц",
    "forms_documents": "документ|документа|документов",
    "forms_days": "день|дня|дней",
}

# Signal counters (section 6.10) that turn a flag into a number for the window header.
REASON_COUNTERS: dict[str, str] = {
    "pass_diff": "diff_words",
    "low_conf_digit": "low_digits",
    "low_conf_text": "low_letters",
    "digits_disagreement": "digit_mismatch",
    "line_count_mismatch": "line_gap",
}

TASK_STUBS: dict[str, dict[str, Any]] = {
    "title_generation": {"title": ""},
    "tags_generation": {"tags": []},
    "follow_up_generation": {"follow_ups": []},
    "query_generation": {"queries": []},
    "autocomplete_generation": {"text": ""},
}


def msg(key: str, **values: Any) -> str:
    """Format a user-facing message.

    :param key: Key in :data:`MESSAGES_RU`.
    :param values: Values for the placeholders.
    :return: The formatted text.
    """
    return MESSAGES_RU[key].format(**values)


def plural_word(n: int, forms_key: str) -> str:
    """Pick the Russian plural form for a number.

    :param n: The number.
    :param forms_key: Key of a ``one|few|many`` entry in :data:`MESSAGES_RU`.
    :return: The matching form.
    """
    one, few, many = MESSAGES_RU[forms_key].split("|")
    n = abs(n)
    if n % 10 == 1 and n % 100 != 11:
        return one
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return few
    return many


def count_phrase(n: int, forms_key: str) -> str:
    """Return ``"<n> <form>"`` with the right plural form."""
    return f"{n} {plural_word(n, forms_key)}"


def format_duration(seconds: float | None) -> str:
    """Format a duration for the chat.

    :param seconds: Duration in seconds; ``None`` counts as zero.
    :return: Human-readable duration.
    """
    total_min = round((seconds or 0) / 60)
    if total_min < 1:
        return msg("duration_less_minute")
    hours, minutes = divmod(total_min, 60)
    if not hours:
        return msg("duration_minutes", m=minutes)
    if not minutes:
        return msg("duration_hours", h=hours)
    return msg("duration_hours_minutes", h=hours, m=minutes)


def format_clock(eta_sec: float, now: datetime | None = None) -> str:
    """Return the local wall-clock time after ``eta_sec`` seconds.

    :param eta_sec: Seconds from now.
    :param now: Current time, for tests.
    :return: ``HH:MM``, with the date when it is more than 20 hours away.
    """
    start = now or datetime.now().astimezone()
    moment = start + timedelta(seconds=max(0.0, eta_sec))
    return moment.strftime("%d.%m %H:%M" if eta_sec > 20 * 3600 else "%H:%M")


def normalize_text(text: str) -> str:
    """Normalize a chat message or a title for matching.

    Casefold, ``ё`` to ``е``, punctuation to spaces, collapsed whitespace.

    :param text: Raw text.
    :return: Normalized text.
    """
    folded = unicodedata.normalize("NFC", text).casefold().replace("ё", "е")
    cleaned = "".join(" " if unicodedata.category(ch).startswith("P") else ch for ch in folded)
    return " ".join(cleaned.split())


@dataclass(frozen=True)
class Command:
    """A parsed chat message."""

    kind: str  # review | queue | files | draft | fix | ingest | help
    query: str = ""
    page: int = 0


def parse_command(text: str) -> Command:
    """Recognize a chat command (section 14).

    :param text: The user's message.
    :return: The command; ``help`` when nothing matches.
    """
    norm = normalize_text(text)
    if not norm:
        return Command("help")
    if norm == normalize_text(MESSAGES_RU["cmd_review"]):
        return Command("review")
    if norm == normalize_text(MESSAGES_RU["cmd_queue"]):
        return Command("queue")
    for kind in ("files", "draft", "ingest"):
        prefix = normalize_text(MESSAGES_RU[f"cmd_{kind}"]) + " "
        if norm.startswith(prefix):
            return Command(kind, norm[len(prefix) :].strip())
    fix_prefix = normalize_text(MESSAGES_RU["cmd_fix"]) + " "
    if norm.startswith(fix_prefix):
        words = "|".join(re.escape(w) for w in MESSAGES_RU["cmd_page_words"].split("|"))
        match = re.fullmatch(rf"(?P<name>.+?) (?:{words}) ?(?P<page>\d+)", norm[len(fix_prefix) :])
        if match:
            return Command("fix", match["name"], int(match["page"]))
    return Command("help")


def task_stub(task: str) -> str:
    """Answer an OpenWebUI background task without calling the service (section 3.3).

    :param task: Task name from ``__task__``.
    :return: The JSON answer OpenWebUI expects, or an empty string.
    """
    stub = TASK_STUBS.get(task)
    if stub is None:
        return ""
    if "title" in stub:
        stub = {"title": MESSAGES_RU["model_name"]}
    return json.dumps(stub, ensure_ascii=False)


_USER_QUERY_RE = re.compile(r"<user_query>\s*(.*?)\s*</user_query>", re.DOTALL | re.IGNORECASE)
_INJECTED_BLOCK_RE = re.compile(
    r"<(attached_files|context|source|sources)\b[^>]*>.*?</\1\s*>", re.DOTALL | re.IGNORECASE
)


def user_query(content: str) -> str:
    """Return what the user typed, without the blocks OpenWebUI adds around it.

    OpenWebUI inserts an ``<attached_files>`` block for files of the chat and, depending on the
    version and settings, the RAG prompt with ``<context>`` and ``<user_query>`` blocks.

    :param content: Content of the user message.
    :return: The user's own text.
    """
    match = _USER_QUERY_RE.search(content)
    if match:
        return match.group(1)
    return _INJECTED_BLOCK_RE.sub(" ", content).strip()


def last_user_text(body: Mapping[str, Any]) -> str:
    """Return the text of the last user message in an OpenWebUI request body."""
    for message in reversed(body.get("messages") or []):
        if not isinstance(message, Mapping) or message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return user_query(content)
        if isinstance(content, list):
            return user_query(
                " ".join(
                    str(part.get("text", ""))
                    for part in content
                    if isinstance(part, Mapping) and part.get("type") == "text"
                )
            )
        return ""
    return ""


def status_name(status: str | None) -> str:
    """Translate a document status for the chat."""
    return MESSAGES_RU.get(f"status_{status}", str(status))


def describe_reasons(item: Mapping[str, Any]) -> str:
    """Describe why a page is in the review queue, with counters (section 8.2).

    :param item: Review item from the API.
    :return: For example ``"3 расхождения проходов, 2 неуверенные цифры"``.
    """
    signals = item.get("signals") or {}
    reasons = [str(code) for code in item.get("reasons") or []]
    if item.get("status") == "failed" and "failed" not in reasons:
        reasons.append("failed")
    parts: list[str] = []
    for code in reasons:
        counter = REASON_COUNTERS.get(code)
        value = signals.get(counter) if counter else None
        forms_key = f"reason_{code}_n"
        if (
            isinstance(value, int | float)
            and not isinstance(value, bool)
            and value >= 1
            and forms_key in MESSAGES_RU
        ):
            parts.append(count_phrase(int(value), forms_key))
        else:
            parts.append(MESSAGES_RU.get(f"reason_{code}", code))
    return ", ".join(parts) or MESSAGES_RU["reason_none"]


def _summary_document_line(
    doc: Mapping[str, Any], guideline: float | None, *, with_logprobs: bool
) -> str:
    parts = [msg("summary_doc", title=doc.get("title", ""), status=status_name(doc.get("status")))]
    progress = doc.get("progress") or {}
    if doc.get("status") in ("queued", "processing"):
        total = int(progress.get("total") or 0)
        pending = int(progress.get("pending") or 0)
        if total:
            parts.append(msg("summary_doc_recognized", done=total - pending, total=total))
        if doc.get("eta_sec"):
            parts.append(msg("summary_doc_eta", eta=format_clock(float(doc["eta_sec"]))))
    if doc.get("to_review"):
        parts.append(
            msg("summary_doc_to_review", pages=count_phrase(int(doc["to_review"]), "forms_pages"))
        )
    if doc.get("drafts"):
        parts.append(msg("summary_doc_drafts", count=int(doc["drafts"])))
    if doc.get("over_guideline"):
        parts.append(
            msg(
                "summary_doc_over",
                share=round(float(doc.get("flagged_share") or 0) * 100),
                guideline=round(float(guideline or 0) * 100),
                time=format_duration(doc.get("est_review_sec")),
            )
        )
    if with_logprobs and doc.get("no_logprobs_pages"):
        parts.append(
            msg(
                "summary_doc_no_logprobs",
                pages=count_phrase(int(doc["no_logprobs_pages"]), "forms_pages"),
            )
        )
    if doc.get("export_error"):
        parts.append(msg("summary_doc_export_error", error=doc["export_error"]))
    return "; ".join(parts)


def format_summary(summary: Mapping[str, Any], *, with_logprobs: bool) -> str:
    """Format the queue summary for the chat (sections 8.1 and 14).

    Only titles, numbers and statuses go into the text, never page content.

    :param summary: Response of ``GET /api/review/summary``.
    :param with_logprobs: Also report pages processed without token probabilities.
    :return: Multi-line text.
    """
    lines: list[str] = []
    total = int(summary.get("queue_total") or 0)
    if total:
        lines.append(
            msg(
                "summary_queue",
                pages=count_phrase(total, "forms_pages"),
                time=format_duration(summary.get("est_review_sec")),
            )
        )
    else:
        lines.append(msg("summary_queue_empty"))
    guideline = summary.get("load_guideline")
    for doc in summary.get("documents") or []:
        lines.append(_summary_document_line(doc, guideline, with_logprobs=with_logprobs))
    for draft in summary.get("stale_drafts") or []:
        lines.append(
            msg(
                "summary_stale_draft",
                page=draft.get("page_number"),
                title=draft.get("title", ""),
                age=count_phrase(int(draft.get("age_days") or 0), "forms_days"),
            )
        )
    for rule in summary.get("suppressed_rules") or []:
        flag = str(rule.get("flag"))
        lines.append(
            msg(
                "summary_suppressed",
                rule=MESSAGES_RU.get(f"rule_{flag}", flag),
                pages=count_phrase(int(rule.get("pages") or 0), "forms_pages"),
            )
        )
    for waiting in summary.get("waiting_redecide") or []:
        lines.append(
            msg(
                "summary_waiting_redecide",
                title=waiting.get("title", ""),
                pages=count_phrase(int(waiting.get("draft_pages") or 0), "forms_pages"),
            )
        )
    return "\n".join(lines)


# ---- draft restore logic: Python reference of OCR_CORE in REVIEW_CORE_JS (section 8.4) ----


def text_sha256(text: str) -> str:
    """Return the sha256 hex digest of the UTF-8 text, as the service and the window compute it."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def choose_draft_source(
    local: Mapping[str, Any] | None, server: Mapping[str, Any] | None
) -> dict[str, Any]:
    """Decide whether the browser copy or the server draft wins when a page is reopened.

    Mirrors ``chooseDraftSource`` of the window JS (section 8.4):

    * no unsent edits in the browser — the server draft (or the page itself);
    * unsent edits on top of the current server save — the browser copy;
    * the last save went through but its answer was lost (server seq is the expected one plus
      one and the text hash matches) — the browser copy, ``base_seq`` moves to the server seq;
    * otherwise — a real conflict, both versions are shown.

    :param local: Browser copy (``base_seq``, ``rev``, ``sent_rev``, ``in_flight``, …) or ``None``.
    :param server: Server draft (``seq``, ``text_sha256``, …) or ``None``.
    :return: ``{"source": "server" | "local" | "conflict", "base_seq": int, "sent_rev": int}``.
    """
    server_seq = int(server.get("seq") or 0) if server is not None else 0
    if local is None:
        return {"source": "server", "base_seq": server_seq, "sent_rev": 0}
    rev = int(local.get("rev") or 0)
    sent = int(local.get("sent_rev") or 0)
    if rev == sent:
        return {"source": "server", "base_seq": server_seq, "sent_rev": sent}
    if int(local.get("base_seq") or 0) == server_seq:
        return {"source": "local", "base_seq": server_seq, "sent_rev": sent}
    flight = local.get("in_flight")
    if (
        isinstance(flight, Mapping)
        and server is not None
        and server_seq == int(flight.get("expected_seq") or 0) + 1
        and server.get("text_sha256") == flight.get("sha256")
    ):
        return {
            "source": "local",
            "base_seq": server_seq,
            "sent_rev": max(sent, int(flight.get("rev") or 0)),
        }
    return {"source": "conflict", "base_seq": server_seq, "sent_rev": sent}


def apply_draft_ack(local: Mapping[str, Any], *, rev: int | None, seq: int) -> dict[str, Any]:
    """Apply a confirmed draft save to the browser copy (mirror of ``applyDraftAck``).

    A confirmation of an older save never marks newer edits as sent.

    :param local: Browser copy.
    :param rev: Local revision the confirmed save carried.
    :param seq: New server save number.
    :return: Updated copy.
    """
    sent = int(local.get("sent_rev") or 0)
    acked = rev if isinstance(rev, int) else sent
    result = dict(local)
    result["base_seq"] = seq
    result["sent_rev"] = max(sent, acked)
    flight = local.get("in_flight")
    if isinstance(flight, Mapping) and int(flight.get("rev") or 0) <= acked:
        result["in_flight"] = None
    return result


def _with_state(hint: Mapping[str, Any]) -> dict[str, Any]:
    return {**hint, "state": hint.get("state") or "active"}


def restore_state(
    item: Mapping[str, Any], local: Mapping[str, Any] | None, server: Mapping[str, Any] | None
) -> dict[str, Any]:
    """Compute what the editor shows when a page opens (mirror of ``restoreState``).

    :param item: Review item from the API.
    :param local: Browser copy or ``None``.
    :param server: Server draft (``item["draft"]``) or ``None``.
    :return: Editor state: mode (``edit``, ``stale`` or ``conflict``), text, hints, counters.
    """
    choice = choose_draft_source(local, server)
    local_rev = int(local.get("rev") or 0) if local is not None else 0
    if choice["source"] == "conflict" and local is not None:
        return {
            "mode": "conflict",
            "source": "conflict",
            "text": local.get("text") or "",
            "printed_label": local.get("printed_label"),
            "hints": [],
            "hints_outdated": False,
            "choices": local.get("choices") or {},
            "cursor": local.get("cursor") or 0,
            "base_seq": int(local.get("base_seq") or 0),
            "base_attempt_id": local.get("base_attempt_id"),
            "base_revision_id": local.get("base_revision_id"),
            "hints_version": local.get("item_version"),
            "rev": local_rev,
            "sent_rev": int(local.get("sent_rev") or 0),
            "in_flight": local.get("in_flight"),
            "stale_reason": None,
            "dirty": True,
            "conflict": {
                "server_seq": choice["base_seq"],
                "server_text": (server.get("text") or "")
                if server is not None
                else (item.get("text") or ""),
                "server_printed_label": server.get("printed_label")
                if server is not None
                else item.get("printed_label"),
            },
        }
    src = local if choice["source"] == "local" else server
    if src is None:
        return {
            "mode": "edit",
            "source": "item",
            "text": item.get("text") or "",
            "printed_label": item.get("printed_label"),
            "hints": [_with_state(h) for h in item.get("hints") or []],
            "hints_outdated": False,
            "choices": {},
            "cursor": 0,
            "base_seq": 0,
            "base_attempt_id": item.get("base_attempt_id"),
            "base_revision_id": item.get("base_revision_id"),
            "hints_version": item.get("item_version"),
            "rev": local_rev,
            "sent_rev": local_rev,
            "in_flight": None,
            "stale_reason": None,
            "dirty": False,
            "conflict": None,
        }
    same_version = src.get("item_version") == item.get("item_version")
    base_differs = src.get("base_attempt_id") != item.get("base_attempt_id") or src.get(
        "base_revision_id"
    ) != item.get("base_revision_id")
    server_stale = bool(server is not None and server.get("stale"))
    stale = base_differs or server_stale
    stale_reason: str | None = None
    if stale:
        stale_reason = "changed"
        if server_stale and server is not None and server.get("stale_reason"):
            stale_reason = str(server["stale_reason"])
    if choice["source"] == "local" and local is not None:
        sent_rev = int(choice["sent_rev"])
        in_flight = (
            local.get("in_flight")
            if choice["base_seq"] == int(local.get("base_seq") or 0)
            else None
        )
    else:
        sent_rev = local_rev
        in_flight = None
    return {
        "mode": "stale" if stale else "edit",
        "source": choice["source"],
        "text": src.get("text") or "",
        "printed_label": src.get("printed_label"),
        "hints": [_with_state(h) for h in src.get("hints") or []] if same_version else [],
        "hints_outdated": not same_version,
        "choices": src.get("choices") or {},
        "cursor": src.get("cursor") or 0,
        "base_seq": choice["base_seq"],
        "base_attempt_id": src.get("base_attempt_id"),
        "base_revision_id": src.get("base_revision_id"),
        "hints_version": src.get("item_version"),
        "rev": local_rev,
        "sent_rev": sent_rev,
        "in_flight": in_flight,
        "stale_reason": stale_reason,
        "dirty": local_rev != sent_rev,
        "conflict": None,
    }


# ---- errors, files, context ----


class ServiceError(Exception):
    """The OCR service answered with an error status (section 11)."""

    def __init__(self, status_code: int, detail: str) -> None:
        """Create the error.

        :param status_code: HTTP status, 0 when the service was not reached.
        :param detail: Machine code from ``{"detail": ...}``.
        """
        super().__init__(f"{status_code}: {detail}")
        self.status_code = status_code
        self.detail = detail


class ServiceUnavailableError(ServiceError):
    """The OCR service could not be reached."""

    def __init__(self, reason: str) -> None:
        """Create the error.

        :param reason: Short transport error name for the log.
        """
        super().__init__(0, reason)


class OwuiFilesError(Exception):
    """OpenWebUI's internal file storage failed."""


class WindowClosedError(Exception):
    """The review window is gone: the tab was closed or the connection dropped."""


@dataclass(frozen=True)
class Upload:
    """A file attached to the chat."""

    name: str
    data: bytes
    content_type: str


@dataclass(frozen=True)
class StoredFile:
    """A file put into OpenWebUI's storage."""

    id: str
    name: str
    size: int
    content_type: str


def upload_kind(name: str, content_type: str) -> str | None:
    """Classify an attachment.

    :param name: File name.
    :param content_type: MIME type reported by OpenWebUI.
    :return: ``pdf``, ``tiff``, ``image`` or ``None`` when unsupported.
    """
    suffix = Path(name).suffix.casefold()
    ctype = content_type.casefold()
    if suffix == ".pdf" or ctype == "application/pdf":
        return "pdf"
    if suffix in (".tif", ".tiff") or ctype == "image/tiff":
        return "tiff"
    if suffix in (".jpg", ".jpeg", ".png") or ctype in ("image/jpeg", "image/png"):
        return "image"
    return None


def _call_supported(func: Callable[..., Any], **kwargs: Any) -> Any:
    """Call ``func`` with the keyword arguments its signature accepts.

    OpenWebUI renames and adds parameters of its file handlers between versions.
    """
    params = inspect.signature(func).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return func(**kwargs)
    return func(**{name: value for name, value in kwargs.items() if name in params})


def _attr_or_key(obj: Any, name: str) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name)
    return getattr(obj, name, None)


async def _maybe_await(value: Any) -> Any:
    """Return the result of an OpenWebUI call that is sync in some versions and async in others."""
    return await value if inspect.isawaitable(value) else value


class OwuiFiles:
    """Access to OpenWebUI's internal file storage.

    OpenWebUI modules are imported inside the methods only, so the rest of the pipe can be
    tested without OpenWebUI. The exact calls are confirmed by the P0.5 prototype.
    """

    def __init__(self, request: Any, user: Mapping[str, Any]) -> None:
        """Bind the adapter to the current request and user.

        :param request: ``__request__`` of the pipe call.
        :param user: ``__user__`` of the pipe call.
        """
        self._request = request
        self._user = user

    async def read_upload(self, entry: Mapping[str, Any]) -> Upload:
        """Read an attachment of the current message.

        :param entry: One element of ``__files__``.
        :return: Name, bytes and MIME type.
        :raises OwuiFilesError: The file cannot be read.
        """
        raw_nested = entry.get("file")
        nested: Mapping[str, Any] = raw_nested if isinstance(raw_nested, Mapping) else {}
        file_id = str(entry.get("id") or nested.get("id") or "")
        try:
            from open_webui.models.files import Files
            from open_webui.storage.provider import Storage

            model = await _maybe_await(Files.get_file_by_id(file_id))
            if model is None:
                raise OwuiFilesError(f"file {file_id} not found")
            path = await _maybe_await(Storage.get_file(model.path))
            data = await asyncio.to_thread(Path(path).read_bytes)
        except OwuiFilesError:
            raise
        # Broad on purpose: OpenWebUI internals raise arbitrary exception types across versions.
        except Exception as err:
            raise OwuiFilesError(f"cannot read file {file_id}") from err
        meta = model.meta or {}
        name = str(model.filename or entry.get("name") or file_id)
        return Upload(name=name, data=data, content_type=str(meta.get("content_type") or ""))

    async def store(self, name: str, data: bytes, content_type: str) -> StoredFile:
        """Put a file into OpenWebUI's storage for the current user, without processing.

        :param name: File name shown in the chat.
        :param data: File bytes.
        :param content_type: MIME type.
        :return: The stored file.
        :raises OwuiFilesError: OpenWebUI rejected the file.
        """
        try:
            from fastapi import UploadFile
            from open_webui.models.users import Users
            from open_webui.routers import files as files_router
            from starlette.datastructures import Headers

            user = await _maybe_await(Users.get_user_by_id(str(self._user.get("id"))))
            upload = UploadFile(
                file=io.BytesIO(data),
                filename=name,
                headers=Headers({"content-type": content_type}),
            )
            handler = getattr(files_router, "upload_file_handler", None) or files_router.upload_file
            result = _call_supported(
                handler,
                request=self._request,
                file=upload,
                user=user,
                metadata={},
                file_metadata={},
                process=False,
                process_in_background=False,
            )
            if inspect.isawaitable(result):
                result = await result
        # Broad on purpose: OpenWebUI internals raise arbitrary exception types across versions.
        except Exception as err:
            raise OwuiFilesError(f"cannot store {name}") from err
        file_id = _attr_or_key(result, "id")
        if not file_id:
            raise OwuiFilesError(f"no id for stored {name}")
        return StoredFile(id=str(file_id), name=name, size=len(data), content_type=content_type)

    async def delete(self, file_id: str) -> None:
        """Delete a file this pipe stored earlier.

        :param file_id: ID of the stored file.
        :raises OwuiFilesError: OpenWebUI failed to delete it.
        """
        try:
            from open_webui.models.files import Files
            from open_webui.storage.provider import Storage

            model = await _maybe_await(Files.get_file_by_id(file_id))
            if model is not None:
                await _maybe_await(Storage.delete_file(model.path))
            await _maybe_await(Files.delete_file_by_id(file_id))
        # Broad on purpose: OpenWebUI internals raise arbitrary exception types across versions.
        except Exception as err:
            raise OwuiFilesError(f"cannot delete {file_id}") from err


def file_url(file_id: str) -> str:
    """Return the OpenWebUI URL of a stored file's content (same browser session)."""
    return f"/api/v1/files/{file_id}/content"


def file_event_entry(stored: StoredFile) -> dict[str, Any]:
    """Describe a stored file for the ``files`` event."""
    return {
        "type": "file",
        "id": stored.id,
        "name": stored.name,
        "url": f"/api/v1/files/{stored.id}",
        "size": stored.size,
        "content_type": stored.content_type,
    }


class ServiceApi(Protocol):
    """The part of the OCR service API the review session and delivery use."""

    async def get_document(self, document_id: str) -> dict[str, Any]: ...
    async def download(
        self, document_id: str, kind: str, *, version: int | None = None, draft: bool = False
    ) -> tuple[str, bytes, str]: ...
    async def deliveries(self) -> list[dict[str, Any]]: ...
    async def mark_delivered(self, document_id: str, version: int) -> None: ...
    async def summary(self) -> dict[str, Any]: ...
    async def queue(self, limit: int, exclude: Iterable[int]) -> list[dict[str, Any]]: ...
    async def review_item(self, page_id: int) -> dict[str, Any]: ...
    async def page_item(self, document_id: str, page_number: int) -> dict[str, Any]: ...
    async def page_image(
        self, page_id: int, *, long_side: int, clip: Sequence[float] | None = None
    ) -> bytes: ...
    async def put_draft(self, page_id: int, draft: Mapping[str, Any]) -> dict[str, Any]: ...
    async def delete_draft(self, page_id: int, draft_seq: int) -> None: ...
    async def post_review(self, page_id: int, review: Mapping[str, Any]) -> dict[str, Any]: ...
    async def ocr_region(self, page_id: int, rect: Mapping[str, Any]) -> dict[str, Any]: ...
    async def retry(self, page_id: int) -> dict[str, Any]: ...


class ChatOutput:
    """Collects the reply and shows progress in the status line.

    The reply is returned from the pipe as one text; no ``message`` events are used, so
    streaming and non-streaming chats show the same content.
    """

    def __init__(self, emitter: EventEmitter | None) -> None:
        """Create the collector.

        :param emitter: ``__event_emitter__`` or ``None``.
        """
        self._emitter = emitter
        self._parts: list[str] = []
        self._status_shown = False

    def write(self, text: str) -> None:
        """Append a paragraph to the reply."""
        if text:
            self._parts.append(text)

    @property
    def text(self) -> str:
        """The reply so far."""
        return "\n\n".join(self._parts)

    async def status(self, description: str, *, done: bool = False) -> None:
        """Show a status line above the reply."""
        if self._emitter is None:
            return
        self._status_shown = True
        await _emit(
            self._emitter, {"type": "status", "data": {"description": description, "done": done}}
        )

    async def close_status(self) -> None:
        """Hide the status line if one was shown."""
        if self._emitter is not None and self._status_shown:
            await _emit(
                self._emitter,
                {"type": "status", "data": {"description": "", "done": True, "hidden": True}},
            )


async def _emit(emitter: EventEmitter, event: dict[str, Any]) -> bool:
    try:
        await emitter(event)
    # Broad on purpose: the emitter is OpenWebUI's socket layer; a failed status update must
    # not break the review.
    except Exception:
        logger.warning("event emitter failed for %s", event.get("type"), exc_info=True)
        return False
    return True


@dataclass
class PipeContext:
    """What one pipe call works with."""

    client: ServiceApi
    out: ChatOutput
    build_wait_s: int
    autosave_idle_s: int
    event_call: EventCall | None
    emitter: EventEmitter | None
    files: OwuiFiles | None
    poll_s: float = BUILD_POLL_S
    #: After an upload, wait this long for a fast document (text PDF) to finish and attach
    #: its files in the same reply; 0 answers at once.
    upload_wait_s: int = 0


def _data_uri(data: bytes) -> str:
    if data.startswith(b"\x89PNG"):
        mime = "image/png"
    elif data.lstrip()[:5] in (b"<svg ", b"<?xml"):
        mime = "image/svg+xml"
    else:
        mime = "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


async def deliver_build(
    ctx: PipeContext,
    document_id: str,
    *,
    version: int | None,
    kinds: Sequence[str] = ("pdf", "docx"),
    draft: bool = False,
) -> bool:
    """Attach the files of one build to the current chat (section 9.7, steps 1–3).

    :param ctx: Pipe context.
    :param document_id: Document ID.
    :param version: Build version; ``None`` for the latest or for the draft.
    :param kinds: File kinds to attach.
    :param draft: Attach the draft DOCX.
    :return: ``True`` when every file was stored and the ``files`` event went out.
    """
    if ctx.files is None or ctx.emitter is None:
        return False
    entries: list[dict[str, Any]] = []
    for kind in kinds:
        try:
            name, data, content_type = await ctx.client.download(
                document_id, kind, version=version, draft=draft
            )
            stored = await ctx.files.store(name, data, content_type)
        except (ServiceError, OwuiFilesError):
            logger.warning("delivery of %s/%s failed", document_id, kind, exc_info=True)
            return False
        entries.append(file_event_entry(stored))
    return await _emit(ctx.emitter, {"type": "files", "data": {"files": entries}})


async def deliver_ready(ctx: PipeContext) -> list[str]:
    """Attach every final build that is not yet in a chat, then confirm it (section 9.7).

    A delivery is confirmed only after its files reached the chat, so a failure repeats the
    delivery next time instead of losing it.

    :param ctx: Pipe context.
    :return: Titles of the delivered documents.
    """
    if ctx.files is None or ctx.emitter is None:
        return []
    delivered: list[str] = []
    for entry in await ctx.client.deliveries():
        document_id = str(entry["document_id"])
        title = str(entry.get("title", ""))
        version = int(entry["version"])
        if not await deliver_build(ctx, document_id, version=version):
            ctx.out.write(msg("delivery_failed", title=title))
            continue
        try:
            await ctx.client.mark_delivered(document_id, version)
        except ServiceError:
            logger.warning("delivery confirmation of %s v%s failed", document_id, version)
        delivered.append(title)
    if delivered:
        ctx.out.write(msg("delivered", titles=", ".join(f"«{t}»" for t in delivered)))
    return delivered


async def wait_builds(ctx: PipeContext, documents: Mapping[str, str]) -> list[str]:
    """Wait for final builds of documents that became ready, at most ``build_wait_s``.

    :param ctx: Pipe context.
    :param documents: Document IDs to titles.
    :return: IDs of documents that reached ``done``.
    """
    pending = dict(documents)
    finished: list[str] = []
    deadline = time.monotonic() + max(0, ctx.build_wait_s)
    while pending:
        for document_id in list(pending):
            try:
                status = (await ctx.client.get_document(document_id)).get("status")
            except ServiceError:
                continue
            if status == "done":
                finished.append(document_id)
                del pending[document_id]
            elif status not in ("ready", "building"):
                del pending[document_id]
        remaining = deadline - time.monotonic()
        if not pending or remaining <= 0:
            break
        await ctx.out.status(
            msg("status_building_docs", docs=count_phrase(len(pending), "forms_documents"))
        )
        await asyncio.sleep(min(ctx.poll_s, remaining))
    return finished


async def wait_document(
    ctx: PipeContext, document_id: str, title: str, timeout_s: float
) -> str | None:
    """Wait for a just uploaded document to finish, at most ``timeout_s``.

    The wait stops early when the service's ETA is beyond the timeout: a scanned document of
    hundreds of pages should not hold the reply for nothing.

    :param ctx: Pipe context.
    :param document_id: Document ID.
    :param title: Title for the status line.
    :param timeout_s: Longest wait in seconds.
    :return: The last status seen, or ``None`` when the service did not answer.
    """
    deadline = time.monotonic() + max(0.0, timeout_s)
    while True:
        try:
            doc = await ctx.client.get_document(document_id)
        except ServiceError:
            return None
        status = str(doc.get("status") or "")
        remaining = deadline - time.monotonic()
        if status not in ("queued", "processing", "ready", "building") or remaining <= 0:
            return status
        eta = doc.get("eta_sec")
        if status in ("queued", "processing") and isinstance(eta, int | float) and eta > remaining:
            return status
        await ctx.out.status(msg("status_upload_wait", title=title, status=status_name(status)))
        await asyncio.sleep(min(ctx.poll_s, remaining))


# ---- review session (sections 8 and 14) ----


@dataclass
class PageOutcome:
    """How the window left a page."""

    kind: str  # done | skip | back | stop | reshow
    page_id: int
    target: int | None = None
    item: dict[str, Any] | None = None


class ReviewSession:
    """One pass of the review window over the queue or over a single page."""

    def __init__(self, ctx: PipeContext, *, single: bool = False) -> None:
        """Create the session.

        :param ctx: Pipe context.
        :param single: Review exactly one page (the «исправить» command).
        """
        self._ctx = ctx
        self._single = single
        self._acks: list[dict[str, Any]] = []
        self._outbox_acks: list[dict[str, Any]] = []
        self._history: list[int] = []
        self._skipped: set[int] = set()
        self._done: set[int] = set()
        self._refetch: dict[int, Callable[[], Awaitable[dict[str, Any]]]] = {}
        self._temp_files: list[str] = []
        self._intro: str | None = None
        self.reviewed = 0
        self.total = 0
        self.ready_documents: dict[str, str] = {}

    # -- window calls --

    async def _call(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        if self._ctx.event_call is None:
            raise WindowClosedError("no event caller")
        message = dict(payload)
        message["acks"], self._acks = self._acks, []
        message["outbox_ack"], self._outbox_acks = self._outbox_acks, []
        try:
            result = await self._ctx.event_call(
                {"type": "execute", "data": {"code": window_code(message)}}
            )
        # Broad on purpose: OpenWebUI's event caller raises arbitrary exception types when the
        # browser goes away; for the review that is the same as a closed tab.
        except Exception as err:
            raise WindowClosedError("event call failed") from err
        if not isinstance(result, dict) or "error" in result:
            raise WindowClosedError(str(result)[:200])
        return result

    async def _close_window(self) -> None:
        try:
            await self._call({"op": "close"})
        except WindowClosedError:
            logger.info("review window was already gone at close")

    # -- outbox of unsent "Done" answers (section 8.4) --

    async def flush_outbox(self) -> None:
        """Send the «Готово» answers the browser kept while it had no connection.

        :raises WindowClosedError: The window did not answer.
        """
        result = await self._call({"op": "collect_outbox"})
        for entry in result.get("entries") or []:
            if not isinstance(entry, Mapping):
                continue
            action = await self._send_outbox_entry(entry)
            self._outbox_acks.append({"request_id": entry.get("request_id"), **action})

    async def _send_outbox_entry(self, entry: Mapping[str, Any]) -> dict[str, Any]:
        page_id = entry.get("page_id")
        review = entry.get("review")
        if not isinstance(page_id, int) or not isinstance(review, Mapping):
            return {"action": "remove"}
        try:
            await self._ctx.client.post_review(page_id, review)
        except ServiceUnavailableError:
            return {"action": "keep"}
        except ServiceError as err:
            if err.detail == "document_building":
                return {"action": "keep"}
            if err.detail == "stale":
                return await self._save_stale_answer(page_id, review)
            if err.detail in ("draft_conflict", "invalid_transition"):
                return {"action": "to_local"}
            if err.status_code in (400, 404, 422):
                return {"action": "remove"}
            return {"action": "keep"}
        return {"action": "remove"}

    async def _save_stale_answer(self, page_id: int, review: Mapping[str, Any]) -> dict[str, Any]:
        """Keep a stale «Готово» as a stale draft instead of throwing the text away."""
        draft = {
            "expected_seq": int(review.get("draft_seq") or 0),
            "item_version": review.get("item_version"),
            "base_attempt_id": review.get("base_attempt_id"),
            "base_revision_id": review.get("base_revision_id"),
            "text": review.get("text", ""),
            "printed_label": review.get("printed_label"),
            "choices": review.get("choices") or {},
            "hints": [],
            "cursor": 0,
            "keep_stale": True,
        }
        try:
            saved = await self._ctx.client.put_draft(page_id, draft)
        except ServiceUnavailableError:
            return {"action": "keep"}
        except ServiceError as err:
            if err.detail in ("draft_conflict", "invalid_transition"):
                return {"action": "to_local"}
            return {"action": "keep"}
        return {"action": "remove", "seq": saved.get("seq")}

    # -- queue and single page --

    async def run_queue(self) -> None:
        """Run the review over the whole queue (section 14, «Проверить накопленное»)."""
        ctx = self._ctx
        try:
            await self.flush_outbox()
        except WindowClosedError:
            ctx.out.write(msg("review_paused_unknown"))
            return
        await deliver_ready(ctx)
        summary = await ctx.client.summary()
        intro = format_summary(summary, with_logprobs=False)
        ctx.out.write(intro)
        self._intro = intro
        self.total = int(summary.get("queue_total") or 0)
        try:
            await self._queue_loop()
        except WindowClosedError:
            await self._pause()
            return
        except ServiceUnavailableError:
            await self._close_window()
            raise
        await self._close_window()
        await self._finish()

    async def run_single(self, item: dict[str, Any], document_id: str, page_number: int) -> None:
        """Review one page opened with «исправить» (scope ``full``).

        :param item: Item from ``GET /api/documents/{id}/pages/{n}/item``.
        :param document_id: Document ID.
        :param page_number: 1-based page number.
        """
        page_id = int(item["page_id"])
        self._refetch[page_id] = lambda: self._ctx.client.page_item(document_id, page_number)
        self.total = 1
        try:
            await self.flush_outbox()
            await self.review_page(page_id, item)
        except WindowClosedError:
            self._ctx.out.write(msg("review_paused_unknown"))
            return
        except ServiceUnavailableError:
            await self._close_window()
            raise
        await self._close_window()
        await self._finish()

    async def _queue_loop(self) -> None:
        forced: list[int] = []
        while True:
            if forced:
                batch = [forced.pop()]
            else:
                items = await self._ctx.client.queue(
                    QUEUE_LIMIT, sorted(self._skipped | self._done)
                )
                if not items:
                    return
                batch = [int(entry["page_id"]) for entry in items]
                self.total = max(self.total, self._position() - 1 + len(batch))
            for page_id in batch:
                outcome = await self.review_page(page_id)
                if outcome.kind == "stop":
                    return
                if outcome.kind == "back" and outcome.target is not None:
                    forced.append(outcome.target)
                    break

    def _position(self) -> int:
        return self.reviewed + len(self._skipped) + 1

    async def _queue_left(self) -> int | None:
        try:
            return int((await self._ctx.client.summary()).get("queue_total") or 0)
        except ServiceError:
            return None

    async def _pause(self) -> None:
        left = await self._queue_left()
        if left is None:
            self._ctx.out.write(msg("review_paused_unknown"))
        else:
            self._ctx.out.write(msg("review_paused", left=count_phrase(left, "forms_pages")))

    async def _finish(self) -> None:
        ctx = self._ctx
        built = await wait_builds(ctx, self.ready_documents)
        await deliver_ready(ctx)
        left = await self._queue_left()
        if self.reviewed == 0 and not built and not left:
            ctx.out.write(msg("review_empty"))
            return
        ctx.out.write(
            msg(
                "review_done",
                pages=count_phrase(self.reviewed, "forms_pages"),
                docs=count_phrase(len(built), "forms_documents"),
                left=left if left is not None else "?",
            )
        )

    async def _fetch_item(self, page_id: int) -> dict[str, Any]:
        fetch = self._refetch.get(page_id)
        if fetch is not None:
            return await fetch()
        return await self._ctx.client.review_item(page_id)

    async def _fresh_item(self, page_id: int) -> dict[str, Any] | None:
        try:
            return await self._fetch_item(page_id)
        except ServiceError:
            return None

    # -- one page --

    async def review_page(self, page_id: int, item: dict[str, Any] | None = None) -> PageOutcome:
        """Show one page and process window events until the page is left.

        :param page_id: Page ID.
        :param item: Already fetched review item, if any.
        :return: How the page was left.
        :raises WindowClosedError: The window went away.
        """
        if item is None:
            try:
                item = await self._fetch_item(page_id)
            except ServiceUnavailableError:
                raise
            except ServiceError:
                logger.info("page %s left the queue before it was shown", page_id)
                self._skipped.add(page_id)
                return PageOutcome("skip", page_id)
        images = await self._images(item)
        payload = await self._show_payload(item, images, reuse_images=False)
        try:
            while True:
                event = await self._call(payload)
                outcome = await self._handle(item, event)
                if outcome is None:
                    payload = {"op": "continue"}
                    continue
                if outcome.kind == "reshow" and outcome.item is not None:
                    item = outcome.item
                    payload = await self._show_payload(item, None, reuse_images=True)
                    continue
                return outcome
        finally:
            await self._drop_temp_files()

    async def _show_payload(
        self, item: Mapping[str, Any], images: Mapping[str, Any] | None, *, reuse_images: bool
    ) -> dict[str, Any]:
        position = min(self._position(), max(self.total, 1))
        total = max(self.total, position)
        page = dict(item)
        page["header"] = {
            "counter": msg("counter", pos=position, total=total),
            "title": str(item.get("title", "")),
            "page_label": msg("page_label", n=item.get("page_number", "?")),
            "reasons": describe_reasons(item),
            "scope_note": msg("scope_full") if item.get("scope") == "full" else "",
        }
        payload: dict[str, Any] = {
            "op": "show",
            "page": page,
            "images": images,
            "reuse_images": reuse_images,
            "can_back": bool(self._history) and not self._single,
            "autosave_idle_s": self._ctx.autosave_idle_s,
        }
        if self._intro:
            payload["intro"] = self._intro
            self._intro = None
        await self._ctx.out.status(msg("status_review", pos=position, total=total))
        return payload

    async def _images(self, item: Mapping[str, Any]) -> dict[str, Any]:
        """Fetch the page image and line crops once per page (section 8.3)."""
        client = self._ctx.client
        page_id = int(item["page_id"])
        page_side = int(item.get("page_long_side") or DEFAULT_PAGE_LONG_SIDE)
        crop_side = int(item.get("crop_long_side") or DEFAULT_CROP_LONG_SIDE)
        raw: dict[str, bytes] = {}
        try:
            raw["page"] = await client.page_image(page_id, long_side=page_side)
        except ServiceError:
            logger.warning("page image %s is unavailable", page_id)
        for hint in item.get("hints") or []:
            crop = hint.get("crop")
            if not crop:
                continue
            try:
                raw[f"crop:{hint['id']}"] = await client.page_image(
                    page_id, long_side=crop_side, clip=crop
                )
            except ServiceError:
                logger.warning(
                    "crop for hint %s of page %s is unavailable", hint.get("id"), page_id
                )
        inline_size = sum(len(data) for data in raw.values()) * 4 // 3
        refs: dict[str, str] = {}
        if inline_size > MAX_INLINE_IMAGES_BYTES and self._ctx.files is not None:
            refs = await self._store_images(page_id, raw)
        for key, data in raw.items():
            refs.setdefault(key, _data_uri(data))
        return {
            "page": refs.get("page"),
            "crops": {
                key.split(":", 1)[1]: ref for key, ref in refs.items() if key.startswith("crop:")
            },
        }

    async def _store_images(self, page_id: int, raw: Mapping[str, bytes]) -> dict[str, str]:
        """Fallback for large images: serve them from OpenWebUI's file store (section 8.3)."""
        refs: dict[str, str] = {}
        if self._ctx.files is None:
            return refs
        for key, data in raw.items():
            name = f"ocr-review-{page_id}-{key.replace(':', '-')}.jpg"
            try:
                stored = await self._ctx.files.store(name, data, "image/jpeg")
            except OwuiFilesError:
                logger.warning("could not store review image %s", name, exc_info=True)
                continue
            self._temp_files.append(stored.id)
            refs[key] = file_url(stored.id)
        return refs

    async def _drop_temp_files(self) -> None:
        files, self._temp_files = self._temp_files, []
        if self._ctx.files is None:
            return
        for file_id in files:
            try:
                await self._ctx.files.delete(file_id)
            except OwuiFilesError:
                logger.warning("could not delete review image %s", file_id)

    # -- window events (section 8.3) --

    async def _handle(
        self, item: Mapping[str, Any], event: Mapping[str, Any]
    ) -> PageOutcome | None:
        etype = str(event.get("type"))
        page_id = int(item["page_id"])
        ack: dict[str, Any] = {
            "event_id": event.get("id"),
            "page_id": event.get("page_id"),
            "type": etype,
        }
        draft = event.get("draft") if isinstance(event.get("draft"), Mapping) else None
        if etype == "draft":
            event_page = event.get("page_id")
            if isinstance(event_page, int) and draft is not None:
                self._acks.append(await self._save_draft(event_page, draft, ack, event.get("rev")))
            return None
        if etype == "stop":
            # Stop always ends the session, even if the window lost track of its page.
            stop_page = event.get("page_id")
            if isinstance(stop_page, int) and draft is not None:
                self._acks.append(await self._save_draft(stop_page, draft, ack, event.get("rev")))
            else:
                self._acks.append({**ack, "status": "ok"})
            return PageOutcome("stop", page_id)
        if event.get("page_id") != page_id:
            self._acks.append({**ack, "status": "error"})
            return None
        if etype == "done":
            return await self._finish_page(item, event, ack, on_conflict=None)
        if etype == "keep_mine":
            if event.get("finish"):
                return await self._finish_page(item, event, ack, on_conflict="keep_mine")
            if draft is not None:
                self._acks.append(await self._save_draft(page_id, draft, ack, event.get("rev")))
            return None
        if etype in ("later", "back", "retry"):
            return await self._leave_page(etype, page_id, draft, ack, event.get("rev"))
        if etype == "region":
            self._acks.append(await self._region(page_id, event.get("rect"), ack))
            return None
        if etype == "discard_draft":
            return await self._discard_draft(page_id, int(event.get("draft_seq") or 0), ack)
        self._acks.append({**ack, "status": "error"})
        return None

    async def _save_draft(
        self, page_id: int, draft: Mapping[str, Any], ack: Mapping[str, Any], rev: Any
    ) -> dict[str, Any]:
        result: dict[str, Any] = {**ack, "rev": rev}
        try:
            saved = await self._ctx.client.put_draft(page_id, draft)
        except ServiceUnavailableError:
            return {**result, "status": "error"}
        except ServiceError as err:
            if err.detail in ("draft_conflict", "stale"):
                fresh = await self._fresh_item(page_id)
                return {
                    **result,
                    "status": err.detail,
                    "item": fresh,
                    "server_draft": (fresh or {}).get("draft"),
                }
            if err.detail == "invalid_transition":
                return {**result, "status": "invalid_transition"}
            return {**result, "status": "error"}
        return {
            **result,
            "status": "ok",
            "seq": saved.get("seq"),
            "text_sha256": saved.get("text_sha256"),
        }

    async def _finish_page(
        self,
        item: Mapping[str, Any],
        event: Mapping[str, Any],
        ack: dict[str, Any],
        *,
        on_conflict: str | None,
    ) -> PageOutcome | None:
        """Send «Готово» (section 8.2) and react to the service's answer (section 8.4)."""
        page_id = int(item["page_id"])
        review = event.get("review")
        ack["finish"] = bool(event.get("finish"))
        if not isinstance(review, Mapping):
            self._acks.append({**ack, "status": "error"})
            return None
        ack["request_id"] = review.get("request_id")
        request = dict(review)
        if on_conflict:
            request["on_conflict"] = on_conflict
        try:
            result = await self._ctx.client.post_review(page_id, request)
        except ServiceUnavailableError:
            self._acks.append({**ack, "status": "error"})
            return None
        except ServiceError as err:
            return await self._finish_failed(page_id, event, ack, err)
        self._acks.append({**ack, "status": "ok", "revision_id": result.get("revision_id")})
        self.reviewed += 1
        self._done.add(page_id)
        self._history.append(page_id)
        if result.get("document_status") == "ready":
            self.ready_documents[str(item.get("document_id"))] = str(item.get("title", ""))
        return PageOutcome("done", page_id)

    async def _finish_failed(
        self, page_id: int, event: Mapping[str, Any], ack: dict[str, Any], err: ServiceError
    ) -> PageOutcome | None:
        if err.detail == "document_building":
            # The answer stays in the browser outbox; the text also goes to the server draft.
            result = {**ack, "status": "building"}
            draft = event.get("draft")
            if isinstance(draft, Mapping):
                saved = await self._save_draft(page_id, draft, ack, event.get("rev"))
                if saved["status"] == "ok":
                    result["seq"] = saved["seq"]
                    result["rev"] = event.get("rev")
            self._acks.append(result)
            self._skip(page_id)
            return PageOutcome("skip", page_id)
        if err.detail in ("stale", "draft_conflict"):
            fresh = await self._fresh_item(page_id)
            self._acks.append(
                {
                    **ack,
                    "status": err.detail,
                    "item": fresh,
                    "server_draft": (fresh or {}).get("draft"),
                }
            )
            return None
        if err.detail == "invalid_transition":
            self._acks.append({**ack, "status": "invalid_transition"})
            self._skip(page_id)
            return PageOutcome("skip", page_id)
        self._acks.append({**ack, "status": "error"})
        return None

    def _skip(self, page_id: int) -> None:
        self._skipped.add(page_id)
        self._history.append(page_id)

    async def _leave_page(
        self,
        etype: str,
        page_id: int,
        draft: Mapping[str, Any] | None,
        ack: dict[str, Any],
        rev: Any,
    ) -> PageOutcome | None:
        if etype == "back" and (self._single or not self._history):
            self._acks.append({**ack, "status": "error"})
            return None
        if draft is not None:
            saved = await self._save_draft(page_id, draft, ack, rev)
            self._acks.append(saved)
            if saved["status"] != "ok":
                return None
        if etype == "back":
            if draft is None:
                self._acks.append({**ack, "status": "ok"})
            return PageOutcome("back", page_id, target=self._history.pop())
        if etype == "retry":
            try:
                await self._ctx.client.retry(page_id)
            except ServiceError as err:
                status = "building" if err.detail == "document_building" else "error"
                self._acks.append({**ack, "status": status})
                return None
            if draft is None:
                self._acks.append({**ack, "status": "ok"})
        elif draft is None:
            self._acks.append({**ack, "status": "ok"})
        self._skip(page_id)
        return PageOutcome("skip", page_id)

    async def _region(self, page_id: int, rect: Any, ack: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(rect, Mapping):
            return {**ack, "status": "error"}
        try:
            result = await self._ctx.client.ocr_region(page_id, rect)
        except ServiceError as err:
            return {**ack, "status": "error", "code": err.status_code}
        return {**ack, "status": "ok", "text": str(result.get("text", ""))}

    async def _discard_draft(
        self, page_id: int, draft_seq: int, ack: dict[str, Any]
    ) -> PageOutcome | None:
        try:
            await self._ctx.client.delete_draft(page_id, draft_seq)
        except ServiceUnavailableError:
            self._acks.append({**ack, "status": "error"})
            return None
        except ServiceError as err:
            if err.detail == "draft_conflict":
                fresh = await self._fresh_item(page_id)
                self._acks.append(
                    {
                        **ack,
                        "status": "draft_conflict",
                        "item": fresh,
                        "server_draft": (fresh or {}).get("draft"),
                    }
                )
            else:
                self._acks.append({**ack, "status": "error"})
            return None
        fresh = await self._fresh_item(page_id)
        self._acks.append({**ack, "status": "ok"})
        if fresh is None:
            self._skip(page_id)
            return PageOutcome("skip", page_id)
        return PageOutcome("reshow", page_id, item=fresh)


# ---- review window (section 8.3) ----

# Pure logic shared with the Python references above; tests run it in Node.
REVIEW_CORE_JS = r"""
var OCR_CORE = (function () {
  'use strict';
  var K = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
  ];

  function nz(v) { return v === undefined ? null : v; }
  function rotr(v, n) { return (v >>> n) | (v << (32 - n)); }

  // crypto.subtle exists only in secure contexts; OpenWebUI is often served over plain http.
  function sha256Hex(str) {
    var bytes = new TextEncoder().encode(str);
    var len = bytes.length;
    var total = ((len + 9 + 63) >> 6) << 6;
    var buf = new Uint8Array(total);
    buf.set(bytes);
    buf[len] = 0x80;
    var view = new DataView(buf.buffer);
    view.setUint32(total - 8, Math.floor(len / 0x20000000));
    view.setUint32(total - 4, (len << 3) >>> 0);
    var H = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19];
    var W = new Array(64);
    for (var off = 0; off < total; off += 64) {
      var t;
      for (t = 0; t < 16; t++) W[t] = view.getUint32(off + t * 4);
      for (t = 16; t < 64; t++) {
        var x = W[t - 15], y = W[t - 2];
        var s0 = rotr(x, 7) ^ rotr(x, 18) ^ (x >>> 3);
        var s1 = rotr(y, 17) ^ rotr(y, 19) ^ (y >>> 10);
        W[t] = (W[t - 16] + s0 + W[t - 7] + s1) >>> 0;
      }
      var a = H[0], b = H[1], c = H[2], d = H[3], e = H[4], f = H[5], g = H[6], h = H[7];
      for (t = 0; t < 64; t++) {
        var S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
        var ch = (e & f) ^ (~e & g);
        var t1 = (h + S1 + ch + K[t] + W[t]) >>> 0;
        var S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
        var maj = (a & b) ^ (a & c) ^ (b & c);
        var t2 = (S0 + maj) >>> 0;
        h = g; g = f; f = e; e = (d + t1) >>> 0; d = c; c = b; b = a; a = (t1 + t2) >>> 0;
      }
      H[0] = (H[0] + a) >>> 0; H[1] = (H[1] + b) >>> 0; H[2] = (H[2] + c) >>> 0; H[3] = (H[3] + d) >>> 0;
      H[4] = (H[4] + e) >>> 0; H[5] = (H[5] + f) >>> 0; H[6] = (H[6] + g) >>> 0; H[7] = (H[7] + h) >>> 0;
    }
    return H.map(function (v) { return ('00000000' + v.toString(16)).slice(-8); }).join('');
  }

  // Section 8.3: prefix first, suffix capped so it never overlaps the prefix.
  function shiftHints(hints, oldText, newText) {
    var minLen = Math.min(oldText.length, newText.length);
    var p = 0;
    while (p < minLen && oldText.charCodeAt(p) === newText.charCodeAt(p)) p++;
    var s = 0, maxS = minLen - p;
    while (s < maxS && oldText.charCodeAt(oldText.length - 1 - s) === newText.charCodeAt(newText.length - 1 - s)) s++;
    var oldEnd = oldText.length - s;
    var delta = newText.length - oldText.length;
    return hints.map(function (h) {
      var out = Object.assign({}, h);
      if (oldText === newText || h.end <= p) return out;
      if (h.start >= oldEnd) { out.start = h.start + delta; out.end = h.end + delta; return out; }
      out.state = 'manual';
      return out;
    });
  }

  function chooseDraftSource(local, server) {
    var serverSeq = server ? (server.seq || 0) : 0;
    if (!local) return {source: 'server', base_seq: serverSeq, sent_rev: 0};
    var rev = local.rev || 0, sent = local.sent_rev || 0;
    if (rev === sent) return {source: 'server', base_seq: serverSeq, sent_rev: sent};
    if ((local.base_seq || 0) === serverSeq) return {source: 'local', base_seq: serverSeq, sent_rev: sent};
    var f = local.in_flight;
    if (f && server && serverSeq === (f.expected_seq || 0) + 1 && server.text_sha256 === f.sha256) {
      return {source: 'local', base_seq: serverSeq, sent_rev: Math.max(sent, f.rev || 0)};
    }
    return {source: 'conflict', base_seq: serverSeq, sent_rev: sent};
  }

  function applyDraftAck(local, ack) {
    var sent = local.sent_rev || 0;
    var acked = typeof ack.rev === 'number' ? ack.rev : sent;
    var out = Object.assign({}, local);
    out.base_seq = ack.seq;
    out.sent_rev = Math.max(sent, acked);
    if (local.in_flight && (local.in_flight.rev || 0) <= acked) out.in_flight = null;
    return out;
  }

  function withState(h) { return Object.assign({}, h, {state: h.state || 'active'}); }

  function restoreState(item, local, server) {
    local = local || null;
    server = server || null;
    var choice = chooseDraftSource(local, server);
    var localRev = local ? (local.rev || 0) : 0;
    if (choice.source === 'conflict') {
      return {
        mode: 'conflict', source: 'conflict', text: local.text || '', printed_label: nz(local.printed_label),
        hints: [], hints_outdated: false, choices: local.choices || {}, cursor: local.cursor || 0,
        base_seq: local.base_seq || 0, base_attempt_id: nz(local.base_attempt_id),
        base_revision_id: nz(local.base_revision_id), hints_version: nz(local.item_version),
        rev: localRev, sent_rev: local.sent_rev || 0, in_flight: nz(local.in_flight),
        stale_reason: null, dirty: true,
        conflict: {
          server_seq: choice.base_seq,
          server_text: server ? (server.text || '') : (item.text || ''),
          server_printed_label: server ? nz(server.printed_label) : nz(item.printed_label)
        }
      };
    }
    var src = choice.source === 'local' ? local : server;
    if (!src) {
      return {
        mode: 'edit', source: 'item', text: item.text || '', printed_label: nz(item.printed_label),
        hints: (item.hints || []).map(withState), hints_outdated: false, choices: {}, cursor: 0,
        base_seq: 0, base_attempt_id: nz(item.base_attempt_id), base_revision_id: nz(item.base_revision_id),
        hints_version: nz(item.item_version), rev: localRev, sent_rev: localRev, in_flight: null,
        stale_reason: null, dirty: false, conflict: null
      };
    }
    var sameVersion = nz(src.item_version) === nz(item.item_version);
    var baseDiffers = nz(src.base_attempt_id) !== nz(item.base_attempt_id) ||
      nz(src.base_revision_id) !== nz(item.base_revision_id);
    var serverStale = !!(server && server.stale);
    var stale = baseDiffers || serverStale;
    var staleReason = null;
    if (stale) staleReason = (serverStale && server.stale_reason) ? server.stale_reason : 'changed';
    var sentRev, inFlight;
    if (choice.source === 'local') {
      sentRev = choice.sent_rev;
      inFlight = choice.base_seq === (local.base_seq || 0) ? nz(local.in_flight) : null;
    } else {
      sentRev = localRev;
      inFlight = null;
    }
    return {
      mode: stale ? 'stale' : 'edit', source: choice.source, text: src.text || '',
      printed_label: nz(src.printed_label),
      hints: sameVersion ? (src.hints || []).map(withState) : [], hints_outdated: !sameVersion,
      choices: src.choices || {}, cursor: src.cursor || 0, base_seq: choice.base_seq,
      base_attempt_id: nz(src.base_attempt_id), base_revision_id: nz(src.base_revision_id),
      hints_version: nz(src.item_version), rev: localRev, sent_rev: sentRev, in_flight: inFlight,
      stale_reason: staleReason, dirty: localRev !== sentRev, conflict: null
    };
  }

  return {
    sha256Hex: sha256Hex, shiftHints: shiftHints, chooseDraftSource: chooseDraftSource,
    applyDraftAck: applyDraftAck, restoreState: restoreState
  };
})();
"""

# The overlay itself. Text from the model goes into the page only through textContent and
# value; nothing is ever rendered as HTML.
REVIEW_UI_JS = r"""
var OCR_WINDOW_VERSION = 'ocr-review-1';

function ocrCreateReview(CORE) {
  'use strict';
  var PREFIX = 'ocrReview.';
  var T = {
    saving: 'сохраняю…',
    recognizing: 'распознаю область…',
    done: 'Готово (Ctrl+Enter)',
    later: 'Отложить (Alt+S)',
    back: 'Назад (Alt+P)',
    region: 'Область (Alt+R)',
    retry: 'Перезапустить',
    stop: 'Закончить (Esc)',
    printedLabel: 'напечатанный номер',
    noHint: 'Курсор не на подсказке. Alt+↓ / Alt+↑ — к следующей / предыдущей.',
    noHints: 'Подсказок нет. Проверь страницу по скану.',
    noMoreHints: 'Больше подсказок нет',
    edited: 'изменено вручную',
    emptyVariant: '(без вставки)',
    hintsOutdated: 'Подсказки устарели после смены правил: текст сохранён, варианты A/B недоступны.',
    staleChanged: 'Страница изменилась после твоей правки.',
    staleRecognized: 'Страница распознана заново после твоей правки.',
    keepMine: 'Оставить мой текст',
    takeNew: 'Взять новый',
    takeNewRecognized: 'Взять новое распознавание',
    myText: 'Мой текст',
    currentText: 'Текущий текст страницы',
    conflictTitle: 'Страница изменена в другой вкладке. Выбери версию:',
    fromBrowser: 'Из этого браузера',
    savedVersion: 'Сохранённая',
    takeSaved: 'Взять сохранённый',
    building: 'Идёт сборка документа: «Готово» отправится в следующей проверке, текст сохранён.',
    pending: 'Страница распознаётся заново: правка сохранена.',
    saveError: 'Не удалось сохранить: текст остаётся в браузере, попробуй ещё раз.',
    regionError: 'Не удалось распознать область.',
    noCandidate: 'Страница не распознана: набери текст вручную или нажми «Перезапустить».',
    noImage: 'Скан не загрузился',
    zoomIn: '+', zoomOut: '−', fit: 'По размеру', rotate: 'Повернуть',
    close: '×',
    forceClose: 'Сервис не отвечает. Нажми Esc ещё раз, чтобы закрыть окно: текст сохранён в браузере.',
    reasons: {diff: 'расхождение проходов', low_conf: 'неуверенное чтение', marker: 'модель не прочитала', handwriting: 'рукописная пометка'}
  };
  var CSS = [
    '.ocrr-root{position:fixed;inset:0;z-index:2147483000;display:flex;flex-direction:column;background:#f3f3f1;color:#1d1d1d;font:14px/1.4 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}',
    '.ocrr-root *{box-sizing:border-box}',
    '.ocrr-hidden{display:none!important}',
    '.ocrr-head{display:flex;gap:12px;align-items:center;padding:8px 12px;background:#fff;border-bottom:1px solid #ddd;flex-wrap:wrap}',
    '.ocrr-counter,.ocrr-title{font-weight:600}',
    '.ocrr-reasons{color:#8a4b00}',
    '.ocrr-scope{color:#b00020;font-weight:600}',
    '.ocrr-status{margin-left:auto;color:#666}',
    '.ocrr-banner,.ocrr-notice,.ocrr-intro{padding:8px 12px;border-bottom:1px solid #ddd;display:flex;gap:8px;align-items:flex-start;flex-wrap:wrap}',
    '.ocrr-banner{background:#fff4d6}',
    '.ocrr-notice{background:#e8f1ff}',
    '.ocrr-intro{background:#fff;white-space:pre-wrap;max-height:30vh;overflow:auto}',
    '.ocrr-grow{flex:1}',
    '.ocrr-main{flex:1;display:flex;min-height:0}',
    '.ocrr-scan{flex:1;position:relative;overflow:hidden;background:#5a5a5a;cursor:grab}',
    '.ocrr-scan.ocrr-region-on{cursor:crosshair}',
    '.ocrr-img{position:absolute;left:0;top:0;transform-origin:50% 50%;user-select:none;max-width:none;max-height:none}',
    '.ocrr-selrect{position:absolute;border:2px dashed #ff3b30;background:rgba(255,59,48,.1);pointer-events:none}',
    '.ocrr-noimg{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;color:#eee}',
    '.ocrr-tools{position:absolute;left:8px;bottom:8px;display:flex;gap:4px}',
    '.ocrr-right{flex:1;display:flex;flex-direction:column;min-width:0;position:relative;border-left:1px solid #ccc;background:#fff}',
    '.ocrr-hint{padding:8px 12px;border-bottom:1px solid #eee;min-height:44px;max-height:35%;overflow:auto}',
    '.ocrr-hreasons{font-weight:600}',
    '.ocrr-variants{display:flex;gap:8px;flex-wrap:wrap;margin-top:6px}',
    '.ocrr-vtext{margin-left:6px;font-family:ui-monospace,Consolas,monospace;white-space:pre-wrap}',
    '.ocrr-crop{display:block;max-width:100%;margin-top:6px;border:1px solid #ccc}',
    '.ocrr-editor{flex:1;position:relative;min-height:0}',
    '.ocrr-mirror,.ocrr-ta{position:absolute;inset:0;width:100%;height:100%;margin:0;padding:10px 12px;border:0;font:15px/1.5 ui-monospace,"Cascadia Mono",Consolas,"DejaVu Sans Mono",monospace;white-space:pre-wrap;overflow-wrap:break-word;word-break:normal;overflow-y:scroll;letter-spacing:normal;tab-size:4}',
    '.ocrr-mirror{color:#111;z-index:1;pointer-events:none}',
    '.ocrr-ta{z-index:2;background:transparent;color:transparent;caret-color:#111;resize:none;outline:none}',
    '.ocrr-ta::selection{background:rgba(0,110,255,.28);color:transparent}',
    '.ocrr-mirror mark{color:inherit;padding:0;margin:0;border-radius:2px}',
    '.ocrr-r-diff{background:#cfe3ff}',
    '.ocrr-r-low_conf{background:#fff0a6}',
    '.ocrr-r-marker{background:#ffc9c9}',
    '.ocrr-r-handwriting{background:#e6d4ff}',
    '.ocrr-mirror mark.ocrr-chosen{background:#d7f5d7}',
    '.ocrr-mirror mark.ocrr-active{box-shadow:0 0 0 2px #2563eb}',
    '.ocrr-plain{flex:1;width:100%;border:1px solid #ddd;padding:10px 12px;font:15px/1.5 ui-monospace,Consolas,monospace;resize:none;background:#fafafa;white-space:pre-wrap}',
    '.ocrr-conflict{flex:1;display:flex;gap:8px;padding:8px;min-height:0}',
    '.ocrr-col{flex:1;display:flex;flex-direction:column;min-width:0;gap:6px}',
    '.ocrr-coltitle{font-weight:600}',
    '.ocrr-tabs{display:flex;gap:4px;padding:4px 12px}',
    '.ocrr-foot{display:flex;gap:8px;padding:8px 12px;border-top:1px solid #ddd;align-items:center;flex-wrap:wrap}',
    '.ocrr-label{width:170px;padding:4px 6px;border:1px solid #bbb;border-radius:4px}',
    '.ocrr-btn{padding:5px 10px;border:1px solid #bbb;border-radius:6px;background:#f7f7f7;cursor:pointer;color:#111;font:inherit}',
    '.ocrr-btn:disabled{opacity:.45;cursor:default}',
    '.ocrr-primary{background:#2563eb;border-color:#2563eb;color:#fff}',
    '.ocrr-on{background:#2563eb;border-color:#2563eb;color:#fff}',
    '.ocrr-picked{outline:2px solid #16a34a}',
    '.ocrr-muted{color:#777}',
    '.ocrr-blocker{position:absolute;inset:0;background:rgba(255,255,255,.6);display:flex;align-items:center;justify-content:center;font-size:18px;z-index:5}'
  ].join('\n');

  var R = {
    version: OCR_WINDOW_VERSION, events: [], waiter: null, blocked: false, root: null, ps: null,
    images: {page: null, crops: {}}, mem: {}, nextId: 1, ui: null, lastCallAt: Date.now(),
    idleTimer: null, noticeTimer: null, stopAt: 0, view: {s: 1, tx: 0, ty: 0, rot: 0},
    regionMode: false, drag: null
  };

  function nz(v) { return v === undefined ? null : v; }

  // ---- storage: localStorage with an in-memory fallback ----
  function storeGet(key) {
    try {
      var raw = window.localStorage.getItem(PREFIX + key);
      if (raw !== null) return JSON.parse(raw);
    } catch (e) { /* storage blocked: use the in-memory copy */ }
    return Object.prototype.hasOwnProperty.call(R.mem, key) ? R.mem[key] : null;
  }
  function storeSet(key, value) {
    R.mem[key] = value;
    try { window.localStorage.setItem(PREFIX + key, JSON.stringify(value)); } catch (e) { /* in-memory copy only */ }
  }
  function storeDel(key) {
    delete R.mem[key];
    try { window.localStorage.removeItem(PREFIX + key); } catch (e) { /* nothing to remove */ }
  }
  function loadLocal(pageId) { return pageId == null ? null : storeGet('draft.' + pageId); }
  function saveLocal(local) { local.saved_at = Date.now(); storeSet('draft.' + local.page_id, local); }
  function dropLocal(pageId) { storeDel('draft.' + pageId); }
  function newLocal(pageId) { return {page_id: pageId, base_seq: 0, rev: 0, sent_rev: 0, in_flight: null}; }
  function readOutbox() { var v = storeGet('outbox'); return Array.isArray(v) ? v : []; }
  function writeOutbox(list) { storeSet('outbox', list); }
  function outboxAdd(pageId, review) {
    var box = readOutbox();
    box.push({request_id: review.request_id, page_id: pageId, review: review, created_at: Date.now()});
    writeOutbox(box);
  }
  function outboxFind(box, requestId) {
    for (var i = 0; i < box.length; i++) if (box[i].request_id === requestId) return i;
    return -1;
  }
  function outboxUpdateSeq(requestId, seq) {
    var box = readOutbox(), i = outboxFind(box, requestId);
    if (i >= 0) { box[i].review.draft_seq = seq; writeOutbox(box); }
  }
  function outboxRemove(requestId) {
    var box = readOutbox(), i = outboxFind(box, requestId);
    if (i >= 0) { box.splice(i, 1); writeOutbox(box); }
  }

  function uuid() {
    try {
      if (window.crypto && typeof window.crypto.randomUUID === 'function') return window.crypto.randomUUID();
    } catch (e) { /* randomUUID needs a secure context */ }
    var b = new Uint8Array(16);
    if (window.crypto && window.crypto.getRandomValues) window.crypto.getRandomValues(b);
    else for (var i = 0; i < 16; i++) b[i] = Math.floor(Math.random() * 256);
    b[6] = (b[6] & 0x0f) | 0x40;
    b[8] = (b[8] & 0x3f) | 0x80;
    var h = Array.prototype.map.call(b, function (x) { return ('0' + x.toString(16)).slice(-2); }).join('');
    return h.slice(0, 8) + '-' + h.slice(8, 12) + '-' + h.slice(12, 16) + '-' + h.slice(16, 20) + '-' + h.slice(20);
  }

  // ---- DOM helpers: text only through textContent / value ----
  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  }
  function button(label, onClick, cls) {
    var b = el('button', 'ocrr-btn' + (cls ? ' ' + cls : ''), label);
    b.type = 'button';
    b.addEventListener('click', function (ev) { ev.preventDefault(); onClick(); });
    return b;
  }
  function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); }
  function show(node, on) {
    var cls = (node.className || '').split(' ').filter(function (c) { return c && c !== 'ocrr-hidden'; });
    if (!on) cls.push('ocrr-hidden');
    node.className = cls.join(' ');
  }

  // ---- events to the pipe: FIFO, consecutive drafts of one page collapse ----
  function isTerminal(ev) {
    return ev.type === 'done' || ev.type === 'later' || ev.type === 'back' || ev.type === 'retry' ||
      ev.type === 'stop' || ev.type === 'discard_draft' || (ev.type === 'keep_mine' && ev.finish);
  }
  function pushEvent(ev) {
    ev.id = R.nextId++;
    var q = R.events;
    var tail = q.length ? q[q.length - 1] : null;
    if (tail && tail.type === 'draft' && tail.page_id === ev.page_id && (ev.type === 'draft' || isTerminal(ev))) q.pop();
    q.push(ev);
    deliver();
  }
  function deliver() {
    if (!R.waiter) return;
    var ev = takeEvent();
    if (!ev) return;
    var w = R.waiter;
    R.waiter = null;
    w(ev);
  }
  function takeEvent() {
    while (R.events.length) {
      var ev = R.events.shift();
      if (finalizeEvent(ev)) return ev;
    }
    return null;
  }
  // Save numbers are filled in when the pipe takes the event: by then every earlier answer
  // has been applied, so the number is the server's current one.
  function finalizeEvent(ev) {
    var ps = R.ps;
    if (ev.type === 'draft' && ps && ps.page_id === ev.page_id && ps.mode !== 'edit') return false;
    var local = loadLocal(ev.page_id);
    var base = local ? (local.base_seq || 0) : 0;
    if (ev.draft) {
      ev.draft.expected_seq = base;
      if (local) {
        local.in_flight = {expected_seq: base, rev: ev.rev || 0, sha256: CORE.sha256Hex(ev.draft.text || '')};
        saveLocal(local);
      }
    }
    if (ev.review) {
      ev.review.draft_seq = base;
      outboxUpdateSeq(ev.review.request_id, base);
    }
    if (ev.type === 'discard_draft') ev.draft_seq = base;
    return true;
  }

  // ---- answers from the pipe ----
  function applyOutboxAck(list) {
    if (!list || !list.length) return;
    var box = readOutbox();
    list.forEach(function (a) {
      var i = outboxFind(box, a.request_id);
      if (i < 0 || a.action === 'keep') return;
      var entry = box[i];
      box.splice(i, 1);
      var review = entry.review || {};
      var local = loadLocal(entry.page_id);
      if (a.action === 'to_local') {
        if (!local) {
          local = newLocal(entry.page_id);
          local.base_seq = review.draft_seq || 0;
        }
        if (local.text !== review.text) {
          Object.assign(local, {
            text: review.text || '', printed_label: nz(review.printed_label), choices: review.choices || {},
            hints: [], cursor: 0, item_version: nz(review.item_version),
            base_attempt_id: nz(review.base_attempt_id), base_revision_id: nz(review.base_revision_id)
          });
        }
        local.rev = Math.max(local.rev || 0, local.sent_rev || 0) + 1;
        saveLocal(local);
      } else if (local && local.text === review.text) {
        if (a.seq != null) {
          local.base_seq = a.seq;
          local.sent_rev = local.rev || 0;
          local.in_flight = null;
          saveLocal(local);
        } else {
          dropLocal(entry.page_id);
        }
      }
    });
    writeOutbox(box);
  }

  function applyAck(ack) {
    var pid = ack.page_id;
    var cur = R.ps && R.ps.page_id === pid ? R.ps : null;
    var finishing = ack.type === 'done' || (ack.type === 'keep_mine' && ack.finish);
    var local;
    if (ack.status === 'ok') {
      if (finishing) { outboxRemove(ack.request_id); dropLocal(pid); return; }
      if (ack.type === 'discard_draft') { dropLocal(pid); return; }
      if (ack.type === 'region') { setStatus(''); if (cur) insertAtCursor(ack.text || ''); return; }
      if (ack.seq != null) {
        local = loadLocal(pid);
        if (local) saveLocal(CORE.applyDraftAck(local, {rev: ack.rev, seq: ack.seq}));
      }
      return;
    }
    if (ack.status === 'building') {
      if (ack.seq != null) {
        outboxUpdateSeq(ack.request_id, ack.seq);
        local = loadLocal(pid);
        if (local) saveLocal(CORE.applyDraftAck(local, {rev: ack.rev, seq: ack.seq}));
      }
      notice(T.building);
      return;
    }
    if (finishing) outboxRemove(ack.request_id);
    local = loadLocal(pid);
    if (local && local.in_flight) { local.in_flight = null; saveLocal(local); }
    if (ack.status === 'draft_conflict') { if (cur) enterConflict(ack.server_draft || null, ack.item || null); return; }
    if (ack.status === 'stale') {
      if (cur) {
        if (ack.item) cur.item = Object.assign({}, ack.item, {header: cur.item.header});
        cur.pendingFinish = finishing;
        var sd = cur.item && cur.item.draft;
        enterStale(sd && sd.stale_reason ? sd.stale_reason : 'changed');
      }
      return;
    }
    if (ack.status === 'invalid_transition') { notice(T.pending); return; }
    if (ack.type === 'region') { setStatus(''); notice(T.regionError); return; }
    notice(T.saveError);
  }

  // ---- overlay ----
  function buildRoot() {
    var ui = R.ui = {};
    var root = el('div', 'ocrr-root');
    root.setAttribute('role', 'dialog');
    var style = el('style', null, CSS);
    root.appendChild(style);

    var head = el('div', 'ocrr-head');
    ui.counter = el('span', 'ocrr-counter');
    ui.title = el('span', 'ocrr-title');
    ui.pageLabel = el('span', 'ocrr-page');
    ui.reasons = el('span', 'ocrr-reasons');
    ui.scope = el('span', 'ocrr-scope');
    ui.status = el('span', 'ocrr-status');
    head.appendChild(ui.counter); head.appendChild(ui.title); head.appendChild(ui.pageLabel);
    head.appendChild(ui.reasons); head.appendChild(ui.scope); head.appendChild(ui.status);
    head.appendChild(button(T.stop, requestStop));

    ui.intro = el('div', 'ocrr-intro ocrr-hidden');
    ui.banner = el('div', 'ocrr-banner ocrr-hidden');
    ui.notice = el('div', 'ocrr-notice ocrr-hidden');

    var main = el('div', 'ocrr-main');
    ui.scan = el('div', 'ocrr-scan');
    ui.img = el('img', 'ocrr-img');
    ui.img.alt = '';
    ui.img.draggable = false;
    ui.sel = el('div', 'ocrr-selrect ocrr-hidden');
    ui.noImg = el('div', 'ocrr-noimg ocrr-hidden', T.noImage);
    var tools = el('div', 'ocrr-tools');
    tools.appendChild(button(T.zoomOut, function () { zoomAt(0.8); }));
    tools.appendChild(button(T.zoomIn, function () { zoomAt(1.25); }));
    tools.appendChild(button(T.fit, fitImage));
    tools.appendChild(button(T.rotate, function () { R.view.rot = (R.view.rot + 90) % 360; fitImage(); }));
    ui.regionBtn = button(T.region, function () { toggleRegion(); });
    tools.appendChild(ui.regionBtn);
    ui.scan.appendChild(ui.img); ui.scan.appendChild(ui.sel); ui.scan.appendChild(ui.noImg); ui.scan.appendChild(tools);

    var right = el('div', 'ocrr-right');
    ui.hint = el('div', 'ocrr-hint');
    ui.tabs = el('div', 'ocrr-tabs ocrr-hidden');
    ui.editor = el('div', 'ocrr-editor');
    ui.mirror = el('div', 'ocrr-mirror');
    ui.ta = el('textarea', 'ocrr-ta');
    ui.ta.spellcheck = false;
    ui.ta.setAttribute('autocomplete', 'off');
    ui.editor.appendChild(ui.mirror); ui.editor.appendChild(ui.ta);
    ui.current = el('textarea', 'ocrr-plain ocrr-hidden');
    ui.current.readOnly = true;
    ui.conflict = el('div', 'ocrr-conflict ocrr-hidden');
    var foot = el('div', 'ocrr-foot');
    ui.label = el('input', 'ocrr-label');
    ui.label.type = 'text';
    ui.label.placeholder = T.printedLabel;
    ui.doneBtn = button(T.done, doDone, 'ocrr-primary');
    ui.laterBtn = button(T.later, doLater);
    ui.backBtn = button(T.back, doBack);
    ui.retryBtn = button(T.retry, doRetry);
    foot.appendChild(ui.label); foot.appendChild(ui.doneBtn); foot.appendChild(ui.laterBtn);
    foot.appendChild(ui.backBtn); foot.appendChild(ui.retryBtn);
    ui.blocker = el('div', 'ocrr-blocker ocrr-hidden', T.saving);
    right.appendChild(ui.hint); right.appendChild(ui.tabs); right.appendChild(ui.editor);
    right.appendChild(ui.current); right.appendChild(ui.conflict); right.appendChild(foot);
    right.appendChild(ui.blocker);

    main.appendChild(ui.scan); main.appendChild(right);
    root.appendChild(head); root.appendChild(ui.intro); root.appendChild(ui.banner);
    root.appendChild(ui.notice); root.appendChild(main);
    document.body.appendChild(root);
    R.root = root;

    ui.ta.addEventListener('input', onInput);
    ui.ta.addEventListener('scroll', function () { ui.mirror.scrollTop = ui.ta.scrollTop; });
    ui.ta.addEventListener('keyup', function () { updateActive(false); });
    ui.ta.addEventListener('click', function () { updateActive(false); });
    ui.ta.addEventListener('select', function () { updateActive(false); });
    ui.label.addEventListener('input', onLabelInput);
    ui.img.addEventListener('load', fitImage);
    ui.scan.addEventListener('wheel', onWheel, {passive: false});
    ui.scan.addEventListener('mousedown', onScanDown);
    window.addEventListener('mousemove', onScanMove);
    window.addEventListener('mouseup', onScanUp);
    window.addEventListener('keydown', onKey, true);
  }
  function ensureRoot() { if (!R.root) buildRoot(); }

  function destroy() {
    window.removeEventListener('keydown', onKey, true);
    window.removeEventListener('mousemove', onScanMove);
    window.removeEventListener('mouseup', onScanUp);
    clearTimeout(R.idleTimer);
    clearTimeout(R.noticeTimer);
    if (R.root && R.root.parentNode) R.root.parentNode.removeChild(R.root);
    R.root = null; R.ui = null; R.ps = null; R.events = []; R.blocked = false;
    if (R.waiter) { var w = R.waiter; R.waiter = null; w({error: 'closed'}); }
  }

  function setStatus(text) { if (R.ui) R.ui.status.textContent = text || ''; }
  function notice(text) {
    if (!R.ui) return;
    R.ui.notice.textContent = text;
    show(R.ui.notice, true);
    clearTimeout(R.noticeTimer);
    R.noticeTimer = setTimeout(function () { if (R.ui) show(R.ui.notice, false); }, 8000);
  }
  function showIntro(text) {
    var box = R.ui.intro;
    clear(box);
    var body = el('div', 'ocrr-grow', text);
    box.appendChild(body);
    box.appendChild(button(T.close, function () { show(box, false); }));
    show(box, true);
  }

  // ---- page ----
  function showPage(call) {
    var item = call.page;
    if (!(call.reuse_images && R.ps && R.ps.page_id === item.page_id)) R.images = call.images || {page: null, crops: {}};
    var local = loadLocal(item.page_id);
    var st = CORE.restoreState(item, local, item.draft || null);
    R.ps = {
      page_id: item.page_id, item: item, text: st.text, hints: st.hints, choices: st.choices || {},
      hints_version: st.hints_version, hints_outdated: st.hints_outdated,
      base_attempt_id: st.base_attempt_id, base_revision_id: st.base_revision_id,
      mode: st.mode, stale_reason: st.stale_reason, conflict: st.conflict, opened_at: Date.now(),
      activeId: null, pendingChoice: null, pendingFinish: false, canBack: !!call.can_back,
      idle: call.autosave_idle_s || 10, tab: 'mine'
    };
    var copy = Object.assign(local || newLocal(item.page_id), {
      base_seq: st.base_seq, rev: st.rev, sent_rev: st.sent_rev,
      // No pipe call can still carry an older save when a new show arrives: resend it.
      in_flight: st.dirty ? null : st.in_flight
    });
    R.blocked = false;
    R.ui.ta.value = st.text;
    R.ui.label.value = st.printed_label || '';
    if (st.mode !== 'conflict') fillLocal(copy);
    saveLocal(copy);
    var pos = Math.min(st.cursor || 0, st.text.length);
    try { R.ui.ta.setSelectionRange(pos, pos); } catch (e) { /* not focusable yet */ }
    R.ui.ta.scrollTop = 0;
    show(R.ui.intro, false);
    if (call.intro) showIntro(call.intro);
    loadImage();
    renderAll();
    try { R.ui.ta.focus(); } catch (e) { /* ignore */ }
    if (st.mode === 'edit' && st.dirty) queueDraft();
  }

  function fillLocal(local) {
    var ps = R.ps, ui = R.ui;
    local.page_id = ps.page_id;
    local.text = ui.ta.value;
    local.printed_label = ui.label.value || null;
    local.hints = ps.hints;
    local.choices = ps.choices;
    local.cursor = ui.ta.selectionStart || 0;
    local.item_version = nz(ps.hints_version);
    local.base_attempt_id = nz(ps.base_attempt_id);
    local.base_revision_id = nz(ps.base_revision_id);
  }
  function touchLocal() {
    var ps = R.ps;
    var local = loadLocal(ps.page_id) || newLocal(ps.page_id);
    local.rev = (local.rev || 0) + 1;
    fillLocal(local);
    saveLocal(local);
    return local;
  }
  function draftPayload() {
    var ps = R.ps, ui = R.ui;
    return {
      item_version: nz(ps.hints_version), base_attempt_id: nz(ps.base_attempt_id),
      base_revision_id: nz(ps.base_revision_id), text: ui.ta.value, printed_label: ui.label.value || null,
      choices: ps.choices, hints: ps.hints, cursor: ui.ta.selectionStart || 0,
      scope: nz(ps.item.scope)
    };
  }
  function reviewPayload() {
    var ps = R.ps, ui = R.ui;
    return {
      request_id: uuid(), item_version: nz(ps.item.item_version), base_attempt_id: nz(ps.base_attempt_id),
      base_revision_id: nz(ps.base_revision_id), text: ui.ta.value, printed_label: ui.label.value || null,
      duration_s: Math.round((Date.now() - ps.opened_at) / 1000), choices: ps.choices
    };
  }
  function hasUnsent() {
    var ps = R.ps;
    if (!ps || ps.mode !== 'edit') return false;
    var local = loadLocal(ps.page_id);
    return !!local && (local.rev || 0) !== (local.sent_rev || 0);
  }

  function renderAll() {
    renderHeader();
    renderMode();
    renderMirror();
    updateActive(true);
  }
  function renderHeader() {
    var hd = R.ps.item.header || {};
    R.ui.counter.textContent = hd.counter || '';
    R.ui.title.textContent = hd.title || '';
    R.ui.pageLabel.textContent = hd.page_label || '';
    R.ui.reasons.textContent = hd.reasons || '';
    R.ui.scope.textContent = hd.scope_note || '';
  }
  function renderMode() {
    var ps = R.ps, ui = R.ui;
    clear(ui.banner); clear(ui.tabs); clear(ui.conflict);
    var lines = [];
    if (ps.mode === 'edit' && ps.item.no_candidate && !ui.ta.value) lines.push(T.noCandidate);
    if (ps.mode === 'edit' && ps.hints_outdated) lines.push(T.hintsOutdated);
    if (ps.mode === 'stale') {
      var recognized = ps.stale_reason === 'recognized_again';
      ui.banner.appendChild(el('span', 'ocrr-grow', recognized ? T.staleRecognized : T.staleChanged));
      ui.banner.appendChild(button(T.keepMine, keepMine, 'ocrr-primary'));
      ui.banner.appendChild(button(recognized ? T.takeNewRecognized : T.takeNew, takeNew));
      ui.tabs.appendChild(button(T.myText, function () { ps.tab = 'mine'; renderMode(); }, ps.tab === 'mine' ? 'ocrr-on' : ''));
      ui.tabs.appendChild(button(T.currentText, function () { ps.tab = 'current'; renderMode(); }, ps.tab === 'current' ? 'ocrr-on' : ''));
      ui.current.value = ps.item.text || '';
    }
    if (ps.mode === 'conflict' && ps.conflict) {
      ui.banner.appendChild(el('span', 'ocrr-grow', T.conflictTitle));
      var mine = el('div', 'ocrr-col');
      var mineText = el('textarea', 'ocrr-plain'); mineText.readOnly = true; mineText.value = ui.ta.value;
      mine.appendChild(el('div', 'ocrr-coltitle', T.fromBrowser)); mine.appendChild(mineText);
      mine.appendChild(button(T.keepMine, chooseMine, 'ocrr-primary'));
      var saved = el('div', 'ocrr-col');
      var savedText = el('textarea', 'ocrr-plain'); savedText.readOnly = true; savedText.value = ps.conflict.server_text || '';
      saved.appendChild(el('div', 'ocrr-coltitle', T.savedVersion)); saved.appendChild(savedText);
      saved.appendChild(button(T.takeSaved, chooseSaved));
      ui.conflict.appendChild(mine); ui.conflict.appendChild(saved);
    }
    lines.forEach(function (line) { ui.banner.appendChild(el('div', 'ocrr-grow', line)); });
    show(ui.banner, ui.banner.firstChild != null);
    show(ui.tabs, ps.mode === 'stale');
    show(ui.conflict, ps.mode === 'conflict');
    show(ui.editor, ps.mode === 'edit' || (ps.mode === 'stale' && ps.tab === 'mine'));
    show(ui.current, ps.mode === 'stale' && ps.tab === 'current');
    var editing = ps.mode === 'edit';
    ui.ta.readOnly = !editing || R.blocked;
    ui.label.readOnly = !editing || R.blocked;
    ui.doneBtn.disabled = !editing;
    ui.laterBtn.disabled = ps.mode === 'conflict';
    ui.backBtn.disabled = !ps.canBack;
    show(ui.retryBtn, ps.item.status === 'failed');
    show(ui.blocker, R.blocked);
  }
  function block() { R.blocked = true; clearTimeout(R.idleTimer); if (R.ps) renderMode(); }
  function unblock() { R.blocked = false; if (R.ps && R.ui) renderMode(); }

  // ---- highlighting: a div under a transparent textarea ----
  function primaryReason(h) {
    var order = ['marker', 'diff', 'low_conf', 'handwriting'];
    var rs = h.reasons || [];
    for (var i = 0; i < order.length; i++) if (rs.indexOf(order[i]) >= 0) return order[i];
    return 'diff';
  }
  function visibleHints() {
    var ps = R.ps;
    if (!ps || ps.mode !== 'edit' || ps.hints_outdated) return [];
    return ps.hints.filter(function (h) { return h.state !== 'manual' && h.end > h.start; })
      .sort(function (a, b) { return a.start - b.start; });
  }
  function renderMirror() {
    var m = R.ui.mirror, text = R.ui.ta.value;
    clear(m);
    var pos = 0;
    visibleHints().forEach(function (h) {
      if (h.start < pos || h.end > text.length) return;
      if (h.start > pos) m.appendChild(document.createTextNode(text.slice(pos, h.start)));
      var mk = el('mark', 'ocrr-r-' + primaryReason(h), text.slice(h.start, h.end));
      if (h.state && h.state.indexOf('chosen') === 0) mk.className += ' ocrr-chosen';
      if (h.id === R.ps.activeId) mk.className += ' ocrr-active';
      mk.setAttribute('data-id', String(h.id));
      m.appendChild(mk);
      pos = h.end;
    });
    // A trailing space keeps the last empty line as tall as in the textarea.
    m.appendChild(document.createTextNode(text.slice(pos) + ' '));
    m.scrollTop = R.ui.ta.scrollTop;
  }
  function findMark(id) {
    var marks = R.ui.mirror.getElementsByTagName('mark');
    for (var i = 0; i < marks.length; i++) if (marks[i].getAttribute('data-id') === String(id)) return marks[i];
    return null;
  }
  function activeHint() {
    var ps = R.ps;
    if (!ps || ps.activeId == null) return null;
    for (var i = 0; i < ps.hints.length; i++) if (ps.hints[i].id === ps.activeId) return ps.hints[i];
    return null;
  }
  function updateActive(force) {
    var ps = R.ps;
    if (!ps) return;
    var pos = R.ui.ta.selectionStart || 0, found = null, hs = visibleHints();
    for (var i = 0; i < hs.length; i++) if (hs[i].start <= pos && pos <= hs[i].end) { found = hs[i]; break; }
    var id = found ? found.id : null;
    if (!force && id === ps.activeId) return;
    var prev = ps.activeId != null ? findMark(ps.activeId) : null;
    if (prev) prev.className = prev.className.replace(' ocrr-active', '');
    ps.activeId = id;
    var mk = id != null ? findMark(id) : null;
    if (mk && mk.className.indexOf('ocrr-active') < 0) mk.className += ' ocrr-active';
    renderHintPanel(found, hs.length);
  }
  function renderHintPanel(h, count) {
    var ui = R.ui, ps = R.ps;
    clear(ui.hint);
    if (!h) {
      ui.hint.appendChild(el('div', 'ocrr-muted', ps.mode === 'edit' ? (count ? T.noHint : T.noHints) : ''));
      return;
    }
    var words = (h.reasons || []).map(function (r) { return T.reasons[r] || r; }).join(', ');
    ui.hint.appendChild(el('div', 'ocrr-hreasons', words));
    if (h.variants && h.state !== 'manual') {
      var row = el('div', 'ocrr-variants');
      ['a', 'b'].forEach(function (k) {
        var v = h.variants[k];
        if (typeof v !== 'string') return;
        var b = button((k === 'a' ? 'A' : 'B') + ' (Alt+' + (k === 'a' ? '1' : '2') + ')', function () { applyVariant(k); },
          h.state === 'chosen_' + k ? 'ocrr-picked' : '');
        b.appendChild(el('span', 'ocrr-vtext', v === '' ? T.emptyVariant : v));
        row.appendChild(b);
      });
      ui.hint.appendChild(row);
    }
    var src = h.crop && R.images && R.images.crops ? R.images.crops[String(h.id)] : null;
    if (src) {
      var im = el('img', 'ocrr-crop');
      im.alt = '';
      im.src = src;
      ui.hint.appendChild(im);
    }
  }
  function scrollToHint(h) {
    var mk = findMark(h.id);
    if (!mk) return;
    R.ui.ta.scrollTop = Math.max(0, mk.offsetTop - R.ui.ta.clientHeight / 3);
    R.ui.mirror.scrollTop = R.ui.ta.scrollTop;
  }

  // ---- editing ----
  function onInput() {
    var ps = R.ps;
    if (!ps) return;
    var oldText = ps.text, newText = R.ui.ta.value;
    var hints = CORE.shiftHints(ps.hints, oldText, newText);
    if (ps.pendingChoice) {
      var pc = ps.pendingChoice;
      ps.pendingChoice = null;
      hints = hints.map(function (h) {
        return h.id === pc.id ? Object.assign({}, h, {start: pc.start, end: pc.start + pc.len, state: 'chosen_' + pc.variant}) : h;
      });
    }
    ps.text = newText;
    ps.hints = hints;
    touchLocal();
    renderMirror();
    updateActive(true);
    scheduleIdle();
  }
  function onLabelInput() {
    if (!R.ps || R.ps.mode !== 'edit') return;
    touchLocal();
    scheduleIdle();
  }
  function scheduleIdle() {
    clearTimeout(R.idleTimer);
    var ps = R.ps;
    if (!ps) return;
    R.idleTimer = setTimeout(queueDraft, Math.max(1, ps.idle) * 1000);
  }
  function queueDraft() {
    var ps = R.ps;
    if (!ps || ps.mode !== 'edit' || R.blocked) return;
    var local = loadLocal(ps.page_id);
    if (!local || (local.rev || 0) === (local.sent_rev || 0)) return;
    if (local.in_flight && local.in_flight.rev === local.rev) return;
    pushEvent({type: 'draft', page_id: ps.page_id, rev: local.rev, draft: draftPayload()});
  }
  function replaceRange(start, end, text) {
    var ta = R.ui.ta;
    ta.focus();
    ta.setSelectionRange(start, end);
    var ok = false;
    try {
      ok = text.length ? document.execCommand('insertText', false, text) : document.execCommand('delete');
    } catch (e) { ok = false; }
    if (!ok) {
      // Fallback without the native undo stack.
      ta.setRangeText(text, start, end, 'end');
      onInput();
    }
  }
  function insertAtCursor(text) {
    if (!R.ps || R.ps.mode !== 'edit' || !text) return;
    var ta = R.ui.ta;
    replaceRange(ta.selectionStart || 0, ta.selectionEnd || 0, text);
  }
  function applyVariant(which) {
    var ps = R.ps, h = activeHint();
    if (!ps || R.blocked || ps.mode !== 'edit' || !h || !h.variants || h.state === 'manual') return;
    var v = h.variants[which];
    if (typeof v !== 'string') return;
    ps.choices[String(h.id)] = which;
    ps.pendingChoice = {id: h.id, start: h.start, len: v.length, variant: which};
    replaceRange(h.start, h.end, v);
    if (ps.pendingChoice) {
      // The text already was this variant, so no input event fired.
      ps.pendingChoice = null;
      ps.hints = ps.hints.map(function (x) { return x.id === h.id ? Object.assign({}, x, {state: 'chosen_' + which}) : x; });
      touchLocal();
      renderMirror();
    }
    try { R.ui.ta.setSelectionRange(h.start, h.start); } catch (e) { /* ignore */ }
    updateActive(true);
    queueDraft();
  }
  function navHint(dir) {
    var ps = R.ps;
    if (!ps || R.blocked) return;
    var hs = visibleHints();
    var cur = activeHint();
    var pos = cur ? cur.start : (R.ui.ta.selectionStart || 0);
    var target = null, i;
    if (dir > 0) { for (i = 0; i < hs.length; i++) if (hs[i].start > pos) { target = hs[i]; break; } }
    else { for (i = hs.length - 1; i >= 0; i--) if (hs[i].start < pos) { target = hs[i]; break; } }
    if (!target) { notice(T.noMoreHints); return; }
    R.ui.ta.focus();
    R.ui.ta.setSelectionRange(target.start, target.start);
    updateActive(true);
    scrollToHint(target);
    queueDraft();
  }

  // ---- page actions ----
  function terminal(ev) { block(); pushEvent(ev); }
  function doDone() {
    var ps = R.ps;
    if (!ps || R.blocked || ps.mode !== 'edit') return;
    var local = loadLocal(ps.page_id) || newLocal(ps.page_id);
    fillLocal(local);
    saveLocal(local);
    var review = reviewPayload();
    outboxAdd(ps.page_id, review);
    terminal({type: 'done', page_id: ps.page_id, rev: local.rev || 0, review: review, draft: draftPayload()});
  }
  function doLater() {
    var ps = R.ps;
    if (!ps || R.blocked || ps.mode === 'conflict') return;
    var local = loadLocal(ps.page_id);
    terminal({type: 'later', page_id: ps.page_id, rev: local ? local.rev || 0 : 0, draft: ps.mode === 'edit' ? draftPayload() : null});
  }
  function doBack() {
    var ps = R.ps;
    if (!ps || R.blocked || !ps.canBack) return;
    var local = loadLocal(ps.page_id);
    terminal({type: 'back', page_id: ps.page_id, rev: local ? local.rev || 0 : 0, draft: hasUnsent() ? draftPayload() : null});
  }
  function doRetry() {
    var ps = R.ps;
    if (!ps || R.blocked || ps.item.status !== 'failed') return;
    var local = loadLocal(ps.page_id);
    terminal({type: 'retry', page_id: ps.page_id, rev: local ? local.rev || 0 : 0, draft: hasUnsent() ? draftPayload() : null});
  }
  function requestStop() {
    if (R.regionMode) { toggleRegion(false); return; }
    if (R.blocked && !R.waiter) {
      // The pipe is busy or gone; a second Esc closes the window, the text stays in localStorage.
      if (R.stopAt && Date.now() - R.stopAt < 4000) { destroy(); return; }
      R.stopAt = Date.now();
      notice(T.forceClose);
      return;
    }
    var ps = R.ps;
    var local = ps ? loadLocal(ps.page_id) : null;
    terminal({type: 'stop', page_id: ps ? ps.page_id : null, rev: local ? local.rev || 0 : 0, draft: hasUnsent() ? draftPayload() : null});
  }
  function enterStale(reason) {
    var ps = R.ps;
    ps.mode = 'stale';
    ps.stale_reason = reason;
    ps.tab = 'mine';
    renderAll();
  }
  function enterConflict(serverDraft, item) {
    var ps = R.ps;
    if (item) ps.item = Object.assign({}, item, {header: ps.item.header});
    var sd = serverDraft || (ps.item && ps.item.draft) || null;
    ps.mode = 'conflict';
    ps.conflict = {
      server_seq: sd ? sd.seq || 0 : 0,
      server_text: sd ? sd.text || '' : (ps.item.text || ''),
      server_printed_label: sd ? nz(sd.printed_label) : nz(ps.item.printed_label)
    };
    renderAll();
  }
  function keepMine() {
    var ps = R.ps;
    if (!ps || R.blocked) return;
    var it = ps.item;
    ps.base_attempt_id = nz(it.base_attempt_id);
    ps.base_revision_id = nz(it.base_revision_id);
    if (nz(ps.hints_version) !== nz(it.item_version)) ps.hints_outdated = true;
    ps.mode = 'edit';
    var local = touchLocal();
    var ev = {type: 'keep_mine', page_id: ps.page_id, rev: local.rev, draft: draftPayload(), finish: ps.pendingFinish};
    renderAll();
    if (ps.pendingFinish) {
      ps.pendingFinish = false;
      ev.review = reviewPayload();
      outboxAdd(ps.page_id, ev.review);
      terminal(ev);
    } else {
      pushEvent(ev);
    }
  }
  function takeNew() {
    var ps = R.ps;
    if (!ps || R.blocked) return;
    terminal({type: 'discard_draft', page_id: ps.page_id});
  }
  function chooseMine() {
    var ps = R.ps;
    if (!ps || !ps.conflict) return;
    var local = loadLocal(ps.page_id) || newLocal(ps.page_id);
    local.base_seq = ps.conflict.server_seq;
    local.in_flight = null;
    saveLocal(local);
    ps.conflict = null;
    var it = ps.item;
    var sameBase = nz(ps.base_attempt_id) === nz(it.base_attempt_id) && nz(ps.base_revision_id) === nz(it.base_revision_id);
    ps.mode = sameBase ? 'edit' : 'stale';
    ps.stale_reason = 'changed';
    touchLocal();
    renderAll();
    if (ps.mode === 'edit') queueDraft();
  }
  function chooseSaved() {
    var ps = R.ps;
    if (!ps || !ps.conflict) return;
    var seq = ps.conflict.server_seq;
    var st = CORE.restoreState(ps.item, null, ps.item.draft || null);
    ps.conflict = null;
    ps.mode = st.mode; ps.stale_reason = st.stale_reason; ps.hints = st.hints; ps.choices = st.choices || {};
    ps.hints_version = st.hints_version; ps.hints_outdated = st.hints_outdated;
    ps.base_attempt_id = st.base_attempt_id; ps.base_revision_id = st.base_revision_id;
    ps.text = st.text;
    R.ui.ta.value = st.text;
    R.ui.label.value = st.printed_label || '';
    var local = loadLocal(ps.page_id) || newLocal(ps.page_id);
    local.base_seq = seq;
    local.rev = (local.rev || 0) + 1;
    local.sent_rev = local.rev;
    local.in_flight = null;
    fillLocal(local);
    saveLocal(local);
    renderAll();
  }

  // ---- scan: zoom, pan, rotate, region ----
  function loadImage() {
    var src = R.images && R.images.page;
    show(R.ui.noImg, !src);
    show(R.ui.img, !!src);
    if (src && R.ui.img.getAttribute('src') !== src) R.ui.img.src = src;
    else if (src) fitImage();
  }
  function applyView() {
    var v = R.view;
    R.ui.img.style.transform = 'translate(' + v.tx + 'px,' + v.ty + 'px) rotate(' + v.rot + 'deg) scale(' + v.s + ')';
  }
  function fitImage() {
    var img = R.ui.img, pane = R.ui.scan;
    var w = img.naturalWidth, h = img.naturalHeight;
    if (!w || !h) return;
    img.style.width = w + 'px';
    img.style.height = h + 'px';
    var W = pane.clientWidth, H = pane.clientHeight;
    var turned = R.view.rot % 180 !== 0;
    var bw = turned ? h : w, bh = turned ? w : h;
    R.view.s = Math.min(W / bw, H / bh) * 0.98;
    R.view.tx = (W - w) / 2;
    R.view.ty = (H - h) / 2;
    applyView();
  }
  function zoomAt(factor, cx, cy) {
    var img = R.ui.img, pane = R.ui.scan, v = R.view;
    var w = img.naturalWidth, h = img.naturalHeight;
    if (!w || !h) return;
    var s = Math.max(0.05, Math.min(20, v.s * factor));
    var f = s / v.s;
    if (cx == null) { cx = pane.clientWidth / 2; cy = pane.clientHeight / 2; }
    var Cx = v.tx + w / 2, Cy = v.ty + h / 2;
    v.tx += (cx - Cx) * (1 - f);
    v.ty += (cy - Cy) * (1 - f);
    v.s = s;
    applyView();
  }
  function onWheel(ev) {
    ev.preventDefault();
    var r = R.ui.scan.getBoundingClientRect();
    zoomAt(ev.deltaY < 0 ? 1.15 : 1 / 1.15, ev.clientX - r.left, ev.clientY - r.top);
  }
  function toggleRegion(on) {
    R.regionMode = on == null ? !R.regionMode : !!on;
    var cls = R.ui.scan.className.replace(' ocrr-region-on', '');
    R.ui.scan.className = R.regionMode ? cls + ' ocrr-region-on' : cls;
    R.ui.regionBtn.className = R.ui.regionBtn.className.replace(' ocrr-on', '') + (R.regionMode ? ' ocrr-on' : '');
  }
  function onScanDown(ev) {
    if (ev.button !== 0 || !R.ps) return;
    ev.preventDefault();
    var r = R.ui.scan.getBoundingClientRect();
    R.drag = {region: R.regionMode, x: ev.clientX, y: ev.clientY, tx: R.view.tx, ty: R.view.ty, left: r.left, top: r.top};
  }
  function onScanMove(ev) {
    var d = R.drag;
    if (!d || !R.ui) return;
    if (!d.region) {
      R.view.tx = d.tx + ev.clientX - d.x;
      R.view.ty = d.ty + ev.clientY - d.y;
      applyView();
      return;
    }
    var s = R.ui.sel.style;
    s.left = (Math.min(d.x, ev.clientX) - d.left) + 'px';
    s.top = (Math.min(d.y, ev.clientY) - d.top) + 'px';
    s.width = Math.abs(ev.clientX - d.x) + 'px';
    s.height = Math.abs(ev.clientY - d.y) + 'px';
    show(R.ui.sel, true);
  }
  // Display rotation only: map a point of the rotated picture back to page fractions.
  function unrotate(u, v, rot) {
    if (rot === 90) return {x: v, y: 1 - u};
    if (rot === 180) return {x: 1 - u, y: 1 - v};
    if (rot === 270) return {x: 1 - v, y: u};
    return {x: u, y: v};
  }
  function clamp01(x) { return Math.max(0, Math.min(1, x)); }
  function onScanUp(ev) {
    var d = R.drag;
    R.drag = null;
    if (!d || !d.region || !R.ui) return;
    show(R.ui.sel, false);
    toggleRegion(false);
    var r = R.ui.img.getBoundingClientRect();
    if (!r.width || !r.height) return;
    var p1 = unrotate((d.x - r.left) / r.width, (d.y - r.top) / r.height, R.view.rot);
    var p2 = unrotate((ev.clientX - r.left) / r.width, (ev.clientY - r.top) / r.height, R.view.rot);
    var rect = {
      x0: clamp01(Math.min(p1.x, p2.x)), y0: clamp01(Math.min(p1.y, p2.y)),
      x1: clamp01(Math.max(p1.x, p2.x)), y1: clamp01(Math.max(p1.y, p2.y))
    };
    if (rect.x1 - rect.x0 < 0.005 || rect.y1 - rect.y0 < 0.003 || !R.ps || R.ps.mode !== 'edit') return;
    setStatus(T.recognizing);
    pushEvent({type: 'region', page_id: R.ps.page_id, rect: rect});
  }

  // ---- keyboard: commands use Alt or Ctrl, plain keys type into the editor ----
  function onKey(ev) {
    if (!R.root) return;
    // OpenWebUI's own shortcuts must not fire under the overlay.
    ev.stopPropagation();
    var code = ev.code;
    var alt = ev.altKey && !ev.ctrlKey && !ev.metaKey;
    var ctrl = (ev.ctrlKey || ev.metaKey) && !ev.altKey;
    var handled = true;
    if (ev.key === 'Escape') requestStop();
    else if (ctrl && ev.key === 'Enter') doDone();
    else if (alt && code === 'ArrowDown') navHint(1);
    else if (alt && code === 'ArrowUp') navHint(-1);
    else if (alt && code === 'Digit1') applyVariant('a');
    else if (alt && code === 'Digit2') applyVariant('b');
    else if (alt && code === 'KeyS') doLater();
    else if (alt && code === 'KeyR') toggleRegion();
    else if (alt && code === 'KeyP') doBack();
    else handled = false;
    if (handled) ev.preventDefault();
  }

  // ---- one execute call ----
  R.handle = function (call) {
    R.lastCallAt = Date.now();
    if (R.waiter) {
      // Another review took over this tab; release the old pipe call.
      var old = R.waiter;
      R.waiter = null;
      old({error: 'superseded'});
    }
    if (call.op === 'collect_outbox') {
      R.events = [];
      return Promise.resolve({type: 'outbox', entries: readOutbox()});
    }
    applyOutboxAck(call.outbox_ack || []);
    (call.acks || []).forEach(applyAck);
    if (call.op === 'close') {
      destroy();
      return Promise.resolve({type: 'closed'});
    }
    ensureRoot();
    if (call.op === 'show') showPage(call);
    else unblock();
    if (call.notice) notice(call.notice);
    return new Promise(function (resolve) {
      R.waiter = resolve;
      deliver();
    });
  };
  R.destroy = destroy;
  return R;
}
"""

REVIEW_ENTRY_JS = r"""
var R = window.__ocrReview;
if (!R || R.version !== OCR_WINDOW_VERSION) {
  if (R && typeof R.destroy === 'function') { try { R.destroy(); } catch (e) { /* old window */ } }
  R = window.__ocrReview = ocrCreateReview(OCR_CORE);
}
return await R.handle(CALL);
"""

REVIEW_WINDOW_JS = REVIEW_CORE_JS + REVIEW_UI_JS + REVIEW_ENTRY_JS


def window_code(payload: Mapping[str, Any]) -> str:
    """Build the code of one ``execute`` call: the payload as a JSON literal plus the static JS.

    OpenWebUI runs it as the body of an async function; its return value comes back to the pipe.

    :param payload: Data for this call; inserted only as JSON, never as code.
    :return: JavaScript source.
    """
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    data = data.replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    return f"const CALL = {data};\n{REVIEW_WINDOW_JS}"


# ---- shared-with-prototype: end ----

UPLOAD_FIELD = "files"
HTTP_TIMEOUT_S = 60.0
LONG_TIMEOUT_S = 900.0


def _error_detail(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return f"http_{response.status_code}"
    detail = body.get("detail") if isinstance(body, Mapping) else None
    return detail if isinstance(detail, str) else f"http_{response.status_code}"


def filename_from_disposition(header: str) -> str | None:
    """Extract the file name from a ``Content-Disposition`` header.

    :param header: Header value.
    :return: Decoded file name or ``None``.
    """
    match = re.search(r"filename\*\s*=\s*(?:UTF-8|utf-8)''([^;]+)", header)
    if match:
        return unquote(match.group(1).strip())
    match = re.search(r'filename\s*=\s*"([^"]+)"', header) or re.search(
        r"filename\s*=\s*([^;]+)", header
    )
    return match.group(1).strip() if match else None


def _path_id(value: str) -> str:
    return quote(value, safe="")


class ServiceClient:
    """Async client of the OCR service REST API (section 11)."""

    def __init__(self, http: httpx.AsyncClient) -> None:
        """Wrap an HTTP client that already carries the base URL and ``X-API-Key``.

        :param http: The HTTP client.
        """
        self._http = http

    async def _send(
        self, method: str, path: str, *, timeout: float | None = None, **kwargs: Any
    ) -> httpx.Response:
        if timeout is not None:
            kwargs["timeout"] = timeout
        try:
            response = await self._http.request(method, path, **kwargs)
        except httpx.HTTPError as err:
            raise ServiceUnavailableError(type(err).__name__) from err
        if response.status_code >= 400:
            raise ServiceError(response.status_code, _error_detail(response))
        return response

    async def _json(self, method: str, path: str, **kwargs: Any) -> Any:
        response = await self._send(method, path, **kwargs)
        if response.status_code == 204 or not response.content:
            return None
        try:
            return response.json()
        except ValueError as err:
            raise ServiceError(response.status_code, "invalid_json") from err

    async def create_document(
        self, uploads: Sequence[Upload], *, ingest: bool, order: Sequence[str] | None
    ) -> dict[str, Any]:
        """``POST /api/documents``."""
        files = [
            (UPLOAD_FIELD, (u.name, u.data, u.content_type or "application/octet-stream"))
            for u in uploads
        ]
        data = {"ingest": "true" if ingest else "false", "priority": "0"}
        if order:
            data["order"] = json.dumps(list(order), ensure_ascii=False)
        result: dict[str, Any] = await self._json(
            "POST", "/api/documents", files=files, data=data, timeout=LONG_TIMEOUT_S
        )
        return result

    async def list_documents(self) -> list[dict[str, Any]]:
        """``GET /api/documents``."""
        return list(await self._json("GET", "/api/documents") or [])

    async def get_document(self, document_id: str) -> dict[str, Any]:
        """``GET /api/documents/{id}``."""
        return dict(await self._json("GET", f"/api/documents/{_path_id(document_id)}"))

    async def request_draft(self, document_id: str) -> dict[str, Any]:
        """``POST /api/documents/{id}/draft``."""
        return dict(await self._json("POST", f"/api/documents/{_path_id(document_id)}/draft"))

    async def request_ingest(self, document_id: str) -> dict[str, Any]:
        """``POST /api/documents/{id}/ingest``."""
        result = await self._json("POST", f"/api/documents/{_path_id(document_id)}/ingest")
        return dict(result or {})

    async def download(
        self, document_id: str, kind: str, *, version: int | None = None, draft: bool = False
    ) -> tuple[str, bytes, str]:
        """``GET /api/documents/{id}/files/{kind}``.

        :return: File name, bytes and MIME type.
        """
        params: dict[str, str] = {}
        if draft:
            params["draft"] = "true"
        elif version is not None:
            params["version"] = str(version)
        response = await self._send(
            "GET",
            f"/api/documents/{_path_id(document_id)}/files/{kind}",
            params=params,
            timeout=LONG_TIMEOUT_S,
        )
        name = filename_from_disposition(response.headers.get("content-disposition", ""))
        content_type = response.headers.get("content-type", "application/octet-stream")
        return name or f"{document_id}.{kind}", response.content, content_type.split(";")[0]

    async def get_build(self, build_id: int) -> dict[str, Any]:
        """``GET /api/builds/{id}``."""
        return dict(await self._json("GET", f"/api/builds/{int(build_id)}"))

    async def deliveries(self) -> list[dict[str, Any]]:
        """``GET /api/deliveries``."""
        return list(await self._json("GET", "/api/deliveries") or [])

    async def mark_delivered(self, document_id: str, version: int) -> None:
        """``POST /api/documents/{id}/delivered``."""
        await self._send(
            "POST",
            f"/api/documents/{_path_id(document_id)}/delivered",
            json={"version": version},
        )

    async def summary(self) -> dict[str, Any]:
        """``GET /api/review/summary``."""
        return dict(await self._json("GET", "/api/review/summary"))

    async def queue(self, limit: int, exclude: Iterable[int]) -> list[dict[str, Any]]:
        """``GET /api/review/queue``."""
        params = {"limit": str(limit)}
        excluded = ",".join(str(int(page_id)) for page_id in exclude)
        if excluded:
            params["exclude"] = excluded
        return list(await self._json("GET", "/api/review/queue", params=params) or [])

    async def review_item(self, page_id: int) -> dict[str, Any]:
        """``GET /api/review/items/{page_id}``."""
        return dict(await self._json("GET", f"/api/review/items/{int(page_id)}"))

    async def page_item(self, document_id: str, page_number: int) -> dict[str, Any]:
        """``GET /api/documents/{id}/pages/{n}/item``."""
        path = f"/api/documents/{_path_id(document_id)}/pages/{int(page_number)}/item"
        return dict(await self._json("GET", path))

    async def page_image(
        self, page_id: int, *, long_side: int, clip: Sequence[float] | None = None
    ) -> bytes:
        """``GET /api/pages/{id}/image`` as JPEG."""
        params = {"long_side": str(long_side), "format": "jpeg"}
        if clip:
            params["clip"] = ",".join(format(float(v), ".6f") for v in clip)
        response = await self._send("GET", f"/api/pages/{int(page_id)}/image", params=params)
        return response.content

    async def put_draft(self, page_id: int, draft: Mapping[str, Any]) -> dict[str, Any]:
        """``PUT /api/pages/{id}/draft``."""
        return dict(await self._json("PUT", f"/api/pages/{int(page_id)}/draft", json=dict(draft)))

    async def delete_draft(self, page_id: int, draft_seq: int) -> None:
        """``DELETE /api/pages/{id}/draft``."""
        await self._send(
            "DELETE", f"/api/pages/{int(page_id)}/draft", params={"draft_seq": str(draft_seq)}
        )

    async def post_review(self, page_id: int, review: Mapping[str, Any]) -> dict[str, Any]:
        """``POST /api/pages/{id}/review``."""
        return dict(
            await self._json("POST", f"/api/pages/{int(page_id)}/review", json=dict(review))
        )

    async def ocr_region(self, page_id: int, rect: Mapping[str, Any]) -> dict[str, Any]:
        """``POST /api/pages/{id}/ocr-region``."""
        body = {key: float(rect[key]) for key in ("x0", "y0", "x1", "y1")}
        return dict(
            await self._json(
                "POST", f"/api/pages/{int(page_id)}/ocr-region", json=body, timeout=LONG_TIMEOUT_S
            )
        )

    async def retry(self, page_id: int) -> dict[str, Any]:
        """``POST /api/pages/{id}/retry``."""
        return dict(await self._json("POST", f"/api/pages/{int(page_id)}/retry") or {})


def match_documents(documents: Iterable[Mapping[str, Any]], query: str) -> list[dict[str, Any]]:
    """Find documents by the beginning of the title (section 14).

    An exact title match wins over prefix matches, so a title that is the prefix of another
    stays reachable.

    :param documents: Documents from ``GET /api/documents``.
    :param query: Normalized name from the command.
    :return: Matching documents.
    """
    wanted = normalize_text(query)
    if not wanted:
        return []
    docs = [dict(d) for d in documents]
    exact = [d for d in docs if normalize_text(str(d.get("title", ""))) == wanted]
    if exact:
        return exact
    return [d for d in docs if normalize_text(str(d.get("title", ""))).startswith(wanted)]


def _user_ingest(user: Mapping[str, Any]) -> bool:
    valves = user.get("valves")
    if valves is None:
        return False
    if isinstance(valves, Mapping):
        return bool(valves.get("ingest_new_documents", False))
    return bool(getattr(valves, "ingest_new_documents", False))


class _Commands:
    """Handlers of chat commands for one pipe call."""

    def __init__(self, ctx: PipeContext, client: ServiceClient, user: Mapping[str, Any]) -> None:
        self._ctx = ctx
        self._client = client
        self._user = user

    async def run(self, command: Command, files: Sequence[Mapping[str, Any]] | None) -> None:
        if files and command.kind == "help":
            await self.queue_files(files)
            return
        handlers: dict[str, Callable[[Command], Awaitable[None]]] = {
            "review": self.review,
            "queue": self.show_queue,
            "files": self.send_files,
            "draft": self.draft,
            "fix": self.fix,
            "ingest": self.ingest,
        }
        handler = handlers.get(command.kind)
        if handler is None:
            self._ctx.out.write(msg("help"))
            return
        await handler(command)

    async def _resolve(self, query: str) -> dict[str, Any] | None:
        matches = match_documents(await self._client.list_documents(), query)
        if not matches:
            self._ctx.out.write(msg("not_found", query=query))
            return None
        if len(matches) > 1:
            items = "\n".join(
                msg("ambiguous_item", title=d.get("title", ""), status=status_name(d.get("status")))
                for d in matches
            )
            self._ctx.out.write(msg("ambiguous", items=items))
            return None
        return matches[0]

    async def queue_files(self, entries: Sequence[Mapping[str, Any]]) -> None:
        """Queue the attached files: each PDF or TIFF alone, all pictures as one document."""
        out = self._ctx.out
        files = self._ctx.files
        singles: list[Upload] = []
        images: list[Upload] = []
        for entry in entries:
            name = str(entry.get("name") or (entry.get("file") or {}).get("filename") or "")
            if files is None:
                out.write(msg("read_failed", name=name))
                continue
            try:
                upload = await files.read_upload(entry)
            except OwuiFilesError:
                logger.warning("cannot read attachment", exc_info=True)
                out.write(msg("read_failed", name=name))
                continue
            kind = upload_kind(upload.name, upload.content_type)
            if kind is None:
                out.write(msg("unsupported_file", name=upload.name))
            elif kind == "image":
                images.append(upload)
            else:
                singles.append(upload)
        groups: list[list[Upload]] = [[u] for u in singles]
        if images:
            # Index prefixes keep the order unambiguous even for equal file names.
            groups.append(
                [
                    Upload(f"{i:04d}_{u.name}", u.data, u.content_type)
                    for i, u in enumerate(images, start=1)
                ]
            )
        for group in groups:
            await self._queue_group(group)

    async def _queue_group(self, group: Sequence[Upload]) -> None:
        out = self._ctx.out
        is_image_set = upload_kind(group[0].name, group[0].content_type) == "image"
        order = [u.name for u in group] if is_image_set else None
        try:
            created = await self._client.create_document(
                group, ingest=_user_ingest(self._user), order=order
            )
        except ServiceUnavailableError:
            raise
        except ServiceError as err:
            out.write(msg("upload_failed", name=group[0].name, error=err.detail))
            return
        title = str(created.get("title", group[0].name))
        document_id = str(created["id"])
        if created.get("duplicate"):
            out.write(msg("duplicate", title=title, status=status_name(created.get("status"))))
            if created.get("status") == "done":
                # The same file again: the files are already built, hand them over.
                await self._deliver_document(document_id, title)
            return
        pages = count_phrase(int(created.get("page_count") or 0), "forms_pages")
        eta: float | None = None
        try:
            eta = (await self._client.get_document(document_id)).get("eta_sec")
        except ServiceError:
            logger.info("no ETA for %s", document_id)
        if eta:
            out.write(msg("queued", title=title, pages=pages, eta=format_clock(float(eta))))
        else:
            out.write(msg("queued_no_eta", title=title, pages=pages))
        if self._ctx.upload_wait_s > 0:
            status = await wait_document(self._ctx, document_id, title, self._ctx.upload_wait_s)
            if status == "done":
                await self._deliver_document(document_id, title)

    async def review(self, command: Command) -> None:
        await ReviewSession(self._ctx).run_queue()

    async def show_queue(self, command: Command) -> None:
        summary = await self._client.summary()
        self._ctx.out.write(format_summary(summary, with_logprobs=True))
        await deliver_ready(self._ctx)

    async def send_files(self, command: Command) -> None:
        doc = await self._resolve(command.query)
        if doc is None:
            return
        await self._deliver_document(str(doc["id"]), str(doc.get("title", "")))

    async def _deliver_document(self, document_id: str, title: str) -> None:
        """Attach the latest final build of a document to the chat and confirm the delivery."""
        details = await self._client.get_document(document_id)
        versions = [
            int(b["version"])
            for b in details.get("builds") or []
            if b.get("kind", "final") == "final" and b.get("status") == "done"
        ]
        if not versions:
            self._ctx.out.write(msg("files_none", title=title))
            return
        version = max(versions)
        if not await deliver_build(self._ctx, document_id, version=version):
            self._ctx.out.write(msg("delivery_failed", title=title))
            return
        try:
            await self._client.mark_delivered(document_id, version)
        except ServiceError:
            logger.warning("delivery confirmation of %s failed", document_id)
        self._ctx.out.write(msg("files_delivered", title=title, version=version))

    async def draft(self, command: Command) -> None:
        doc = await self._resolve(command.query)
        if doc is None:
            return
        out = self._ctx.out
        title = str(doc.get("title", ""))
        try:
            build_id = int((await self._client.request_draft(str(doc["id"])))["build_id"])
        except ServiceUnavailableError:
            raise
        except ServiceError as err:
            if err.detail == "invalid_transition":
                out.write(msg("draft_not_allowed", title=title))
                return
            raise
        deadline = time.monotonic() + max(0, self._ctx.build_wait_s)
        while True:
            build = await self._client.get_build(build_id)
            if build.get("status") == "done":
                break
            if build.get("status") == "failed":
                out.write(msg("draft_failed", title=title, error=build.get("error") or "?"))
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                out.write(msg("draft_timeout", title=title))
                return
            await out.status(msg("status_draft", title=title))
            await asyncio.sleep(min(self._ctx.poll_s, remaining))
        if await deliver_build(
            self._ctx, str(doc["id"]), version=None, kinds=("docx",), draft=True
        ):
            out.write(msg("draft_delivered", title=title))
        else:
            out.write(msg("delivery_failed", title=title))

    async def fix(self, command: Command) -> None:
        doc = await self._resolve(command.query)
        if doc is None:
            return
        title = str(doc.get("title", ""))
        try:
            item = await self._client.page_item(str(doc["id"]), command.page)
        except ServiceUnavailableError:
            raise
        except ServiceError as err:
            if err.detail == "invalid_transition":
                self._ctx.out.write(msg("fix_pending", title=title, page=command.page))
            elif err.status_code == 404:
                self._ctx.out.write(msg("fix_no_page", title=title, page=command.page))
            else:
                raise
            return
        await ReviewSession(self._ctx, single=True).run_single(item, str(doc["id"]), command.page)

    async def ingest(self, command: Command) -> None:
        doc = await self._resolve(command.query)
        if doc is None:
            return
        try:
            await self._client.request_ingest(str(doc["id"]))
        except ServiceUnavailableError:
            raise
        except ServiceError as err:
            if err.detail == "ingester_disabled":
                self._ctx.out.write(msg("ingest_disabled"))
                return
            raise
        self._ctx.out.write(msg("ingest_requested", title=doc.get("title", "")))


class Pipe:
    """OpenWebUI pipe «OCR-архив»: the only user interface of the OCR archive (section 14)."""

    class Valves(BaseModel):
        """Admin settings."""

        backend_url: str = Field(default="http://host.docker.internal:8090")
        api_key: str = Field(default="")
        allowed_user_ids: list[str] = Field(default_factory=list)
        build_wait_s: int = Field(default=900)
        autosave_idle_s: int = Field(default=10)
        # Seconds to wait after an upload for a fast document to finish; its files are then
        # attached in the same reply. A scanned document with a longer ETA is not waited for.
        upload_wait_s: int = Field(default=60)

    class UserValves(BaseModel):
        """Per-user settings."""

        ingest_new_documents: bool = Field(default=False)

    # Poll interval while waiting for builds; tests shorten it.
    build_poll_s: float = BUILD_POLL_S

    def __init__(self) -> None:
        """Create the pipe with default valves."""
        self.valves = self.Valves()

    def pipes(self) -> list[dict[str, str]]:
        """Expose the pipe as one model."""
        return [{"id": "ocr_archive", "name": MESSAGES_RU["model_name"]}]

    def _allowed_ids(self) -> set[str]:
        """IDs from the valve, tolerant to the OpenWebUI form: one comma-separated string,
        quotes and spaces around the values."""
        ids: set[str] = set()
        for value in self.valves.allowed_user_ids:
            ids.update(part.strip().strip("\"'") for part in str(value).split(","))
        ids.discard("")
        return ids

    def _is_allowed(self, user: Mapping[str, Any]) -> bool:
        user_id = str(user.get("id") or "")
        if user_id and user_id in self._allowed_ids():
            return True
        # The ID goes to the server log only, never into the chat (section 3.3).
        logger.warning("access denied: user id %r is not in allowed_user_ids", user_id)
        return False

    def _make_client(self) -> httpx.AsyncClient:
        # trust_env=False: the backend is on the Docker network; HTTP_PROXY of the OpenWebUI
        # container (set for its outbound traffic) must not route these requests to a proxy.
        return httpx.AsyncClient(
            base_url=self.valves.backend_url.rstrip("/"),
            headers={"X-API-Key": self.valves.api_key},
            timeout=HTTP_TIMEOUT_S,
            trust_env=False,
        )

    def _make_files(self, request: Any, user: Mapping[str, Any]) -> OwuiFiles | None:
        return OwuiFiles(request, user)

    async def pipe(
        self,
        body: dict[str, Any],
        __user__: dict[str, Any] | None = None,
        __files__: list[dict[str, Any]] | None = None,
        __task__: str | None = None,
        __request__: Any = None,
        __event_emitter__: EventEmitter | None = None,
        __event_call__: EventCall | None = None,
    ) -> str:
        """Route one chat message: queue files, show the queue, run a review session or a command."""
        user: Mapping[str, Any] = __user__ or {}
        if not self._is_allowed(user):
            return task_stub(__task__) if __task__ else msg("access_denied")
        if __task__:
            return task_stub(__task__)
        out = ChatOutput(__event_emitter__)
        command = parse_command(last_user_text(body))
        try:
            async with self._make_client() as http:
                client = ServiceClient(http)
                ctx = PipeContext(
                    client=client,
                    out=out,
                    build_wait_s=self.valves.build_wait_s,
                    autosave_idle_s=self.valves.autosave_idle_s,
                    event_call=__event_call__,
                    emitter=__event_emitter__,
                    files=self._make_files(__request__, user),
                    poll_s=self.build_poll_s,
                    upload_wait_s=self.valves.upload_wait_s,
                )
                await _Commands(ctx, client, user).run(command, __files__)
        except ServiceUnavailableError:
            logger.warning("OCR service is unavailable")
            out.write(msg("service_down"))
        except ServiceError as err:
            logger.warning("OCR service error %s %s", err.status_code, err.detail)
            out.write(msg("service_error", code=err.status_code))
        await out.close_status()
        return out.text
