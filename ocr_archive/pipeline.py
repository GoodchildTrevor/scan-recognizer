"""Processing of one page: two passes, decision, review item; re-deciding and reprocessing.

Sections 6.1–6.11. The model is never queried inside a database transaction.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pymupdf

from ocr_archive import pdfpages, pdftext
from ocr_archive.build.markers import hard_markers_in
from ocr_archive.checks import attempt_defects, compare_passes, line_count_mismatch, merge_parts
from ocr_archive.db import repo
from ocr_archive.db.connection import connect, immediate
from ocr_archive.exceptions import (
    StatusTransitionError,
    VlmInvalidResponseError,
    VlmRequestError,
    VlmTimeoutError,
    VlmTransportError,
)
from ocr_archive.lines import (
    detect_lines,
    find_split_gap,
    line_detection_from_json,
    line_detection_to_json,
    lines_version,
)
from ocr_archive.models import (
    AttemptRole,
    Decision,
    Defect,
    DocumentStatus,
    FlagCode,
    LineDetection,
    PageAttempts,
    PageStatus,
    PageTrack,
    PipelineContext,
    ReviewScope,
    TextLayer,
    VlmResult,
)
from ocr_archive.review import item_to_json, review_item
from ocr_archive.settings import ProfileConfig, Settings
from ocr_archive.signals import page_signals
from ocr_archive.status import recompute_status
from ocr_archive.textnorm import has_tables
from ocr_archive.versions import PDF_TEXT_PROFILE
from ocr_archive.vlm import FRAGMENT_PROMPT

logger = logging.getLogger(__name__)

INVALID_RESPONSE_RETRIES = 2
#: Timeouts of one page before it is decided failed: generation time is reproducible, so a
#: page that needs more than vlm.timeout_s would otherwise be retried forever (section 12.3).
MAX_TIMEOUT_RETRIES = 3
FAILED_REASON = "failed"

__all__ = [
    "decide",
    "process_page",
    "recompute_status",
    "redecide_document",
    "redecide_one",
    "reprocess_document",
]


# --- decision ----------------------------------------------------------------------------------


def _signal_flags(signals: dict[str, float | bool], settings: Settings) -> set[FlagCode]:
    cfg = settings.signals
    flags: set[FlagCode] = set()
    if float(signals.get("diff_words", 0)) >= cfg.min_diff_words:
        flags.add(FlagCode.PASS_DIFF)
    if "low_digits" in signals and float(signals["low_digits"]) >= cfg.min_low_digits:
        flags.add(FlagCode.LOW_CONF_DIGIT)
    if "low_letters" in signals and float(signals["low_letters"]) >= cfg.min_low_letters:
        flags.add(FlagCode.LOW_CONF_TEXT)
    return flags


def decide(
    page: PageAttempts, lines: LineDetection, settings: Settings, rules_version: str
) -> Decision:
    """Decide a page from its two passes (section 6.9).

    A pass whose request failed permanently counts as not working; a page without any
    working pass is decided ``failed`` without a candidate.
    """
    a1, a2 = page.first, page.second
    cfg = settings.checks
    d1 = attempt_defects(a1, lines, cfg) if a1 is not None else {Defect.EMPTY}
    d2 = attempt_defects(a2, lines, cfg) if a2 is not None else {Defect.EMPTY}
    flags: set[FlagCode] = set()
    candidate: VlmResult | None
    candidate_id: int | None
    candidate_pass: int | None
    comparison = None
    if d1 and d2:
        if a2 is not None and a2.text.strip():
            candidate, candidate_id, candidate_pass = a2, page.second_attempt_id, 2
        elif a1 is not None:
            candidate, candidate_id, candidate_pass = a1, page.first_attempt_id, 1
        else:
            candidate, candidate_id, candidate_pass = a2, page.second_attempt_id, 2
        flags.add(FlagCode.BROKEN)
    elif d2:
        candidate, candidate_id, candidate_pass = a1, page.first_attempt_id, 1
        flags.add(FlagCode.UNVERIFIED)
    elif d1:
        candidate, candidate_id, candidate_pass = a2, page.second_attempt_id, 2
        flags.add(FlagCode.UNVERIFIED)
    else:
        assert_pair = (a1, a2)
        first, second = assert_pair
        if first is None or second is None:  # unreachable: both have no defects
            raise ValueError("decide() needs both passes when neither is defective")
        if settings.decide.candidate == "pass1":
            candidate, candidate_id, candidate_pass = first, page.first_attempt_id, 1
        else:
            candidate, candidate_id, candidate_pass = second, page.second_attempt_id, 2
        comparison = compare_passes(first.text, second.text)
        if not comparison.digits_equal:
            flags.add(FlagCode.DIGITS_DISAGREEMENT)
        if not comparison.tables_equal:
            flags.add(FlagCode.TABLE_MISMATCH)

    if candidate is None:
        return Decision(
            status=PageStatus.FAILED,
            candidate_attempt_id=None,
            flags=frozenset(),
            signals={},
            rules_version=rules_version,
        )
    signals = page_signals(candidate, comparison, lines, checks=cfg)
    if comparison is not None:
        flags |= _signal_flags(signals, settings)
    if line_count_mismatch(candidate.text, lines, cfg):
        flags.add(FlagCode.LINE_COUNT_MISMATCH)
    if hard_markers_in(candidate.text, settings.decide.hard_markers):
        flags.add(FlagCode.MARKERS)
    enabled = set(settings.decide.flags)
    suppressed = flags - enabled
    active = flags & enabled
    return Decision(
        status=PageStatus.FLAGGED if active else PageStatus.OK,
        candidate_attempt_id=candidate_id,
        flags=frozenset(active),
        signals=signals,
        rules_version=rules_version,
        suppressed_flags=frozenset(suppressed),
        candidate_pass=candidate_pass,
    )


def signals_payload(
    decision: Decision, page: PageAttempts, lines: LineDetection, settings: Settings
) -> dict[str, Any]:
    """Content of ``pages.signals_json``: all counters, also below thresholds."""
    payload: dict[str, Any] = dict(decision.signals)
    payload["suppressed_flags"] = sorted(f.value for f in decision.suppressed_flags)
    payload["candidate_pass"] = decision.candidate_pass
    payload["defects"] = {
        "pass1": sorted(attempt_defects(page.first, lines, settings.checks))
        if page.first is not None
        else None,
        "pass2": sorted(attempt_defects(page.second, lines, settings.checks))
        if page.second is not None
        else None,
    }
    return payload


# --- helpers -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class _PageInfo:
    page_id: int
    document_id: str
    page_index: int
    status: str
    generation: int
    track: str | None
    pdf_path: Path
    current_attempt_id: int | None


def _page_info(conn: sqlite3.Connection, page_id: int, data_dir: Path) -> _PageInfo:
    row = conn.execute(
        "SELECT p.*, d.source_path FROM pages p JOIN documents d ON d.id = p.document_id "
        "WHERE p.id = ?",
        (page_id,),
    ).fetchone()
    if row is None:
        raise KeyError(page_id)
    return _PageInfo(
        page_id=int(row["id"]),
        document_id=str(row["document_id"]),
        page_index=int(row["page_index"]),
        status=str(row["status"]),
        generation=int(row["ocr_generation"]),
        track=row["track"],
        pdf_path=data_dir / str(row["source_path"]),
        current_attempt_id=row["current_attempt_id"],
    )


def page_size(pdf_path: Path, page_index: int) -> tuple[float, float]:
    """Width and height of a page in PDF points."""
    with pymupdf.open(pdf_path) as doc:
        rect = doc[page_index].rect
        return float(rect.width), float(rect.height)


def page_lines(
    conn: sqlite3.Connection, info_page_id: int, pdf_path: Path, page_index: int
) -> LineDetection:
    """Return the stored line detection, recomputing it when the detector version changed."""
    row = repo.get_page(conn, info_page_id)
    version = lines_version()
    if row is not None and row["line_boxes_json"] and row["line_boxes_version"] == version:
        return line_detection_from_json(str(row["line_boxes_json"]))
    detection = detect_lines(pdf_path, page_index)
    with immediate(conn):
        repo.set_line_boxes(conn, info_page_id, line_detection_to_json(detection), version)
    return detection


def load_page_attempts(conn: sqlite3.Connection, page_row: sqlite3.Row) -> PageAttempts:
    """Attempts of the current generation that take part in the decision."""
    rows = repo.latest_decision_attempts(conn, int(page_row["id"]), int(page_row["ocr_generation"]))
    first = next((r for r in reversed(rows) if r["pass_no"] == 1), None)
    second_rows = [r for r in rows if r["pass_no"] == 2]
    second = next((r for r in reversed(second_rows) if r["role"] == AttemptRole.COMBINED), None)
    if second is None and second_rows:
        second = second_rows[-1]
    return PageAttempts(
        page_id=int(page_row["id"]),
        page_index=int(page_row["page_index"]),
        track=PageTrack(page_row["track"] or PageTrack.VLM),
        first=repo.attempt_to_result(first) if first is not None else None,
        second=repo.attempt_to_result(second) if second is not None else None,
        first_attempt_id=int(first["id"]) if first is not None else None,
        second_attempt_id=int(second["id"]) if second is not None else None,
    )


def _working_attempt_id(
    conn: sqlite3.Connection, page_id: int, lines: LineDetection, settings: Settings
) -> int | None:
    """Last attempt without defects, of any generation, for a failed page (section 6.9)."""
    rows = conn.execute(
        "SELECT * FROM attempts WHERE page_id = ? AND role IN ('full', 'combined') "
        "ORDER BY id DESC",
        (page_id,),
    ).fetchall()
    for row in rows:
        if not attempt_defects(repo.attempt_to_result(row), lines, settings.checks):
            return int(row["id"])
    return None


def _now() -> datetime:
    return datetime.now(UTC)


# --- one page ----------------------------------------------------------------------------------


class _PageAbortError(Exception):
    """The page changed under the worker (new generation): drop the result silently."""


async def _call_model(
    ctx: PipelineContext,
    png: bytes,
    profile: ProfileConfig,
    *,
    user_text: str,
    prompt: str | None = None,
) -> VlmResult:
    for attempt in range(INVALID_RESPONSE_RETRIES + 1):
        try:
            return await ctx.vlm.transcribe(png, profile, user_text=user_text, prompt=prompt)
        except VlmInvalidResponseError:
            if attempt == INVALID_RESPONSE_RETRIES:
                raise
            logger.warning("invalid model response, repeating (%d)", attempt + 1)
    raise AssertionError("unreachable")


def _save_attempt(
    ctx: PipelineContext,
    info: _PageInfo,
    *,
    profile_name: str,
    pass_no: int,
    role: AttemptRole,
    region: dict[str, Any] | None,
    result: VlmResult,
) -> int:
    conn = connect(ctx.db_path)
    try:
        with immediate(conn):
            attempt_id = repo.insert_attempt(
                conn,
                page_id=info.page_id,
                profile_id=ctx.profile_ids[profile_name],
                generation=info.generation,
                pass_no=pass_no,
                role=role,
                region=region,
                result=result,
            )
            repo.clear_retry(conn, info.page_id)
    finally:
        conn.close()
    if attempt_id is None:
        raise _PageAbortError
    return attempt_id


async def _render(
    ctx: PipelineContext, info: _PageInfo, long_side: int, clip: pymupdf.Rect | None = None
) -> bytes:
    settings = ctx.settings
    return await asyncio.to_thread(
        pdfpages.render_cached,
        settings.cache_dir,
        info.page_id,
        info.pdf_path,
        info.page_index,
        long_side,
        clip=clip,
        max_gb=settings.render_cache.max_gb,
    )


async def _first_pass(
    ctx: PipelineContext, info: _PageInfo, saved: list[sqlite3.Row]
) -> tuple[VlmResult, int]:
    reuse = next((r for r in saved if r["role"] == AttemptRole.FULL), None)
    if reuse is not None:
        return repo.attempt_to_result(reuse), int(reuse["id"])
    profile = ctx.settings.profiles["pass1"]
    png = await _render(ctx, info, profile.image_long_side)
    result = await _call_model(ctx, png, profile, user_text=f"Страница {info.page_index + 1}.")
    attempt_id = _save_attempt(
        ctx,
        info,
        profile_name="pass1",
        pass_no=1,
        role=AttemptRole.FULL,
        region=None,
        result=result,
    )
    return result, attempt_id


def _split_rects(info: _PageInfo, lines: LineDetection, settings: Settings) -> list[pymupdf.Rect]:
    y = find_split_gap(info.pdf_path, info.page_index, lines, settings.checks)
    if y is None:
        return []
    with pymupdf.open(info.pdf_path) as doc:
        rect = doc[info.page_index].rect
    return [
        pymupdf.Rect(rect.x0, rect.y0, rect.x1, y),
        pymupdf.Rect(rect.x0, y, rect.x1, rect.y1),
    ]


async def _second_pass_split(
    ctx: PipelineContext, info: _PageInfo, rects: list[pymupdf.Rect], saved: list[sqlite3.Row]
) -> tuple[VlmResult, int]:
    profile = ctx.settings.profiles["pass2"]
    parts: list[VlmResult] = []
    part_ids: list[int] = []
    total = len(rects)
    saved_parts = {
        json.loads(r["region_json"])["part"]: r
        for r in saved
        if r["role"] == AttemptRole.PART and r["region_json"]
    }
    for number, rect in enumerate(rects, start=1):
        reuse = saved_parts.get(number)
        if reuse is not None:
            parts.append(repo.attempt_to_result(reuse))
            part_ids.append(int(reuse["id"]))
            continue
        png = await _render(ctx, info, profile.image_long_side, clip=rect)
        result = await _call_model(
            ctx,
            png,
            profile,
            user_text=f"Фрагмент страницы {info.page_index + 1}, часть {number} из {total}.",
            prompt=FRAGMENT_PROMPT,
        )
        region = {"part": number, "of": total, "rect": [rect.x0, rect.y0, rect.x1, rect.y1]}
        part_ids.append(
            _save_attempt(
                ctx,
                info,
                profile_name="pass2",
                pass_no=2,
                role=AttemptRole.PART,
                region=region,
                result=result,
            )
        )
        parts.append(result)
    combined = merge_parts(parts)
    combined_id = _save_attempt(
        ctx,
        info,
        profile_name="pass2",
        pass_no=2,
        role=AttemptRole.COMBINED,
        region={"parts": part_ids},
        result=combined,
    )
    return combined, combined_id


async def _second_pass(
    ctx: PipelineContext,
    info: _PageInfo,
    lines: LineDetection,
    first_text: str | None,
    saved: list[sqlite3.Row],
) -> tuple[VlmResult, int]:
    combined = next((r for r in saved if r["role"] == AttemptRole.COMBINED), None)
    if combined is not None:
        return repo.attempt_to_result(combined), int(combined["id"])
    full = next((r for r in saved if r["role"] == AttemptRole.FULL), None)
    if full is not None:
        return repo.attempt_to_result(full), int(full["id"])
    settings = ctx.settings
    profile = settings.profiles["pass2"]
    auto = profile.split == "auto"
    if (
        auto
        and lines.reliable
        and len(lines.boxes) >= settings.checks.split_min_lines
        and first_text is not None
        and not has_tables(first_text)
    ):
        rects = await asyncio.to_thread(_split_rects, info, lines, settings)
        if rects:
            return await _second_pass_split(ctx, info, rects, saved)
    if any(r["role"] == AttemptRole.PART for r in saved):
        # Parts of an interrupted relaxed split: finish the same split.
        rects = await asyncio.to_thread(_split_rects, info, lines, settings)
        if rects:
            return await _second_pass_split(ctx, info, rects, saved)
    png = await _render(ctx, info, profile.image_long_side)
    try:
        result = await _call_model(ctx, png, profile, user_text=f"Страница {info.page_index + 1}.")
    except VlmRequestError as err:
        if not (err.context_exceeded and auto and lines.reliable):
            raise
        # One retry with a relaxed split: no minimum line count and tables allowed (12.3).
        rects = await asyncio.to_thread(_split_rects, info, lines, settings)
        if not rects:
            raise
        logger.info("page %d: context exceeded, retrying pass 2 split", info.page_id)
        return await _second_pass_split(ctx, info, rects, saved)
    attempt_id = _save_attempt(
        ctx,
        info,
        profile_name="pass2",
        pass_no=2,
        role=AttemptRole.FULL,
        region=None,
        result=result,
    )
    return result, attempt_id


def _schedule_retry(ctx: PipelineContext, info: _PageInfo, *, limit: int | None = None) -> bool:
    """Schedule the next attempt with exponential backoff.

    :param limit: when set, no retry is scheduled once ``retry_count`` reached it.
    :return: whether a retry was scheduled.
    """
    conn = connect(ctx.db_path)
    try:
        with immediate(conn):
            row = repo.get_page(conn, info.page_id)
            count = int(row["retry_count"]) if row is not None else 0
            if limit is not None and count >= limit:
                return False
            delay = min(float(2**count), ctx.settings.vlm.max_backoff_s)
            delay += random.uniform(0, delay * 0.1)
            retry_at = (_now() + timedelta(seconds=delay)).isoformat(timespec="seconds")
            repo.schedule_retry(conn, info.page_id, info.generation, retry_at)
            return True
    finally:
        conn.close()


def _write_page_result(
    ctx: PipelineContext,
    info: _PageInfo,
    *,
    decision: Decision,
    payload: dict[str, Any] | None,
    review: dict[str, Any] | None,
    error: str | None = None,
) -> bool:
    conn = connect(ctx.db_path)
    try:
        with immediate(conn):
            written = repo.write_decision(
                conn,
                info.page_id,
                expected_status=PageStatus.PENDING,
                expected_generation=info.generation,
                allow_draft=True,
                status=decision.status,
                candidate_attempt_id=decision.candidate_attempt_id,
                flags=[f.value for f in decision.flags],
                signals=payload,
                review=review,
                rules_version=decision.rules_version
                if decision.status != PageStatus.FAILED
                else None,
                error=error,
            )
            if written:
                recompute_status(conn, info.document_id, ctx.rules_version)
        return written
    finally:
        conn.close()


async def _process_pdf_text(ctx: PipelineContext, info: _PageInfo) -> PageStatus:
    text = await asyncio.to_thread(pdftext.pdf_text_markdown, info.pdf_path, info.page_index)
    result = VlmResult(
        text=text,
        finish_reason="stop",
        tokens_in=None,
        tokens_out=None,
        duration_ms=0,
        token_probs=None,
    )
    attempt_id = _save_attempt(
        ctx,
        info,
        profile_name=PDF_TEXT_PROFILE,
        pass_no=0,
        role=AttemptRole.PDF_TEXT,
        region=None,
        result=result,
    )
    decision = Decision(
        status=PageStatus.OK,
        candidate_attempt_id=attempt_id,
        flags=frozenset(),
        signals={},
        rules_version=ctx.rules_version,
    )
    conn = connect(ctx.db_path)
    try:
        row = repo.get_page(conn, info.page_id)
        payload = (
            json.loads(row["signals_json"]) if row is not None and row["signals_json"] else None
        )
    finally:
        conn.close()
    written = _write_page_result(ctx, info, decision=decision, payload=payload, review=None)
    return PageStatus.OK if written else PageStatus(info.status)


def _failed_item(
    ctx: PipelineContext,
    info: _PageInfo,
    page: PageAttempts,
    decision: Decision,
    lines: LineDetection,
) -> dict[str, Any]:
    item = review_item(
        page,
        decision,
        lines,
        ReviewScope.HINTS,
        ctx.settings,
        page_size=page_size(info.pdf_path, info.page_index),
    )
    data = item_to_json(item, rules_version=ctx.rules_version)
    data["reasons"] = [FAILED_REASON]
    return data


async def process_page(page_id: int, ctx: PipelineContext) -> PageStatus:
    """Process one pending page: fast track, or two passes, decision and review item.

    :return: the resulting page status; ``pending`` when the model was unavailable and the page
        is scheduled for a later retry, or when the result was dropped (new generation).
    """
    conn = connect(ctx.db_path)
    try:
        info = _page_info(conn, page_id, ctx.settings.data_dir)
        if info.status != PageStatus.PENDING:
            return PageStatus(info.status)
        if info.track == PageTrack.PDF_TEXT:
            conn.close()
            return await _process_pdf_text(ctx, info)
        saved1 = repo.find_attempts(conn, page_id, ctx.profile_ids["pass1"], info.generation, 1)
        saved2 = repo.find_attempts(conn, page_id, ctx.profile_ids["pass2"], info.generation, 2)
        lines = await asyncio.to_thread(page_lines, conn, page_id, info.pdf_path, info.page_index)
    finally:
        conn.close()

    errors: list[str] = []
    first: VlmResult | None = None
    first_id: int | None = None
    second: VlmResult | None = None
    second_id: int | None = None
    try:
        try:
            first, first_id = await _first_pass(ctx, info, saved1)
        except VlmRequestError as err:
            errors.append(f"pass 1: {err}")
        try:
            second, second_id = await _second_pass(
                ctx, info, lines, first.text if first is not None else None, saved2
            )
        except VlmRequestError as err:
            errors.append(f"pass 2: {err}")
    except VlmTransportError as err:
        limit = MAX_TIMEOUT_RETRIES if isinstance(err, VlmTimeoutError) else None
        if _schedule_retry(ctx, info, limit=limit):
            logger.warning("page %d: model unavailable (%s), retry later", page_id, err)
            return PageStatus.PENDING
        page = PageAttempts(
            page_id=page_id,
            page_index=info.page_index,
            track=PageTrack.VLM,
            first=first,
            second=second,
            first_attempt_id=first_id,
            second_attempt_id=second_id,
        )
        return await asyncio.to_thread(
            _finish_failed, ctx, info, page, lines, f"{limit} timeouts: {err}"
        )
    except _PageAbortError:
        logger.info("page %d: generation changed during recognition, result dropped", page_id)
        return PageStatus.PENDING

    page = PageAttempts(
        page_id=page_id,
        page_index=info.page_index,
        track=PageTrack.VLM,
        first=first,
        second=second,
        first_attempt_id=first_id,
        second_attempt_id=second_id,
    )
    if errors:
        return await asyncio.to_thread(_finish_failed, ctx, info, page, lines, "; ".join(errors))
    decision = decide(page, lines, ctx.settings, ctx.rules_version)
    payload = signals_payload(decision, page, lines, ctx.settings)
    review = None
    if decision.status == PageStatus.FLAGGED:
        item = review_item(
            page,
            decision,
            lines,
            ReviewScope.HINTS,
            ctx.settings,
            page_size=page_size(info.pdf_path, info.page_index),
        )
        review = item_to_json(item, rules_version=ctx.rules_version)
    written = _write_page_result(ctx, info, decision=decision, payload=payload, review=review)
    if not written:
        logger.info("page %d: decision dropped, the page changed meanwhile", page_id)
        return PageStatus.PENDING
    return decision.status


def _finish_failed(
    ctx: PipelineContext, info: _PageInfo, page: PageAttempts, lines: LineDetection, error: str
) -> PageStatus:
    conn = connect(ctx.db_path)
    try:
        candidate_id = _working_attempt_id(conn, info.page_id, lines, ctx.settings)
        candidate_row = repo.get_attempt(conn, candidate_id) if candidate_id else None
    finally:
        conn.close()
    candidate = repo.attempt_to_result(candidate_row) if candidate_row is not None else None
    failed_page = PageAttempts(
        page_id=page.page_id,
        page_index=page.page_index,
        track=page.track,
        first=candidate,
        second=None,
        first_attempt_id=candidate_id,
        second_attempt_id=None,
    )
    decision = Decision(
        status=PageStatus.FAILED,
        candidate_attempt_id=candidate_id,
        flags=frozenset(),
        signals={},
        rules_version=ctx.rules_version,
    )
    review = _failed_item(ctx, info, failed_page, decision, lines)
    logger.warning("page %d failed permanently: %s", info.page_id, error[:300])
    written = _write_page_result(
        ctx, info, decision=decision, payload={"error": error[:1000]}, review=review, error=error
    )
    return PageStatus.FAILED if written else PageStatus.PENDING


# --- re-deciding -------------------------------------------------------------------------------


def redecide_page(
    conn: sqlite3.Connection, page_row: sqlite3.Row, ctx: PipelineContext, pdf_path: Path
) -> bool:
    """Apply sections 6.9 and 6.13 again to stored attempts, without the model.

    :return: whether the new decision was written.
    """
    page_id = int(page_row["id"])
    lines = page_lines(conn, page_id, pdf_path, int(page_row["page_index"]))
    page = load_page_attempts(conn, page_row)
    decision = decide(page, lines, ctx.settings, ctx.rules_version)
    payload = signals_payload(decision, page, lines, ctx.settings)
    review = None
    if decision.status == PageStatus.FLAGGED:
        item = review_item(
            page,
            decision,
            lines,
            ReviewScope.HINTS,
            ctx.settings,
            page_size=page_size(pdf_path, int(page_row["page_index"])),
        )
        review = item_to_json(item, rules_version=ctx.rules_version)
    with immediate(conn):
        written = repo.write_decision(
            conn,
            page_id,
            expected_status=str(page_row["status"]),
            expected_generation=int(page_row["ocr_generation"]),
            allow_draft=False,
            status=decision.status,
            candidate_attempt_id=decision.candidate_attempt_id,
            flags=[f.value for f in decision.flags],
            signals=payload,
            review=review,
            rules_version=ctx.rules_version,
        )
        recompute_status(conn, str(page_row["document_id"]), ctx.rules_version)
    return written


def redecide_one(conn: sqlite3.Connection, document_id: str, ctx: PipelineContext) -> bool:
    """Re-decide the next page of the document that needs it; return False if there is none."""
    row = repo.next_page_to_redecide(conn, document_id, ctx.rules_version)
    if row is None:
        return False
    doc = repo.get_document(conn, document_id)
    if doc is None:
        return False
    redecide_page(conn, row, ctx, ctx.settings.data_dir / str(doc["source_path"]))
    return True


def redecide_document(document_id: str, ctx: PipelineContext) -> DocumentStatus:
    """Re-decide all pages of a document that need it (section 6.11)."""
    conn = connect(ctx.db_path)
    try:
        while redecide_one(conn, document_id, ctx):
            pass
        with immediate(conn):
            return recompute_status(conn, document_id, ctx.rules_version)
    finally:
        conn.close()


def reprocess_document(
    document_id: str,
    ctx: PipelineContext,
    *,
    include_reviewed: bool = False,
    include_pdf_text: bool = False,
) -> DocumentStatus:
    """Send vlm pages back to recognition with a new generation (section 6.11).

    :param include_pdf_text: also take the text of text-layer pages from the PDF again, after
        a change of the extraction code.
    :raises StatusTransitionError: ``document_building`` while the document is being built.
    :raises KeyError: unknown document.
    """
    conn = connect(ctx.db_path)
    try:
        with immediate(conn):
            doc = repo.get_document(conn, document_id)
            if doc is None:
                raise KeyError(document_id)
            if doc["status"] == DocumentStatus.BUILDING:
                raise StatusTransitionError("document_building")
            repo.reprocess_pages(
                conn,
                document_id,
                include_reviewed=include_reviewed,
                include_pdf_text=include_pdf_text,
            )
            return recompute_status(conn, document_id, ctx.rules_version, force=True)
    finally:
        conn.close()


# --- page classification -----------------------------------------------------------------------


def classify_pages(conn: sqlite3.Connection, document_id: str, settings: Settings) -> None:
    """Determine text layer kind and track of every page of a document (section 6.1)."""
    doc = repo.get_document(conn, document_id)
    if doc is None:
        raise KeyError(document_id)
    pdf_path = settings.data_dir / str(doc["source_path"])
    for row in repo.pages_of_document(conn, document_id):
        if row["track"] is not None:
            continue
        index = int(row["page_index"])
        layer = pdfpages.text_layer_kind(pdf_path, index)
        signals: dict[str, Any] | None = None
        if layer == TextLayer.VISIBLE and settings.fast_track.enabled:
            check = pdftext.fast_track_check(pdf_path, index, layer, settings.fast_track)
            track = check.track
            if check.reason:
                signals = {"fast_track_reason": check.reason}
        else:
            track = PageTrack.VLM
        with immediate(conn):
            repo.set_page_track(
                conn, int(row["id"]), text_layer=layer, track=track, signals=signals
            )
