"""``ocr page``: run both passes on one page of a file and print the decision, storing nothing."""

from __future__ import annotations

import json
from pathlib import Path

from ocr_archive import pdfpages
from ocr_archive.checks import merge_parts
from ocr_archive.lines import detect_lines, find_split_gap
from ocr_archive.models import PageAttempts, PageTrack, VlmResult
from ocr_archive.pipeline import decide
from ocr_archive.settings import Settings
from ocr_archive.textnorm import has_tables
from ocr_archive.versions import rules_version
from ocr_archive.vlm import FRAGMENT_PROMPT
from ocr_archive.vlm.client import VlmClient
from ocr_archive.vlm.logprobs import low_confidence_spans


async def debug_page(settings: Settings, pdf_path: Path, page_index: int) -> str:
    """Return a human-readable report of one page; page text is printed only here, on request."""
    import pymupdf

    client = VlmClient(settings.vlm, settings.llama_api_key.get_secret_value())
    try:
        lines = detect_lines(pdf_path, page_index)
        p1, p2 = settings.profiles["pass1"], settings.profiles["pass2"]
        user = f"Страница {page_index + 1}."
        first = await client.transcribe(
            pdfpages.render_page(pdf_path, page_index, p1.image_long_side), p1, user_text=user
        )
        second: VlmResult
        gap = None
        if (
            p2.split == "auto"
            and lines.reliable
            and len(lines.boxes) >= settings.checks.split_min_lines
            and not has_tables(first.text)
        ):
            gap = find_split_gap(pdf_path, page_index, lines, settings.checks)
        if gap is not None:
            with pymupdf.open(pdf_path) as doc:
                rect = doc[page_index].rect
            rects = [
                pymupdf.Rect(rect.x0, rect.y0, rect.x1, gap),
                pymupdf.Rect(rect.x0, gap, rect.x1, rect.y1),
            ]
            parts = []
            for number, clip in enumerate(rects, start=1):
                png = pdfpages.render_page(pdf_path, page_index, p2.image_long_side, clip=clip)
                parts.append(
                    await client.transcribe(
                        png,
                        p2,
                        prompt=FRAGMENT_PROMPT,
                        user_text=f"Фрагмент страницы {page_index + 1}, часть {number} из 2.",
                    )
                )
            second = merge_parts(parts)
        else:
            second = await client.transcribe(
                pdfpages.render_page(pdf_path, page_index, p2.image_long_side), p2, user_text=user
            )
    finally:
        await client.aclose()
    page = PageAttempts(
        page_id=0,
        page_index=page_index,
        track=PageTrack.VLM,
        first=first,
        second=second,
        first_attempt_id=1,
        second_attempt_id=2,
    )
    decision = decide(page, lines, settings, rules_version(settings))
    candidate = second if decision.candidate_attempt_id == 2 else first
    spans = (
        low_confidence_spans(candidate.token_probs, settings.checks)
        if candidate.token_probs
        else []
    )
    out = [
        f"lines: {len(lines.boxes)} reliable={lines.reliable} columns={lines.columns_detected} "
        f"skew={lines.skew_deg:.2f} split_at={gap}",
        f"status: {decision.status} flags: {sorted(decision.flags)} "
        f"suppressed: {sorted(decision.suppressed_flags)}",
        "signals: " + json.dumps(decision.signals, ensure_ascii=False),
        "uncertain spans: "
        + ", ".join(
            f"[{s.start}:{s.end}] {candidate.text[s.start : s.end]!r} p={s.min_p:.2f}"
            for s in spans
        ),
        "--- candidate text ---",
        candidate.text,
    ]
    return "\n".join(out)
