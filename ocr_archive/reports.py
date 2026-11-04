"""Pilot and spot-check reports (section 15). They describe; they promise no error rate."""

from __future__ import annotations

import json
import sqlite3
import statistics
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from ocr_archive.db import repo
from ocr_archive.models import FlagCode, ReviewScope
from ocr_archive.settings import Settings


def _median(values: list[int]) -> float | None:
    return float(statistics.median(values)) if values else None


def _fmt_s(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0f} s"


def pilot_report(conn: sqlite3.Connection, document_id: str, settings: Settings) -> str:
    """Markdown report of a pilot document (section 15.2)."""
    doc = repo.get_document(conn, document_id)
    if doc is None:
        raise KeyError(document_id)
    pages = repo.pages_of_document(conn, document_id)
    revisions = repo.revisions_of_document(conn, document_id)
    last_revision: dict[int, sqlite3.Row] = {}
    for revision in revisions:
        last_revision[int(revision["page_id"])] = revision
    edited_pages = {pid for pid, r in last_revision.items() if r["action"] == "edit"}

    total = len(pages)
    flagged_pages = [p for p in pages if json.loads(p["flags_json"] or "[]")]
    per_flag: Counter[str] = Counter()
    per_flag_edited: Counter[str] = Counter()
    suppressed: Counter[str] = Counter()
    no_logprobs = 0
    with_crops = 0
    for page in pages:
        flags = json.loads(page["flags_json"] or "[]")
        for flag in flags:
            per_flag[flag] += 1
            if int(page["id"]) in edited_pages:
                per_flag_edited[flag] += 1
        signals = json.loads(page["signals_json"] or "{}")
        suppressed.update(signals.get("suppressed_flags", []))
        if signals.get("no_logprobs"):
            no_logprobs += 1
        review = json.loads(page["review_json"] or "{}")
        if any(h.get("crop") for h in review.get("hints", [])):
            with_crops += 1

    samples = repo.samples_of_document(conn, document_id)
    done_samples = [s for s in samples if s["done_at"]]
    vlm_samples = [s for s in done_samples if s["track"] == "vlm"]
    fast_samples = [s for s in done_samples if s["track"] == "pdf_text"]

    durations = {
        scope: [
            int(r["duration_s"])
            for r in revisions
            if r["scope"] == scope and r["duration_s"] is not None
        ]
        for scope in (ReviewScope.HINTS, ReviewScope.FULL)
    }
    median_hints = _median(durations[ReviewScope.HINTS])
    share = len(flagged_pages) / total if total else 0.0
    forecast = None
    if median_hints is not None:
        forecast = share * total * median_hints

    lines = [
        f"# Pilot report: {doc['title']}",
        "",
        f"Document `{document_id}`, {total} pages, generated "
        f"{datetime.now(UTC).isoformat(timespec='seconds')}.",
        "This report describes how the system worked on this document. "
        "It is not a statistical guarantee of an error rate.",
        "",
        "## Flags",
        "",
        f"- Pages with flags: {len(flagged_pages)} of {total} ({share:.1%}); "
        f"guideline {settings.review.load_guideline:.0%}.",
        f"- Pages without logprobs: {no_logprobs}.",
        f"- Flagged pages with line crops: {with_crops} of {len(flagged_pages)}.",
        "",
        "| Rule | Pages | Corrected by the reviewer |",
        "|---|---|---|",
    ]
    for flag in FlagCode:
        if per_flag[flag.value] or flag in settings.decide.flags:
            lines.append(
                f"| {flag.value} | {per_flag[flag.value]} | {per_flag_edited[flag.value]} |"
            )
    disabled = [f for f in FlagCode if f not in settings.decide.flags]
    if disabled:
        lines += ["", "Switched-off rules (computed, not queued):"]
        lines += [f"- {f.value}: {suppressed[f.value]} pages not reviewed" for f in disabled]
    lines += [
        "",
        "## Spot checks",
        "",
        f"- Pages without flags read in full: {len(vlm_samples)}, corrected "
        f"{sum(1 for s in vlm_samples if int(s['page_id']) in edited_pages)}.",
        f"- Fast-track pages read in full: {len(fast_samples)}, corrected "
        f"{sum(1 for s in fast_samples if int(s['page_id']) in edited_pages)}.",
        "",
        "## Review time",
        "",
    ]
    for scope, values in durations.items():
        lines.append(
            f"- {scope.value}: {len(values)} pages, median {_fmt_s(_median(values))}, "
            f"total {_fmt_s(float(sum(values)) if values else None)}."
        )
    lines.append(f"- Forecast for a document of this size: {_fmt_s(forecast)} of review.")
    return "\n".join(lines) + "\n"


def audit_report(conn: sqlite3.Connection, document_id: str | None) -> str:
    """Spot-check summary: corrected pages and their signals, without intervals (15.4)."""
    samples = repo.samples_of_document(conn, document_id)
    done = [s for s in samples if s["done_at"]]
    lines = [
        "# Spot-check report" + (f": {document_id}" if document_id else ""),
        "",
        f"Sampled pages: {len(samples)}, reviewed: {len(done)}.",
        "",
        "| Document | Page | Track | Corrected | Signals |",
        "|---|---|---|---|---|",
    ]
    corrected = 0
    for sample in done:
        revision = (
            repo.get_revision(conn, int(sample["current_revision_id"]))
            if sample["current_revision_id"] is not None
            else None
        )
        edited = revision is not None and revision["action"] == "edit"
        corrected += int(edited)
        signals = json.loads(sample["signals_json"] or "{}")
        brief = ", ".join(
            f"{k}={v}"
            for k, v in sorted(signals.items())
            if k in ("diff_words", "digit_mismatch", "low_digits", "low_letters", "line_gap")
        )
        lines.append(
            f"| {sample['document_id']} | {int(sample['page_index']) + 1} | {sample['track']} | "
            f"{'yes' if edited else 'no'} | {brief} |"
        )
    lines += ["", f"Corrected: {corrected} of {len(done)} reviewed sample pages."]
    return "\n".join(lines) + "\n"


def write_report(settings: Settings, prefix: str, content: str) -> Path:
    """Write ``reports/<prefix>-<date>.md`` and return its path."""
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y-%m-%d")
    path = settings.reports_dir / f"{prefix}-{stamp}.md"
    path.write_text(content, encoding="utf-8")
    return path
