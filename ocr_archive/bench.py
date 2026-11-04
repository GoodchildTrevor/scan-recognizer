"""Comparison of recognition profiles on the gold pages (section 15.1)."""

from __future__ import annotations

import logging
import sqlite3
import statistics
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from rapidfuzz.distance import Levenshtein

from ocr_archive import pdfpages
from ocr_archive.db import repo
from ocr_archive.exceptions import VlmRequestError, VlmTransportError
from ocr_archive.settings import Settings
from ocr_archive.textnorm import counted_lines, numbers, soft_normalize
from ocr_archive.vlm.client import VlmClient

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PageScore:
    page_id: int
    cer: float
    numbers_equal: bool
    line_recall: float
    seconds: float


def cer(reference: str, hypothesis: str) -> float:
    """Character error rate on softly normalized texts."""
    ref = soft_normalize(reference)
    hyp = soft_normalize(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return Levenshtein.distance(ref, hyp) / len(ref)


def line_recall(reference: str, hypothesis: str) -> float:
    """Share of reference lines found in the hypothesis (soft-normalized, order ignored)."""
    ref_lines = [soft_normalize(line) for line in counted_lines(reference)]
    if not ref_lines:
        return 1.0
    hyp_lines = {soft_normalize(line) for line in counted_lines(hypothesis)}
    return sum(1 for line in ref_lines if line in hyp_lines) / len(ref_lines)


def score_page(page_id: int, gold: str, text: str, seconds: float) -> PageScore:
    return PageScore(
        page_id, cer(gold, text), numbers(gold) == numbers(text), line_recall(gold, text), seconds
    )


async def run_bench(conn: sqlite3.Connection, settings: Settings, profiles: list[str]) -> str:
    """Transcribe every gold page with each profile and return a Markdown report."""
    client = VlmClient(settings.vlm, settings.llama_api_key.get_secret_value())
    gold = repo.gold_pages(conn)
    sections = [
        f"# Bench {datetime.now(UTC).isoformat(timespec='seconds')}",
        "",
        f"Gold pages: {len(gold)}. The gold set compares settings; it does not measure the "
        "error rate of the archive.",
        "",
        "| Profile | Long side | CER | Numbers equal | Line recall | s/page median | s/page p95 |",
        "|---|---|---|---|---|---|---|",
    ]
    try:
        for name in profiles:
            profile = settings.profiles[name]
            scores: list[PageScore] = []
            for row in gold:
                pdf_path = settings.data_dir / str(row["source_path"])
                png = pdfpages.render_page(
                    pdf_path, int(row["page_index"]), profile.image_long_side
                )
                try:
                    result = await client.transcribe(
                        png, profile, user_text=f"Страница {int(row['page_index']) + 1}."
                    )
                except (VlmTransportError, VlmRequestError) as err:
                    logger.warning("bench page %d failed: %s", row["page_id"], err)
                    continue
                scores.append(
                    score_page(
                        int(row["page_id"]),
                        str(row["text"]),
                        result.text,
                        result.duration_ms / 1000,
                    )
                )
            if not scores:
                sections.append(f"| {name} | {profile.image_long_side} | - | - | - | - | - |")
                continue
            secs = sorted(s.seconds for s in scores)
            p95 = secs[min(len(secs) - 1, round(0.95 * (len(secs) - 1)))]
            sections.append(
                f"| {name} | {profile.image_long_side} | "
                f"{statistics.mean(s.cer for s in scores):.4f} | "
                f"{sum(s.numbers_equal for s in scores)}/{len(scores)} | "
                f"{statistics.mean(s.line_recall for s in scores):.3f} | "
                f"{statistics.median(secs):.1f} | {p95:.1f} |"
            )
    finally:
        await client.aclose()
    return "\n".join(sections) + "\n"


def gold_file_name(document_id: str, page_index: int) -> str:
    return f"{document_id}_p{page_index + 1:04d}.txt"


def export_gold(
    conn: sqlite3.Connection, settings: Settings, document_id: str, page_numbers: list[int]
) -> list[Path]:
    """Write the first-pass text of the pages for proofreading into ``gold/``."""
    settings.gold_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for number in page_numbers:
        page = repo.get_page_by_index(conn, document_id, number - 1)
        if page is None:
            raise KeyError(number)
        attempts = conn.execute(
            "SELECT text FROM attempts WHERE page_id = ? AND pass_no IN (0, 1) "
            "AND role IN ('full', 'pdf_text') ORDER BY generation DESC, id DESC LIMIT 1",
            (page["id"],),
        ).fetchone()
        path = settings.gold_dir / gold_file_name(document_id, int(page["page_index"]))
        path.write_text(str(attempts["text"]) if attempts else "", encoding="utf-8")
        written.append(path)
    return written


def import_gold(conn: sqlite3.Connection, settings: Settings) -> int:
    """Load proofread files ``<document_id>_pNNNN.txt`` from ``gold/``."""
    count = 0
    for path in sorted(settings.gold_dir.glob("*_p*.txt")):
        document_id, _, number = path.stem.rpartition("_p")
        if not number.isdigit():
            continue
        page = repo.get_page_by_index(conn, document_id, int(number) - 1)
        if page is None:
            logger.warning("gold file %s does not match a page", path.name)
            continue
        repo.upsert_gold(conn, int(page["id"]), path.read_text(encoding="utf-8"))
        count += 1
    return count
