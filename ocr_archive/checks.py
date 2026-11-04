"""Checks of one attempt, merging of split parts and comparison of two passes (6.5, 6.7, 6.8)."""

from __future__ import annotations

import difflib

from rapidfuzz.distance import Levenshtein

from ocr_archive.models import Defect, LineDetection, PassComparison, VlmResult
from ocr_archive.settings import ChecksConfig
from ocr_archive.textnorm import counted_lines, has_tables, numbers, soft_normalize, table_shape
from ocr_archive.vlm.logprobs import merge_token_probs

LOOP_MIN_REPEATS = 4
LOOP_MIN_LINE_CHARS = 15
LOOP_TAIL_CHARS = 200
LOOP_TAIL_REPEATS = 3
EMPTY_MIN_LINES = 3


def _has_loop(text: str) -> bool:
    run_line, run_len = None, 0
    for raw in text.split("\n"):
        line = raw.strip()
        if not line:
            run_line, run_len = None, 0
            continue
        if line == run_line:
            run_len += 1
        else:
            run_line, run_len = line, 1
        if run_len >= LOOP_MIN_REPEATS and len(line) >= LOOP_MIN_LINE_CHARS:
            return True
    tail = text[-LOOP_TAIL_CHARS:]
    return len(text) >= LOOP_TAIL_CHARS and text.count(tail) >= LOOP_TAIL_REPEATS


def attempt_defects(result: VlmResult, lines: LineDetection, cfg: ChecksConfig) -> set[Defect]:
    """Return defects of an attempt: truncated, loop, empty (section 6.5)."""
    defects: set[Defect] = set()
    if result.finish_reason == "length":
        defects.add(Defect.TRUNCATED)
    if _has_loop(result.text):
        defects.add(Defect.LOOP)
    if len(result.text.strip()) < cfg.empty_min_chars and len(lines.boxes) >= EMPTY_MIN_LINES:
        defects.add(Defect.EMPTY)
    return defects


def merge_parts(parts: list[VlmResult]) -> VlmResult:
    """Join split parts into the combined attempt (section 6.7).

    Texts are joined with a newline; finish_reason is ``length`` if any part was cut; token
    offsets of later parts shift by the preceding text plus the separator.
    """
    texts = [part.text for part in parts]
    shifts: list[int] = []
    offset = 0
    for text in texts:
        shifts.append(offset)
        offset += len(text) + 1
    finish = "length" if any(p.finish_reason == "length" for p in parts) else "stop"

    def _sum(values: list[int | None]) -> int | None:
        return None if any(v is None for v in values) else sum(v or 0 for v in values)

    return VlmResult(
        text="\n".join(texts),
        finish_reason=finish,
        tokens_in=_sum([p.tokens_in for p in parts]),
        tokens_out=_sum([p.tokens_out for p in parts]),
        duration_ms=sum(p.duration_ms for p in parts),
        token_probs=merge_token_probs(
            [(p.token_probs, shift) for p, shift in zip(parts, shifts, strict=True)]
        ),
    )


def _unmatched(a: list[str], b: list[str]) -> tuple[list[str], list[str]]:
    matcher = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    rest_a: list[str] = []
    rest_b: list[str] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "equal":
            rest_a.extend(a[i1:i2])
            rest_b.extend(b[j1:j2])
    return rest_a, rest_b


def diff_words(first: str, second: str) -> int:
    """Words outside matching blocks of the two softly normalized texts, without words that
    contain digits (numbers are compared separately)."""
    words_a = soft_normalize(first).split()
    words_b = soft_normalize(second).split()
    rest_a, rest_b = _unmatched(words_a, words_b)
    return sum(1 for word in rest_a + rest_b if not any(ch.isdigit() for ch in word))


def compare_passes(first: str, second: str) -> PassComparison:
    """Compare two transcriptions of one page (section 6.8)."""
    nums_a, nums_b = numbers(first), numbers(second)
    rest_a, rest_b = _unmatched(nums_a, nums_b)
    return PassComparison(
        similarity=float(
            Levenshtein.normalized_similarity(soft_normalize(first), soft_normalize(second))
        ),
        diff_words=diff_words(first, second),
        digits_equal=nums_a == nums_b,
        digit_mismatch=len(rest_a) + len(rest_b),
        tables_equal=table_shape(first) == table_shape(second),
    )


def line_gap(text: str, lines: LineDetection) -> int | None:
    """Difference between text lines and detected lines, or None without a reliable detector."""
    if not lines.reliable:
        return None
    return abs(len(counted_lines(text)) - len(lines.boxes))


def line_count_mismatch(text: str, lines: LineDetection, cfg: ChecksConfig) -> bool:
    """Flag rule of section 6.9: reliable detector, no tables, gap ≥ abs and > ratio tolerance."""
    if not lines.reliable or has_tables(text):
        return False
    gap = line_gap(text, lines) or 0
    return gap >= cfg.line_abs_tolerance and gap > cfg.line_ratio_tolerance * len(lines.boxes)
