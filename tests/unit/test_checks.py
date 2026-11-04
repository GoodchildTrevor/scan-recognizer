"""Attempt defects, part merging, pass comparison, token probabilities (6.5–6.8)."""

from __future__ import annotations

import math

import pytest

from ocr_archive.checks import (
    attempt_defects,
    compare_passes,
    diff_words,
    line_count_mismatch,
    merge_parts,
)
from ocr_archive.models import Defect, LineBox, LineDetection, TokenProbs, VlmResult
from ocr_archive.settings import ChecksConfig
from ocr_archive.vlm.logprobs import (
    low_confidence_spans,
    merge_token_probs,
    parse_logprobs,
    token_stats,
)

CFG = ChecksConfig()


def lines_of(n: int, reliable: bool = True) -> LineDetection:
    boxes = [LineBox(50, 50 + 20 * i, 500, 62 + 20 * i) for i in range(n)]
    return LineDetection(boxes=boxes, reliable=reliable, columns_detected=False, skew_deg=0.0)


def result(text: str, finish: str = "stop", probs: TokenProbs | None = None) -> VlmResult:
    return VlmResult(text, finish, 10, 10, 5, probs)


@pytest.mark.p2
def test_defects() -> None:
    assert attempt_defects(result("x" * 100, "length"), lines_of(3), CFG) == {Defect.TRUNCATED}
    looped = "\n".join(["Одна и та же строка повторяется"] * 4)
    assert Defect.LOOP in attempt_defects(result(looped), lines_of(4), CFG)
    tail = "abcdefghij" * 20
    assert Defect.LOOP in attempt_defects(result("начало " + tail * 3), lines_of(3), CFG)
    assert attempt_defects(result("коротко"), lines_of(3), CFG) == {Defect.EMPTY}
    assert attempt_defects(result("коротко"), lines_of(2), CFG) == set()


@pytest.mark.p2
def test_merge_parts() -> None:
    p1 = TokenProbs(1, 1, [(0, 3, 0.5, "letter"), (4, 6, 0.8, "digit")])
    p2 = TokenProbs(2, 0, [(0, 4, 0.4, "letter")])
    merged = merge_parts([result("abc 12", "stop", p1), result("defg", "length", p2)])
    assert merged.text == "abc 12\ndefg"
    assert merged.finish_reason == "length"
    assert merged.token_probs is not None
    assert merged.token_probs.tokens[-1] == (7, 11, 0.4, "letter")
    assert merged.token_probs.letter_tokens == 3
    assert merged.duration_ms == 10
    assert merge_parts([result("a", "stop", p1), result("b", "stop", None)]).token_probs is None


@pytest.mark.p2
def test_compare_passes() -> None:
    first = "Расход составил 125 тонн в «марте» месяце"
    second = 'Расход составил 126 тонн в "марте" месяце'
    cmp = compare_passes(first, second)
    assert not cmp.digits_equal
    assert cmp.digit_mismatch == 2
    assert cmp.diff_words == 0
    assert cmp.tables_equal
    assert diff_words("один два три", "один пять три") == 2


@pytest.mark.p2
def test_line_count_mismatch() -> None:
    text = "\n".join(f"строка {i}" for i in range(20))
    assert not line_count_mismatch(text, lines_of(22), CFG)
    assert line_count_mismatch(text, lines_of(24), CFG)
    assert not line_count_mismatch(text, lines_of(24, reliable=False), CFG)
    assert not line_count_mismatch(text + "\n| a | b |", lines_of(30), CFG)


def _entries(tokens: list[tuple[str, float]]) -> list[dict[str, object]]:
    return [{"token": t, "logprob": math.log(p)} for t, p in tokens]


@pytest.mark.p1
def test_parse_logprobs_spans_and_stats() -> None:
    tokens = [
        ("Итого", 0.995),
        (" ", 1.0),
        ("12", 0.8),
        ("3", 0.85),
        (" ", 1.0),
        ("руб", 0.5),
        (".", 0.3),
    ]
    text = "".join(t for t, _ in tokens)
    probs = parse_logprobs(text, _entries(tokens))
    assert probs is not None
    assert probs.letter_tokens == 2 and probs.digit_tokens == 2
    assert [t[3] for t in probs.tokens] == ["digit", "digit", "letter"]
    spans = low_confidence_spans(probs, CFG)
    assert [(s.start, s.end) for s in spans] == [(6, 9), (10, 13)]
    stats = token_stats(probs, CFG)
    assert stats.digit_low == 2 and stats.letter_low == 1
    assert stats.digit_min_p == pytest.approx(0.8)


@pytest.mark.p1
def test_parse_logprobs_mismatch_drops() -> None:
    assert parse_logprobs("другой текст", _entries([("текст", 0.5)])) is None
    assert parse_logprobs("текст", None) is None


@pytest.mark.p1
def test_merge_token_probs_requires_all_parts() -> None:
    assert merge_token_probs([(None, 0)]) is None
