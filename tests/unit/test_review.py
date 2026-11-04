"""Review item: hints, variants, crops, item version, hint shifting, UTF-16 (6.13, 8.3, 11)."""

from __future__ import annotations

import math
import re

import pytest

from ocr_archive.models import (
    Decision,
    FlagCode,
    Hint,
    LineBox,
    LineDetection,
    PageAttempts,
    PageStatus,
    PageTrack,
    ReviewScope,
    VlmResult,
)
from ocr_archive.pipeline import decide
from ocr_archive.review import (
    REASON_DIFF,
    REASON_HANDWRITING,
    REASON_LOW_CONF,
    REASON_MARKER,
    Utf16Map,
    hint_to_api,
    item_version,
    review_item,
    shift_hints,
)
from ocr_archive.settings import Settings
from ocr_archive.vlm.logprobs import parse_logprobs

pytestmark = pytest.mark.p3
S = Settings()
PAGE_SIZE = (600.0, 800.0)


def lines_for(text: str, reliable: bool = True, widths: list[float] | None = None) -> LineDetection:
    rows = [line for line in text.split("\n") if line.strip()]
    boxes = []
    for i, line in enumerate(rows):
        width = widths[i] if widths else 6.0 * len(line)
        boxes.append(LineBox(50, 50 + 20 * i, 50 + width, 62 + 20 * i))
    return LineDetection(boxes=boxes, reliable=reliable, columns_detected=False, skew_deg=0.0)


def res(text: str, low: dict[str, float] | None = None) -> VlmResult:
    tokens = re.findall(r"\s+|\S+", text)
    entries = [{"token": t, "logprob": math.log((low or {}).get(t, 1.0))} for t in tokens]
    return VlmResult(text, "stop", 1, 1, 1, parse_logprobs(text, entries))


def item_for(
    first: str,
    second: str,
    *,
    low: dict[str, float] | None = None,
    lines: LineDetection | None = None,
    scope: ReviewScope = ReviewScope.HINTS,
):
    page = PageAttempts(1, 0, PageTrack.VLM, res(first), res(second, low), 11, 22)
    detection = lines or lines_for(second)
    decision = decide(page, detection, S, "rv")
    return decision, review_item(page, decision, detection, scope, S, page_size=PAGE_SIZE)


BASE = "Протокол заседания комиссии\nприсутствовали все члены комиссии\nрешение принято"


def test_overlapping_sources_merge_into_one_hint() -> None:
    second = BASE.replace("заседания", "заседаня [неразб.]")
    _, item = item_for(BASE, second, low={"заседаня": 0.4})
    hint = item.hints[0]
    assert hint.reasons == {REASON_DIFF, REASON_LOW_CONF, REASON_MARKER}
    assert len([h for h in item.hints if hint.start < h.end and h.start < hint.end]) == 1


def test_variants_a_b_raw_substrings() -> None:
    second = BASE.replace("присутствовали", "присуствовали")
    decision, item = item_for(BASE, second)
    hint = next(h for h in item.hints if REASON_DIFF in h.reasons)
    assert hint.variants == {"a": "присутствовали", "b": "присуствовали"}
    candidate = second if decision.candidate_attempt_id == 22 else BASE
    assert candidate[hint.start : hint.end] in hint.variants.values()


def test_empty_range_widened_with_insert_variants() -> None:
    first = BASE.replace("все члены", "все постоянные члены")
    _, item = item_for(first, BASE)
    hint = next(h for h in item.hints if REASON_DIFF in h.reasons)
    assert hint.end > hint.start
    assert hint.variants is not None
    assert "постоянные" in hint.variants["a"] and "постоянные" not in hint.variants["b"]
    assert BASE[hint.start : hint.end] == hint.variants["b"]


def test_inserted_line_widened_to_line() -> None:
    first = BASE.replace("\nрешение", "\nслушали доклад\nрешение")
    _, item = item_for(first, BASE)
    hint = next(h for h in item.hints if REASON_DIFF in h.reasons)
    assert hint.variants is not None
    assert "слушали доклад" in hint.variants["a"]
    assert "\n" in hint.variants["a"]


def test_diff_inside_table_or_formula_has_no_variants() -> None:
    table1 = BASE + "\n<table><tr><td>Итого</td><td>сумма</td></tr></table>"
    table2 = BASE + "\n<table><tr><td>Итог</td><td>сумма</td></tr></table>"
    _, item = item_for(table1, table2)
    diff_hints = [h for h in item.hints if REASON_DIFF in h.reasons]
    assert diff_hints and all(h.variants is None for h in diff_hints)
    f1 = BASE + "\nформула \\( a + b \\) конец"
    f2 = BASE + "\nформула \\( a - b \\) конец"
    _, item = item_for(f1, f2)
    assert all(h.variants is None for h in item.hints if REASON_DIFF in h.reasons)


def test_handwriting_hint_only_on_flagged_page() -> None:
    text = BASE + "\n[рук.: согласовано]"
    _, clean = item_for(text, text)
    assert not any(REASON_HANDWRITING in h.reasons for h in clean.hints)
    flagged = text.replace("решение", "решение 5")
    decision, item = item_for(text.replace("решение", "решение 6"), flagged)
    assert decision.flags
    assert any(REASON_HANDWRITING in h.reasons for h in item.hints)


def test_digit_with_p_08_gives_flag_and_hint() -> None:
    text = BASE.replace("решение принято", "решение принято 15 голосами")
    decision, item = item_for(text, text, low={"15": 0.8})
    assert FlagCode.LOW_CONF_DIGIT in decision.flags
    hint = next(h for h in item.hints if REASON_LOW_CONF in h.reasons)
    assert text[hint.start : hint.end] == "15"


def test_crops_only_with_confident_binding() -> None:
    second = BASE.replace("присутствовали", "присуствовали")
    _, item = item_for(BASE, second)
    assert all(h.crop is not None for h in item.hints)
    _, unreliable = item_for(BASE, second, lines=lines_for(second, reliable=False))
    assert all(h.crop is None for h in unreliable.hints)
    table = second + "\n| a | b |"
    _, with_table = item_for(BASE + "\n| a | b |", table, lines=lines_for(table))
    assert all(h.crop is None for h in with_table.hints)


def test_missing_plus_split_line_fails_full_plausibility() -> None:
    lines = [f"строка текста номер {i} длинная достаточно" for i in range(8)]
    text = "\n".join(lines)
    widths = [6.0 * len(line) for line in lines]
    widths[2] = widths[2] * 2.2  # one box holds two text lines
    widths[5] = widths[5] * 0.45  # one text line spread over two boxes
    detection = lines_for(text, widths=widths)
    other = text.replace("номер 3", "номер 3 ")
    _, item = item_for(other, text.replace("длинная", "длиная"), lines=detection)
    assert item.hints and all(h.crop is None for h in item.hints)


def test_item_version_changes_with_hints() -> None:
    hint = Hint(1, 0, 3, frozenset({REASON_DIFF}), None, None)
    v1 = item_version(22, None, "rv", [hint])
    assert v1 != item_version(22, None, "rv", [])
    assert v1 != item_version(22, None, "rv2", [hint])
    assert v1 == item_version(22, None, "rv", [hint])


def _hint(start: int, end: int) -> Hint:
    return Hint(1, start, end, frozenset({REASON_DIFF}), {"a": "x", "b": "y"}, None)


@pytest.mark.parametrize(
    ("old", "new", "expected"),
    [
        ("ab[cd]ef", "Xab[cd]ef", (3, 7, False)),  # insert before
        ("ab[cd]ef", "ab[cd]efX", (2, 6, False)),  # insert after
        ("ab[cd]ef", "ab[cXd]ef", (2, 6, True)),  # insert inside
        ("abcdef", "abXcdef", (3, 5, False)),  # insert exactly at the start: shifted
        ("abcdef", "abcdXef", (2, 4, False)),  # insert exactly at the end: unchanged
        ("abcdef", "abef", (2, 4, True)),  # deletion of the hint
        ("abcdef", "abQQef", (2, 4, True)),  # replacement
    ],
)
def test_shift_hints(old: str, new: str, expected: tuple[int, int, bool]) -> None:
    start = old.index("[") if "[" in old else 2
    end = old.index("]") + 1 if "]" in old else 4
    shifted = shift_hints([_hint(start, end)], old, new)[0]
    if expected[2]:
        assert shifted.edited and shifted.variants is None
    else:
        assert (shifted.start, shifted.end, shifted.edited) == expected


def test_shift_hints_repeated_characters() -> None:
    old, new = "ааааа", "аааааа"
    hint = _hint(2, 3)
    shifted = shift_hints([hint], old, new)[0]
    # prefix wins: the insertion lands at the end, after the hint
    assert (shifted.start, shifted.end, shifted.edited) == (2, 3, False)


def test_utf16_offsets_for_astral_symbol() -> None:
    text = "𝑥 = 5, слово"
    start = text.index("слово")
    mapping = Utf16Map(text)
    api = hint_to_api(Hint(1, start, start + 5, frozenset({REASON_DIFF}), None, None), mapping)
    utf16 = text.encode("utf-16-le")
    assert utf16[api["start"] * 2 : api["end"] * 2].decode("utf-16-le") == "слово"
    assert mapping.from_utf16(api["start"]) == start


def test_failed_page_item_without_candidate() -> None:
    page = PageAttempts(1, 0, PageTrack.VLM, None, None, None, None)
    decision = Decision(PageStatus.FAILED, None, frozenset(), {}, "rv")
    item = review_item(page, decision, lines_for(BASE), ReviewScope.HINTS, S)
    assert item.hints == [] and item.base_attempt_id is None
