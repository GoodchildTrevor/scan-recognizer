"""Review item of a page: hints, variants, crops and item version (section 6.13)."""

from __future__ import annotations

import bisect
import difflib
import hashlib
import json
import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from typing import Any

from ocr_archive.build.markers import find_markers, handwriting_markers, hard_markers_in
from ocr_archive.lines import line_plausible
from ocr_archive.models import (
    Decision,
    Hint,
    LineDetection,
    PageAttempts,
    ReviewItem,
    ReviewScope,
    VlmResult,
)
from ocr_archive.settings import Settings
from ocr_archive.textnorm import (
    counted_lines,
    formula_spans,
    has_tables,
    html_table_spans,
    is_table_separator,
    unify_quotes_dashes,
)
from ocr_archive.vlm.logprobs import low_confidence_spans

REASON_DIFF = "diff"
REASON_LOW_CONF = "low_conf"
REASON_MARKER = "marker"
REASON_HANDWRITING = "handwriting"
SPOT_CHECK_REASON = "spot_check"

#: Share of plausible lines required to show line crops; the PDF layer needs only 0.9.
CROP_MIN_SHARE = 1.0

_TOKEN_RE = re.compile(r"\S+")


# --- UTF-16 offsets ----------------------------------------------------------------------------


class Utf16Map:
    """Convert code point offsets of one text to UTF-16 code units and back (section 11)."""

    def __init__(self, text: str) -> None:
        self._wide = [i for i, ch in enumerate(text) if ord(ch) > 0xFFFF]
        self._length = len(text)

    def to_utf16(self, index: int) -> int:
        """Code point index → UTF-16 index."""
        return index + bisect.bisect_left(self._wide, index)

    def from_utf16(self, index: int) -> int:
        """UTF-16 index → code point index; an index inside a surrogate pair rounds down."""
        cp = index
        for wide in self._wide:
            if self.to_utf16(wide) < index:
                cp -= 1
            else:
                break
        return max(0, min(cp, self._length))


def to_utf16_offsets(text: str, offsets: Iterable[int]) -> list[int]:
    """Convert code point offsets into UTF-16 code units, as JavaScript counts them."""
    mapping = Utf16Map(text)
    return [mapping.to_utf16(offset) for offset in offsets]


def hint_to_api(hint: Hint, mapping: Utf16Map) -> dict[str, Any]:
    """Serialize a hint for the API with UTF-16 offsets."""
    return {
        "id": hint.hint_id,
        "start": mapping.to_utf16(hint.start),
        "end": mapping.to_utf16(hint.end),
        "reasons": sorted(hint.reasons),
        "variants": hint.variants,
        "crop": list(hint.crop) if hint.crop else None,
    }


# --- tokens and diff ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Token:
    start: int
    end: int
    key: str


def _tokens(text: str) -> list[_Token]:
    return [
        _Token(m.start(), m.end(), unify_quotes_dashes(unicodedata.normalize("NFC", m.group())))
        for m in _TOKEN_RE.finditer(text)
    ]


@dataclass
class _Edit:
    start: int  # range in the candidate replaced by text from the other pass
    end: int
    replacement: str
    other_start: int  # range in the other pass, for the markup check
    other_end: int


@dataclass
class _RawHint:
    start: int
    end: int
    reasons: set[str]
    edits: list[_Edit] = field(default_factory=list)


def _line_bounds(text: str, pos: int) -> tuple[int, int]:
    start = text.rfind("\n", 0, pos) + 1
    end = text.find("\n", pos)
    return start, len(text) if end == -1 else end


def hint_diff(candidate: str, other: str, merge_gap_words: int) -> list[_RawHint]:
    """Pass differences on raw text: one hint per unequal block of whitespace-separated tokens.

    Keys compare tokens after NFC and unified quotes/dashes; positions stay in the raw texts.
    Neighbouring blocks on one line with fewer than ``merge_gap_words`` matched tokens between
    them are merged. An empty range in the candidate is widened to the neighbouring word, or to
    the whole line when the insertion is a separate line.
    """
    cand = _tokens(candidate)
    oth = _tokens(other)
    matcher = difflib.SequenceMatcher(
        a=[t.key for t in cand], b=[t.key for t in oth], autojunk=False
    )
    blocks = [op[1:] for op in matcher.get_opcodes() if op[0] != "equal"]
    merged: list[tuple[int, int, int, int]] = []
    for block in blocks:
        if merged:
            pi1, pi2, pj1, _ = merged[-1]
            i1 = block[0]
            gap_tokens = i1 - pi2
            between_start = cand[pi2 - 1].end if pi2 > 0 else 0
            between_end = cand[i1].start if i1 < len(cand) else len(candidate)
            same_line = "\n" not in candidate[between_start:between_end]
            if gap_tokens < merge_gap_words and same_line:
                merged[-1] = (pi1, block[1], pj1, block[3])
                continue
        merged.append(block)

    hints: list[_RawHint] = []
    for i1, i2, j1, j2 in merged:
        o_start = oth[j1].start if j2 > j1 else (oth[j1 - 1].end if j1 > 0 else 0)
        o_end = oth[j2 - 1].end if j2 > j1 else o_start
        inserted = other[o_start:o_end]
        if i2 > i1:
            start, end = cand[i1].start, cand[i2 - 1].end
            hints.append(
                _RawHint(start, end, {REASON_DIFF}, [_Edit(start, end, inserted, o_start, o_end)])
            )
            continue
        hints.append(_insert_hint(candidate, other, cand, oth, i1, j1, j2, o_start, o_end))
    return hints


def _insert_hint(
    candidate: str,
    other: str,
    cand: list[_Token],
    oth: list[_Token],
    i1: int,
    j1: int,
    j2: int,
    o_start: int,
    o_end: int,
) -> _RawHint:
    """Hint for text present only in the other pass: widen to a neighbour word or line."""
    inserted = other[o_start:o_end]
    sep_before = other[oth[j1 - 1].end : o_start] if j1 > 0 else " "
    sep_after = other[o_end : oth[j2].start] if j2 < len(oth) else " "
    sep_before = sep_before or " "
    sep_after = sep_after or " "
    if not cand:
        return _RawHint(
            0, len(candidate), {REASON_DIFF}, [_Edit(0, 0, inserted + sep_after, o_start, o_end)]
        )
    if i1 > 0:
        prev = cand[i1 - 1]
        line_start, line_end = _line_bounds(candidate, prev.start)
        separate_line = "\n" in sep_before
        if separate_line or i1 == len(cand) or cand[i1].start > line_end:
            start, end = (line_start, prev.end) if separate_line else (prev.start, prev.end)
        else:
            start, end = prev.start, prev.end
        edit = _Edit(prev.end, prev.end, sep_before + inserted, o_start, o_end)
        return _RawHint(start, end, {REASON_DIFF}, [edit])
    nxt = cand[0]
    line_start, line_end = _line_bounds(candidate, nxt.start)
    separate_line = "\n" in sep_after
    start, end = (nxt.start, line_end) if separate_line else (nxt.start, nxt.end)
    edit = _Edit(nxt.start, nxt.start, inserted + sep_after, o_start, o_end)
    return _RawHint(start, end, {REASON_DIFF}, [edit])


# --- merging and variants ----------------------------------------------------------------------


def merge_hints(raw: list[_RawHint]) -> list[_RawHint]:
    """Merge overlapping and touching ranges into one hint with the union of reasons."""
    result: list[_RawHint] = []
    for hint in sorted(raw, key=lambda h: (h.start, h.end)):
        if result and hint.start <= result[-1].end:
            last = result[-1]
            last.end = max(last.end, hint.end)
            last.reasons |= hint.reasons
            last.edits.extend(hint.edits)
        else:
            result.append(_RawHint(hint.start, hint.end, set(hint.reasons), list(hint.edits)))
    return result


def _markup_spans(text: str) -> list[tuple[int, int]]:
    spans = html_table_spans(text) + formula_spans(text)
    offset = 0
    for line in text.split("\n"):
        if is_table_separator(line):
            spans.append((offset, offset + len(line)))
        offset += len(line) + 1
    return spans


def _touches(spans: list[tuple[int, int]], start: int, end: int) -> bool:
    if start == end:
        return any(s <= start < e for s, e in spans)
    return any(start < e and s < end for s, e in spans)


def _variants(
    hint: _RawHint,
    candidate: str,
    cand_markup: list[tuple[int, int]],
    other_markup: list[tuple[int, int]],
    candidate_is_first: bool,
) -> dict[str, str] | None:
    if REASON_DIFF not in hint.reasons or not hint.edits:
        return None
    for edit in hint.edits:
        if _touches(cand_markup, edit.start, edit.end) or _touches(
            other_markup, edit.other_start, edit.other_end
        ):
            return None
    original = candidate[hint.start : hint.end]
    pieces: list[str] = []
    cursor = hint.start
    for edit in sorted(hint.edits, key=lambda e: (e.start, e.end)):
        if edit.start < cursor:
            return None  # overlapping edits cannot be applied unambiguously
        pieces.append(candidate[cursor : edit.start])
        pieces.append(edit.replacement)
        cursor = edit.end
    pieces.append(candidate[cursor : hint.end])
    other_variant = "".join(pieces)
    if candidate_is_first:
        return {"a": original, "b": other_variant}
    return {"a": other_variant, "b": original}


def _strip_range(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


# --- crops -------------------------------------------------------------------------------------


def _counted_line_ranges(text: str) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    offset = 0
    kept = counted_lines(text)
    kept_iter = iter(kept)
    next_kept = next(kept_iter, None)
    for line in text.split("\n"):
        if next_kept is not None and line == next_kept:
            ranges.append((offset, offset + len(line)))
            next_kept = next(kept_iter, None)
        offset += len(line) + 1
    return ranges


def binding_is_confident(text: str, lines: LineDetection) -> bool:
    """Confident line binding for crops: reliable, no tables, equal counts, all lines plausible."""
    if not lines.reliable or has_tables(text):
        return False
    kept = counted_lines(text)
    if len(kept) != len(lines.boxes):
        return False
    return line_plausible(kept, lines.boxes, min_share=CROP_MIN_SHARE)


def bind_crops(
    hints: list[Hint],
    text: str,
    lines: LineDetection,
    page_size: tuple[float, float] | None,
) -> list[Hint]:
    """Attach line crops to hints when the binding is confident; otherwise no crops at all."""
    if page_size is None or not binding_is_confident(text, lines):
        return [replace(h, crop=None) for h in hints]
    width, height = page_size
    ranges = _counted_line_ranges(text)
    boxes = lines.boxes
    block_x0 = min(b.x0 for b in boxes)
    block_x1 = max(b.x1 for b in boxes)
    result = []
    for hint in hints:
        covered = [k for k, (s, e) in enumerate(ranges) if hint.start < e and s < hint.end]
        if hint.start == hint.end:
            covered = [k for k, (s, e) in enumerate(ranges) if s <= hint.start <= e]
        if not covered or hint.reasons <= {REASON_HANDWRITING}:
            result.append(replace(hint, crop=None))
            continue
        first = max(0, covered[0] - 1)
        last = min(len(boxes) - 1, covered[-1] + 1)
        y0 = min(boxes[k].y0 for k in range(first, last + 1))
        y1 = max(boxes[k].y1 for k in range(first, last + 1))
        crop = (
            max(0.0, block_x0 / width),
            max(0.0, y0 / height),
            min(1.0, block_x1 / width),
            min(1.0, y1 / height),
        )
        rounded = (round(crop[0], 5), round(crop[1], 5), round(crop[2], 5), round(crop[3], 5))
        result.append(replace(hint, crop=rounded))
    return result


# --- item --------------------------------------------------------------------------------------


def hints_to_json(hints: list[Hint]) -> list[dict[str, Any]]:
    """Serialize hints with code point offsets (storage format of review_json)."""
    return [
        {
            "id": h.hint_id,
            "start": h.start,
            "end": h.end,
            "reasons": sorted(h.reasons),
            "variants": h.variants,
            "crop": list(h.crop) if h.crop else None,
        }
        for h in hints
    ]


def _crop_tuple(value: Any) -> tuple[float, float, float, float] | None:
    if not value:
        return None
    x0, y0, x1, y1 = (float(v) for v in value)
    return x0, y0, x1, y1


def hints_from_json(data: list[dict[str, Any]]) -> list[Hint]:
    return [
        Hint(
            hint_id=int(h["id"]),
            start=int(h["start"]),
            end=int(h["end"]),
            reasons=frozenset(h["reasons"]),
            variants=h.get("variants"),
            crop=_crop_tuple(h.get("crop")),
        )
        for h in data
    ]


def item_version(
    candidate_attempt_id: int | None,
    revision_id: int | None,
    rules_version: str,
    hints: list[Hint],
) -> str:
    """sha256 of the candidate, the current revision, the rules version and the hints."""
    payload = json.dumps(
        [candidate_attempt_id, revision_id, rules_version, hints_to_json(hints)],
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _candidate(page: PageAttempts, decision: Decision) -> tuple[VlmResult | None, VlmResult | None]:
    """Return (candidate, other pass) according to the decision."""
    if decision.candidate_attempt_id is None:
        return None, None
    if decision.candidate_attempt_id == page.first_attempt_id:
        return page.first, page.second
    return page.second, page.first


def review_item(
    page: PageAttempts,
    decision: Decision,
    lines: LineDetection,
    scope: ReviewScope,
    settings: Settings,
    *,
    page_size: tuple[float, float] | None = None,
    revision_id: int | None = None,
) -> ReviewItem:
    """Build the review item of a page: the whole page with hints (section 6.13).

    :param page_size: page width and height in PDF points, needed for crops.
    :param revision_id: current revision; when set, the editor starts from it and has no hints.
    """
    candidate, other = _candidate(page, decision)
    reasons = sorted(f.value for f in decision.flags)
    if scope == ReviewScope.FULL and not reasons:
        reasons = [SPOT_CHECK_REASON]
    hints: list[Hint] = []
    if candidate is not None and revision_id is None:
        hints = _build_hints(candidate, other, decision, lines, settings, page_size, page)
    return ReviewItem(
        page_id=page.page_id,
        scope=scope,
        reasons=reasons,
        hints=hints,
        item_version=item_version(
            decision.candidate_attempt_id, revision_id, decision.rules_version, hints
        ),
        base_attempt_id=decision.candidate_attempt_id,
        base_revision_id=revision_id,
        text_a=page.first.text if page.first else None,
        text_b=page.second.text if page.second else None,
    )


def _build_hints(
    candidate: VlmResult,
    other: VlmResult | None,
    decision: Decision,
    lines: LineDetection,
    settings: Settings,
    page_size: tuple[float, float] | None,
    page: PageAttempts,
) -> list[Hint]:
    text = candidate.text
    raw: list[_RawHint] = []
    compared = "diff_words" in decision.signals
    if other is not None and compared:
        raw.extend(hint_diff(text, other.text, settings.hints.merge_gap_words))
    if candidate.token_probs is not None:
        for span in low_confidence_spans(candidate.token_probs, settings.checks):
            start, end = _strip_range(text, span.start, span.end)
            if end > start:
                raw.append(_RawHint(start, end, {REASON_LOW_CONF}))
    for marker in hard_markers_in(text, settings.decide.hard_markers):
        raw.append(_RawHint(marker.start, marker.end, {REASON_MARKER}))
    if decision.flags:
        for marker in handwriting_markers(text, settings.decide.hard_markers):
            raw.append(_RawHint(marker.start, marker.end, {REASON_HANDWRITING}))
    merged = merge_hints(raw)
    candidate_is_first = page.first is candidate
    cand_markup = _markup_spans(text)
    other_markup = _markup_spans(other.text) if other is not None else []
    hints = [
        Hint(
            hint_id=index,
            start=h.start,
            end=h.end,
            reasons=frozenset(h.reasons),
            variants=_variants(h, text, cand_markup, other_markup, candidate_is_first),
            crop=None,
        )
        for index, h in enumerate(merged, start=1)
    ]
    return bind_crops(hints, text, lines, page_size)


def item_to_json(item: ReviewItem, *, rules_version: str) -> dict[str, Any]:
    """Storage format of ``pages.review_json``."""
    return {
        "scope": item.scope.value,
        "reasons": item.reasons,
        "item_version": item.item_version,
        "base_attempt_id": item.base_attempt_id,
        "base_revision_id": item.base_revision_id,
        "rules_version": rules_version,
        "hints": hints_to_json(item.hints),
    }


# --- hints after edits -------------------------------------------------------------------------


def shift_hints(hints: list[Hint], old_text: str, new_text: str) -> list[Hint]:
    """Move hints after a manual edit; reference implementation for the window JS (section 8.3).

    The common prefix is computed first, the suffix is capped so it never overlaps it. Hints
    entirely before the change stay (an insertion exactly at their end does not touch them),
    hints entirely after it shift (an insertion exactly at their start shifts them), the others
    are marked as edited manually.
    """
    p = 0
    limit = min(len(old_text), len(new_text))
    while p < limit and old_text[p] == new_text[p]:
        p += 1
    s = 0
    max_suffix = limit - p
    while s < max_suffix and old_text[len(old_text) - 1 - s] == new_text[len(new_text) - 1 - s]:
        s += 1
    old_end = len(old_text) - s
    delta = len(new_text) - len(old_text)
    result = []
    for hint in hints:
        if hint.edited:
            result.append(hint)
        elif hint.end <= p:
            result.append(hint)
        elif hint.start >= old_end:
            result.append(replace(hint, start=hint.start + delta, end=hint.end + delta))
        else:
            result.append(replace(hint, edited=True, variants=None))
    return result


def marker_ranges(text: str) -> list[tuple[int, int]]:
    """Ranges of all markers, used to mark draft hints."""
    return [(m.start, m.end) for m in find_markers(text)]
