"""Enums and value objects shared by the pipeline, the review and the build."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ocr_archive.settings import Settings
    from ocr_archive.vlm.client import VlmClient


class DocumentStatus(StrEnum):
    QUEUED = "queued"
    PROCESSING = "processing"
    NEEDS_REVIEW = "needs_review"
    READY = "ready"
    BUILDING = "building"
    DONE = "done"
    FAILED = "failed"


class PageStatus(StrEnum):
    PENDING = "pending"
    OK = "ok"
    FLAGGED = "flagged"
    FAILED = "failed"
    REVIEWED = "reviewed"


class TextLayer(StrEnum):
    NONE = "none"
    HIDDEN = "hidden"
    VISIBLE = "visible"


class PageTrack(StrEnum):
    VLM = "vlm"
    PDF_TEXT = "pdf_text"


class FlagCode(StrEnum):
    BROKEN = "broken"
    UNVERIFIED = "unverified"
    DIGITS_DISAGREEMENT = "digits_disagreement"
    TABLE_MISMATCH = "table_mismatch"
    LINE_COUNT_MISMATCH = "line_count_mismatch"
    MARKERS = "markers"
    PASS_DIFF = "pass_diff"
    LOW_CONF_DIGIT = "low_conf_digit"
    LOW_CONF_TEXT = "low_conf_text"


#: Flags that cannot be switched off: without them a broken transcription passes unreviewed.
MANDATORY_FLAGS = frozenset({FlagCode.BROKEN, FlagCode.UNVERIFIED})


class Defect(StrEnum):
    TRUNCATED = "truncated"
    LOOP = "loop"
    EMPTY = "empty"


class ReviewScope(StrEnum):
    HINTS = "hints"
    FULL = "full"


class BuildKind(StrEnum):
    FINAL = "final"
    DRAFT = "draft"


class BuildStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class IngestStatus(StrEnum):
    NONE = "none"
    PENDING = "pending"
    DONE = "done"
    FAILED = "failed"


class AttemptRole(StrEnum):
    FULL = "full"
    PART = "part"
    COMBINED = "combined"
    PDF_TEXT = "pdf_text"


@dataclass(frozen=True)
class FastTrackCheck:
    track: PageTrack
    reason: str | None  # images | too_short | bad_chars | math_fonts; None for pdf_text


@dataclass(frozen=True)
class LineBox:
    x0: float
    y0: float
    x1: float
    y1: float  # PDF points, page coordinate system


@dataclass(frozen=True)
class LineDetection:
    boxes: list[LineBox]
    reliable: bool
    columns_detected: bool
    skew_deg: float


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    min_p: float


@dataclass(frozen=True)
class TokenStats:  # computed at decision time from TokenProbs and checks.*_token_p, section 6.6
    letter_min_p: float
    letter_low: int
    digit_min_p: float
    digit_low: int
    digit_tokens: int


@dataclass(frozen=True)
class TokenProbs:  # stored raw data, attempts.token_probs_json
    letter_tokens: int
    digit_tokens: int
    tokens: list[tuple[int, int, float, str]]  # (start, end, p, kind) for tokens with p < 0.99


@dataclass(frozen=True)
class VlmResult:
    text: str
    finish_reason: str | None
    tokens_in: int | None
    tokens_out: int | None
    duration_ms: int
    token_probs: TokenProbs | None  # None when logprobs are unavailable or do not match the text


@dataclass(frozen=True)
class PassComparison:
    similarity: float
    diff_words: int
    digits_equal: bool
    digit_mismatch: int
    tables_equal: bool


@dataclass(frozen=True)
class Decision:
    status: PageStatus
    candidate_attempt_id: int | None  # None for a failed page without a working attempt
    flags: frozenset[FlagCode]
    signals: dict[str, float | bool]  # counters of section 6.10, no_logprobs flag
    rules_version: str
    suppressed_flags: frozenset[FlagCode] = frozenset()
    candidate_pass: int | None = None  # 1 or 2; which pass became the candidate


@dataclass(frozen=True)
class Hint:
    hint_id: int
    start: int  # character range in the candidate text, end exclusive
    end: int
    reasons: frozenset[str]  # diff | low_conf | marker | handwriting
    variants: dict[str, str] | None  # {"a": ..., "b": ...} when reasons include diff
    crop: tuple[float, float, float, float] | None  # page fractions x0, y0, x1, y1
    edited: bool = False  # set by shift_hints when a manual edit touched the hint


@dataclass(frozen=True)
class ReviewItem:
    page_id: int
    scope: ReviewScope
    reasons: list[str]  # flag codes, or "spot_check"
    hints: list[Hint]  # empty when the editor starts from a revision
    item_version: str
    base_attempt_id: int | None
    base_revision_id: int | None
    text_a: str | None = None
    text_b: str | None = None


@dataclass(frozen=True)
class PageAttempts:
    page_id: int
    page_index: int
    track: PageTrack
    first: VlmResult | None  # pass 1; None if the request failed permanently
    second: VlmResult | None  # pass 2; the combined attempt when the page was split
    first_attempt_id: int | None
    second_attempt_id: int | None


@dataclass(frozen=True)
class PipelineContext:
    settings: Settings
    db_path: Path
    vlm: VlmClient
    rules_version: str
    profile_ids: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class BuildContext:
    settings: Settings
    db_path: Path
    outputs_dir: Path
    export_dir: Path
