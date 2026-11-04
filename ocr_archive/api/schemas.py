"""Request and response models of the REST API (section 11)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class DocumentCreated(BaseModel):
    id: str
    title: str
    status: str
    page_count: int
    duplicate: bool


class DraftBuildCreated(BaseModel):
    build_id: int


class IngestRequested(BaseModel):
    ingest_status: str


class StatusResponse(BaseModel):
    status: str


class BuildInfo(BaseModel):
    id: int
    document_id: str
    kind: str
    version: int
    status: str
    error: str | None = None


class Delivery(BaseModel):
    document_id: str
    title: str
    version: int


class DeliveredRequest(BaseModel):
    version: int = Field(ge=1)


class QueueEntry(BaseModel):
    page_id: int
    document_id: str
    title: str
    page_number: int
    scope: str
    reasons: list[str]
    has_draft: bool


class DraftRequest(BaseModel):
    expected_seq: int = Field(ge=0)
    item_version: str
    base_attempt_id: int | None = None
    base_revision_id: int | None = None
    text: str
    printed_label: str | None = None
    choices: Any = None
    hints: Any = None
    cursor: int | None = None
    keep_stale: bool = False
    scope: Literal["hints", "full"] | None = None  # scope the review was started with (8.1)


class DraftSaved(BaseModel):
    seq: int
    text_sha256: str


class ReviewRequest(BaseModel):
    request_id: str = Field(min_length=8, max_length=64)
    item_version: str
    base_attempt_id: int | None = None
    base_revision_id: int | None = None
    draft_seq: int = Field(default=0, ge=0)
    text: str
    printed_label: str | None = None
    duration_s: int | None = Field(default=None, ge=0)
    choices: Any = None
    on_conflict: Literal["keep_mine"] | None = None


class ReviewResponse(BaseModel):
    revision_id: int
    document_status: str


class RegionRequest(BaseModel):
    x0: float = Field(ge=0, le=1)
    y0: float = Field(ge=0, le=1)
    x1: float = Field(ge=0, le=1)
    y1: float = Field(ge=0, le=1)


class LowConfSpan(BaseModel):
    start: int
    end: int
    min_p: float


class RegionResponse(BaseModel):
    text: str
    low_conf: list[LowConfSpan]


class Health(BaseModel):
    db: str
    vlm: str
    worker_alive: bool
