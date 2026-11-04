"""Transactional operations shared by the REST API and the CLI (sections 5.1, 8, 9.6, 11, 15.4).

Every write runs in ``BEGIN IMMEDIATE`` and ends with the status recomputation of section 5.1.
"""

from __future__ import annotations

import hashlib
import json
import random
import sqlite3
import statistics
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from ocr_archive.db import repo
from ocr_archive.db.connection import immediate, read_snapshot, utc_now
from ocr_archive.exceptions import DraftConflictError, StaleRevisionError, StatusTransitionError
from ocr_archive.models import (
    MANDATORY_FLAGS,
    Decision,
    DocumentStatus,
    FlagCode,
    Hint,
    LineDetection,
    PageStatus,
    PageTrack,
    ReviewScope,
)
from ocr_archive.pipeline import load_page_attempts, page_lines, page_size
from ocr_archive.review import (
    SPOT_CHECK_REASON,
    Utf16Map,
    hint_to_api,
    hints_from_json,
    item_version,
    review_item,
)
from ocr_archive.settings import Settings
from ocr_archive.status import recompute_status

FAILED_REASON = "failed"
DRAFT_REASON = "draft"

_EMPTY_LINES = LineDetection(boxes=[], reliable=False, columns_detected=False, skew_deg=0.0)


def text_sha256(text: str) -> str:
    """sha256 hex of the UTF-8 text, as the window computes it (section 8.4)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _require_page(conn: sqlite3.Connection, page_id: int) -> sqlite3.Row:
    page = repo.get_page(conn, page_id)
    if page is None:
        raise KeyError(page_id)
    return page


def _require_document(conn: sqlite3.Connection, document_id: str) -> sqlite3.Row:
    doc = repo.get_document(conn, document_id)
    if doc is None:
        raise KeyError(document_id)
    return doc


def default_scope(conn: sqlite3.Connection, page: sqlite3.Row) -> ReviewScope:
    """Scope of a page without a draft: spot check → full, flag or failure → hints, else full."""
    if repo.open_sample(conn, int(page["id"])):
        return ReviewScope.FULL
    if page["status"] in (PageStatus.FLAGGED, PageStatus.FAILED):
        return ReviewScope.HINTS
    return ReviewScope.FULL


# --- review item -------------------------------------------------------------------------------


@dataclass
class ItemView:
    """Everything the review window needs for one page."""

    page_id: int
    document_id: str
    title: str
    page_number: int
    page_count: int
    status: str
    scope: ReviewScope
    reasons: list[str]
    signals: dict[str, Any]
    item_version: str
    base_attempt_id: int | None
    base_revision_id: int | None
    text: str
    hints: list[Hint]
    text_a: str | None
    text_b: str | None
    printed_label: str | None
    draft: dict[str, Any] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def no_candidate(self) -> bool:
        return self.base_attempt_id is None and self.base_revision_id is None

    def to_api(self) -> dict[str, Any]:
        """Serialize for the API; hint offsets in UTF-16 code units."""
        mapping = Utf16Map(self.text)
        return {
            "page_id": self.page_id,
            "document_id": self.document_id,
            "title": self.title,
            "page_number": self.page_number,
            "page_count": self.page_count,
            "status": self.status,
            "scope": self.scope.value,
            "reasons": self.reasons,
            "signals": self.signals,
            "item_version": self.item_version,
            "base_attempt_id": self.base_attempt_id,
            "base_revision_id": self.base_revision_id,
            "text": self.text,
            "text_a": self.text_a,
            "text_b": self.text_b,
            "no_candidate": self.no_candidate,
            "printed_label": self.printed_label,
            "hints": [hint_to_api(h, mapping) for h in self.hints],
            "draft": self.draft,
        } | self.extra


def _reasons(page: sqlite3.Row, scope: ReviewScope) -> list[str]:
    flags = json.loads(page["flags_json"] or "[]")
    if page["status"] == PageStatus.FAILED:
        return [FAILED_REASON, *flags]
    if flags:
        return list(flags)
    return [SPOT_CHECK_REASON] if scope == ReviewScope.FULL else []


def _decision_from_page(page: sqlite3.Row, rules_version: str) -> Decision:
    flags = frozenset(
        FlagCode(f)
        for f in json.loads(page["flags_json"] or "[]")
        if f in FlagCode._value2member_map_
    )
    signals = json.loads(page["signals_json"] or "{}")
    return Decision(
        status=PageStatus(page["status"])
        if page["status"] != PageStatus.REVIEWED
        else PageStatus.OK,
        candidate_attempt_id=page["current_attempt_id"],
        flags=flags,
        signals={k: v for k, v in signals.items() if isinstance(v, int | float | bool)},
        rules_version=page["rules_version"] or rules_version,
    )


def _draft_info(conn: sqlite3.Connection, draft: sqlite3.Row, page: sqlite3.Row) -> dict[str, Any]:
    base = (draft["base_attempt_id"], draft["base_revision_id"])
    current = (page["current_attempt_id"], page["current_revision_id"])
    stale = bool(draft["stale"]) or base != current
    reason: str | None = None
    if draft["stale"]:
        reason = "keep_stale"
    elif base != current:
        reason = "changed"
        if page["current_revision_id"] is None and page["current_attempt_id"] is not None:
            current_attempt = repo.get_attempt(conn, int(page["current_attempt_id"]))
            base_attempt = (
                repo.get_attempt(conn, int(draft["base_attempt_id"]))
                if draft["base_attempt_id"] is not None
                else None
            )
            if current_attempt is not None and (
                base_attempt is None
                or int(current_attempt["generation"]) > int(base_attempt["generation"])
            ):
                reason = "recognized_again"
    return {
        "seq": int(draft["seq"]),
        "text": draft["text"],
        "text_sha256": text_sha256(str(draft["text"])),
        "printed_label": draft["printed_label"],
        "choices": json.loads(draft["choices_json"]) if draft["choices_json"] else None,
        "hints": json.loads(draft["hints_json"]) if draft["hints_json"] else None,
        "cursor": draft["cursor"],
        "item_version": draft["item_version"],
        "base_attempt_id": draft["base_attempt_id"],
        "base_revision_id": draft["base_revision_id"],
        "scope": draft["scope"],
        "stale": stale,
        "stale_reason": reason if stale else None,
        "updated_at": draft["updated_at"],
    }


def build_item_view(
    conn: sqlite3.Connection,
    page_id: int,
    settings: Settings,
    rules_version: str,
    *,
    scope: ReviewScope | None = None,
) -> ItemView:
    """Assemble the review item of a page (section 6.13) with its draft.

    :param scope: forced scope, e.g. ``full`` for the "fix" command.
    :raises StatusTransitionError: ``invalid_transition`` for a pending page.
    :raises KeyError: unknown page.
    """
    page = _require_page(conn, page_id)
    if page["status"] == PageStatus.PENDING:
        raise StatusTransitionError("invalid_transition", "page is still being recognized")
    doc = _require_document(conn, str(page["document_id"]))
    draft = repo.get_draft(conn, page_id)
    if scope is None:
        scope = (
            ReviewScope(draft["scope"])
            if draft is not None and draft["scope"]
            else default_scope(conn, page)
        )
    reasons = _reasons(page, scope)
    signals = json.loads(page["signals_json"] or "{}")
    attempts = load_page_attempts(conn, page)
    text_a = attempts.first.text if attempts.first else None
    text_b = attempts.second.text if attempts.second else None
    revision_id = page["current_revision_id"]
    candidate_id = page["current_attempt_id"]
    page_rules = page["rules_version"] or rules_version
    hints: list[Hint] = []
    if revision_id is not None:
        revision = repo.get_revision(conn, int(revision_id))
        text = str(revision["text"]) if revision is not None else ""
        version = item_version(candidate_id, revision_id, page_rules, [])
    else:
        candidate = repo.get_attempt(conn, int(candidate_id)) if candidate_id else None
        text = str(candidate["text"]) if candidate is not None else ""
        stored = json.loads(page["review_json"]) if page["review_json"] else None
        if stored is not None and stored.get("base_attempt_id") == candidate_id:
            hints = hints_from_json(stored.get("hints", []))
            version = str(stored["item_version"])
        elif candidate is not None and page["track"] == PageTrack.VLM:
            pdf_path = settings.data_dir / str(doc["source_path"])
            index = int(page["page_index"])
            lines = page_lines(conn, page_id, pdf_path, index)
            item = review_item(
                attempts,
                _decision_from_page(page, rules_version),
                lines,
                scope,
                settings,
                page_size=page_size(pdf_path, index),
            )
            hints, version = item.hints, item.item_version
        else:
            version = item_version(candidate_id, None, page_rules, [])
    return ItemView(
        page_id=page_id,
        document_id=str(doc["id"]),
        title=str(doc["title"]),
        page_number=int(page["page_index"]) + 1,
        page_count=int(doc["page_count"]),
        status=str(page["status"]),
        scope=scope,
        reasons=reasons,
        signals=signals,
        item_version=version,
        base_attempt_id=candidate_id if revision_id is None else page["current_attempt_id"],
        base_revision_id=revision_id,
        text=text,
        hints=hints,
        text_a=text_a,
        text_b=text_b,
        printed_label=page["printed_label"],
        draft=_draft_info(conn, draft, page) if draft is not None else None,
        extra={
            # Image sizes for the window: whole page, and the pass2 render used for crops.
            "page_long_side": settings.review.page_long_side,
            "crop_long_side": settings.profiles["pass2"].image_long_side,
        },
    )


def item_for_page_number(
    conn: sqlite3.Connection,
    document_id: str,
    page_number: int,
    settings: Settings,
    rules_version: str,
) -> ItemView:
    """Item with scope ``full`` for the "fix" command; page numbers are 1-based."""
    page = repo.get_page_by_index(conn, document_id, page_number - 1)
    if page is None:
        raise KeyError(page_number)
    return build_item_view(conn, int(page["id"]), settings, rules_version, scope=ReviewScope.FULL)


# --- drafts ------------------------------------------------------------------------------------


@dataclass(frozen=True)
class DraftInput:
    expected_seq: int
    item_version: str
    base_attempt_id: int | None
    base_revision_id: int | None
    text: str
    printed_label: str | None = None
    choices: Any = None
    hints: Any = None
    cursor: int | None = None
    keep_stale: bool = False
    scope: str | None = None


def save_draft(conn: sqlite3.Connection, page_id: int, data: DraftInput) -> int:
    """Save a draft with a conditional update on its seq (section 8.4); allowed during a build.

    :return: the new seq.
    :raises StatusTransitionError: ``invalid_transition`` for a pending page.
    :raises DraftConflictError: the draft was saved from another tab.
    :raises StaleRevisionError: the base changed and ``keep_stale`` is not set.
    """
    with immediate(conn):
        page = _require_page(conn, page_id)
        if page["status"] == PageStatus.PENDING:
            raise StatusTransitionError("invalid_transition", "draft is frozen while recognizing")
        base = (data.base_attempt_id, data.base_revision_id)
        current = (page["current_attempt_id"], page["current_revision_id"])
        if base != current and not data.keep_stale:
            raise StaleRevisionError("page changed since the window loaded it")
        # A base that no longer exists cannot be referenced; the draft is stale either way.
        base_attempt = data.base_attempt_id
        if base_attempt is not None and repo.get_attempt(conn, base_attempt) is None:
            base_attempt = None
        base_revision = data.base_revision_id
        if base_revision is not None and repo.get_revision(conn, base_revision) is None:
            base_revision = None
        fields = {
            "item_version": data.item_version,
            "base_attempt_id": base_attempt,
            "base_revision_id": base_revision,
            "text": data.text,
            "printed_label": data.printed_label,
            "choices": data.choices,
            "hints": data.hints,
            "cursor": data.cursor,
            "stale": data.keep_stale,
            "scope": data.scope,
        }
        draft = repo.get_draft(conn, page_id)
        if draft is None:
            if data.expected_seq != 0:
                raise DraftConflictError("draft no longer exists")
            fields["scope"] = data.scope or default_scope(conn, page).value
            repo.insert_draft(conn, page_id, fields)
            recompute_status(conn, str(page["document_id"]))
            return 1
        new_seq = repo.update_draft(conn, page_id, data.expected_seq, fields)
        if new_seq is None:
            raise DraftConflictError("draft was saved from another tab")
        return new_seq


def delete_draft(conn: sqlite3.Connection, page_id: int, draft_seq: int) -> None:
    """Delete a draft ("take new"); a repeated delete does nothing.

    :raises DraftConflictError: the server draft has another seq.
    """
    with immediate(conn):
        page = _require_page(conn, page_id)
        draft = repo.get_draft(conn, page_id)
        if draft is None:
            return
        if int(draft["seq"]) != draft_seq:
            raise DraftConflictError("draft was saved from another tab")
        repo.delete_draft(conn, page_id)
        recompute_status(conn, str(page["document_id"]))


# --- "Done" ------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ReviewInput:
    request_id: str
    item_version: str
    base_attempt_id: int | None
    base_revision_id: int | None
    draft_seq: int
    text: str
    printed_label: str | None = None
    duration_s: int | None = None
    choices: Any = None
    on_conflict: str | None = None


@dataclass(frozen=True)
class ReviewResult:
    revision_id: int
    document_status: str
    created: bool


def submit_review(conn: sqlite3.Connection, page_id: int, data: ReviewInput) -> ReviewResult:
    """Close a page with a revision of its full text (section 8, 10.1).

    :raises StatusTransitionError: ``document_building`` or ``invalid_transition``.
    :raises DraftConflictError: the server draft has another seq than ``draft_seq``.
    :raises StaleRevisionError: the base changed and ``on_conflict`` is not ``keep_mine``.
    """
    with immediate(conn):
        existing = repo.get_revision_by_request(conn, data.request_id)
        page = _require_page(conn, page_id)
        doc = _require_document(conn, str(page["document_id"]))
        if existing is not None:
            return ReviewResult(int(existing["id"]), str(doc["status"]), False)
        if doc["status"] == DocumentStatus.BUILDING:
            raise StatusTransitionError("document_building")
        if page["status"] == PageStatus.PENDING:
            raise StatusTransitionError("invalid_transition", "page is still being recognized")
        draft = repo.get_draft(conn, page_id)
        server_seq = int(draft["seq"]) if draft is not None else 0
        if server_seq != data.draft_seq:
            raise DraftConflictError("draft was saved from another tab")
        current = (page["current_attempt_id"], page["current_revision_id"])
        base = (data.base_attempt_id, data.base_revision_id)
        if base != current and data.on_conflict != "keep_mine":
            raise StaleRevisionError("page changed since the window loaded it")
        if page["current_revision_id"] is not None:
            revision = repo.get_revision(conn, int(page["current_revision_id"]))
            base_text = str(revision["text"]) if revision is not None else None
        elif page["current_attempt_id"] is not None:
            attempt = repo.get_attempt(conn, int(page["current_attempt_id"]))
            base_text = str(attempt["text"]) if attempt is not None else None
        else:
            base_text = None
        scope = (
            draft["scope"]
            if draft is not None and draft["scope"]
            else default_scope(conn, page).value
        )
        label = data.printed_label if data.printed_label is not None else page["printed_label"]
        revision_id = repo.insert_revision(
            conn,
            page_id=page_id,
            request_id=data.request_id,
            base_attempt_id=current[0],
            base_revision_id=current[1],
            action="accept" if data.text == base_text else "edit",
            scope=scope,
            text=data.text,
            printed_label=label,
            choices=data.choices,
            duration_s=data.duration_s,
        )
        repo.mark_page_reviewed(conn, page_id, revision_id, label)
        repo.delete_draft(conn, page_id)
        repo.close_sample(conn, page_id)
        status = recompute_status(conn, str(doc["id"]))
        return ReviewResult(revision_id, status.value, True)


def retry_page(conn: sqlite3.Connection, page_id: int) -> str:
    """Send a failed page back to recognition; its draft is kept (section 8.2).

    :return: the page status after the call.
    :raises StatusTransitionError: ``document_building``.
    """
    with immediate(conn):
        page = _require_page(conn, page_id)
        doc = _require_document(conn, str(page["document_id"]))
        if doc["status"] == DocumentStatus.BUILDING:
            raise StatusTransitionError("document_building")
        if repo.retry_page(conn, page_id):
            recompute_status(conn, str(doc["id"]))
            return PageStatus.PENDING.value
        return str(page["status"])


# --- document requests -------------------------------------------------------------------------


def request_rebuild(conn: sqlite3.Connection, document_id: str) -> str:
    """Set ``rebuild_requested`` and recompute; a failed document is recomputed too."""
    with immediate(conn):
        doc = _require_document(conn, document_id)
        if doc["status"] == DocumentStatus.BUILDING:
            raise StatusTransitionError("document_building")
        repo.set_document_flag(conn, document_id, "rebuild_requested", 1)
        return recompute_status(conn, document_id, force=True).value


def request_redecide(conn: sqlite3.Connection, document_id: str) -> str:
    """Set ``redecide_requested`` and recompute (section 6.11)."""
    with immediate(conn):
        _require_document(conn, document_id)
        repo.set_document_flag(conn, document_id, "redecide_requested", 1)
        return recompute_status(conn, document_id).value


def request_draft_build(conn: sqlite3.Connection, document_id: str) -> int:
    """Create a pending draft build, or return the one already pending or running (9.6)."""
    with immediate(conn):
        doc = _require_document(conn, document_id)
        if doc["status"] not in (DocumentStatus.NEEDS_REVIEW, DocumentStatus.READY):
            raise StatusTransitionError("invalid_transition", "draft needs needs_review or ready")
        active = repo.active_draft_build(conn, document_id)
        if active is not None:
            return int(active["id"])
        build_id, _ = repo.insert_build(conn, document_id, "draft", "pending")
        return build_id


def request_ingest(conn: sqlite3.Connection, document_id: str, settings: Settings) -> str:
    """Request loading into the knowledge base (section 5.1, D9).

    :raises StatusTransitionError: ``ingester_disabled``.
    """
    if not settings.ingester.enabled:
        raise StatusTransitionError("ingester_disabled")
    with immediate(conn):
        doc = _require_document(conn, document_id)
        repo.set_document_flag(conn, document_id, "ingest_requested", 1)
        build = repo.latest_done_build(conn, document_id)
        if build is not None and doc["status"] == DocumentStatus.DONE:
            ingest = repo.get_ingest_for_build(conn, int(build["id"]))
            if ingest is None or ingest["status"] == "failed":
                repo.create_ingest(conn, int(build["id"]))
                repo.set_ingest_status(conn, document_id, "pending")
        refreshed = _require_document(conn, document_id)
        return str(refreshed["ingest_status"])


def audit_sample(
    conn: sqlite3.Connection,
    document_id: str,
    n: int,
    *,
    track: str | None = None,
    rng: random.Random | None = None,
) -> list[int]:
    """Put random ``ok`` pages without flags into a spot check (section 15.4)."""
    chooser = rng or random.Random()
    with immediate(conn):
        _require_document(conn, document_id)
        rows = conn.execute(
            "SELECT id FROM pages WHERE document_id = ? AND status = 'ok' AND flags_json = '[]' "
            "AND (? IS NULL OR track = ?) "
            "AND id NOT IN (SELECT page_id FROM samples WHERE done_at IS NULL) ORDER BY page_index",
            (document_id, track, track),
        ).fetchall()
        ids = [int(r["id"]) for r in rows]
        chosen = sorted(chooser.sample(ids, min(n, len(ids))))
        for page_id in chosen:
            repo.insert_sample(conn, page_id)
        recompute_status(conn, document_id)
    return chosen


# --- queue and summary -------------------------------------------------------------------------


def review_queue(
    conn: sqlite3.Connection, *, limit: int, exclude: list[int]
) -> list[dict[str, Any]]:
    """Queue of section 8.1, excluding pages skipped in this review session."""
    rows = repo.review_queue(conn, exclude, limit)
    result = []
    for row in rows:
        flags = json.loads(row["flags_json"] or "[]")
        if row["has_draft"]:
            scope = row["draft_scope"] or (
                ReviewScope.FULL if row["is_sample"] else ReviewScope.HINTS
            )
        elif row["is_sample"] and row["status"] not in (PageStatus.FLAGGED, PageStatus.FAILED):
            scope = ReviewScope.FULL
        else:
            scope = ReviewScope.HINTS
        if row["status"] == PageStatus.FAILED:
            reasons = [FAILED_REASON, *flags]
        elif flags and row["status"] == PageStatus.FLAGGED:
            reasons = flags
        elif row["is_sample"]:
            reasons = [SPOT_CHECK_REASON]
        else:
            reasons = [DRAFT_REASON]
        result.append(
            {
                "page_id": int(row["page_id"]),
                "document_id": str(row["document_id"]),
                "title": str(row["title"]),
                "page_number": int(row["page_index"]) + 1,
                "scope": str(scope),
                "reasons": reasons,
                "has_draft": bool(row["has_draft"]),
            }
        )
    return result


def _median_review_s(conn: sqlite3.Connection, scope: str, settings: Settings) -> float:
    durations = repo.recent_review_durations(conn, scope)
    return (
        float(statistics.median(durations)) if durations else settings.review.default_page_review_s
    )


def document_view(
    conn: sqlite3.Connection,
    document_id: str,
    settings: Settings,
    *,
    queue: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Document with progress, load, review time and ETA (GET /api/documents/{id}).

    :param queue: queue entries of this document, when the caller already has the queue;
        loaded here otherwise.
    """
    with read_snapshot(conn):
        doc = _require_document(conn, document_id)
        progress = repo.page_progress(conn, document_id)
        flagged, decided = repo.pages_with_flags_count(conn, document_id)
        decided_all = progress["total"] - progress["pending"]
        share = flagged / decided_all if decided_all else 0.0
        if queue is None:
            queue = [
                q
                for q in review_queue(conn, limit=100000, exclude=[])
                if q["document_id"] == document_id
            ]
        n_full = sum(1 for q in queue if q["scope"] == ReviewScope.FULL)
        n_hints = len(queue) - n_full
        est = n_hints * _median_review_s(
            conn, ReviewScope.HINTS, settings
        ) + n_full * _median_review_s(conn, ReviewScope.FULL, settings)
        sec_per_page = repo.average_page_seconds(conn)
        builds = [
            {
                "id": int(b["id"]),
                "kind": b["kind"],
                "version": int(b["version"]),
                "status": b["status"],
                "error": b["error"],
            }
            for b in repo.builds_of_document(conn, document_id)
        ]
        return {
            "id": doc["id"],
            "title": doc["title"],
            "status": doc["status"],
            "page_count": int(doc["page_count"]),
            "progress": progress,
            "flagged_share": round(share, 4),
            "over_guideline": share > settings.review.load_guideline,
            "to_review": len(queue),
            "est_review_sec": round(est),
            "no_logprobs_pages": repo.count_no_logprobs(conn, document_id),
            "sec_per_page": sec_per_page,
            "eta_sec": _eta_sec(conn, document_id, sec_per_page),
            "builds": builds,
            "ingest_status": doc["ingest_status"],
            "delivered_version": doc["delivered_version"],
            "export_name": doc["export_name"],
            "export_error": doc["export_error"],
            "error": doc["error"],
            "created_at": doc["created_at"],
            "decided_vlm_pages": decided,
        }


def _eta_by_document(conn: sqlite3.Connection, sec_per_page: float) -> dict[str, int]:
    """ETA of every processing or queued document: pending pages of the document and of the
    documents ahead of it, times seconds per page; one count query per document."""
    ahead = 0
    result: dict[str, int] = {}
    for row in repo.documents_in_status(conn, [DocumentStatus.PROCESSING, DocumentStatus.QUEUED]):
        ahead += repo.count_pending(conn, str(row["id"]))
        result[str(row["id"])] = round(ahead * sec_per_page)
    return result


def _eta_sec(conn: sqlite3.Connection, document_id: str, sec_per_page: float | None) -> int | None:
    """Pending pages of this document and of the documents ahead of it, times seconds per page."""
    if sec_per_page is None:
        return None
    return _eta_by_document(conn, sec_per_page).get(document_id, 0)


def list_documents(
    conn: sqlite3.Connection, status: str | None, settings: Settings
) -> list[dict[str, Any]]:
    """Short list of documents with progress (GET /api/documents)."""
    with read_snapshot(conn):
        sec_per_page = repo.average_page_seconds(conn)
        etas = _eta_by_document(conn, sec_per_page) if sec_per_page is not None else {}
        return [
            {
                "id": row["id"],
                "title": row["title"],
                "status": row["status"],
                "page_count": int(row["page_count"]),
                "progress": repo.page_progress(conn, str(row["id"])),
                "eta_sec": etas.get(str(row["id"]))
                if row["status"] in (DocumentStatus.QUEUED, DocumentStatus.PROCESSING)
                else None,
                "created_at": row["created_at"],
            }
            for row in repo.list_documents(conn, status)
        ]


def review_summary(
    conn: sqlite3.Connection, settings: Settings, rules_version: str
) -> dict[str, Any]:
    """Data of "What's in the queue" and of the review start (sections 8.1, 15.3)."""
    with read_snapshot(conn):
        queue = review_queue(conn, limit=100000, exclude=[])
        per_doc: dict[str, list[dict[str, Any]]] = {}
        for entry in queue:
            per_doc.setdefault(entry["document_id"], []).append(entry)
        documents = []
        for row in repo.list_documents(conn):
            doc_id = str(row["id"])
            if (
                row["status"] == DocumentStatus.DONE
                and doc_id not in per_doc
                and not row["export_error"]
            ):
                continue
            view = document_view(conn, doc_id, settings, queue=per_doc.get(doc_id, []))
            documents.append(
                {
                    key: view[key]
                    for key in (
                        "id",
                        "title",
                        "status",
                        "progress",
                        "eta_sec",
                        "to_review",
                        "flagged_share",
                        "over_guideline",
                        "est_review_sec",
                        "no_logprobs_pages",
                        "export_error",
                        "error",
                    )
                }
                | {"drafts": view["progress"]["drafts"]}
            )
        cutoff = (datetime.now(UTC) - timedelta(days=settings.review.stale_draft_days)).isoformat(
            timespec="seconds"
        )
        now = datetime.now(UTC)
        stale = [
            {
                "document_id": row["document_id"],
                "title": row["title"],
                "page_number": int(row["page_index"]) + 1,
                "age_days": (now - datetime.fromisoformat(str(row["created_at"]))).days,
            }
            for row in repo.drafts_created_before(conn, cutoff)
        ]
        counts = repo.suppressed_flag_counts(conn)
        disabled = [
            f for f in FlagCode if f not in settings.decide.flags and f not in MANDATORY_FLAGS
        ]
        suppressed = [{"flag": f.value, "pages": counts.get(f.value, 0)} for f in disabled]
        waiting = []
        for row in repo.list_documents(conn):
            if row["redecide_requested"]:
                drafts = repo.count_draft_pages_waiting_redecide(
                    conn, str(row["id"]), rules_version
                )
                if drafts:
                    waiting.append(
                        {"document_id": row["id"], "title": row["title"], "draft_pages": drafts}
                    )
        n_full = sum(1 for q in queue if q["scope"] == ReviewScope.FULL)
        est = (len(queue) - n_full) * _median_review_s(
            conn, ReviewScope.HINTS, settings
        ) + n_full * _median_review_s(conn, ReviewScope.FULL, settings)
        return {
            "queue_total": len(queue),
            "est_review_sec": round(est),
            "load_guideline": settings.review.load_guideline,
            "documents": documents,
            "stale_drafts": stale,
            "suppressed_rules": suppressed,
            "waiting_redecide": waiting,
            "generated_at": utc_now(),
        }
