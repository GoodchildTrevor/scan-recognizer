"""All SQL queries of the service. Parameters are always bound, never formatted into SQL."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import asdict
from typing import Any

from ocr_archive.db.connection import utc_now
from ocr_archive.models import (
    BuildKind,
    BuildStatus,
    DocumentStatus,
    IngestStatus,
    PageStatus,
    TokenProbs,
    VlmResult,
)

Row = sqlite3.Row


def _placeholders(count: int) -> str:
    return ",".join("?" * count)


# --- control -----------------------------------------------------------------------------------


def get_control(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM control WHERE key = ?", (key,)).fetchone()
    return None if row is None else str(row["value"])


def set_control(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO control (key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )


# --- profiles ----------------------------------------------------------------------------------


def ensure_profile(conn: sqlite3.Connection, profile_id: str, name: str, config_json: str) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO profiles (id, name, config_json, created_at) VALUES (?, ?, ?, ?)",
        (profile_id, name, config_json, utc_now()),
    )


# --- documents ---------------------------------------------------------------------------------


def find_document_by_sha(conn: sqlite3.Connection, sha256: str) -> Row | None:
    return conn.execute("SELECT * FROM documents WHERE source_sha256 = ?", (sha256,)).fetchone()


def insert_document(
    conn: sqlite3.Connection,
    *,
    document_id: str,
    title: str,
    source_sha256: str,
    source_path: str,
    original_name: str,
    page_count: int,
    priority: int,
    ingest_requested: bool,
) -> None:
    """Insert a queued document and its pending pages."""
    now = utc_now()
    conn.execute(
        "INSERT INTO documents (id, title, source_sha256, source_path, original_name, page_count, "
        "status, priority, ingest_requested, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            document_id,
            title,
            source_sha256,
            source_path,
            original_name,
            page_count,
            DocumentStatus.QUEUED,
            priority,
            int(ingest_requested),
            now,
            now,
        ),
    )
    conn.executemany(
        "INSERT INTO pages (document_id, page_index, updated_at) VALUES (?, ?, ?)",
        [(document_id, index, now) for index in range(page_count)],
    )


def get_document(conn: sqlite3.Connection, document_id: str) -> Row | None:
    return conn.execute("SELECT * FROM documents WHERE id = ?", (document_id,)).fetchone()


def list_documents(conn: sqlite3.Connection, status: str | None = None) -> list[Row]:
    if status:
        return conn.execute(
            "SELECT * FROM documents WHERE status = ? ORDER BY priority DESC, created_at",
            (status,),
        ).fetchall()
    return conn.execute("SELECT * FROM documents ORDER BY priority DESC, created_at").fetchall()


def set_document_status(
    conn: sqlite3.Connection, document_id: str, status: str, *, error: str | None = None
) -> None:
    conn.execute(
        "UPDATE documents SET status = ?, error = ?, updated_at = ? WHERE id = ?",
        (status, error, utc_now(), document_id),
    )


def transition_document(
    conn: sqlite3.Connection, document_id: str, from_status: str, to_status: str
) -> bool:
    """Conditionally move a document from one status to another; return whether it moved."""
    cur = conn.execute(
        "UPDATE documents SET status = ?, updated_at = ? WHERE id = ? AND status = ?",
        (to_status, utc_now(), document_id, from_status),
    )
    return cur.rowcount == 1


def set_document_flag(conn: sqlite3.Connection, document_id: str, column: str, value: int) -> None:
    """Set one of the boolean request columns of a document."""
    if column not in {"rebuild_requested", "redecide_requested", "ingest_requested"}:
        raise ValueError(f"unknown document flag {column}")
    conn.execute(
        f"UPDATE documents SET {column} = ?, updated_at = ? WHERE id = ?",
        (value, utc_now(), document_id),
    )


def set_ingest_status(conn: sqlite3.Connection, document_id: str, status: str) -> None:
    conn.execute(
        "UPDATE documents SET ingest_status = ?, updated_at = ? WHERE id = ?",
        (status, utc_now(), document_id),
    )


def next_document_for_work(conn: sqlite3.Connection) -> Row | None:
    """Return the next document of step 6: processing first, then queued; by priority and age."""
    return conn.execute(
        "SELECT * FROM documents WHERE status IN ('processing', 'queued') "
        "ORDER BY CASE status WHEN 'processing' THEN 0 ELSE 1 END, priority DESC, created_at "
        "LIMIT 1"
    ).fetchone()


def documents_in_status(conn: sqlite3.Connection, statuses: Iterable[str]) -> list[Row]:
    values = list(statuses)
    return conn.execute(
        f"SELECT * FROM documents WHERE status IN ({_placeholders(len(values))}) "
        "ORDER BY priority DESC, created_at",
        values,
    ).fetchall()


def set_delivered(conn: sqlite3.Connection, document_id: str, version: int) -> None:
    conn.execute(
        "UPDATE documents SET delivered_version = MAX(COALESCE(delivered_version, 0), ?), "
        "updated_at = ? WHERE id = ?",
        (version, utc_now(), document_id),
    )


def deliveries(conn: sqlite3.Connection) -> list[Row]:
    """Documents with a final build newer than the delivered one."""
    return conn.execute(
        "SELECT d.id AS document_id, d.title, MAX(b.version) AS version FROM documents d "
        "JOIN builds b ON b.document_id = d.id AND b.kind = 'final' AND b.status = 'done' "
        "AND b.pruned_at IS NULL "
        "GROUP BY d.id HAVING MAX(b.version) > COALESCE(d.delivered_version, 0) "
        "ORDER BY d.priority DESC, d.created_at"
    ).fetchall()


# --- pages -------------------------------------------------------------------------------------


def get_page(conn: sqlite3.Connection, page_id: int) -> Row | None:
    return conn.execute("SELECT * FROM pages WHERE id = ?", (page_id,)).fetchone()


def get_page_by_index(conn: sqlite3.Connection, document_id: str, page_index: int) -> Row | None:
    return conn.execute(
        "SELECT * FROM pages WHERE document_id = ? AND page_index = ?", (document_id, page_index)
    ).fetchone()


def pages_of_document(conn: sqlite3.Connection, document_id: str) -> list[Row]:
    return conn.execute(
        "SELECT * FROM pages WHERE document_id = ? ORDER BY page_index", (document_id,)
    ).fetchall()


def count_pages_without_track(conn: sqlite3.Connection, document_id: str) -> int:
    row = conn.execute(
        "SELECT COUNT(*) FROM pages WHERE document_id = ? AND track IS NULL", (document_id,)
    ).fetchone()
    return int(row[0])


def set_page_track(
    conn: sqlite3.Connection,
    page_id: int,
    *,
    text_layer: str,
    track: str,
    signals: dict[str, Any] | None,
) -> None:
    conn.execute(
        "UPDATE pages SET text_layer = ?, track = ?, signals_json = ?, updated_at = ? WHERE id = ?",
        (text_layer, track, _dumps_or_none(signals), utc_now(), page_id),
    )


def next_pending_pages(
    conn: sqlite3.Connection, document_id: str, now: str, limit: int
) -> list[Row]:
    return conn.execute(
        "SELECT * FROM pages WHERE document_id = ? AND status = 'pending' "
        "AND (retry_at IS NULL OR retry_at <= ?) ORDER BY page_index LIMIT ?",
        (document_id, now, limit),
    ).fetchall()


def count_pending(conn: sqlite3.Connection, document_id: str) -> int:
    row = conn.execute(
        "SELECT COUNT(*) FROM pages WHERE document_id = ? AND status = 'pending'", (document_id,)
    ).fetchone()
    return int(row[0])


def earliest_retry_at(conn: sqlite3.Connection) -> str | None:
    row = conn.execute(
        "SELECT MIN(retry_at) FROM pages WHERE status = 'pending' AND retry_at IS NOT NULL"
    ).fetchone()
    return None if row[0] is None else str(row[0])


_NEEDS_REDECIDE = (
    "p.track = 'vlm' AND p.status IN ('ok', 'flagged') AND p.current_revision_id IS NULL "
    "AND NOT EXISTS (SELECT 1 FROM drafts d WHERE d.page_id = p.id) "
    "AND (p.rules_version IS NULL OR p.rules_version != ?)"
)


def count_pages_to_redecide(conn: sqlite3.Connection, document_id: str, rules_version: str) -> int:
    row = conn.execute(
        f"SELECT COUNT(*) FROM pages p WHERE p.document_id = ? AND {_NEEDS_REDECIDE}",
        (document_id, rules_version),
    ).fetchone()
    return int(row[0])


def next_page_to_redecide(
    conn: sqlite3.Connection, document_id: str, rules_version: str
) -> Row | None:
    return conn.execute(
        f"SELECT p.* FROM pages p WHERE p.document_id = ? AND {_NEEDS_REDECIDE} "
        "ORDER BY p.page_index LIMIT 1",
        (document_id, rules_version),
    ).fetchone()


def count_draft_pages_waiting_redecide(
    conn: sqlite3.Connection, document_id: str, rules_version: str
) -> int:
    """Pages that would need re-deciding but are held by a draft (section 6.11)."""
    row = conn.execute(
        "SELECT COUNT(*) FROM pages p JOIN drafts d ON d.page_id = p.id "
        "WHERE p.document_id = ? AND p.track = 'vlm' AND p.status IN ('ok', 'flagged') "
        "AND p.current_revision_id IS NULL "
        "AND (p.rules_version IS NULL OR p.rules_version != ?)",
        (document_id, rules_version),
    ).fetchone()
    return int(row[0])


def set_line_boxes(conn: sqlite3.Connection, page_id: int, boxes_json: str, version: str) -> None:
    conn.execute(
        "UPDATE pages SET line_boxes_json = ?, line_boxes_version = ? WHERE id = ?",
        (boxes_json, version, page_id),
    )


def schedule_retry(conn: sqlite3.Connection, page_id: int, generation: int, retry_at: str) -> bool:
    cur = conn.execute(
        "UPDATE pages SET retry_at = ?, retry_count = retry_count + 1, updated_at = ? "
        "WHERE id = ? AND status = 'pending' AND ocr_generation = ?",
        (retry_at, utc_now(), page_id, generation),
    )
    return cur.rowcount == 1


def clear_retry(conn: sqlite3.Connection, page_id: int) -> None:
    conn.execute(
        "UPDATE pages SET retry_at = NULL, retry_count = 0 WHERE id = ? AND retry_count > 0",
        (page_id,),
    )


def write_decision(
    conn: sqlite3.Connection,
    page_id: int,
    *,
    expected_status: str,
    expected_generation: int,
    allow_draft: bool,
    status: str,
    candidate_attempt_id: int | None,
    flags: Sequence[str],
    signals: dict[str, Any] | None,
    review: dict[str, Any] | None,
    rules_version: str | None,
    error: str | None = None,
) -> bool:
    """Write a page decision conditionally (section 6.11).

    The update applies only if nobody pressed "Done", started a draft, or sent the page to
    reprocess/retry since the worker read it. A draft does not block a page in ``pending``.

    :return: whether the decision was written.
    """
    draft_clause = (
        ""
        if allow_draft
        else (" AND NOT EXISTS (SELECT 1 FROM drafts d WHERE d.page_id = pages.id)")
    )
    cur = conn.execute(
        "UPDATE pages SET status = ?, current_attempt_id = ?, flags_json = ?, signals_json = ?, "
        "review_json = ?, rules_version = ?, error = ?, retry_at = NULL, retry_count = 0, "
        "updated_at = ? "
        "WHERE id = ? AND current_revision_id IS NULL AND status = ? AND ocr_generation = ?"
        + draft_clause,
        (
            status,
            candidate_attempt_id,
            json.dumps(sorted(flags)),
            _dumps_or_none(signals),
            _dumps_or_none(review),
            rules_version,
            error,
            utc_now(),
            page_id,
            expected_status,
            expected_generation,
        ),
    )
    return cur.rowcount == 1


def update_review_json(conn: sqlite3.Connection, page_id: int, review: dict[str, Any]) -> None:
    conn.execute("UPDATE pages SET review_json = ? WHERE id = ?", (json.dumps(review), page_id))


def retry_page(conn: sqlite3.Connection, page_id: int) -> bool:
    """Move a failed page back to pending with a new OCR generation."""
    cur = conn.execute(
        "UPDATE pages SET status = 'pending', ocr_generation = ocr_generation + 1, "
        "retry_at = NULL, retry_count = 0, error = NULL, updated_at = ? "
        "WHERE id = ? AND status = 'failed'",
        (utc_now(), page_id),
    )
    return cur.rowcount == 1


def reprocess_pages(
    conn: sqlite3.Connection,
    document_id: str,
    *,
    include_reviewed: bool,
    include_pdf_text: bool = False,
) -> list[int]:
    """Send vlm pages back to pending with a new generation (section 6.11); return their ids.

    With ``include_pdf_text`` the text-layer pages go back too: their text is taken from the
    PDF again, which a change of the extraction code requires.
    """
    statuses = ["ok", "flagged", "failed"] + (["reviewed"] if include_reviewed else [])
    tracks = ["vlm"] + (["pdf_text"] if include_pdf_text else [])
    rows = conn.execute(
        f"SELECT id FROM pages WHERE document_id = ? "
        f"AND track IN ({_placeholders(len(tracks))}) "
        f"AND status IN ({_placeholders(len(statuses))})",
        (document_id, *tracks, *statuses),
    ).fetchall()
    ids = [int(r["id"]) for r in rows]
    if not ids:
        return ids
    marks = _placeholders(len(ids))
    revision_clause = ", current_revision_id = NULL" if include_reviewed else ""
    conn.execute(
        "UPDATE pages SET status = 'pending', ocr_generation = ocr_generation + 1, "
        "current_attempt_id = NULL, flags_json = '[]', signals_json = NULL, review_json = NULL, "
        "rules_version = NULL, retry_at = NULL, retry_count = 0, error = NULL, updated_at = ?"
        f"{revision_clause} WHERE id IN ({marks})",
        (utc_now(), *ids),
    )
    conn.execute(f"DELETE FROM drafts WHERE page_id IN ({marks})", ids)
    return ids


def page_progress(conn: sqlite3.Connection, document_id: str) -> dict[str, int]:
    """Counts of pages by status, drafts and open spot checks."""
    counts = {status.value: 0 for status in PageStatus}
    for row in conn.execute(
        "SELECT status, COUNT(*) AS n FROM pages WHERE document_id = ? GROUP BY status",
        (document_id,),
    ):
        counts[str(row["status"])] = int(row["n"])
    counts["total"] = sum(counts[s.value] for s in PageStatus)
    counts["drafts"] = int(
        conn.execute(
            "SELECT COUNT(*) FROM drafts d JOIN pages p ON p.id = d.page_id "
            "WHERE p.document_id = ?",
            (document_id,),
        ).fetchone()[0]
    )
    counts["samples_open"] = int(
        conn.execute(
            "SELECT COUNT(*) FROM samples s JOIN pages p ON p.id = s.page_id "
            "WHERE p.document_id = ? AND s.done_at IS NULL",
            (document_id,),
        ).fetchone()[0]
    )
    return counts


def status_inputs(conn: sqlite3.Connection, document_id: str) -> dict[str, int]:
    """Counters used by the status recomputation of section 5.1."""
    row = conn.execute(
        "SELECT "
        "SUM(status = 'pending') AS pending, "
        "SUM(status IN ('flagged', 'failed')) AS open_flags, "
        "(SELECT COUNT(*) FROM drafts d JOIN pages p2 ON p2.id = d.page_id "
        " WHERE p2.document_id = ?1) AS drafts, "
        "(SELECT COUNT(*) FROM samples s JOIN pages p3 ON p3.id = s.page_id "
        " WHERE p3.document_id = ?1 AND s.done_at IS NULL) AS samples_open "
        "FROM pages WHERE document_id = ?1",
        (document_id,),
    ).fetchone()
    return {key: int(row[key] or 0) for key in ("pending", "open_flags", "drafts", "samples_open")}


def page_texts(conn: sqlite3.Connection, document_id: str) -> list[Row]:
    """Current text source of every page: revision first, then candidate attempt (section 5.2)."""
    return conn.execute(
        "SELECT p.id AS page_id, p.page_index, p.printed_label, p.status, p.track, p.text_layer, "
        "p.current_attempt_id, p.current_revision_id, p.flags_json, p.signals_json, "
        "p.review_json, p.rules_version, p.line_boxes_json, "
        "r.text AS revision_text, r.scope AS revision_scope, a.text AS attempt_text, "
        "a.profile_id AS attempt_profile_id "
        "FROM pages p "
        "LEFT JOIN revisions r ON r.id = p.current_revision_id "
        "LEFT JOIN attempts a ON a.id = p.current_attempt_id "
        "WHERE p.document_id = ? ORDER BY p.page_index",
        (document_id,),
    ).fetchall()


# --- attempts ----------------------------------------------------------------------------------


def token_probs_to_json(probs: TokenProbs | None) -> str | None:
    if probs is None:
        return None
    return json.dumps(
        {
            "counts": {"letter": probs.letter_tokens, "digit": probs.digit_tokens},
            "tokens": [list(t) for t in probs.tokens],
        },
        ensure_ascii=False,
    )


def token_probs_from_json(raw: str | None) -> TokenProbs | None:
    if raw is None:
        return None
    data = json.loads(raw)
    return TokenProbs(
        letter_tokens=int(data["counts"]["letter"]),
        digit_tokens=int(data["counts"]["digit"]),
        tokens=[(int(s), int(e), float(p), str(k)) for s, e, p, k in data["tokens"]],
    )


def insert_attempt(
    conn: sqlite3.Connection,
    *,
    page_id: int,
    profile_id: str,
    generation: int,
    pass_no: int,
    role: str,
    region: dict[str, Any] | None,
    result: VlmResult,
) -> int | None:
    """Insert an attempt if the page is still at the given OCR generation; return its id."""
    cur = conn.execute(
        "INSERT INTO attempts (page_id, profile_id, generation, pass_no, role, region_json, text, "
        "finish_reason, tokens_in, tokens_out, duration_ms, token_probs_json, created_at) "
        "SELECT ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ? "
        "WHERE EXISTS (SELECT 1 FROM pages WHERE id = ? AND ocr_generation = ?)",
        (
            page_id,
            profile_id,
            generation,
            pass_no,
            role,
            _dumps_or_none(region),
            result.text,
            result.finish_reason,
            result.tokens_in,
            result.tokens_out,
            result.duration_ms,
            token_probs_to_json(result.token_probs),
            utc_now(),
            page_id,
            generation,
        ),
    )
    return int(cur.lastrowid) if cur.rowcount == 1 and cur.lastrowid is not None else None


def find_attempts(
    conn: sqlite3.Connection, page_id: int, profile_id: str, generation: int, pass_no: int
) -> list[Row]:
    """Saved attempts of one pass that can be reused after a restart (section 6.4)."""
    return conn.execute(
        "SELECT * FROM attempts WHERE page_id = ? AND profile_id = ? AND generation = ? "
        "AND pass_no = ? ORDER BY id",
        (page_id, profile_id, generation, pass_no),
    ).fetchall()


def get_attempt(conn: sqlite3.Connection, attempt_id: int) -> Row | None:
    return conn.execute("SELECT * FROM attempts WHERE id = ?", (attempt_id,)).fetchone()


def latest_decision_attempts(conn: sqlite3.Connection, page_id: int, generation: int) -> list[Row]:
    """Attempts of the generation that take part in the decision: full, combined and pdf_text."""
    return conn.execute(
        "SELECT * FROM attempts WHERE page_id = ? AND generation = ? "
        "AND role IN ('full', 'combined', 'pdf_text') ORDER BY id",
        (page_id, generation),
    ).fetchall()


def attempt_to_result(row: Row) -> VlmResult:
    return VlmResult(
        text=str(row["text"]),
        finish_reason=row["finish_reason"],
        tokens_in=row["tokens_in"],
        tokens_out=row["tokens_out"],
        duration_ms=int(row["duration_ms"] or 0),
        token_probs=token_probs_from_json(row["token_probs_json"]),
    )


def average_page_seconds(conn: sqlite3.Connection, limit: int = 200) -> float | None:
    """Mean model time per page over the latest attempts, for ETA estimates."""
    row = conn.execute(
        "SELECT AVG(s) FROM (SELECT SUM(duration_ms) / 1000.0 AS s FROM attempts "
        "WHERE role IN ('full', 'part') GROUP BY page_id, generation "
        "ORDER BY MAX(id) DESC LIMIT ?)",
        (limit,),
    ).fetchone()
    return None if row[0] is None else float(row[0])


# --- revisions ---------------------------------------------------------------------------------


def get_revision(conn: sqlite3.Connection, revision_id: int) -> Row | None:
    return conn.execute("SELECT * FROM revisions WHERE id = ?", (revision_id,)).fetchone()


def get_revision_by_request(conn: sqlite3.Connection, request_id: str) -> Row | None:
    return conn.execute("SELECT * FROM revisions WHERE request_id = ?", (request_id,)).fetchone()


def insert_revision(
    conn: sqlite3.Connection,
    *,
    page_id: int,
    request_id: str,
    base_attempt_id: int | None,
    base_revision_id: int | None,
    action: str,
    scope: str,
    text: str,
    printed_label: str | None,
    choices: Any,
    duration_s: int | None,
) -> int:
    cur = conn.execute(
        "INSERT INTO revisions (page_id, request_id, base_attempt_id, base_revision_id, action, "
        "scope, text, printed_label, choices_json, duration_s, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            page_id,
            request_id,
            base_attempt_id,
            base_revision_id,
            action,
            scope,
            text,
            printed_label,
            _dumps_or_none(choices),
            duration_s,
            utc_now(),
        ),
    )
    return int(cur.lastrowid or 0)


def mark_page_reviewed(
    conn: sqlite3.Connection, page_id: int, revision_id: int, printed_label: str | None
) -> None:
    conn.execute(
        "UPDATE pages SET status = 'reviewed', current_revision_id = ?, printed_label = ?, "
        "updated_at = ? WHERE id = ?",
        (revision_id, printed_label, utc_now(), page_id),
    )


def recent_review_durations(conn: sqlite3.Connection, scope: str, limit: int = 200) -> list[int]:
    rows = conn.execute(
        "SELECT duration_s FROM revisions WHERE scope = ? AND duration_s IS NOT NULL "
        "ORDER BY id DESC LIMIT ?",
        (scope, limit),
    ).fetchall()
    return [int(r["duration_s"]) for r in rows]


def revisions_of_document(conn: sqlite3.Connection, document_id: str) -> list[Row]:
    return conn.execute(
        "SELECT r.*, p.page_index, p.flags_json, p.signals_json, p.track FROM revisions r "
        "JOIN pages p ON p.id = r.page_id WHERE p.document_id = ? ORDER BY r.id",
        (document_id,),
    ).fetchall()


# --- drafts ------------------------------------------------------------------------------------


def get_draft(conn: sqlite3.Connection, page_id: int) -> Row | None:
    return conn.execute("SELECT * FROM drafts WHERE page_id = ?", (page_id,)).fetchone()


def insert_draft(conn: sqlite3.Connection, page_id: int, fields: dict[str, Any]) -> int:
    now = utc_now()
    conn.execute(
        "INSERT INTO drafts (page_id, item_version, base_attempt_id, base_revision_id, text, "
        "printed_label, choices_json, hints_json, cursor, scope, seq, stale, created_at, "
        "updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?)",
        (
            page_id,
            fields["item_version"],
            fields["base_attempt_id"],
            fields["base_revision_id"],
            fields["text"],
            fields["printed_label"],
            _dumps_or_none(fields.get("choices")),
            _dumps_or_none(fields.get("hints")),
            fields.get("cursor"),
            fields.get("scope"),
            int(bool(fields.get("stale"))),
            now,
            now,
        ),
    )
    return 1


def update_draft(
    conn: sqlite3.Connection, page_id: int, expected_seq: int, fields: dict[str, Any]
) -> int | None:
    """Conditionally update a draft at ``expected_seq``; return the new seq or None."""
    cur = conn.execute(
        "UPDATE drafts SET item_version = ?, base_attempt_id = ?, base_revision_id = ?, text = ?, "
        "printed_label = ?, choices_json = ?, hints_json = ?, cursor = ?, "
        "scope = COALESCE(?, scope), seq = seq + 1, stale = ?, updated_at = ? "
        "WHERE page_id = ? AND seq = ?",
        (
            fields["item_version"],
            fields["base_attempt_id"],
            fields["base_revision_id"],
            fields["text"],
            fields["printed_label"],
            _dumps_or_none(fields.get("choices")),
            _dumps_or_none(fields.get("hints")),
            fields.get("cursor"),
            fields.get("scope"),
            int(bool(fields.get("stale"))),
            utc_now(),
            page_id,
            expected_seq,
        ),
    )
    return expected_seq + 1 if cur.rowcount == 1 else None


def delete_draft(conn: sqlite3.Connection, page_id: int) -> None:
    conn.execute("DELETE FROM drafts WHERE page_id = ?", (page_id,))


def drafts_created_before(conn: sqlite3.Connection, cutoff: str) -> list[Row]:
    return conn.execute(
        "SELECT d.page_id, d.created_at, p.page_index, p.document_id, doc.title FROM drafts d "
        "JOIN pages p ON p.id = d.page_id JOIN documents doc ON doc.id = p.document_id "
        "WHERE d.created_at < ? ORDER BY d.created_at",
        (cutoff,),
    ).fetchall()


# --- samples -----------------------------------------------------------------------------------


def insert_sample(conn: sqlite3.Connection, page_id: int) -> bool:
    cur = conn.execute(
        "INSERT INTO samples (page_id, created_at) VALUES (?, ?) "
        "ON CONFLICT(page_id) DO UPDATE SET done_at = NULL, created_at = excluded.created_at "
        "WHERE samples.done_at IS NOT NULL",
        (page_id, utc_now()),
    )
    return cur.rowcount == 1


def close_sample(conn: sqlite3.Connection, page_id: int) -> None:
    conn.execute(
        "UPDATE samples SET done_at = ? WHERE page_id = ? AND done_at IS NULL",
        (utc_now(), page_id),
    )


def open_sample(conn: sqlite3.Connection, page_id: int) -> bool:
    row = conn.execute(
        "SELECT 1 FROM samples WHERE page_id = ? AND done_at IS NULL", (page_id,)
    ).fetchone()
    return row is not None


def samples_of_document(conn: sqlite3.Connection, document_id: str | None) -> list[Row]:
    query = (
        "SELECT s.*, p.page_index, p.document_id, p.track, p.signals_json, p.flags_json, "
        "p.current_revision_id, p.current_attempt_id FROM samples s JOIN pages p "
        "ON p.id = s.page_id"
    )
    if document_id is None:
        return conn.execute(query + " ORDER BY s.id").fetchall()
    return conn.execute(query + " WHERE p.document_id = ? ORDER BY s.id", (document_id,)).fetchall()


# --- review queue ------------------------------------------------------------------------------


def review_queue(conn: sqlite3.Connection, exclude: Sequence[int], limit: int) -> list[Row]:
    """Pages in the order of section 8.1: started drafts, flagged/failed, spot checks."""
    exclude_clause = f" AND p.id NOT IN ({_placeholders(len(exclude))})" if exclude else ""
    query = (
        "SELECT p.id AS page_id, p.document_id, p.page_index, p.status, p.flags_json, "
        "doc.title, d.page_id IS NOT NULL AS has_draft, d.scope AS draft_scope, "
        "s.page_id IS NOT NULL AS is_sample, "
        "CASE WHEN d.page_id IS NOT NULL THEN 1 "
        "     WHEN p.status IN ('flagged', 'failed') THEN 2 "
        "     ELSE 3 END AS grp "
        "FROM pages p JOIN documents doc ON doc.id = p.document_id "
        "LEFT JOIN drafts d ON d.page_id = p.id "
        "LEFT JOIN samples s ON s.page_id = p.id AND s.done_at IS NULL "
        "WHERE doc.status IN ('processing', 'needs_review', 'done') AND p.status != 'pending' "
        "AND (d.page_id IS NOT NULL OR p.status IN ('flagged', 'failed') OR s.page_id IS NOT NULL)"
        + exclude_clause
        + " ORDER BY grp, doc.priority DESC, doc.created_at, p.page_index LIMIT ?"
    )
    return conn.execute(query, (*exclude, limit)).fetchall()


def pages_with_flags_count(conn: sqlite3.Connection, document_id: str) -> tuple[int, int]:
    """Return (pages flagged at decision, decided vlm pages) for flagged_share (section 5.1)."""
    row = conn.execute(
        "SELECT SUM(flags_json != '[]') AS flagged, "
        "SUM(track = 'vlm' AND status != 'pending') AS decided "
        "FROM pages WHERE document_id = ?",
        (document_id,),
    ).fetchone()
    return int(row["flagged"] or 0), int(row["decided"] or 0)


def count_no_logprobs(conn: sqlite3.Connection, document_id: str) -> int:
    row = conn.execute(
        "SELECT COUNT(*) FROM pages WHERE document_id = ? "
        "AND json_extract(signals_json, '$.no_logprobs') = 1",
        (document_id,),
    ).fetchone()
    return int(row[0])


def suppressed_flag_counts(conn: sqlite3.Connection) -> dict[str, int]:
    """Pages per switched-off flag that did not reach the queue because of it (section 6.9)."""
    counts: dict[str, int] = {}
    rows = conn.execute(
        "SELECT value AS flag, COUNT(*) AS n FROM pages, "
        "json_each(json_extract(pages.signals_json, '$.suppressed_flags')) "
        "WHERE pages.signals_json IS NOT NULL GROUP BY value"
    ).fetchall()
    for row in rows:
        counts[str(row["flag"])] = int(row["n"])
    return counts


# --- builds ------------------------------------------------------------------------------------


def next_build_version(conn: sqlite3.Connection, document_id: str) -> int:
    row = conn.execute(
        "SELECT COALESCE(MAX(version), 0) + 1 FROM builds WHERE document_id = ?", (document_id,)
    ).fetchone()
    return int(row[0])


def insert_build(
    conn: sqlite3.Connection, document_id: str, kind: str, status: str
) -> tuple[int, int]:
    """Insert a build with the next version; return (build id, version)."""
    version = next_build_version(conn, document_id)
    cur = conn.execute(
        "INSERT INTO builds (document_id, version, kind, status, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (document_id, version, kind, status, utc_now()),
    )
    return int(cur.lastrowid or 0), version


def get_build(conn: sqlite3.Connection, build_id: int) -> Row | None:
    return conn.execute("SELECT * FROM builds WHERE id = ?", (build_id,)).fetchone()


def builds_of_document(conn: sqlite3.Connection, document_id: str) -> list[Row]:
    return conn.execute(
        "SELECT * FROM builds WHERE document_id = ? ORDER BY version", (document_id,)
    ).fetchall()


def latest_done_build(
    conn: sqlite3.Connection, document_id: str, kind: str = BuildKind.FINAL
) -> Row | None:
    return conn.execute(
        "SELECT * FROM builds WHERE document_id = ? AND kind = ? AND status = 'done' "
        "AND pruned_at IS NULL ORDER BY version DESC LIMIT 1",
        (document_id, kind),
    ).fetchone()


def latest_final_snapshot_build(conn: sqlite3.Connection, document_id: str) -> Row | None:
    """Last successful final build, even if its files were pruned: its snapshot defines rule 4."""
    return conn.execute(
        "SELECT * FROM builds WHERE document_id = ? AND kind = 'final' AND status = 'done' "
        "ORDER BY version DESC LIMIT 1",
        (document_id,),
    ).fetchone()


def build_by_version(conn: sqlite3.Connection, document_id: str, version: int) -> Row | None:
    return conn.execute(
        "SELECT * FROM builds WHERE document_id = ? AND version = ?", (document_id, version)
    ).fetchone()


def active_draft_build(conn: sqlite3.Connection, document_id: str) -> Row | None:
    return conn.execute(
        "SELECT * FROM builds WHERE document_id = ? AND kind = 'draft' "
        "AND status IN ('pending', 'running') ORDER BY id LIMIT 1",
        (document_id,),
    ).fetchone()


def next_pending_draft_build(conn: sqlite3.Connection) -> Row | None:
    return conn.execute(
        "SELECT * FROM builds WHERE kind = 'draft' AND status = 'pending' ORDER BY id LIMIT 1"
    ).fetchone()


def next_ready_document(conn: sqlite3.Connection) -> Row | None:
    return conn.execute(
        "SELECT * FROM documents WHERE status = 'ready' ORDER BY priority DESC, created_at LIMIT 1"
    ).fetchone()


def set_build_running(conn: sqlite3.Connection, build_id: int) -> bool:
    cur = conn.execute(
        "UPDATE builds SET status = 'running' WHERE id = ? AND status = 'pending'", (build_id,)
    )
    return cur.rowcount == 1


def finish_build(
    conn: sqlite3.Connection,
    build_id: int,
    *,
    status: str,
    output_dir: str | None = None,
    manifest: dict[str, Any] | None = None,
    error: str | None = None,
) -> None:
    conn.execute(
        "UPDATE builds SET status = ?, output_dir = COALESCE(?, output_dir), "
        "manifest_json = COALESCE(?, manifest_json), error = ? WHERE id = ?",
        (
            status,
            output_dir,
            None if manifest is None else json.dumps(manifest, ensure_ascii=False),
            error,
            build_id,
        ),
    )


def interrupt_running_builds(conn: sqlite3.Connection) -> int:
    cur = conn.execute(
        "UPDATE builds SET status = 'failed', error = 'interrupted' WHERE status = 'running'"
    )
    return cur.rowcount


def live_output_dirs(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute(
        "SELECT output_dir FROM builds WHERE status = 'done' AND pruned_at IS NULL "
        "AND output_dir IS NOT NULL"
    ).fetchall()
    return {str(r["output_dir"]) for r in rows}


def builds_to_prune(
    conn: sqlite3.Connection, document_id: str, keep_final: int, keep_drafts: int
) -> list[Row]:
    result: list[Row] = []
    for kind, keep in ((BuildKind.FINAL, keep_final), (BuildKind.DRAFT, keep_drafts)):
        rows = conn.execute(
            "SELECT * FROM builds WHERE document_id = ? AND kind = ? AND status = 'done' "
            "AND pruned_at IS NULL ORDER BY version DESC",
            (document_id, kind),
        ).fetchall()
        result.extend(rows[keep:])
    return result


def mark_pruned(conn: sqlite3.Connection, build_id: int) -> None:
    conn.execute("UPDATE builds SET pruned_at = ? WHERE id = ?", (utc_now(), build_id))


# --- export ------------------------------------------------------------------------------------


def export_name_taken(conn: sqlite3.Connection, name: str, document_id: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM documents WHERE export_name = ? AND id != ?", (name, document_id)
    ).fetchone()
    return row is not None


def set_export_name(conn: sqlite3.Connection, document_id: str, name: str) -> None:
    conn.execute("UPDATE documents SET export_name = ? WHERE id = ?", (name, document_id))


def set_export_result(
    conn: sqlite3.Connection, document_id: str, *, version: int | None, error: str | None
) -> None:
    if version is not None:
        conn.execute(
            "UPDATE documents SET exported_version = ?, export_error = NULL, "
            "export_attempt_at = ? WHERE id = ?",
            (version, utc_now(), document_id),
        )
    else:
        conn.execute(
            "UPDATE documents SET export_error = ?, export_attempt_at = ? WHERE id = ?",
            (error, utc_now(), document_id),
        )


def documents_needing_export(conn: sqlite3.Connection, attempt_before: str) -> list[Row]:
    """Documents whose last final build is not exported, last attempt older than the cutoff."""
    return conn.execute(
        "SELECT d.*, MAX(b.version) AS last_version FROM documents d JOIN builds b "
        "ON b.document_id = d.id AND b.kind = 'final' AND b.status = 'done' "
        "AND b.pruned_at IS NULL "
        "GROUP BY d.id HAVING MAX(b.version) > COALESCE(d.exported_version, 0) "
        "AND (d.export_attempt_at IS NULL OR d.export_attempt_at < ?) "
        "ORDER BY d.priority DESC, d.created_at",
        (attempt_before,),
    ).fetchall()


def exported_documents(conn: sqlite3.Connection) -> list[Row]:
    """Rows of the export index (section 9.7)."""
    return conn.execute(
        "SELECT d.id, d.title, d.export_name, d.exported_version, d.page_count, b.created_at "
        "AS built_at, b.manifest_json FROM documents d JOIN builds b ON b.document_id = d.id "
        "AND b.version = d.exported_version WHERE d.exported_version IS NOT NULL "
        "ORDER BY d.export_name"
    ).fetchall()


# --- ingests -----------------------------------------------------------------------------------


def create_ingest(conn: sqlite3.Connection, build_id: int) -> bool:
    """Create the single ingest of a final build, or move a failed one back to pending."""
    cur = conn.execute(
        "INSERT INTO ingests (build_id, status, created_at) VALUES (?, 'pending', ?) "
        "ON CONFLICT(build_id) DO UPDATE SET status = 'pending', error = NULL "
        "WHERE ingests.status = 'failed'",
        (build_id, utc_now()),
    )
    return cur.rowcount == 1


def next_pending_ingest(conn: sqlite3.Connection) -> Row | None:
    return conn.execute(
        "SELECT i.*, b.document_id, b.output_dir, b.version FROM ingests i "
        "JOIN builds b ON b.id = i.build_id WHERE i.status = 'pending' ORDER BY i.id LIMIT 1"
    ).fetchone()


def get_ingest_for_build(conn: sqlite3.Connection, build_id: int) -> Row | None:
    return conn.execute("SELECT * FROM ingests WHERE build_id = ?", (build_id,)).fetchone()


def finish_ingest(
    conn: sqlite3.Connection,
    ingest_id: int,
    *,
    status: str,
    response: dict[str, Any] | None,
    error: str | None,
) -> None:
    conn.execute(
        "UPDATE ingests SET status = ?, response_json = ?, error = ? WHERE id = ?",
        (status, _dumps_or_none(response), error, ingest_id),
    )


# --- gold --------------------------------------------------------------------------------------


def upsert_gold(conn: sqlite3.Connection, page_id: int, text: str) -> None:
    conn.execute(
        "INSERT INTO gold (page_id, text, created_at) VALUES (?, ?, ?) "
        "ON CONFLICT(page_id) DO UPDATE SET text = excluded.text",
        (page_id, text, utc_now()),
    )


def gold_pages(conn: sqlite3.Connection) -> list[Row]:
    return conn.execute(
        "SELECT g.page_id, g.text, p.document_id, p.page_index, d.source_path FROM gold g "
        "JOIN pages p ON p.id = g.page_id JOIN documents d ON d.id = p.document_id "
        "ORDER BY p.document_id, p.page_index"
    ).fetchall()


# --- helpers -----------------------------------------------------------------------------------


def _dumps_or_none(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "__dataclass_fields__"):
        value = asdict(value)
    return json.dumps(value, ensure_ascii=False)


def loads(raw: str | None, default: Any = None) -> Any:
    """Parse a JSON column, returning ``default`` for NULL."""
    return default if raw is None else json.loads(raw)


__all__ = [
    "BuildStatus",
    "IngestStatus",
]
