"""Fakes for testing the OpenWebUI pipe without OpenWebUI and without the OCR service."""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from email.parser import BytesParser
from email.policy import default as email_policy
from pathlib import Path
from types import ModuleType
from typing import Any
from urllib.parse import parse_qs, quote

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[2]
PIPE_PATH = ROOT / "owui" / "ocr_archive_pipe.py"
PROTOTYPE_PATH = ROOT / "prototype" / "owui_window_pipe.py"


def load_module(name: str, path: Path) -> ModuleType:
    """Import a single-file OpenWebUI function by path."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


op = load_module("ocr_archive_pipe", PIPE_PATH)

USER_ID = "u1"


def parse_window_code(code: str) -> dict[str, Any]:
    """Extract the JSON payload from the code of one execute call."""
    first_line, _, _ = code.partition("\n")
    prefix = "const CALL = "
    assert first_line.startswith(prefix) and first_line.endswith(";")
    payload: dict[str, Any] = json.loads(first_line[len(prefix) : -1])
    return payload


# ---- fake OCR service ----


@dataclass
class Call:
    """One HTTP request the pipe made."""

    method: str
    path: str
    params: dict[str, str]
    body: Any


@dataclass
class FakeService:
    """Stateful stand-in for the REST API of section 11."""

    documents: dict[str, dict[str, Any]] = field(default_factory=dict)
    pages: dict[int, dict[str, Any]] = field(default_factory=dict)
    drafts: dict[int, dict[str, Any]] = field(default_factory=dict)
    revisions: dict[str, int] = field(default_factory=dict)
    builds: dict[int, dict[str, Any]] = field(default_factory=dict)
    pending_deliveries: list[dict[str, Any]] = field(default_factory=list)
    delivered: list[tuple[str, int]] = field(default_factory=list)
    uploads: list[dict[str, Any]] = field(default_factory=list)
    calls: list[Call] = field(default_factory=list)
    errors: dict[tuple[str, str], list[tuple[int, str]]] = field(default_factory=dict)
    summary_extra: dict[str, Any] = field(default_factory=dict)
    images: dict[int, bytes] = field(default_factory=dict)
    create_extra: dict[str, Any] = field(default_factory=dict)  # merged into uploaded documents
    ingest_enabled: bool = True
    region_text: str = "[область]"
    next_revision: int = 100

    # -- setup --

    def add_document(
        self, doc_id: str, title: str, status: str = "needs_review", **extra: Any
    ) -> None:
        self.documents[doc_id] = {
            "id": doc_id,
            "title": title,
            "status": status,
            "page_count": 0,
            "builds": [],
            "eta_sec": None,
            "created_at": "2026-11-02T20:00:00Z",
            **extra,
        }

    def add_page(
        self,
        page_id: int,
        doc_id: str,
        number: int,
        text: str,
        *,
        status: str = "flagged",
        hints: list[dict[str, Any]] | None = None,
        **extra: Any,
    ) -> None:
        self.documents[doc_id]["page_count"] += 1
        self.pages[page_id] = {
            "page_id": page_id,
            "document_id": doc_id,
            "page_number": number,
            "status": status,
            "scope": "hints",
            "reasons": ["pass_diff"],
            "signals": {"diff_words": 3},
            "item_version": f"v1-{page_id}",
            "base_attempt_id": page_id * 10,
            "base_revision_id": None,
            "text": text,
            "printed_label": None,
            "hints": hints or [],
            **extra,
        }

    def fail(self, method: str, path: str, status: int, detail: str, times: int = 1) -> None:
        """Make the next ``times`` requests to ``method path`` fail."""
        self.errors.setdefault((method, path), []).extend([(status, detail)] * times)

    def calls_to(self, method: str, pattern: str) -> list[Call]:
        return [c for c in self.calls if c.method == method and re.fullmatch(pattern, c.path)]

    # -- model --

    def item(self, page_id: int, scope: str | None = None) -> dict[str, Any]:
        page = self.pages[page_id]
        doc = self.documents[page["document_id"]]
        draft = self.drafts.get(page_id)
        if draft is not None:
            draft = dict(draft)
            changed = (draft["base_attempt_id"], draft["base_revision_id"]) != (
                page["base_attempt_id"],
                page["base_revision_id"],
            )
            draft["stale"] = changed or bool(draft.get("keep_stale"))
            draft["stale_reason"] = (
                ("keep_stale" if draft.get("keep_stale") else "changed") if draft["stale"] else None
            )
        return {
            "page_id": page_id,
            "document_id": page["document_id"],
            "title": doc["title"],
            "page_number": page["page_number"],
            "page_count": doc["page_count"],
            "status": page["status"],
            "scope": scope or page["scope"],
            "reasons": page["reasons"],
            "signals": page["signals"],
            "item_version": page["item_version"],
            "base_attempt_id": page["base_attempt_id"],
            "base_revision_id": page["base_revision_id"],
            "text": page["text"],
            "text_a": page["text"],
            "text_b": page["text"],
            "no_candidate": page["base_attempt_id"] is None,
            "printed_label": page["printed_label"],
            "hints": [] if page["base_revision_id"] else page["hints"],
            "draft": draft,
        }

    def queue_ids(self, exclude: set[int]) -> list[int]:
        ids = [
            pid
            for pid, page in self.pages.items()
            if pid not in exclude
            and (
                page["status"] in ("flagged", "failed")
                or (pid in self.drafts and page["status"] != "pending")
            )
        ]
        return sorted(ids, key=lambda pid: (pid not in self.drafts, pid))

    # -- HTTP --

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        params = {k: v[0] for k, v in parse_qs(request.url.query.decode()).items()}
        body: Any = None
        content_type = request.headers.get("content-type", "")
        if content_type.startswith("application/json") and request.content:
            body = json.loads(request.content)
        elif content_type.startswith("multipart/form-data"):
            body = _parse_multipart(content_type, request.content)
        self.calls.append(Call(request.method, path, params, body))
        queued = self.errors.get((request.method, path))
        if queued:
            status, detail = queued.pop(0)
            return httpx.Response(status, json={"detail": detail})
        for method, pattern, func in self._routes():
            if method != request.method:
                continue
            match = re.fullmatch(pattern, path)
            if match:
                return func(match, params, body)
        return httpx.Response(404, json={"detail": "not_found"})

    def _routes(
        self,
    ) -> list[tuple[str, str, Callable[[re.Match[str], dict[str, str], Any], httpx.Response]]]:
        doc = r"/api/documents/(?P<doc>[^/]+)"
        page = r"/api/pages/(?P<page>\d+)"
        return [
            ("POST", r"/api/documents", self._create),
            ("GET", r"/api/documents", self._list),
            ("GET", doc, self._get_doc),
            ("POST", doc + "/draft", self._draft_build),
            ("POST", doc + "/ingest", self._ingest),
            ("POST", doc + "/delivered", self._delivered),
            ("GET", doc + r"/files/(?P<kind>\w+)", self._file),
            ("GET", doc + r"/pages/(?P<n>\d+)/item", self._page_item),
            ("GET", r"/api/builds/(?P<build>\d+)", self._build),
            ("GET", r"/api/deliveries", self._deliveries),
            ("GET", r"/api/review/summary", self._summary),
            ("GET", r"/api/review/queue", self._queue),
            ("GET", r"/api/review/items/(?P<page>\d+)", self._review_item),
            ("GET", page + "/image", self._image),
            ("PUT", page + "/draft", self._put_draft),
            ("DELETE", page + "/draft", self._delete_draft),
            ("POST", page + "/review", self._review),
            ("POST", page + "/ocr-region", self._region),
            ("POST", page + "/retry", self._retry),
        ]

    def _create(self, m: re.Match[str], p: dict[str, str], body: Any) -> httpx.Response:
        self.uploads.append(body)
        first = body["files"][0]["filename"]
        for existing in self.documents.values():
            if existing.get("source_name") == first:
                return httpx.Response(201, json={**_doc_brief(existing), "duplicate": True})
        doc_id = f"doc{len(self.documents) + 1}"
        self.add_document(
            doc_id, Path(first).stem, status="queued", source_name=first, eta_sec=3600
        )
        self.documents[doc_id]["page_count"] = 214
        self.documents[doc_id].update(self.create_extra)
        return httpx.Response(201, json={**_doc_brief(self.documents[doc_id]), "duplicate": False})

    def _list(self, m: re.Match[str], p: dict[str, str], body: Any) -> httpx.Response:
        return httpx.Response(200, json=[_doc_brief(d) for d in self.documents.values()])

    def _get_doc(self, m: re.Match[str], p: dict[str, str], body: Any) -> httpx.Response:
        doc = self.documents.get(m["doc"])
        if doc is None:
            return httpx.Response(404, json={"detail": "not_found"})
        sequence = doc.get("status_sequence")
        if sequence:
            doc["status"] = sequence.pop(0)
        return httpx.Response(200, json={k: v for k, v in doc.items() if k != "status_sequence"})

    def _draft_build(self, m: re.Match[str], p: dict[str, str], body: Any) -> httpx.Response:
        doc = self.documents[m["doc"]]
        if doc["status"] not in ("needs_review", "ready"):
            return httpx.Response(409, json={"detail": "invalid_transition"})
        build_id = len(self.builds) + 1
        self.builds[build_id] = {
            "id": build_id,
            "document_id": doc["id"],
            "kind": "draft",
            "status": "pending",
            "status_sequence": list(doc.get("draft_sequence", ["running", "done"])),
        }
        return httpx.Response(202, json={"build_id": build_id})

    def _build(self, m: re.Match[str], p: dict[str, str], body: Any) -> httpx.Response:
        build = self.builds[int(m["build"])]
        if build["status_sequence"]:
            build["status"] = build["status_sequence"].pop(0)
        return httpx.Response(
            200, json={k: v for k, v in build.items() if k != "status_sequence"} | {"error": None}
        )

    def _ingest(self, m: re.Match[str], p: dict[str, str], body: Any) -> httpx.Response:
        if not self.ingest_enabled:
            return httpx.Response(409, json={"detail": "ingester_disabled"})
        return httpx.Response(202, json={"ingest_status": "pending"})

    def _delivered(self, m: re.Match[str], p: dict[str, str], body: Any) -> httpx.Response:
        self.delivered.append((m["doc"], int(body["version"])))
        self.pending_deliveries = [
            d for d in self.pending_deliveries if d["document_id"] != m["doc"]
        ]
        return httpx.Response(204)

    def _file(self, m: re.Match[str], p: dict[str, str], body: Any) -> httpx.Response:
        doc = self.documents[m["doc"]]
        kind = m["kind"]
        suffix = {"pdf": ".pdf", "docx": ".docx", "chunks": ".jsonl"}[kind]
        name = doc["title"] + (" (черновик)" if p.get("draft") == "true" else "") + suffix
        return httpx.Response(
            200,
            content=f"{kind}:{doc['id']}:{p.get('version', '')}".encode(),
            headers={
                "content-type": "application/octet-stream",
                "content-disposition": f"attachment; filename*=UTF-8''{quote(name)}",
            },
        )

    def _page_item(self, m: re.Match[str], p: dict[str, str], body: Any) -> httpx.Response:
        for pid, page in self.pages.items():
            if page["document_id"] == m["doc"] and page["page_number"] == int(m["n"]):
                if page["status"] == "pending":
                    return httpx.Response(409, json={"detail": "invalid_transition"})
                return httpx.Response(200, json=self.item(pid, scope="full"))
        return httpx.Response(404, json={"detail": "not_found"})

    def _deliveries(self, m: re.Match[str], p: dict[str, str], body: Any) -> httpx.Response:
        return httpx.Response(200, json=self.pending_deliveries)

    def _summary(self, m: re.Match[str], p: dict[str, str], body: Any) -> httpx.Response:
        total = len(self.queue_ids(set()))
        summary = {
            "queue_total": total,
            "est_review_sec": total * 60,
            "load_guideline": 0.1,
            "documents": [
                {
                    "id": d["id"],
                    "title": d["title"],
                    "status": d["status"],
                    "progress": {"total": d["page_count"], "pending": 0},
                    "eta_sec": None,
                    "to_review": total,
                    "drafts": len(self.drafts),
                    "flagged_share": 0.05,
                    "over_guideline": False,
                    "est_review_sec": total * 60,
                    "no_logprobs_pages": 0,
                    "export_error": None,
                }
                for d in self.documents.values()
            ],
            "stale_drafts": [],
            "suppressed_rules": [],
            "waiting_redecide": [],
        }
        summary.update(self.summary_extra)
        return httpx.Response(200, json=summary)

    def _queue(self, m: re.Match[str], p: dict[str, str], body: Any) -> httpx.Response:
        exclude = {int(x) for x in p.get("exclude", "").split(",") if x}
        limit = int(p.get("limit", "50"))
        items = []
        for pid in self.queue_ids(exclude)[:limit]:
            page = self.pages[pid]
            items.append(
                {
                    "page_id": pid,
                    "document_id": page["document_id"],
                    "title": self.documents[page["document_id"]]["title"],
                    "page_number": page["page_number"],
                    "scope": page["scope"],
                    "reasons": page["reasons"],
                    "has_draft": pid in self.drafts,
                }
            )
        return httpx.Response(200, json=items)

    def _review_item(self, m: re.Match[str], p: dict[str, str], body: Any) -> httpx.Response:
        pid = int(m["page"])
        if pid not in self.pages:
            return httpx.Response(404, json={"detail": "not_found"})
        if self.pages[pid]["status"] == "pending":
            return httpx.Response(409, json={"detail": "invalid_transition"})
        return httpx.Response(200, json=self.item(pid))

    def _image(self, m: re.Match[str], p: dict[str, str], body: Any) -> httpx.Response:
        data = self.images.get(int(m["page"]), b"\xff\xd8fake-jpeg")
        return httpx.Response(200, content=data, headers={"content-type": "image/jpeg"})

    def _put_draft(self, m: re.Match[str], p: dict[str, str], body: Any) -> httpx.Response:
        pid = int(m["page"])
        page = self.pages[pid]
        if page["status"] == "pending":
            return httpx.Response(409, json={"detail": "invalid_transition"})
        current = self.drafts.get(pid)
        current_seq = current["seq"] if current else 0
        if int(body.get("expected_seq") or 0) != current_seq:
            return httpx.Response(409, json={"detail": "draft_conflict"})
        base = (body.get("base_attempt_id"), body.get("base_revision_id"))
        if base != (page["base_attempt_id"], page["base_revision_id"]) and not body.get(
            "keep_stale"
        ):
            return httpx.Response(409, json={"detail": "stale"})
        text = body["text"]
        self.drafts[pid] = {
            **{k: body.get(k) for k in body if k != "expected_seq"},
            "seq": current_seq + 1,
            "text_sha256": op.text_sha256(text),
            "updated_at": "2026-11-03T07:00:00Z",
        }
        return httpx.Response(
            200, json={"seq": current_seq + 1, "text_sha256": op.text_sha256(text)}
        )

    def _delete_draft(self, m: re.Match[str], p: dict[str, str], body: Any) -> httpx.Response:
        pid = int(m["page"])
        current = self.drafts.get(pid)
        if current is not None and current["seq"] != int(p.get("draft_seq", "0")):
            return httpx.Response(409, json={"detail": "draft_conflict"})
        self.drafts.pop(pid, None)
        return httpx.Response(204)

    def _review(self, m: re.Match[str], p: dict[str, str], body: Any) -> httpx.Response:
        pid = int(m["page"])
        page = self.pages[pid]
        doc = self.documents[page["document_id"]]
        if body["request_id"] in self.revisions:
            return httpx.Response(
                200,
                json={
                    "revision_id": self.revisions[body["request_id"]],
                    "document_status": doc["status"],
                },
            )
        if doc["status"] == "building":
            return httpx.Response(409, json={"detail": "document_building"})
        if page["status"] == "pending":
            return httpx.Response(409, json={"detail": "invalid_transition"})
        current = self.drafts.get(pid)
        if (current["seq"] if current else 0) != int(body.get("draft_seq") or 0):
            return httpx.Response(409, json={"detail": "draft_conflict"})
        base = (body.get("base_attempt_id"), body.get("base_revision_id"))
        if base != (page["base_attempt_id"], page["base_revision_id"]) and (
            body.get("on_conflict") != "keep_mine"
        ):
            return httpx.Response(409, json={"detail": "stale"})
        revision_id = self.next_revision
        self.next_revision += 1
        self.revisions[body["request_id"]] = revision_id
        page.update(status="reviewed", base_revision_id=revision_id, text=body["text"])
        self.drafts.pop(pid, None)
        still_open = [
            q
            for q, other in self.pages.items()
            if other["document_id"] == doc["id"]
            and (other["status"] in ("flagged", "failed", "pending") or q in self.drafts)
        ]
        if not still_open:
            doc["status"] = "ready"
        return httpx.Response(
            201, json={"revision_id": revision_id, "document_status": doc["status"]}
        )

    def _region(self, m: re.Match[str], p: dict[str, str], body: Any) -> httpx.Response:
        return httpx.Response(200, json={"text": self.region_text, "low_conf": []})

    def _retry(self, m: re.Match[str], p: dict[str, str], body: Any) -> httpx.Response:
        page = self.pages[int(m["page"])]
        if self.documents[page["document_id"]]["status"] == "building":
            return httpx.Response(409, json={"detail": "document_building"})
        if page["status"] == "failed":
            page["status"] = "pending"
        return httpx.Response(202, json={"status": page["status"]})


def _doc_brief(doc: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "id": doc["id"],
        "title": doc["title"],
        "status": doc["status"],
        "page_count": doc["page_count"],
        "created_at": doc.get("created_at"),
    }


def _parse_multipart(content_type: str, content: bytes) -> dict[str, Any]:
    message = BytesParser(policy=email_policy).parsebytes(
        b"Content-Type: " + content_type.encode() + b"\r\n\r\n" + content
    )
    fields: dict[str, Any] = {"files": []}
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        filename = part.get_filename()
        payload = part.get_payload(decode=True)
        if filename is not None:
            fields["files"].append({"field": name, "filename": filename, "data": payload})
        else:
            fields[str(name)] = payload.decode("utf-8") if isinstance(payload, bytes) else payload
    return fields


# ---- fake OpenWebUI pieces ----


class FakeFiles:
    """Stand-in for :class:`OwuiFiles`."""

    def __init__(self) -> None:
        self.stored: list[Any] = []
        self.deleted: list[str] = []
        self.fail_store = False

    async def read_upload(self, entry: Mapping[str, Any]) -> Any:
        return op.Upload(name=entry["name"], data=entry["_data"], content_type=entry["_type"])

    async def store(self, name: str, data: bytes, content_type: str) -> Any:
        if self.fail_store:
            raise op.OwuiFilesError("disk full")
        stored = op.StoredFile(
            id=f"f{len(self.stored) + 1}", name=name, size=len(data), content_type=content_type
        )
        self.stored.append(stored)
        return stored

    async def delete(self, file_id: str) -> None:
        self.deleted.append(file_id)


class Emitter:
    """Records ``__event_emitter__`` events."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    async def __call__(self, event: dict[str, Any]) -> None:
        self.events.append(event)

    def of_type(self, kind: str) -> list[dict[str, Any]]:
        return [e for e in self.events if e.get("type") == kind]


Step = Callable[["FakeWindow", dict[str, Any]], dict[str, Any]] | dict[str, Any]


class FakeWindow:
    """Scripted ``__event_call__``: follows the window protocol of section 8.3."""

    def __init__(self, steps: list[Step] | None = None, outbox: list[dict[str, Any]] | None = None):
        self.steps = list(steps or [])
        self.outbox = list(outbox or [])
        self.payloads: list[dict[str, Any]] = []
        self.acks: list[dict[str, Any]] = []
        self.outbox_acks: list[dict[str, Any]] = []
        self.page: dict[str, Any] | None = None
        self.seqs: dict[int, int] = {}
        self.server_draft: dict[str, Any] | None = None
        self.next_id = 1
        self.rev = 0
        self.exhausted = False
        self.closed = False
        self.fail_with: dict[str, Any] | None = None

    async def __call__(self, event: dict[str, Any]) -> Any:
        assert event["type"] == "execute"
        payload = parse_window_code(event["data"]["code"])
        self.payloads.append(payload)
        self.outbox_acks.extend(payload.get("outbox_ack") or [])
        for ack in payload.get("acks") or []:
            self._apply_ack(ack)
        if payload["op"] == "collect_outbox":
            return {"type": "outbox", "entries": self.outbox}
        if payload["op"] == "close":
            self.closed = True
            return {"type": "closed"}
        if payload["op"] == "show":
            self.page = payload["page"]
            draft = self.page.get("draft")
            self.seqs[self.page["page_id"]] = draft["seq"] if draft else 0
        if self.fail_with is not None:
            return self.fail_with
        if not self.steps:
            self.exhausted = True
            return {"error": "script exhausted"}
        step = self.steps.pop(0)
        return step(self, payload) if callable(step) else step

    def _apply_ack(self, ack: dict[str, Any]) -> None:
        self.acks.append(ack)
        if ack.get("seq") is not None and ack.get("status") in ("ok", "building"):
            self.seqs[int(ack["page_id"])] = int(ack["seq"])
        item = ack.get("item")
        if item and self.page and item["page_id"] == self.page["page_id"]:
            for key in ("base_attempt_id", "base_revision_id", "item_version", "text"):
                self.page[key] = item[key]
        if ack.get("status") == "draft_conflict":
            self.server_draft = ack.get("server_draft")

    @property
    def shown_pages(self) -> list[int]:
        return [p["page"]["page_id"] for p in self.payloads if p["op"] == "show"]

    def event(self, kind: str, **fields: Any) -> dict[str, Any]:
        assert self.page is not None
        self.next_id += 1
        return {
            "type": kind,
            "id": self.next_id,
            "page_id": self.page["page_id"],
            "rev": self.rev,
            **fields,
        }

    def draft_body(self, text: str | None = None) -> dict[str, Any]:
        assert self.page is not None
        return {
            "expected_seq": self.seqs.get(self.page["page_id"], 0),
            "item_version": self.page["item_version"],
            "base_attempt_id": self.page["base_attempt_id"],
            "base_revision_id": self.page["base_revision_id"],
            "text": self.page["text"] if text is None else text,
            "printed_label": None,
            "choices": {},
            "hints": [],
            "cursor": 0,
        }

    def review_body(self, text: str | None = None) -> dict[str, Any]:
        assert self.page is not None
        return {
            "request_id": uuid.uuid4().hex,
            "item_version": self.page["item_version"],
            "base_attempt_id": self.page["base_attempt_id"],
            "base_revision_id": self.page["base_revision_id"],
            "draft_seq": self.seqs.get(self.page["page_id"], 0),
            "text": self.page["text"] if text is None else text,
            "printed_label": None,
            "duration_s": 3,
            "choices": {},
        }


def draft(text: str) -> Step:
    def step(w: FakeWindow, payload: dict[str, Any]) -> dict[str, Any]:
        w.rev += 1
        return w.event("draft", draft=w.draft_body(text))

    return step


def done(text: str | None = None) -> Step:
    def step(w: FakeWindow, payload: dict[str, Any]) -> dict[str, Any]:
        w.rev += 1
        review = w.review_body(text)
        return w.event("done", review=review, draft=w.draft_body(review["text"]))

    return step


def leave(kind: str, text: str | None = None) -> Step:
    """``later``, ``back``, ``retry`` or ``stop``; with ``text`` the event carries a draft."""

    def step(w: FakeWindow, payload: dict[str, Any]) -> dict[str, Any]:
        return w.event(kind, draft=w.draft_body(text) if text is not None else None)

    return step


def region(rect: dict[str, float]) -> Step:
    def step(w: FakeWindow, payload: dict[str, Any]) -> dict[str, Any]:
        return w.event("region", rect=rect)

    return step


def keep_mine(text: str, *, finish: bool) -> Step:
    def step(w: FakeWindow, payload: dict[str, Any]) -> dict[str, Any]:
        w.rev += 1
        fields: dict[str, Any] = {"draft": w.draft_body(text), "finish": finish}
        if finish:
            fields["review"] = w.review_body(text)
        return w.event("keep_mine", **fields)

    return step


def discard() -> Step:
    def step(w: FakeWindow, payload: dict[str, Any]) -> dict[str, Any]:
        assert w.page is not None
        return w.event("discard_draft", draft_seq=w.seqs.get(w.page["page_id"], 0))

    return step


# ---- pipe under test ----


class PipeForTest(op.Pipe):
    """The real pipe wired to the fake service and fake file store."""

    build_poll_s = 0.01

    def __init__(self, service: FakeService, files: FakeFiles | None = None) -> None:
        super().__init__()
        self.service = service
        self.files = files or FakeFiles()
        self.valves.allowed_user_ids = [USER_ID]
        self.valves.api_key = "secret"
        self.valves.upload_wait_s = 0  # tests of the wait set it explicitly

    def _make_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            transport=httpx.MockTransport(self.service.handler),
            base_url="http://ocr.test",
            headers={"X-API-Key": self.valves.api_key},
        )

    def _make_files(self, request: Any, user: Mapping[str, Any]) -> Any:
        return self.files


async def run_pipe(
    pipe: PipeForTest,
    text: str,
    *,
    window: FakeWindow | None = None,
    emitter: Emitter | None = None,
    files: list[dict[str, Any]] | None = None,
    user_id: str = USER_ID,
    task: str | None = None,
) -> str:
    """Send one chat message to the pipe."""
    body = {"messages": [{"role": "user", "content": text}], "stream": True}
    result: str = await pipe.pipe(
        body,
        __user__={"id": user_id, "valves": {"ingest_new_documents": False}},
        __files__=files,
        __task__=task,
        __request__=None,
        __event_emitter__=emitter or Emitter(),
        __event_call__=window,
    )
    return result


# ---- Node.js for the window JS ----


def node_binary() -> str | None:
    """Node.js for the JS tests: ``OCR_TEST_NODE`` (an Electron binary works too) or ``node``."""
    return os.environ.get("OCR_TEST_NODE") or shutil.which("node")


def run_node(script: str, tmp_path: Path, *args: str) -> Any:
    """Run a script in Node.js and return the JSON printed on its last stdout line."""
    node = node_binary()
    if node is None:
        pytest.skip("Node.js is not available; set OCR_TEST_NODE to run the window JS tests")
    path = tmp_path / "test.js"
    path.write_text(script, encoding="utf-8")
    env = {**os.environ, "ELECTRON_RUN_AS_NODE": "1"}
    proc = subprocess.run(
        [node, str(path), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
        env=env,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    assert lines, proc.stderr
    return json.loads(lines[-1])
