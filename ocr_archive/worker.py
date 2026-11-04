"""The single worker: lock, recovery, heartbeat thread and the work loop (section 12)."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import shutil
import signal
import sqlite3
import sys
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import FrameType
from typing import IO, Any

from ocr_archive.build import build_document
from ocr_archive.build.export import export_document
from ocr_archive.db import repo
from ocr_archive.db.connection import connect, immediate, open_db, utc_now
from ocr_archive.exceptions import BuildError, ExportError, IngestError, StatusTransitionError
from ocr_archive.ingester_client import upload_chunks
from ocr_archive.models import (
    BuildContext,
    BuildKind,
    DocumentStatus,
    PipelineContext,
)
from ocr_archive.pipeline import classify_pages, process_page, redecide_one
from ocr_archive.settings import Settings
from ocr_archive.status import RULES_VERSION_KEY, recompute_status
from ocr_archive.versions import PDF_TEXT_PROFILE, canonical_json, profile_config, profile_id
from ocr_archive.versions import rules_version as compute_rules_version
from ocr_archive.vlm.client import VlmClient

logger = logging.getLogger(__name__)

HEARTBEAT_KEY = "worker_heartbeat"
HEARTBEAT_INTERVAL_S = 10.0
EXPORT_RETRY_INTERVAL = timedelta(minutes=10)
PROFILE_NAMES = ("pass1", "pass2", "region", PDF_TEXT_PROFILE)


class WorkerLockedError(RuntimeError):
    """Another worker holds the lock."""


def acquire_lock(path: Path) -> IO[str]:
    """Take an exclusive non-blocking lock for the whole process lifetime.

    The OS releases it when the process dies, so there is neither a false "death" during a long
    model request nor a race at start. flock works only on a local Linux disk (section 12.4); the
    msvcrt fallback exists for development on Windows.

    :raises WorkerLockedError: another worker is running.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    try:
        import fcntl

        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as err:
            handle.close()
            raise WorkerLockedError("another worker is running") from err
    except ImportError:
        import msvcrt

        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)  # type: ignore[attr-defined,unused-ignore]
        except OSError as err:
            handle.close()
            raise WorkerLockedError("another worker is running") from err
    return handle


def register_profiles(conn: sqlite3.Connection, settings: Settings) -> dict[str, str]:
    """Compute profile ids (OCR versions) and record them in ``profiles``."""
    ids: dict[str, str] = {}
    with immediate(conn):
        for name in PROFILE_NAMES:
            pid = profile_id(name, settings)
            repo.ensure_profile(conn, pid, name, canonical_json(profile_config(name, settings)))
            ids[name] = pid
    return ids


def recover(conn: sqlite3.Connection, settings: Settings) -> None:
    """Restore a consistent state after a crash (section 12.1)."""
    with immediate(conn):
        interrupted = repo.interrupt_running_builds(conn)
        for doc in repo.documents_in_status(conn, [DocumentStatus.BUILDING]):
            repo.set_document_status(conn, str(doc["id"]), DocumentStatus.READY)
        live = repo.live_output_dirs(conn)
    if interrupted:
        logger.warning("%d interrupted builds marked failed", interrupted)
    outputs = settings.outputs_dir
    if outputs.exists():
        for doc_dir in outputs.iterdir():
            if not doc_dir.is_dir():
                continue
            for version_dir in doc_dir.iterdir():
                rel = version_dir.relative_to(settings.data_dir).as_posix()
                if version_dir.is_dir() and rel not in live:
                    shutil.rmtree(version_dir, ignore_errors=True)
                    logger.info("removed unreferenced output %s", rel)
    if settings.export_dir.exists():
        for tmp in settings.export_dir.rglob(".*.tmp"):
            with contextlib.suppress(OSError):
                tmp.unlink()


def recompute_all(conn: sqlite3.Connection, rules_version: str) -> None:
    """Recompute every document except queued and building; done only with a pending request."""
    for doc in repo.list_documents(conn):
        status = doc["status"]
        if status in (DocumentStatus.QUEUED, DocumentStatus.BUILDING, DocumentStatus.FAILED):
            continue
        if status == DocumentStatus.DONE and not (
            doc["redecide_requested"] or doc["rebuild_requested"]
        ):
            continue
        with immediate(conn):
            recompute_status(conn, str(doc["id"]), rules_version)


class Heartbeat(threading.Thread):
    """Writes ``control.worker_heartbeat`` every 10 s from its own thread and connection, so
    neither a model request nor a synchronous build delays it."""

    def __init__(self, db_path: Path, interval: float = HEARTBEAT_INTERVAL_S) -> None:
        super().__init__(name="heartbeat", daemon=True)
        self._db_path = db_path
        self._interval = interval
        self._stop = threading.Event()

    def run(self) -> None:
        conn = connect(self._db_path)
        try:
            while not self._stop.is_set():
                try:
                    with immediate(conn):
                        repo.set_control(conn, HEARTBEAT_KEY, utc_now())
                except sqlite3.Error:
                    logger.warning("heartbeat write failed", exc_info=True)
                self._stop.wait(self._interval)
        finally:
            conn.close()

    def stop(self) -> None:
        self._stop.set()


def heartbeat_age_s(conn: sqlite3.Connection) -> float | None:
    """Age of the last heartbeat in seconds, or None if there was none."""
    value = repo.get_control(conn, HEARTBEAT_KEY)
    if value is None:
        return None
    return (datetime.now(UTC) - datetime.fromisoformat(value)).total_seconds()


@dataclass
class Worker:
    """State of the running worker."""

    settings: Settings
    ctx: PipelineContext
    build_ctx: BuildContext
    conn: sqlite3.Connection
    stopping: threading.Event = field(default_factory=threading.Event)

    async def run_forever(self) -> None:
        """Do one unit of work per round, sleep when idle."""
        while not self.stopping.is_set():
            try:
                did_work = await self.round()
            except asyncio.CancelledError:
                raise
            except Exception:  # broad on purpose: one failing unit must not stop the worker loop
                logger.exception("worker round failed")
                did_work = False
            if not did_work:
                await asyncio.sleep(self._idle_delay())

    def _idle_delay(self) -> float:
        delay = self.settings.worker.poll_interval_s
        earliest = repo.earliest_retry_at(self.conn)
        if earliest is not None:
            wait = (datetime.fromisoformat(earliest) - datetime.now(UTC)).total_seconds()
            delay = min(delay, max(0.1, wait))
        return delay

    async def round(self) -> bool:
        """One unit of work; return whether anything was done (section 12.1)."""
        if await self._final_build():
            return True
        if await self._retry_export():
            return True
        if await self._draft_build():
            return True
        if await self._ingest():
            return True
        return await self._document_step()

    async def _final_build(self) -> bool:
        doc = repo.next_ready_document(self.conn)
        if doc is None:
            return False
        document_id = str(doc["id"])
        try:
            version = await asyncio.to_thread(
                build_document, document_id, self.build_ctx, kind=BuildKind.FINAL
            )
        except StatusTransitionError:
            return True
        except BuildError as err:
            logger.error("final build of %s failed: %s", document_id, err)
            return True
        await self._export(document_id, version)
        return True

    async def _export(self, document_id: str, version: int) -> None:
        try:
            await asyncio.to_thread(export_document, document_id, version, self.build_ctx)
        except ExportError as err:
            logger.error("export of %s v%d failed: %s", document_id, version, err)

    async def _retry_export(self) -> bool:
        cutoff = (datetime.now(UTC) - EXPORT_RETRY_INTERVAL).isoformat(timespec="seconds")
        rows = repo.documents_needing_export(self.conn, cutoff)
        if not rows:
            return False
        row = rows[0]
        await self._export(str(row["id"]), int(row["last_version"]))
        return True

    async def _draft_build(self) -> bool:
        build = repo.next_pending_draft_build(self.conn)
        if build is None:
            return False
        try:
            await asyncio.to_thread(
                build_document, str(build["document_id"]), self.build_ctx, kind=BuildKind.DRAFT
            )
        except (BuildError, StatusTransitionError) as err:
            logger.error("draft build %d failed: %s", build["id"], err)
        return True

    async def _ingest(self) -> bool:
        row = repo.next_pending_ingest(self.conn)
        if row is None:
            return False
        document_id = str(row["document_id"])
        path = self.settings.data_dir / str(row["output_dir"]) / "chunks.jsonl"
        try:
            response = await upload_chunks(path, document_id, self.settings.ingester)
        except (IngestError, OSError) as err:
            logger.error("ingest of %s failed: %s", document_id, err)
            with immediate(self.conn):
                repo.finish_ingest(
                    self.conn, int(row["id"]), status="failed", response=None, error=str(err)
                )
                repo.set_ingest_status(self.conn, document_id, "failed")
            return True
        with immediate(self.conn):
            repo.finish_ingest(
                self.conn, int(row["id"]), status="done", response=response, error=None
            )
            repo.set_ingest_status(self.conn, document_id, "done")
        return True

    async def _document_step(self) -> bool:
        now = utc_now()
        for doc in repo.documents_in_status(
            self.conn, [DocumentStatus.PROCESSING, DocumentStatus.QUEUED]
        ):
            document_id = str(doc["id"])
            if doc["status"] == DocumentStatus.QUEUED:
                with immediate(self.conn):
                    if not repo.transition_document(
                        self.conn, document_id, DocumentStatus.QUEUED, DocumentStatus.PROCESSING
                    ):
                        continue
            if repo.count_pages_without_track(self.conn, document_id):
                await asyncio.to_thread(self._classify, document_id)
                return True
            pages = repo.next_pending_pages(
                self.conn, document_id, now, self.settings.worker.page_concurrency
            )
            if pages:
                await self._process_pages([int(p["id"]) for p in pages])
                return True
            if repo.count_pending(self.conn, document_id):
                continue  # only pages waiting for retry_at: try the next document
            if await asyncio.to_thread(self._redecide, document_id):
                return True
            with immediate(self.conn):
                status = recompute_status(self.conn, document_id, self.ctx.rules_version)
            if status != DocumentStatus.PROCESSING:
                return True
        return False

    def _classify(self, document_id: str) -> None:
        conn = connect(self.settings.database_path)
        try:
            classify_pages(conn, document_id, self.settings)
        finally:
            conn.close()

    def _redecide(self, document_id: str) -> bool:
        conn = connect(self.settings.database_path)
        try:
            return redecide_one(conn, document_id, self.ctx)
        finally:
            conn.close()

    async def _process_pages(self, page_ids: list[int]) -> None:
        results = await asyncio.gather(
            *(process_page(page_id, self.ctx) for page_id in page_ids), return_exceptions=True
        )
        for page_id, result in zip(page_ids, results, strict=True):
            if isinstance(result, asyncio.CancelledError):
                raise result
            if isinstance(result, BaseException):
                self._fail_page(page_id, result)

    def _fail_page(self, page_id: int, error: BaseException) -> None:
        logger.error("page %d failed with an unexpected error", page_id, exc_info=error)
        with immediate(self.conn):
            page = repo.get_page(self.conn, page_id)
            if page is None or page["status"] != "pending":
                return
            repo.write_decision(
                self.conn,
                page_id,
                expected_status="pending",
                expected_generation=int(page["ocr_generation"]),
                allow_draft=True,
                status="failed",
                candidate_attempt_id=page["current_attempt_id"],
                flags=[],
                signals={"error": f"{error.__class__.__name__}: {error}"[:1000]},
                review={
                    "scope": "hints",
                    "reasons": ["failed"],
                    "hints": [],
                    "item_version": "",
                    "base_attempt_id": page["current_attempt_id"],
                    "base_revision_id": None,
                },
                rules_version=None,
                error=f"{error.__class__.__name__}: {error}"[:1000],
            )
            recompute_status(self.conn, str(page["document_id"]), self.ctx.rules_version)


def _install_signal_handlers(
    worker: Worker, loop: asyncio.AbstractEventLoop, task: asyncio.Task[Any]
) -> None:
    def handler(signum: int, frame: FrameType | None) -> None:
        logger.info("signal %d received, stopping", signum)
        worker.stopping.set()
        loop.call_soon_threadsafe(task.cancel)

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)


async def _main(worker: Worker) -> None:
    loop = asyncio.get_running_loop()
    task = asyncio.ensure_future(worker.run_forever())
    _install_signal_handlers(worker, loop, task)
    try:
        await task
    except asyncio.CancelledError:
        logger.info("worker stopped")
    finally:
        await worker.ctx.vlm.aclose()


def run_worker(settings: Settings) -> int:
    """Run the worker until SIGTERM/SIGINT.

    :return: process exit code: 0 after a signal, 1 when another worker holds the lock.
    """
    try:
        lock = acquire_lock(settings.lock_path)
    except WorkerLockedError:
        print("another worker is running", file=sys.stderr)
        return 1
    conn = open_db(settings.database_path)
    recover(conn, settings)
    version = compute_rules_version(settings)
    profile_ids = register_profiles(conn, settings)
    with immediate(conn):
        repo.set_control(conn, RULES_VERSION_KEY, version)
    recompute_all(conn, version)
    heartbeat = Heartbeat(settings.database_path)
    heartbeat.start()
    vlm = VlmClient(settings.vlm, settings.llama_api_key.get_secret_value())
    ctx = PipelineContext(
        settings=settings,
        db_path=settings.database_path,
        vlm=vlm,
        rules_version=version,
        profile_ids=profile_ids,
    )
    build_ctx = BuildContext(
        settings=settings,
        db_path=settings.database_path,
        outputs_dir=settings.outputs_dir,
        export_dir=settings.export_dir,
    )
    worker = Worker(settings=settings, ctx=ctx, build_ctx=build_ctx, conn=conn)
    logger.info("worker started, rules version %s", version[:12])
    try:
        asyncio.run(_main(worker))
    finally:
        heartbeat.stop()
        conn.close()
        lock.close()
    busy = [t for t in threading.enumerate() if t is not threading.main_thread() and not t.daemon]
    if busy:
        # A build running in a helper thread is not waited for: recovery redoes it at next start.
        logging.shutdown()
        os._exit(0)
    return 0
