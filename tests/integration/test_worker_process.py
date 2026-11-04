"""Worker as a process: single instance lock, restart after kill, build before pages (p2–p4)."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from ocr_archive.db import repo
from ocr_archive.db.connection import connect, immediate, utc_now
from ocr_archive.models import DocumentStatus
from ocr_archive.worker import heartbeat_age_s
from tests.conftest import FakeVlm
from tests.integration.helpers import make_document, make_worker, page_text, synthetic_pages

ROOT = Path(__file__).resolve().parents[2]


def _start_worker(config: Path) -> subprocess.Popen[bytes]:
    env = {**os.environ, "OCR_CONFIG": str(config), "PYTHONPATH": str(ROOT)}
    return subprocess.Popen(
        [sys.executable, "-m", "ocr_archive.cli", "worker"],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _wait_heartbeat(db_path: Path, after: float, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if db_path.exists():
            conn = connect(db_path)
            try:
                age = heartbeat_age_s(conn)
            finally:
                conn.close()
            if age is not None and age < time.time() - after:
                return
        time.sleep(0.3)
    raise AssertionError("worker did not write a heartbeat")


@pytest.mark.p2
def test_single_worker_and_restart_after_kill(tmp_path: Path) -> None:
    data = tmp_path / "data"
    config = tmp_path / "config.yaml"
    config.write_text(
        f"data_dir: {data.as_posix()}\nexport_dir: {(tmp_path / 'export').as_posix()}\n"
        "vlm:\n  base_url: http://127.0.0.1:9/v1\nworker:\n  poll_interval_s: 0.5\n",
        encoding="utf-8",
    )
    started = time.time()
    first = _start_worker(config)
    try:
        _wait_heartbeat(data / "state.db", started)
        second = _start_worker(config)
        assert second.wait(timeout=60) == 1
        assert b"another worker is running" in second.stderr.read()  # type: ignore[union-attr]
    finally:
        first.kill()
        first.wait(timeout=30)
    restarted_at = time.time()
    third = _start_worker(config)
    try:
        _wait_heartbeat(data / "state.db", restarted_at)
        assert third.poll() is None
    finally:
        third.terminate()
        third.wait(timeout=60)


@pytest.mark.p3
async def test_transient_model_error_does_not_block_builds(
    conn, settings, tmp_path: Path, fake_vlm: FakeVlm, monkeypatch: pytest.MonkeyPatch
) -> None:
    import ocr_archive.worker as worker_module

    synth = synthetic_pages(1)
    doc_a, ids = make_document(conn, settings, tmp_path, synth, name="a.pdf")
    doc_b, _ = make_document(
        conn, settings, tmp_path, synthetic_pages(1, n_lines=6), name="b.pdf", title="Второй"
    )
    fake_vlm.texts[(1, 0)] = page_text(synth[0])
    with immediate(conn):
        repo.set_document_status(conn, doc_b, DocumentStatus.READY)
        conn.execute(
            "UPDATE pages SET retry_at = ?, retry_count = 3 WHERE id = ?",
            ("2999-01-01T00:00:00+00:00", ids[0]),
        )
        repo.transition_document(conn, doc_a, "queued", "processing")
    built: list[str] = []

    def fake_build(document_id: str, ctx: object, *, kind: object) -> int:
        built.append(document_id)
        with immediate(conn):
            repo.set_document_status(conn, document_id, DocumentStatus.DONE)
        return 1

    monkeypatch.setattr(worker_module, "build_document", fake_build)
    monkeypatch.setattr(worker_module, "export_document", lambda *a, **k: Path("."))
    worker = make_worker(conn, settings, fake_vlm)
    assert await worker.round()
    assert built == [doc_b]
    assert fake_vlm.calls == []
    assert repo.get_page(conn, ids[0])["status"] == "pending"
    assert utc_now() < "2999"
