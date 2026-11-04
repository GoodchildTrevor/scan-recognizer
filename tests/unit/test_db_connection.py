"""Schema migration when the api and the worker start together (section 12.4)."""

from __future__ import annotations

from pathlib import Path

import pytest

from ocr_archive.db import connection

pytestmark = pytest.mark.p2


def test_migrate_skips_a_step_another_process_applied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "state.db"
    first = connection.connect(db_path)
    second = connection.connect(db_path)
    try:
        # Both processes read version 0; the first one migrates before the second applies it.
        real = connection.schema_version
        stale = {"left": 1}

        def stale_once(conn: object) -> int:
            if conn is second and stale["left"]:
                stale["left"] -= 1
                return 0
            return real(conn)  # type: ignore[arg-type]

        monkeypatch.setattr(connection, "schema_version", stale_once)
        assert connection.migrate(first) == connection.SCHEMA_VERSION
        assert connection.migrate(second) == connection.SCHEMA_VERSION
        assert not second.in_transaction
        assert second.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 0
    finally:
        first.close()
        second.close()
