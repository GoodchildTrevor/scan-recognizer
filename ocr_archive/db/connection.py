"""SQLite connections in WAL mode and schema migrations via PRAGMA user_version."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

SCHEMA_PATH = Path(__file__).with_name("schema.sql")

#: Each entry migrates the schema from version i to i + 1.
MIGRATIONS: list[str] = [SCHEMA_PATH.read_text(encoding="utf-8")]
SCHEMA_VERSION = len(MIGRATIONS)


def utc_now() -> str:
    """Return the current time as ISO 8601 UTC."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def connect(db_path: Path) -> sqlite3.Connection:
    """Open a connection with WAL, foreign keys and a busy timeout.

    The connection is in autocommit mode; write transactions are opened explicitly with
    :func:`immediate`, so a transaction is never held while the model is queried.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, isolation_level=None, check_same_thread=False, timeout=5.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


@contextmanager
def immediate(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """Run a block inside ``BEGIN IMMEDIATE``; commit on success, roll back on error.

    Nested use joins the outer transaction.
    """
    if conn.in_transaction:
        yield conn
        return
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")


@contextmanager
def read_snapshot(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """Run a block inside one read transaction, so all reads see the same state."""
    if conn.in_transaction:
        yield conn
        return
    conn.execute("BEGIN")
    try:
        yield conn
    finally:
        conn.execute("COMMIT")


def schema_version(conn: sqlite3.Connection) -> int:
    """Return ``PRAGMA user_version``."""
    return int(conn.execute("PRAGMA user_version").fetchone()[0])


def migrate(conn: sqlite3.Connection) -> int:
    """Apply missing migrations and return the resulting schema version.

    The api and the worker may start together on an empty database: when another process
    applied the migration first, the version is read again and the step is skipped.
    """
    while (version := schema_version(conn)) < SCHEMA_VERSION:
        try:
            conn.executescript(
                "BEGIN IMMEDIATE;\n"
                + MIGRATIONS[version]
                + f"\nPRAGMA user_version={version + 1};\nCOMMIT;"
            )
        except sqlite3.OperationalError:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            if schema_version(conn) == version:
                raise
    return schema_version(conn)


def open_db(db_path: Path) -> sqlite3.Connection:
    """Connect and migrate."""
    conn = connect(db_path)
    migrate(conn)
    return conn
