"""Regresión B38-C12: toda conexión SQLite del runtime lleva los PRAGMA estándar.

Bug histórico: ninguno de los ``sqlite3.connect`` del runtime ejecutaba PRAGMA
de conexión — sin WAL ni busy_timeout, la amplificación de escritura por
episodio (SMG + EventBus + court_runtime + artifacts) producía "database is
locked" bajo concurrencia; sin foreign_keys=ON no había integridad referencial.

Fuente única de verdad: ``runtime.core.event_log_sqlite.configure_sqlite_connection``.
"""

import sqlite3

from runtime.core.event_log_sqlite import (
    EventLogSQLite,
    configure_sqlite_connection,
)
from runtime.storage.backends.sqlite_store import SQLiteStorageBackend
from runtime.storage.records import StoredEvent

# synchronous: 0=OFF, 1=NORMAL, 2=FULL, 3=EXTRA
_SYNCHRONOUS_NORMAL = 1


def _assert_pragmas(conn: sqlite3.Connection) -> None:
    assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
    assert conn.execute("PRAGMA synchronous").fetchone()[0] == _SYNCHRONOUS_NORMAL
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_helper_sets_all_four_pragmas(tmp_path):
    conn = configure_sqlite_connection(sqlite3.connect(tmp_path / "helper.db"))
    try:
        _assert_pragmas(conn)
    finally:
        conn.close()


def test_helper_returns_same_connection(tmp_path):
    raw = sqlite3.connect(tmp_path / "same.db")
    try:
        assert configure_sqlite_connection(raw) is raw
    finally:
        raw.close()


def test_sqlite_backend_connect_applies_pragmas(tmp_path):
    backend = SQLiteStorageBackend(str(tmp_path / "backend.db"))
    conn = backend._connect()
    try:
        _assert_pragmas(conn)
    finally:
        conn.close()


def test_event_log_leaves_db_in_wal_mode(tmp_path):
    """WAL es persistente en el fichero: tras usar el ledger, cualquier
    conexión posterior ve journal_mode=wal (evidencia de que las conexiones
    internas de EventLogSQLite pasaron por el helper)."""
    db_path = str(tmp_path / "ledger.db")
    ledger = EventLogSQLite(db_path)
    ledger.log_event("smoke", {"run_id": "R-pragma"})
    assert ledger.get_events(limit=1)[0]["event"] == "smoke"

    raw = sqlite3.connect(db_path)
    try:
        assert raw.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    finally:
        raw.close()


def test_backend_roundtrip_still_works_with_pragmas(tmp_path):
    """Smoke: el backend sigue escribiendo/leyendo eventos con los PRAGMA activos."""
    backend = SQLiteStorageBackend(str(tmp_path / "roundtrip.db"))
    backend.append_event(
        StoredEvent(event_type="episode.closed", payload={}, run_id="R-1")
    )
    found = backend.list_events(run_id="R-1", limit=5)
    assert len(found) == 1
    assert found[0].event_type == "episode.closed"
