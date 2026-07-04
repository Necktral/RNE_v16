import sqlite3
from pathlib import Path

from runtime.core.event_log_sqlite import EventLogSQLite


def test_legacy_eve_type_schema_is_migrated(tmp_path: Path):
    db_path = tmp_path / "legacy_events.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                eve_type TEXT NOT NULL,
                payload TEXT,
                timestamp TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "INSERT INTO events (eve_type, payload, timestamp) VALUES ('legacy_evt', '{}', '2026-01-01T00:00:00')"
        )
        conn.commit()

    ledger = EventLogSQLite(str(db_path))
    ledger.log_event("new_evt", {"ok": True}, "2026-01-01T00:00:01")
    events = ledger.get_events(limit=10)
    assert events[0]["event"] == "legacy_evt"
    assert events[1]["event"] == "new_evt"

    with sqlite3.connect(db_path) as conn:
        cols = [row[1] for row in conn.execute("PRAGMA table_info(events)").fetchall()]
        assert "event_type" in cols
        assert "eve_type" not in cols
        legacy_backup_exists = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='events_legacy_eve_type'"
        ).fetchone()
        assert legacy_backup_exists is not None

