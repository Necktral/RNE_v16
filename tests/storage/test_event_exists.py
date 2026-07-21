"""Consultas de existencia exacta del ledger para SQLite y PostgreSQL."""

from __future__ import annotations

from contextlib import nullcontext

from runtime.storage.backends.postgres_store import PostgresStorageBackend
from runtime.storage.backends.sqlite_store import SQLiteStorageBackend
from runtime.storage.records import StoredEvent


def test_sqlite_event_exists_filters_run_type_and_payload_without_window(tmp_path):
    backend = SQLiteStorageBackend(str(tmp_path / "event-exists.db"))
    backend.append_event(
        StoredEvent(
            event_type="episode.closed",
            run_id="run-target",
            payload={"episode_id": "target"},
        )
    )
    for index in range(501):
        backend.append_event(
            StoredEvent(
                event_type="episode.closed",
                run_id="run-target",
                payload={"episode_id": f"other-{index}"},
            )
        )

    assert backend.event_exists(
        event_type="episode.closed",
        run_id="run-target",
        payload_contains={"episode_id": "target"},
    )
    assert not backend.event_exists(
        event_type="episode.closed",
        run_id="run-other",
        payload_contains={"episode_id": "target"},
    )
    assert not backend.event_exists(
        event_type="episode.closed",
        run_id="run-target",
        payload_contains={"episode_id": "missing"},
    )
    backend.close()


class _Cursor:
    def __init__(self, result: bool):
        self.result = result
        self.query = ""
        self.params = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def execute(self, query, params):
        self.query = query
        self.params = list(params)

    def fetchone(self):
        return {"event_exists": self.result}


class _Connection:
    def __init__(self, cursor: _Cursor):
        self._cursor = cursor

    def cursor(self):
        return nullcontext(self._cursor)


def test_postgres_event_exists_uses_exists_and_jsonb_containment(monkeypatch):
    cursor = _Cursor(True)
    backend = object.__new__(PostgresStorageBackend)
    monkeypatch.setattr(backend, "_connect", lambda: nullcontext(_Connection(cursor)))

    assert backend.event_exists(
        event_type="episode.closed",
        run_id="run-target",
        payload_contains={"episode_id": "target"},
    )

    normalized_query = " ".join(cursor.query.split())
    assert normalized_query.startswith("SELECT EXISTS (SELECT 1 FROM ledger_events WHERE")
    assert "event_type = %s" in normalized_query
    assert "run_id = %s" in normalized_query
    assert "payload_jsonb @> %s" in normalized_query
    assert "ORDER BY" not in normalized_query
    assert "LIMIT" not in normalized_query
    assert cursor.params[:2] == ["episode.closed", "run-target"]
    assert cursor.params[2].obj == {"episode_id": "target"}
