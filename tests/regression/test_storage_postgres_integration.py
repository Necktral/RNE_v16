import os
from pathlib import Path
import sqlite3

import psycopg
import pytest

from runtime.storage import StorageConfig, StorageFactory
from runtime.storage.migrations import migrate_sqlite_ledger_to_postgres


RUN_PG_TESTS = os.environ.get("RNFE_RUN_PG_TESTS") == "1"

pytestmark = pytest.mark.skipif(
    not RUN_PG_TESTS,
    reason="Set RNFE_RUN_PG_TESTS=1 para ejecutar tests de integracion PostgreSQL.",
)


def _postgres_dsn() -> str:
    dsn = os.environ.get("RNFE_POSTGRES_DSN")
    if not dsn:
        pytest.skip("RNFE_POSTGRES_DSN no esta definido.")
    return dsn


def _postgres_config(tmp_path: Path) -> StorageConfig:
    return StorageConfig(
        mode="postgres",
        sqlite_db_path=str(tmp_path / "unused.db"),
        postgres_dsn=_postgres_dsn(),
        artifact_root=tmp_path / "artifacts",
        prefer_postgres_reads=True,
        strict_dual_write=False,
    )


def test_postgres_storage_crud_minimo(tmp_path: Path):
    storage = StorageFactory.create_facade(_postgres_config(tmp_path))
    run_id = "pg-run-1"

    storage.append_event(
        event_type="pg.event",
        payload={"value": 1},
        run_id=run_id,
        source="pg_test",
    )
    events = storage.list_events(run_id=run_id, limit=20)
    assert any(evt.event_type == "pg.event" for evt in events)

    storage.write_telemetry_snapshot(
        run_id=run_id,
        metrics={"cpu": 0.42},
    )
    snapshots = storage.list_telemetry_snapshots(run_id=run_id, limit=10)
    assert snapshots and snapshots[0].metrics["cpu"] == 0.42

    storage.append_reasoning_trace(
        run_id=run_id,
        step_index=0,
        family="ABD",
        status="ok",
        detail={"origin": "pg"},
    )
    traces = storage.list_reasoning_traces(run_id=run_id, limit=10)
    assert traces and traces[0].family == "ABD"

    storage.upsert_session_bridge(
        session_id="sess-pg-1",
        episode_id="ep-pg-1",
        channel="cli",
        metadata={"mode": "integration"},
    )
    session = storage.get_session_bridge("sess-pg-1")
    assert session is not None
    assert session.channel == "cli"

    artifact = storage.materialize_artifact(
        run_id=run_id,
        kind="weights",
        content=b"0123456789",
        filename="weights.bin",
        metadata={"phase": "integration"},
    )
    assert Path(artifact.abs_path).exists()
    assert artifact.size_bytes == 10
    listed = storage.list_artifacts(run_id=run_id, kind="weights", limit=10)
    assert listed and listed[0].sha256 == artifact.sha256

    storage.close()


def test_sqlite_to_postgres_migration_idempotente(tmp_path: Path):
    dsn = _postgres_dsn()
    sqlite_db = tmp_path / "legacy_events.db"
    sqlite_path = str(sqlite_db.resolve())

    with sqlite3.connect(sqlite_db) as conn:
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
            "INSERT INTO events (eve_type, payload, timestamp) VALUES (?, ?, ?)",
            ("legacy_a", '{"run_id":"mig-run"}', "2026-01-01T00:00:00"),
        )
        conn.execute(
            "INSERT INTO events (eve_type, payload, timestamp) VALUES (?, ?, ?)",
            ("legacy_b", '{"run_id":"mig-run"}', "2026-01-01T00:00:01"),
        )
        conn.commit()

    migrate_sqlite_ledger_to_postgres(
        sqlite_db_path=str(sqlite_db),
        postgres_dsn=dsn,
    )
    migrate_sqlite_ledger_to_postgres(
        sqlite_db_path=str(sqlite_db),
        postgres_dsn=dsn,
    )

    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT COUNT(*)
            FROM ledger_events
            WHERE legacy_db_path = %s
            """,
            (sqlite_path,),
        )
        total = cur.fetchone()[0]
    assert total == 2
