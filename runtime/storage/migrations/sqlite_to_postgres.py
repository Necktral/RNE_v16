"""Migracion idempotente desde ledger SQLite local hacia PostgreSQL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
from typing import Iterator

from runtime.storage.backends.postgres_store import PostgresStorageBackend
from runtime.storage.records import StoredEvent


def _detect_event_column(conn: sqlite3.Connection) -> str:
    cols = [row[1] for row in conn.execute("PRAGMA table_info(events)").fetchall()]
    if "event_type" in cols:
        return "event_type"
    if "eve_type" in cols:
        return "eve_type"
    raise RuntimeError("No se encontro columna de evento (event_type/eve_type) en SQLite")


def _safe_json(raw: str | None) -> dict:
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if isinstance(data, dict):
        return data
    return {"value": data}


def _iter_events(
    db_path: str,
    *,
    batch_size: int = 1000,
) -> Iterator[list[tuple[int, str, str | None, str]]]:
    with sqlite3.connect(db_path) as conn:
        event_column = _detect_event_column(conn)
        offset = 0
        while True:
            rows = conn.execute(
                f"""
                SELECT id, {event_column}, payload, timestamp
                FROM events
                ORDER BY id ASC
                LIMIT ? OFFSET ?
                """,
                (batch_size, offset),
            ).fetchall()
            if not rows:
                break
            yield rows
            offset += len(rows)


def migrate_sqlite_ledger_to_postgres(
    *,
    sqlite_db_path: str,
    postgres_dsn: str,
    batch_size: int = 1000,
) -> int:
    sqlite_path = Path(sqlite_db_path).resolve()
    legacy_db_path = str(sqlite_path)
    backend = PostgresStorageBackend(postgres_dsn)

    migrated = 0
    for chunk in _iter_events(str(sqlite_path), batch_size=batch_size):
        for row_id, event_type, payload_raw, timestamp in chunk:
            payload = _safe_json(payload_raw)
            event = StoredEvent(
                event_id=f"legacy::{legacy_db_path}::{row_id}",
                event_type=event_type,
                payload=payload,
                timestamp=timestamp,
                run_id=payload.get("run_id") or payload.get("_run_id"),
                source=payload.get("source") or "sqlite_migration",
                legacy_db_path=legacy_db_path,
                legacy_event_id=row_id,
            )
            backend.append_event(event)
            migrated += 1
    backend.close()
    return migrated


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Migra ledger SQLite local a PostgreSQL en modo idempotente."
    )
    parser.add_argument("--sqlite-db", required=True, help="Ruta SQLite origen")
    parser.add_argument("--postgres-dsn", required=True, help="DSN PostgreSQL destino")
    parser.add_argument("--batch-size", type=int, default=1000, help="Tamano de lote")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    migrated = migrate_sqlite_ledger_to_postgres(
        sqlite_db_path=args.sqlite_db,
        postgres_dsn=args.postgres_dsn,
        batch_size=args.batch_size,
    )
    print(f"Migracion completada. Filas procesadas: {migrated}")


if __name__ == "__main__":
    main()
