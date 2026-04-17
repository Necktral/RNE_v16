# event_log_sqlite.py – Persistencia de eventos en SQLite para AEON FENIX-Δ
default_db_path = "aeon_event_log.db"

import json
import os
import sqlite3
import threading
from datetime import datetime


class EventLogSQLite:
    """
    Ledger SQLite con compatibilidad histórica:
    - esquema canónico: `event_type`
    - esquema legacy: `eve_type`
    """

    def __init__(self, db_path=None):
        self.db_path = db_path or os.environ.get("AEON_EVENT_DB", default_db_path)
        self._lock = threading.Lock()
        self._event_column = "event_type"
        self._ensure_schema()

    def _list_columns(self, conn, table_name):
        rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        return [row[1] for row in rows]

    def _ensure_schema(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    payload TEXT,
                    timestamp TEXT NOT NULL
                )
                """
            )
            cols = self._list_columns(conn, "events")
            has_event_type = "event_type" in cols
            has_eve_type = "eve_type" in cols

            # Migracion one-way: eve_type -> event_type
            if has_eve_type and not has_event_type:
                conn.execute("ALTER TABLE events RENAME TO events_legacy_eve_type")
                conn.execute(
                    """
                    CREATE TABLE events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        event_type TEXT NOT NULL,
                        payload TEXT,
                        timestamp TEXT NOT NULL
                    )
                    """
                )
                conn.execute(
                    """
                    INSERT INTO events (id, event_type, payload, timestamp)
                    SELECT id, eve_type, payload, timestamp
                    FROM events_legacy_eve_type
                    """
                )
                has_event_type = True
                has_eve_type = False

            self._event_column = "event_type" if has_event_type else "eve_type"
            conn.commit()

    def log_event(self, event_type, payload, timestamp=None):
        ts = timestamp or datetime.utcnow().isoformat()
        with self._lock, sqlite3.connect(self.db_path) as conn:
            conn.execute(
                f"INSERT INTO events ({self._event_column}, payload, timestamp) VALUES (?, ?, ?)",
                (event_type, json.dumps(payload), ts),
            )
            conn.commit()

    def get_events(self, limit=200, event_types=None):
        query = f"SELECT {self._event_column}, payload, timestamp FROM events"
        params = []
        if event_types:
            placeholders = ",".join(["?"] * len(event_types))
            query += f" WHERE {self._event_column} IN ({placeholders})"
            params.extend(event_types)
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(query, params).fetchall()
        return [
            {
                "event": row[0],
                "payload": json.loads(row[1]) if row[1] else None,
                "timestamp": row[2],
            }
            for row in rows
        ][::-1]
