# event_log_sqlite.py – Persistencia de eventos en SQLite para AEON FENIX-Δ
default_db_path = "aeon_event_log.db"

import sqlite3
import threading
import os
import json
from datetime import datetime

class EventLogSQLite:
    def __init__(self, db_path=None):
        self.db_path = db_path or os.environ.get("AEON_EVENT_DB", default_db_path)
        self._lock = threading.Lock()
        self._ensure_schema()

    def _ensure_schema(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                eve_type TEXT NOT NULL,
                payload TEXT,
                timestamp TEXT NOT NULL
            )
            """)
            conn.commit()

    def log_event(self, eve_type, payload, timestamp=None):
        ts = timestamp or datetime.utcnow().isoformat()
        with self._lock, sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO events (eve_type, payload, timestamp) VALUES (?, ?, ?)",
                (eve_type, json.dumps(payload), ts)
            )
            conn.commit()

    def get_events(self, limit=200, event_types=None):
        query = "SELECT eve_type, payload, timestamp FROM events"
        params = []
        if event_types:
            placeholders = ",".join(["?"] * len(event_types))
            query += f" WHERE eve_type IN ({placeholders})"
            params.extend(event_types)
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(query, params).fetchall()
            return [
                {
                    "event": row[0],
                    "payload": json.loads(row[1]) if row[1] else None,
                    "timestamp": row[2]
                }
                for row in rows
            ][::-1]  # reverse for chronological order
