"""Backend SQLite para storage RNFE."""

from __future__ import annotations

import json
import sqlite3
import threading
from typing import Sequence

from runtime.core.event_log_sqlite import EventLogSQLite

from ..interfaces import StorageBackend
from ..records import (
    ArtifactRecord,
    ReasoningTraceRecord,
    SessionBridgeRecord,
    StoredEvent,
    TelemetrySnapshotRecord,
)


def _safe_json_load(raw: str | None) -> dict:
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if isinstance(data, dict):
        return data
    return {"value": data}


class SQLiteStorageBackend(StorageBackend):
    """Implementacion SQLite local/fallback con compatibilidad legacy."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._ledger = EventLogSQLite(db_path)
        self._ensure_aux_schema()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _ensure_aux_schema(self) -> None:
        with self._lock, self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS telemetry_snapshots (
                    snapshot_id TEXT PRIMARY KEY,
                    run_id TEXT,
                    metrics TEXT NOT NULL,
                    snapshot_ts TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS reasoning_traces (
                    trace_id TEXT PRIMARY KEY,
                    run_id TEXT,
                    step_index INTEGER NOT NULL,
                    family TEXT NOT NULL,
                    status TEXT NOT NULL,
                    detail TEXT,
                    trace_ts TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    episode_id TEXT NOT NULL,
                    channel TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    metadata TEXT
                );

                CREATE TABLE IF NOT EXISTS artifacts (
                    artifact_id TEXT PRIMARY KEY,
                    run_id TEXT,
                    kind TEXT NOT NULL,
                    rel_path TEXT NOT NULL,
                    abs_path TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    mime_type TEXT,
                    metadata TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_telemetry_run_ts
                ON telemetry_snapshots(run_id, snapshot_ts);

                CREATE INDEX IF NOT EXISTS idx_reasoning_run_step
                ON reasoning_traces(run_id, step_index);

                CREATE INDEX IF NOT EXISTS idx_artifacts_run_kind
                ON artifacts(run_id, kind, created_at);
                """
            )
            conn.commit()

    def append_event(self, event: StoredEvent) -> StoredEvent:
        payload = dict(event.payload or {})
        if event.run_id and "run_id" not in payload:
            payload["run_id"] = event.run_id
        if event.source and "source" not in payload:
            payload["source"] = event.source
        with self._lock:
            self._ledger.log_event(event.event_type, payload, event.timestamp)
        return event

    def list_events(
        self,
        *,
        limit: int = 200,
        event_types: Sequence[str] | None = None,
        run_id: str | None = None,
    ) -> list[StoredEvent]:
        rows = self._ledger.get_events(limit=limit, event_types=list(event_types or []))
        events: list[StoredEvent] = []
        for row in rows:
            payload = row.get("payload")
            payload_dict = payload if isinstance(payload, dict) else {"value": payload}
            item = StoredEvent(
                event_type=row.get("event", "unknown"),
                payload=payload_dict,
                timestamp=row.get("timestamp") or "",
                run_id=payload_dict.get("run_id") or payload_dict.get("_run_id"),
                source=payload_dict.get("source"),
            )
            if run_id and item.run_id != run_id:
                continue
            events.append(item)
        return events[-limit:]

    def write_telemetry_snapshot(
        self, snapshot: TelemetrySnapshotRecord
    ) -> TelemetrySnapshotRecord:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO telemetry_snapshots
                (snapshot_id, run_id, metrics, snapshot_ts)
                VALUES (?, ?, ?, ?)
                """,
                (
                    snapshot.snapshot_id,
                    snapshot.run_id,
                    json.dumps(snapshot.metrics),
                    snapshot.timestamp,
                ),
            )
            conn.commit()
        return snapshot

    def list_telemetry_snapshots(
        self, *, run_id: str | None = None, limit: int = 200
    ) -> list[TelemetrySnapshotRecord]:
        query = (
            "SELECT snapshot_id, run_id, metrics, snapshot_ts "
            "FROM telemetry_snapshots"
        )
        params: list[object] = []
        if run_id:
            query += " WHERE run_id = ?"
            params.append(run_id)
        query += " ORDER BY snapshot_ts ASC LIMIT ?"
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [
            TelemetrySnapshotRecord(
                snapshot_id=row[0],
                run_id=row[1],
                metrics=_safe_json_load(row[2]),
                timestamp=row[3],
            )
            for row in rows
        ]

    def append_reasoning_trace(
        self, trace: ReasoningTraceRecord
    ) -> ReasoningTraceRecord:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO reasoning_traces
                (trace_id, run_id, step_index, family, status, detail, trace_ts)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trace.trace_id,
                    trace.run_id,
                    trace.step_index,
                    trace.family,
                    trace.status,
                    json.dumps(trace.detail),
                    trace.timestamp,
                ),
            )
            conn.commit()
        return trace

    def list_reasoning_traces(
        self, *, run_id: str | None = None, limit: int = 200
    ) -> list[ReasoningTraceRecord]:
        query = (
            "SELECT trace_id, run_id, step_index, family, status, detail, trace_ts "
            "FROM reasoning_traces"
        )
        params: list[object] = []
        if run_id:
            query += " WHERE run_id = ?"
            params.append(run_id)
        query += " ORDER BY trace_ts ASC, step_index ASC LIMIT ?"
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [
            ReasoningTraceRecord(
                trace_id=row[0],
                run_id=row[1],
                step_index=row[2],
                family=row[3],
                status=row[4],
                detail=_safe_json_load(row[5]),
                timestamp=row[6],
            )
            for row in rows
        ]

    def register_artifact(self, artifact: ArtifactRecord) -> ArtifactRecord:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO artifacts
                (artifact_id, run_id, kind, rel_path, abs_path, sha256, size_bytes,
                 mime_type, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    artifact.artifact_id,
                    artifact.run_id,
                    artifact.kind,
                    artifact.rel_path,
                    artifact.abs_path,
                    artifact.sha256,
                    artifact.size_bytes,
                    artifact.mime_type,
                    json.dumps(artifact.metadata),
                    artifact.created_at,
                ),
            )
            conn.commit()
        return artifact

    def list_artifacts(
        self,
        *,
        run_id: str | None = None,
        kind: str | None = None,
        limit: int = 200,
    ) -> list[ArtifactRecord]:
        query = (
            "SELECT artifact_id, run_id, kind, rel_path, abs_path, sha256, "
            "size_bytes, mime_type, metadata, created_at FROM artifacts"
        )
        clauses: list[str] = []
        params: list[object] = []
        if run_id:
            clauses.append("run_id = ?")
            params.append(run_id)
        if kind:
            clauses.append("kind = ?")
            params.append(kind)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [
            ArtifactRecord(
                artifact_id=row[0],
                run_id=row[1],
                kind=row[2],
                rel_path=row[3],
                abs_path=row[4],
                sha256=row[5],
                size_bytes=row[6],
                mime_type=row[7],
                metadata=_safe_json_load(row[8]),
                created_at=row[9],
            )
            for row in rows
        ]

    def upsert_session_bridge(
        self, record: SessionBridgeRecord
    ) -> SessionBridgeRecord:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO sessions (session_id, episode_id, channel, timestamp, metadata)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    episode_id=excluded.episode_id,
                    channel=excluded.channel,
                    timestamp=excluded.timestamp,
                    metadata=excluded.metadata
                """,
                (
                    record.session_id,
                    record.episode_id,
                    record.channel,
                    record.timestamp,
                    json.dumps(record.metadata),
                ),
            )
            conn.commit()
        return record

    def get_session_bridge(self, session_id: str) -> SessionBridgeRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT session_id, episode_id, channel, timestamp, metadata
                FROM sessions
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()
        if not row:
            return None
        return SessionBridgeRecord(
            session_id=row[0],
            episode_id=row[1],
            channel=row[2],
            timestamp=row[3],
            metadata=_safe_json_load(row[4]),
        )

    def close(self) -> None:
        # SQLite usa conexiones cortas por operacion; no hay pool que cerrar.
        return None
