"""Backend PostgreSQL para storage RNFE."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence
from uuid import uuid4

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from ..interfaces import StorageBackend
from ..records import (
    ArtifactRecord,
    ReasoningTraceRecord,
    RealityAssessmentRecord,
    RealityBenchRunRecord,
    SessionBridgeRecord,
    StoredEvent,
    TelemetrySnapshotRecord,
)


def _payload_hash(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


class PostgresStorageBackend(StorageBackend):
    """Store transaccional principal en PostgreSQL."""

    def __init__(self, dsn: str, schema_path: str | None = None):
        self.dsn = dsn
        if schema_path:
            self.schema_path = Path(schema_path)
        else:
            self.schema_path = Path(__file__).resolve().parent / "postgres" / "schema.sql"
        self._ensure_schema()

    def _connect(self) -> psycopg.Connection[Any]:
        return psycopg.connect(self.dsn, row_factory=dict_row)

    def _ensure_schema(self) -> None:
        ddl = self.schema_path.read_text(encoding="utf-8")
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(ddl)
            conn.commit()

    def append_event(self, event: StoredEvent) -> StoredEvent:
        event_id = event.event_id or str(uuid4())
        payload = dict(event.payload or {})
        payload_hash = event.payload_hash or _payload_hash(payload)
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO ledger_events
                (event_id, run_id, event_type, payload_jsonb, event_ts, source,
                 legacy_db_path, legacy_event_id, payload_hash)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT DO NOTHING
                RETURNING event_id
                """,
                (
                    event_id,
                    event.run_id,
                    event.event_type,
                    Jsonb(payload),
                    event.timestamp,
                    event.source,
                    event.legacy_db_path,
                    event.legacy_event_id,
                    payload_hash,
                ),
            )
            inserted = cur.fetchone()
            if not inserted and event.legacy_db_path and event.legacy_event_id is not None:
                cur.execute(
                    """
                    SELECT event_id
                    FROM ledger_events
                    WHERE legacy_db_path = %s AND legacy_event_id = %s
                    """,
                    (event.legacy_db_path, event.legacy_event_id),
                )
                found = cur.fetchone()
                if found:
                    event_id = found["event_id"]
            conn.commit()
        event.event_id = event_id
        event.payload_hash = payload_hash
        return event

    def list_events(
        self,
        *,
        limit: int = 200,
        event_types: Sequence[str] | None = None,
        run_id: str | None = None,
    ) -> list[StoredEvent]:
        query = (
            "SELECT event_id, run_id, event_type, payload_jsonb, event_ts, source, "
            "legacy_db_path, legacy_event_id, payload_hash "
            "FROM ledger_events"
        )
        clauses: list[str] = []
        params: list[Any] = []
        if event_types:
            clauses.append("event_type = ANY(%s)")
            params.append(list(event_types))
        if run_id:
            clauses.append("run_id = %s")
            params.append(run_id)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY event_ts ASC, created_at ASC LIMIT %s"
        params.append(limit)
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()
        return [
            StoredEvent(
                event_id=row["event_id"],
                run_id=row["run_id"],
                event_type=row["event_type"],
                payload=dict(row["payload_jsonb"] or {}),
                timestamp=row["event_ts"],
                source=row["source"],
                legacy_db_path=row["legacy_db_path"],
                legacy_event_id=row["legacy_event_id"],
                payload_hash=row["payload_hash"],
            )
            for row in rows
        ]

    def write_telemetry_snapshot(
        self, snapshot: TelemetrySnapshotRecord
    ) -> TelemetrySnapshotRecord:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO telemetry_snapshots
                (snapshot_id, run_id, metrics_jsonb, snapshot_ts)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT(snapshot_id) DO UPDATE SET
                    run_id = excluded.run_id,
                    metrics_jsonb = excluded.metrics_jsonb,
                    snapshot_ts = excluded.snapshot_ts,
                    created_at = now()
                """,
                (
                    snapshot.snapshot_id,
                    snapshot.run_id,
                    Jsonb(snapshot.metrics),
                    snapshot.timestamp,
                ),
            )
            conn.commit()
        return snapshot

    def list_telemetry_snapshots(
        self, *, run_id: str | None = None, limit: int = 200
    ) -> list[TelemetrySnapshotRecord]:
        query = (
            "SELECT snapshot_id, run_id, metrics_jsonb, snapshot_ts "
            "FROM telemetry_snapshots"
        )
        params: list[Any] = []
        if run_id:
            query += " WHERE run_id = %s"
            params.append(run_id)
        query += " ORDER BY snapshot_ts ASC, created_at ASC LIMIT %s"
        params.append(limit)
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()
        return [
            TelemetrySnapshotRecord(
                snapshot_id=row["snapshot_id"],
                run_id=row["run_id"],
                metrics=dict(row["metrics_jsonb"] or {}),
                timestamp=row["snapshot_ts"],
            )
            for row in rows
        ]

    def append_reasoning_trace(
        self, trace: ReasoningTraceRecord
    ) -> ReasoningTraceRecord:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO reasoning_traces
                (trace_id, run_id, step_index, family, status, detail_jsonb, trace_ts)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT(trace_id) DO UPDATE SET
                    run_id = excluded.run_id,
                    step_index = excluded.step_index,
                    family = excluded.family,
                    status = excluded.status,
                    detail_jsonb = excluded.detail_jsonb,
                    trace_ts = excluded.trace_ts,
                    created_at = now()
                """,
                (
                    trace.trace_id,
                    trace.run_id,
                    trace.step_index,
                    trace.family,
                    trace.status,
                    Jsonb(trace.detail),
                    trace.timestamp,
                ),
            )
            conn.commit()
        return trace

    def list_reasoning_traces(
        self, *, run_id: str | None = None, limit: int = 200
    ) -> list[ReasoningTraceRecord]:
        query = (
            "SELECT trace_id, run_id, step_index, family, status, detail_jsonb, trace_ts "
            "FROM reasoning_traces"
        )
        params: list[Any] = []
        if run_id:
            query += " WHERE run_id = %s"
            params.append(run_id)
        query += " ORDER BY trace_ts ASC, step_index ASC LIMIT %s"
        params.append(limit)
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()
        return [
            ReasoningTraceRecord(
                trace_id=row["trace_id"],
                run_id=row["run_id"],
                step_index=row["step_index"],
                family=row["family"],
                status=row["status"],
                detail=dict(row["detail_jsonb"] or {}),
                timestamp=row["trace_ts"],
            )
            for row in rows
        ]

    def register_artifact(self, artifact: ArtifactRecord) -> ArtifactRecord:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO artifacts
                (artifact_id, run_id, kind, rel_path, abs_path, sha256, size_bytes,
                 mime_type, metadata_jsonb, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT(artifact_id) DO UPDATE SET
                    run_id = excluded.run_id,
                    kind = excluded.kind,
                    rel_path = excluded.rel_path,
                    abs_path = excluded.abs_path,
                    sha256 = excluded.sha256,
                    size_bytes = excluded.size_bytes,
                    mime_type = excluded.mime_type,
                    metadata_jsonb = excluded.metadata_jsonb,
                    created_at = excluded.created_at
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
                    Jsonb(artifact.metadata),
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
            "size_bytes, mime_type, metadata_jsonb, created_at FROM artifacts"
        )
        clauses: list[str] = []
        params: list[Any] = []
        if run_id:
            clauses.append("run_id = %s")
            params.append(run_id)
        if kind:
            clauses.append("kind = %s")
            params.append(kind)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY created_at DESC LIMIT %s"
        params.append(limit)
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()
        return [
            ArtifactRecord(
                artifact_id=row["artifact_id"],
                run_id=row["run_id"],
                kind=row["kind"],
                rel_path=row["rel_path"],
                abs_path=row["abs_path"],
                sha256=row["sha256"],
                size_bytes=row["size_bytes"],
                mime_type=row["mime_type"],
                metadata=dict(row["metadata_jsonb"] or {}),
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def upsert_session_bridge(
        self, record: SessionBridgeRecord
    ) -> SessionBridgeRecord:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO sessions
                (session_id, episode_id, channel, metadata_jsonb, timestamp)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT(session_id) DO UPDATE SET
                    episode_id = excluded.episode_id,
                    channel = excluded.channel,
                    metadata_jsonb = excluded.metadata_jsonb,
                    timestamp = excluded.timestamp,
                    updated_at = now()
                """,
                (
                    record.session_id,
                    record.episode_id,
                    record.channel,
                    Jsonb(record.metadata),
                    record.timestamp,
                ),
            )
            conn.commit()
        return record

    def get_session_bridge(self, session_id: str) -> SessionBridgeRecord | None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT session_id, episode_id, channel, metadata_jsonb, timestamp
                FROM sessions
                WHERE session_id = %s
                """,
                (session_id,),
            )
            row = cur.fetchone()
        if not row:
            return None
        return SessionBridgeRecord(
            session_id=row["session_id"],
            episode_id=row["episode_id"],
            channel=row["channel"],
            metadata=dict(row["metadata_jsonb"] or {}),
            timestamp=row["timestamp"],
        )

    def write_reality_assessment(
        self, assessment: RealityAssessmentRecord
    ) -> RealityAssessmentRecord:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO reality_assessments
                (assessment_id, run_id, bench_run_id, episode_id, closure_passed,
                 continuity_score, trace_integrity, collapse_detected, details_jsonb, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT(assessment_id) DO UPDATE SET
                    run_id = excluded.run_id,
                    bench_run_id = excluded.bench_run_id,
                    episode_id = excluded.episode_id,
                    closure_passed = excluded.closure_passed,
                    continuity_score = excluded.continuity_score,
                    trace_integrity = excluded.trace_integrity,
                    collapse_detected = excluded.collapse_detected,
                    details_jsonb = excluded.details_jsonb,
                    created_at = excluded.created_at
                """,
                (
                    assessment.assessment_id,
                    assessment.run_id,
                    assessment.bench_run_id,
                    assessment.episode_id,
                    assessment.closure_passed,
                    assessment.continuity_score,
                    assessment.trace_integrity,
                    assessment.collapse_detected,
                    Jsonb(assessment.details),
                    assessment.created_at,
                ),
            )
            conn.commit()
        return assessment

    def list_reality_assessments(
        self,
        *,
        run_id: str | None = None,
        bench_run_id: str | None = None,
        limit: int = 200,
    ) -> list[RealityAssessmentRecord]:
        query = (
            "SELECT assessment_id, run_id, bench_run_id, episode_id, closure_passed, "
            "continuity_score, trace_integrity, collapse_detected, details_jsonb, created_at "
            "FROM reality_assessments"
        )
        clauses: list[str] = []
        params: list[Any] = []
        if run_id:
            clauses.append("run_id = %s")
            params.append(run_id)
        if bench_run_id:
            clauses.append("bench_run_id = %s")
            params.append(bench_run_id)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY created_at ASC LIMIT %s"
        params.append(limit)
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()
        return [
            RealityAssessmentRecord(
                assessment_id=row["assessment_id"],
                run_id=row["run_id"],
                bench_run_id=row["bench_run_id"],
                episode_id=row["episode_id"],
                closure_passed=bool(row["closure_passed"]),
                continuity_score=float(row["continuity_score"]),
                trace_integrity=bool(row["trace_integrity"]),
                collapse_detected=bool(row["collapse_detected"]),
                details=dict(row["details_jsonb"] or {}),
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def write_reality_bench_run(
        self, bench_run: RealityBenchRunRecord
    ) -> RealityBenchRunRecord:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO reality_bench_runs
                (bench_run_id, run_id, total_episodes, closure_rate, continuity_mean,
                 collapse_count, gate_profile, passed, summary_jsonb, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT(bench_run_id) DO UPDATE SET
                    run_id = excluded.run_id,
                    total_episodes = excluded.total_episodes,
                    closure_rate = excluded.closure_rate,
                    continuity_mean = excluded.continuity_mean,
                    collapse_count = excluded.collapse_count,
                    gate_profile = excluded.gate_profile,
                    passed = excluded.passed,
                    summary_jsonb = excluded.summary_jsonb,
                    created_at = excluded.created_at
                """,
                (
                    bench_run.bench_run_id,
                    bench_run.run_id,
                    bench_run.total_episodes,
                    bench_run.closure_rate,
                    bench_run.continuity_mean,
                    bench_run.collapse_count,
                    bench_run.gate_profile,
                    bench_run.passed,
                    Jsonb(bench_run.summary),
                    bench_run.created_at,
                ),
            )
            conn.commit()
        return bench_run

    def list_reality_bench_runs(
        self, *, run_id: str | None = None, limit: int = 50
    ) -> list[RealityBenchRunRecord]:
        query = (
            "SELECT bench_run_id, run_id, total_episodes, closure_rate, continuity_mean, "
            "collapse_count, gate_profile, passed, summary_jsonb, created_at "
            "FROM reality_bench_runs"
        )
        params: list[Any] = []
        if run_id:
            query += " WHERE run_id = %s"
            params.append(run_id)
        query += " ORDER BY created_at DESC LIMIT %s"
        params.append(limit)
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()
        return [
            RealityBenchRunRecord(
                bench_run_id=row["bench_run_id"],
                run_id=row["run_id"],
                total_episodes=int(row["total_episodes"]),
                closure_rate=float(row["closure_rate"]),
                continuity_mean=float(row["continuity_mean"]),
                collapse_count=int(row["collapse_count"]),
                gate_profile=row["gate_profile"],
                passed=bool(row["passed"]),
                summary=dict(row["summary_jsonb"] or {}),
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def close(self) -> None:
        # Conexion por operacion; no hay pool persistente.
        return None
