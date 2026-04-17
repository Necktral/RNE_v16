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
    EpisodeCertificateRecord,
    MemoryRecord,
    PromotionDecisionRecord,
    ReasoningTraceRecord,
    RealityAssessmentRecord,
    RealityBenchRunRecord,
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

                CREATE TABLE IF NOT EXISTS reality_assessments (
                    assessment_id TEXT PRIMARY KEY,
                    run_id TEXT,
                    bench_run_id TEXT,
                    episode_id TEXT NOT NULL,
                    closure_passed INTEGER NOT NULL,
                    continuity_score REAL NOT NULL,
                    trace_integrity INTEGER NOT NULL,
                    collapse_detected INTEGER NOT NULL,
                    details TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS reality_bench_runs (
                    bench_run_id TEXT PRIMARY KEY,
                    run_id TEXT,
                    total_episodes INTEGER NOT NULL,
                    closure_rate REAL NOT NULL,
                    continuity_mean REAL NOT NULL,
                    collapse_count INTEGER NOT NULL,
                    gate_profile TEXT NOT NULL,
                    passed INTEGER NOT NULL,
                    summary TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS episode_certificates (
                    certificate_id TEXT PRIMARY KEY,
                    episode_id TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    trace_id TEXT NOT NULL,
                    smg_artifacts TEXT NOT NULL,
                    lotf_artifacts TEXT NOT NULL,
                    world_artifacts TEXT NOT NULL,
                    continuity_score REAL NOT NULL,
                    ioc_proxy REAL NOT NULL,
                    risk_score REAL NOT NULL,
                    verdict TEXT NOT NULL,
                    rollback_ready INTEGER NOT NULL,
                    promotion_candidate INTEGER NOT NULL,
                    metadata TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS promotion_decisions (
                    decision_id TEXT PRIMARY KEY,
                    episode_id TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    certificate_id TEXT NOT NULL,
                    verdict TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    rollback_ready INTEGER NOT NULL,
                    metadata TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS memory_records (
                    memory_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    episode_id TEXT NOT NULL,
                    scale TEXT NOT NULL,
                    structure_json TEXT NOT NULL,
                    ttl_seconds INTEGER,
                    no_interference INTEGER NOT NULL,
                    certificate_id TEXT,
                    ioc_proxy REAL,
                    support_count INTEGER NOT NULL,
                    metadata TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_telemetry_run_ts
                ON telemetry_snapshots(run_id, snapshot_ts);

                CREATE INDEX IF NOT EXISTS idx_reasoning_run_step
                ON reasoning_traces(run_id, step_index);

                CREATE INDEX IF NOT EXISTS idx_artifacts_run_kind
                ON artifacts(run_id, kind, created_at);

                CREATE INDEX IF NOT EXISTS idx_reality_assessments_run_bench
                ON reality_assessments(run_id, bench_run_id, created_at);

                CREATE INDEX IF NOT EXISTS idx_reality_bench_runs_run
                ON reality_bench_runs(run_id, created_at);

                CREATE INDEX IF NOT EXISTS idx_episode_certificates_run
                ON episode_certificates(run_id, created_at);

                CREATE INDEX IF NOT EXISTS idx_promotion_decisions_run
                ON promotion_decisions(run_id, created_at);

                CREATE INDEX IF NOT EXISTS idx_memory_records_run_scale
                ON memory_records(run_id, scale, created_at);
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

    def write_reality_assessment(
        self, assessment: RealityAssessmentRecord
    ) -> RealityAssessmentRecord:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO reality_assessments
                (assessment_id, run_id, bench_run_id, episode_id, closure_passed,
                 continuity_score, trace_integrity, collapse_detected, details, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    assessment.assessment_id,
                    assessment.run_id,
                    assessment.bench_run_id,
                    assessment.episode_id,
                    int(assessment.closure_passed),
                    assessment.continuity_score,
                    int(assessment.trace_integrity),
                    int(assessment.collapse_detected),
                    json.dumps(assessment.details),
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
            "continuity_score, trace_integrity, collapse_detected, details, created_at "
            "FROM reality_assessments"
        )
        clauses: list[str] = []
        params: list[object] = []
        if run_id:
            clauses.append("run_id = ?")
            params.append(run_id)
        if bench_run_id:
            clauses.append("bench_run_id = ?")
            params.append(bench_run_id)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY created_at ASC LIMIT ?"
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [
            RealityAssessmentRecord(
                assessment_id=row[0],
                run_id=row[1],
                bench_run_id=row[2],
                episode_id=row[3],
                closure_passed=bool(row[4]),
                continuity_score=float(row[5]),
                trace_integrity=bool(row[6]),
                collapse_detected=bool(row[7]),
                details=_safe_json_load(row[8]),
                created_at=row[9],
            )
            for row in rows
        ]

    def write_reality_bench_run(
        self, bench_run: RealityBenchRunRecord
    ) -> RealityBenchRunRecord:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO reality_bench_runs
                (bench_run_id, run_id, total_episodes, closure_rate, continuity_mean,
                 collapse_count, gate_profile, passed, summary, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    bench_run.bench_run_id,
                    bench_run.run_id,
                    bench_run.total_episodes,
                    bench_run.closure_rate,
                    bench_run.continuity_mean,
                    bench_run.collapse_count,
                    bench_run.gate_profile,
                    int(bench_run.passed),
                    json.dumps(bench_run.summary),
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
            "collapse_count, gate_profile, passed, summary, created_at FROM reality_bench_runs"
        )
        params: list[object] = []
        if run_id:
            query += " WHERE run_id = ?"
            params.append(run_id)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [
            RealityBenchRunRecord(
                bench_run_id=row[0],
                run_id=row[1],
                total_episodes=int(row[2]),
                closure_rate=float(row[3]),
                continuity_mean=float(row[4]),
                collapse_count=int(row[5]),
                gate_profile=row[6],
                passed=bool(row[7]),
                summary=_safe_json_load(row[8]),
                created_at=row[9],
            )
            for row in rows
        ]

    def write_episode_certificate(
        self, certificate: EpisodeCertificateRecord
    ) -> EpisodeCertificateRecord:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO episode_certificates
                (certificate_id, episode_id, run_id, trace_id, smg_artifacts, lotf_artifacts,
                 world_artifacts, continuity_score, ioc_proxy, risk_score, verdict, rollback_ready,
                 promotion_candidate, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    certificate.certificate_id,
                    certificate.episode_id,
                    certificate.run_id,
                    certificate.trace_id,
                    json.dumps(certificate.smg_artifacts),
                    json.dumps(certificate.lotf_artifacts),
                    json.dumps(certificate.world_artifacts),
                    certificate.continuity_score,
                    certificate.ioc_proxy,
                    certificate.risk_score,
                    certificate.verdict,
                    int(certificate.rollback_ready),
                    int(certificate.promotion_candidate),
                    json.dumps(certificate.metadata),
                    certificate.created_at,
                ),
            )
            conn.commit()
        return certificate

    def get_episode_certificate(
        self, *, certificate_id: str | None = None, episode_id: str | None = None
    ) -> EpisodeCertificateRecord | None:
        if not certificate_id and not episode_id:
            raise ValueError("certificate_id o episode_id es obligatorio")
        query = (
            "SELECT certificate_id, episode_id, run_id, trace_id, smg_artifacts, "
            "lotf_artifacts, world_artifacts, continuity_score, ioc_proxy, risk_score, "
            "verdict, rollback_ready, promotion_candidate, metadata, created_at "
            "FROM episode_certificates"
        )
        params: list[object] = []
        if certificate_id:
            query += " WHERE certificate_id = ?"
            params.append(certificate_id)
        elif episode_id:
            query += " WHERE episode_id = ?"
            params.append(episode_id)
        query += " ORDER BY created_at DESC LIMIT 1"
        with self._connect() as conn:
            row = conn.execute(query, params).fetchone()
        if not row:
            return None
        return EpisodeCertificateRecord(
            certificate_id=row[0],
            episode_id=row[1],
            run_id=row[2],
            trace_id=row[3],
            smg_artifacts=_safe_json_load(row[4]),
            lotf_artifacts=_safe_json_load(row[5]),
            world_artifacts=_safe_json_load(row[6]),
            continuity_score=float(row[7]),
            ioc_proxy=float(row[8]),
            risk_score=float(row[9]),
            verdict=row[10],
            rollback_ready=bool(row[11]),
            promotion_candidate=bool(row[12]),
            metadata=_safe_json_load(row[13]),
            created_at=row[14],
        )

    def list_episode_certificates(
        self, *, run_id: str | None = None, limit: int = 200
    ) -> list[EpisodeCertificateRecord]:
        query = (
            "SELECT certificate_id, episode_id, run_id, trace_id, smg_artifacts, "
            "lotf_artifacts, world_artifacts, continuity_score, ioc_proxy, risk_score, "
            "verdict, rollback_ready, promotion_candidate, metadata, created_at "
            "FROM episode_certificates"
        )
        params: list[object] = []
        if run_id:
            query += " WHERE run_id = ?"
            params.append(run_id)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [
            EpisodeCertificateRecord(
                certificate_id=row[0],
                episode_id=row[1],
                run_id=row[2],
                trace_id=row[3],
                smg_artifacts=_safe_json_load(row[4]),
                lotf_artifacts=_safe_json_load(row[5]),
                world_artifacts=_safe_json_load(row[6]),
                continuity_score=float(row[7]),
                ioc_proxy=float(row[8]),
                risk_score=float(row[9]),
                verdict=row[10],
                rollback_ready=bool(row[11]),
                promotion_candidate=bool(row[12]),
                metadata=_safe_json_load(row[13]),
                created_at=row[14],
            )
            for row in rows
        ]

    def write_promotion_decision(
        self, decision: PromotionDecisionRecord
    ) -> PromotionDecisionRecord:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO promotion_decisions
                (decision_id, episode_id, run_id, certificate_id, verdict, reason,
                 rollback_ready, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    decision.decision_id,
                    decision.episode_id,
                    decision.run_id,
                    decision.certificate_id,
                    decision.verdict,
                    decision.reason,
                    int(decision.rollback_ready),
                    json.dumps(decision.metadata),
                    decision.created_at,
                ),
            )
            conn.commit()
        return decision

    def list_promotion_decisions(
        self, *, run_id: str | None = None, limit: int = 200
    ) -> list[PromotionDecisionRecord]:
        query = (
            "SELECT decision_id, episode_id, run_id, certificate_id, verdict, reason, "
            "rollback_ready, metadata, created_at FROM promotion_decisions"
        )
        params: list[object] = []
        if run_id:
            query += " WHERE run_id = ?"
            params.append(run_id)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [
            PromotionDecisionRecord(
                decision_id=row[0],
                episode_id=row[1],
                run_id=row[2],
                certificate_id=row[3],
                verdict=row[4],
                reason=row[5],
                rollback_ready=bool(row[6]),
                metadata=_safe_json_load(row[7]),
                created_at=row[8],
            )
            for row in rows
        ]

    def write_memory_record(self, memory: MemoryRecord) -> MemoryRecord:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO memory_records
                (memory_id, run_id, episode_id, scale, structure_json, ttl_seconds,
                 no_interference, certificate_id, ioc_proxy, support_count, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    memory.memory_id,
                    memory.run_id,
                    memory.episode_id,
                    memory.scale,
                    json.dumps(memory.structure_json),
                    memory.ttl_seconds,
                    int(memory.no_interference),
                    memory.certificate_id,
                    memory.ioc_proxy,
                    memory.support_count,
                    json.dumps(memory.metadata),
                    memory.created_at,
                ),
            )
            conn.commit()
        return memory

    def retrieve_memory_records(
        self,
        *,
        run_id: str | None = None,
        scales: Sequence[str] | None = None,
        min_ioc_proxy: float | None = None,
        limit: int = 50,
    ) -> list[MemoryRecord]:
        query = (
            "SELECT memory_id, run_id, episode_id, scale, structure_json, ttl_seconds, "
            "no_interference, certificate_id, ioc_proxy, support_count, metadata, created_at "
            "FROM memory_records"
        )
        clauses: list[str] = []
        params: list[object] = []
        if run_id:
            clauses.append("run_id = ?")
            params.append(run_id)
        if scales:
            placeholders = ",".join(["?"] * len(scales))
            clauses.append(f"scale IN ({placeholders})")
            params.extend(scales)
        if min_ioc_proxy is not None:
            clauses.append("ioc_proxy >= ?")
            params.append(min_ioc_proxy)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [
            MemoryRecord(
                memory_id=row[0],
                run_id=row[1],
                episode_id=row[2],
                scale=row[3],
                structure_json=_safe_json_load(row[4]),
                ttl_seconds=row[5],
                no_interference=bool(row[6]),
                certificate_id=row[7],
                ioc_proxy=float(row[8]) if row[8] is not None else None,
                support_count=int(row[9]),
                metadata=_safe_json_load(row[10]),
                created_at=row[11],
            )
            for row in rows
        ]

    def close(self) -> None:
        # SQLite usa conexiones cortas por operacion; no hay pool que cerrar.
        return None
