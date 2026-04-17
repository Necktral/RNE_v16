"""Interfaces de persistencia para runtime/storage."""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from .records import (
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


@runtime_checkable
class LedgerStore(Protocol):
    def append_event(self, event: StoredEvent) -> StoredEvent:
        ...

    def list_events(
        self,
        *,
        limit: int = 200,
        event_types: Sequence[str] | None = None,
        run_id: str | None = None,
    ) -> list[StoredEvent]:
        ...


@runtime_checkable
class TelemetryStore(Protocol):
    def write_telemetry_snapshot(
        self, snapshot: TelemetrySnapshotRecord
    ) -> TelemetrySnapshotRecord:
        ...

    def list_telemetry_snapshots(
        self, *, run_id: str | None = None, limit: int = 200
    ) -> list[TelemetrySnapshotRecord]:
        ...


@runtime_checkable
class ReasoningTraceStore(Protocol):
    def append_reasoning_trace(
        self, trace: ReasoningTraceRecord
    ) -> ReasoningTraceRecord:
        ...

    def list_reasoning_traces(
        self, *, run_id: str | None = None, limit: int = 200
    ) -> list[ReasoningTraceRecord]:
        ...


@runtime_checkable
class ArtifactIndexStore(Protocol):
    def register_artifact(self, artifact: ArtifactRecord) -> ArtifactRecord:
        ...

    def list_artifacts(
        self,
        *,
        run_id: str | None = None,
        kind: str | None = None,
        limit: int = 200,
    ) -> list[ArtifactRecord]:
        ...


@runtime_checkable
class SessionStore(Protocol):
    def upsert_session_bridge(
        self, record: SessionBridgeRecord
    ) -> SessionBridgeRecord:
        ...

    def get_session_bridge(self, session_id: str) -> SessionBridgeRecord | None:
        ...


@runtime_checkable
class RealityStore(Protocol):
    def write_reality_assessment(
        self, assessment: RealityAssessmentRecord
    ) -> RealityAssessmentRecord:
        ...

    def list_reality_assessments(
        self,
        *,
        run_id: str | None = None,
        bench_run_id: str | None = None,
        limit: int = 200,
    ) -> list[RealityAssessmentRecord]:
        ...

    def write_reality_bench_run(
        self, bench_run: RealityBenchRunRecord
    ) -> RealityBenchRunRecord:
        ...

    def list_reality_bench_runs(
        self, *, run_id: str | None = None, limit: int = 50
    ) -> list[RealityBenchRunRecord]:
        ...


@runtime_checkable
class CertificationStore(Protocol):
    def write_episode_certificate(
        self, certificate: EpisodeCertificateRecord
    ) -> EpisodeCertificateRecord:
        ...

    def get_episode_certificate(
        self, *, certificate_id: str | None = None, episode_id: str | None = None
    ) -> EpisodeCertificateRecord | None:
        ...

    def list_episode_certificates(
        self, *, run_id: str | None = None, limit: int = 200
    ) -> list[EpisodeCertificateRecord]:
        ...

    def write_promotion_decision(
        self, decision: PromotionDecisionRecord
    ) -> PromotionDecisionRecord:
        ...

    def list_promotion_decisions(
        self, *, run_id: str | None = None, limit: int = 200
    ) -> list[PromotionDecisionRecord]:
        ...


@runtime_checkable
class MemoryStore(Protocol):
    def write_memory_record(self, memory: MemoryRecord) -> MemoryRecord:
        ...

    def retrieve_memory_records(
        self,
        *,
        run_id: str | None = None,
        scales: Sequence[str] | None = None,
        min_ioc_proxy: float | None = None,
        limit: int = 50,
    ) -> list[MemoryRecord]:
        ...


@runtime_checkable
class StorageBackend(
    LedgerStore,
    TelemetryStore,
    ReasoningTraceStore,
    ArtifactIndexStore,
    SessionStore,
    RealityStore,
    CertificationStore,
    MemoryStore,
    Protocol,
):
    def close(self) -> None:
        ...
