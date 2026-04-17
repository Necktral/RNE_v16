"""Interfaces de persistencia para runtime/storage."""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from .records import (
    ArtifactRecord,
    ReasoningTraceRecord,
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
class StorageBackend(
    LedgerStore,
    TelemetryStore,
    ReasoningTraceStore,
    ArtifactIndexStore,
    SessionStore,
    Protocol,
):
    def close(self) -> None:
        ...
