"""Backend hibrido para dual-write (PostgreSQL + SQLite)."""

from __future__ import annotations

from typing import Sequence

from ..interfaces import StorageBackend
from ..records import (
    ArtifactRecord,
    ReasoningTraceRecord,
    SessionBridgeRecord,
    StoredEvent,
    TelemetrySnapshotRecord,
)


class HybridStorageBackend(StorageBackend):
    """Escribe en ambos stores y lee con prioridad configurable."""

    def __init__(
        self,
        *,
        primary: StorageBackend,
        fallback: StorageBackend,
        prefer_primary_reads: bool = True,
        strict_dual_write: bool = False,
    ):
        self.primary = primary
        self.fallback = fallback
        self.prefer_primary_reads = prefer_primary_reads
        self.strict_dual_write = strict_dual_write

    def _dual_write(self, method: str, payload: object) -> object:
        primary_error: Exception | None = None
        fallback_error: Exception | None = None
        primary_result: object | None = None
        fallback_result: object | None = None

        try:
            primary_result = getattr(self.primary, method)(payload)
        except Exception as exc:  # pragma: no cover - cubierto en integracion
            primary_error = exc

        try:
            fallback_result = getattr(self.fallback, method)(payload)
        except Exception as exc:  # pragma: no cover - cubierto en integracion
            fallback_error = exc

        if self.strict_dual_write and (primary_error or fallback_error):
            errors = [e for e in (primary_error, fallback_error) if e]
            raise RuntimeError(
                f"Dual-write estricto fallo en {method}: {[str(e) for e in errors]}"
            )

        if primary_result is not None:
            return primary_result
        if fallback_result is not None:
            return fallback_result
        if primary_error and fallback_error:
            raise RuntimeError(
                f"Dual-write fallo en {method}: {primary_error}; {fallback_error}"
            )
        if primary_error:
            raise primary_error
        if fallback_error:
            raise fallback_error
        return payload

    def _read_with_fallback(self, method: str, **kwargs: object) -> list:
        first = self.primary if self.prefer_primary_reads else self.fallback
        second = self.fallback if self.prefer_primary_reads else self.primary
        try:
            rows = getattr(first, method)(**kwargs)
            if rows:
                return rows
        except Exception:  # pragma: no cover - cubierto en integracion
            pass
        return getattr(second, method)(**kwargs)

    def append_event(self, event: StoredEvent) -> StoredEvent:
        return self._dual_write("append_event", event)  # type: ignore[return-value]

    def list_events(
        self,
        *,
        limit: int = 200,
        event_types: Sequence[str] | None = None,
        run_id: str | None = None,
    ) -> list[StoredEvent]:
        return self._read_with_fallback(
            "list_events",
            limit=limit,
            event_types=event_types,
            run_id=run_id,
        )

    def write_telemetry_snapshot(
        self, snapshot: TelemetrySnapshotRecord
    ) -> TelemetrySnapshotRecord:
        return self._dual_write("write_telemetry_snapshot", snapshot)  # type: ignore[return-value]

    def list_telemetry_snapshots(
        self, *, run_id: str | None = None, limit: int = 200
    ) -> list[TelemetrySnapshotRecord]:
        return self._read_with_fallback(
            "list_telemetry_snapshots",
            run_id=run_id,
            limit=limit,
        )

    def append_reasoning_trace(
        self, trace: ReasoningTraceRecord
    ) -> ReasoningTraceRecord:
        return self._dual_write("append_reasoning_trace", trace)  # type: ignore[return-value]

    def list_reasoning_traces(
        self, *, run_id: str | None = None, limit: int = 200
    ) -> list[ReasoningTraceRecord]:
        return self._read_with_fallback(
            "list_reasoning_traces",
            run_id=run_id,
            limit=limit,
        )

    def register_artifact(self, artifact: ArtifactRecord) -> ArtifactRecord:
        return self._dual_write("register_artifact", artifact)  # type: ignore[return-value]

    def list_artifacts(
        self,
        *,
        run_id: str | None = None,
        kind: str | None = None,
        limit: int = 200,
    ) -> list[ArtifactRecord]:
        return self._read_with_fallback(
            "list_artifacts",
            run_id=run_id,
            kind=kind,
            limit=limit,
        )

    def upsert_session_bridge(
        self, record: SessionBridgeRecord
    ) -> SessionBridgeRecord:
        return self._dual_write("upsert_session_bridge", record)  # type: ignore[return-value]

    def get_session_bridge(self, session_id: str) -> SessionBridgeRecord | None:
        first = self.primary if self.prefer_primary_reads else self.fallback
        second = self.fallback if self.prefer_primary_reads else self.primary
        try:
            row = first.get_session_bridge(session_id)
            if row is not None:
                return row
        except Exception:  # pragma: no cover - cubierto en integracion
            pass
        return second.get_session_bridge(session_id)

    def close(self) -> None:
        self.primary.close()
        self.fallback.close()
