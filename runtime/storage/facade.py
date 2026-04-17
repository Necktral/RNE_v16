"""Fachada de alto nivel para persistencia del runtime."""

from __future__ import annotations

import hashlib
import mimetypes
from pathlib import Path
from typing import Any, Mapping, Sequence
from uuid import uuid4

from .config import StorageConfig
from .interfaces import StorageBackend
from .records import (
    ArtifactRecord,
    ReasoningTraceRecord,
    SessionBridgeRecord,
    StoredEvent,
    TelemetrySnapshotRecord,
    utc_now_iso,
)


class StorageFacade:
    """API estable y backend-agnostica para runtime y exocortex."""

    def __init__(self, *, backend: StorageBackend, config: StorageConfig):
        self.backend = backend
        self.config = config

    def append_event(
        self,
        *,
        event_type: str,
        payload: Mapping[str, Any] | None,
        timestamp: str | None = None,
        run_id: str | None = None,
        source: str | None = None,
        legacy_db_path: str | None = None,
        legacy_event_id: int | None = None,
        event_id: str | None = None,
    ) -> StoredEvent:
        event = StoredEvent(
            event_id=event_id,
            run_id=run_id,
            event_type=event_type,
            payload=dict(payload or {}),
            timestamp=timestamp or utc_now_iso(),
            source=source,
            legacy_db_path=legacy_db_path,
            legacy_event_id=legacy_event_id,
        )
        return self.backend.append_event(event)

    def list_events(
        self,
        *,
        limit: int = 200,
        event_types: Sequence[str] | None = None,
        run_id: str | None = None,
    ) -> list[StoredEvent]:
        return self.backend.list_events(limit=limit, event_types=event_types, run_id=run_id)

    def write_telemetry_snapshot(
        self,
        *,
        metrics: Mapping[str, Any],
        snapshot_id: str | None = None,
        timestamp: str | None = None,
        run_id: str | None = None,
    ) -> TelemetrySnapshotRecord:
        record = TelemetrySnapshotRecord(
            snapshot_id=snapshot_id or str(uuid4()),
            run_id=run_id,
            metrics=dict(metrics),
            timestamp=timestamp or utc_now_iso(),
        )
        return self.backend.write_telemetry_snapshot(record)

    def list_telemetry_snapshots(
        self, *, run_id: str | None = None, limit: int = 200
    ) -> list[TelemetrySnapshotRecord]:
        return self.backend.list_telemetry_snapshots(run_id=run_id, limit=limit)

    def append_reasoning_trace(
        self,
        *,
        family: str,
        status: str,
        step_index: int,
        detail: Mapping[str, Any] | None = None,
        trace_id: str | None = None,
        timestamp: str | None = None,
        run_id: str | None = None,
    ) -> ReasoningTraceRecord:
        record = ReasoningTraceRecord(
            trace_id=trace_id or str(uuid4()),
            run_id=run_id,
            step_index=step_index,
            family=family,
            status=status,
            detail=dict(detail or {}),
            timestamp=timestamp or utc_now_iso(),
        )
        return self.backend.append_reasoning_trace(record)

    def list_reasoning_traces(
        self, *, run_id: str | None = None, limit: int = 200
    ) -> list[ReasoningTraceRecord]:
        return self.backend.list_reasoning_traces(run_id=run_id, limit=limit)

    def register_artifact(
        self,
        *,
        kind: str,
        abs_path: str | Path,
        run_id: str | None = None,
        artifact_id: str | None = None,
        sha256: str | None = None,
        size_bytes: int | None = None,
        mime_type: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        created_at: str | None = None,
    ) -> ArtifactRecord:
        path = Path(abs_path).resolve()
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Artifact path invalida: {path}")

        content = path.read_bytes()
        digest = sha256 or hashlib.sha256(content).hexdigest()
        size = size_bytes if size_bytes is not None else len(content)
        guessed_mime = mime_type or mimetypes.guess_type(path.name)[0]
        try:
            rel_path = str(path.relative_to(self.config.artifact_root))
        except ValueError:
            rel_path = path.name

        record = ArtifactRecord(
            artifact_id=artifact_id or str(uuid4()),
            run_id=run_id,
            kind=kind,
            rel_path=rel_path,
            abs_path=str(path),
            sha256=digest,
            size_bytes=size,
            mime_type=guessed_mime,
            metadata=dict(metadata or {}),
            created_at=created_at or utc_now_iso(),
        )
        return self.backend.register_artifact(record)

    def materialize_artifact(
        self,
        *,
        run_id: str | None,
        kind: str,
        content: bytes | str | Path,
        filename: str | None = None,
        mime_type: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ArtifactRecord:
        if isinstance(content, Path):
            raw = content.read_bytes()
            suffix = content.suffix
        elif isinstance(content, str):
            raw = content.encode("utf-8")
            suffix = Path(filename).suffix if filename else ".txt"
        else:
            raw = bytes(content)
            suffix = Path(filename).suffix if filename else ".bin"

        digest = hashlib.sha256(raw).hexdigest()
        run_segment = run_id or "no-run"
        base_dir = self.config.artifact_root / run_segment / kind / digest[:2] / digest[2:4]
        base_dir.mkdir(parents=True, exist_ok=True)
        target = base_dir / f"{digest}{suffix}"
        target.write_bytes(raw)
        return self.register_artifact(
            kind=kind,
            abs_path=target,
            run_id=run_id,
            sha256=digest,
            size_bytes=len(raw),
            mime_type=mime_type,
            metadata=metadata,
        )

    def list_artifacts(
        self,
        *,
        run_id: str | None = None,
        kind: str | None = None,
        limit: int = 200,
    ) -> list[ArtifactRecord]:
        return self.backend.list_artifacts(run_id=run_id, kind=kind, limit=limit)

    def upsert_session_bridge(
        self,
        *,
        session_id: str,
        episode_id: str,
        channel: str,
        timestamp: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> SessionBridgeRecord:
        record = SessionBridgeRecord(
            session_id=session_id,
            episode_id=episode_id,
            channel=channel,
            timestamp=timestamp or utc_now_iso(),
            metadata=dict(metadata or {}),
        )
        return self.backend.upsert_session_bridge(record)

    def get_session_bridge(self, session_id: str) -> SessionBridgeRecord | None:
        return self.backend.get_session_bridge(session_id)

    def close(self) -> None:
        self.backend.close()
