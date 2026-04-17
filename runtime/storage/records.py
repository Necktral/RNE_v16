"""Tipos de datos persistentes para la capa de storage RNFE."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional


def utc_now_iso() -> str:
    """Devuelve timestamp ISO-8601 en UTC."""
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class StoredEvent:
    event_type: str
    payload: Dict[str, Any]
    timestamp: str = field(default_factory=utc_now_iso)
    run_id: Optional[str] = None
    source: Optional[str] = None
    legacy_db_path: Optional[str] = None
    legacy_event_id: Optional[int] = None
    event_id: Optional[str] = None
    payload_hash: Optional[str] = None


@dataclass(slots=True)
class TelemetrySnapshotRecord:
    snapshot_id: str
    metrics: Dict[str, Any]
    timestamp: str = field(default_factory=utc_now_iso)
    run_id: Optional[str] = None


@dataclass(slots=True)
class ReasoningTraceRecord:
    trace_id: str
    step_index: int
    family: str
    status: str
    detail: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=utc_now_iso)
    run_id: Optional[str] = None


@dataclass(slots=True)
class ArtifactRecord:
    artifact_id: str
    kind: str
    rel_path: str
    abs_path: str
    sha256: str
    size_bytes: int
    created_at: str = field(default_factory=utc_now_iso)
    run_id: Optional[str] = None
    mime_type: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SessionBridgeRecord:
    session_id: str
    episode_id: str
    channel: str
    timestamp: str = field(default_factory=utc_now_iso)
    metadata: Dict[str, Any] = field(default_factory=dict)
