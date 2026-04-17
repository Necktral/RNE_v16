"""Capa de persistencia hibrida para runtime RNFE."""

from __future__ import annotations

import threading

from .config import StorageConfig
from .factory import StorageFactory
from .facade import StorageFacade
from .interfaces import (
    ArtifactIndexStore,
    CertificationStore,
    LedgerStore,
    MemoryStore,
    ReasoningTraceStore,
    RealityStore,
    SessionStore,
    TelemetryStore,
)
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

_storage_lock = threading.Lock()
_storage_singleton: StorageFacade | None = None


def get_storage(
    *,
    config: StorageConfig | None = None,
    refresh: bool = False,
) -> StorageFacade:
    global _storage_singleton
    with _storage_lock:
        if refresh and _storage_singleton is not None:
            _storage_singleton.close()
            _storage_singleton = None
        if _storage_singleton is None:
            effective_config = config or StorageConfig.from_env()
            _storage_singleton = StorageFactory.create_facade(effective_config)
        return _storage_singleton


def reset_storage() -> None:
    global _storage_singleton
    with _storage_lock:
        if _storage_singleton is not None:
            _storage_singleton.close()
            _storage_singleton = None


__all__ = [
    "ArtifactIndexStore",
    "ArtifactRecord",
    "CertificationStore",
    "EpisodeCertificateRecord",
    "LedgerStore",
    "MemoryRecord",
    "MemoryStore",
    "PromotionDecisionRecord",
    "ReasoningTraceRecord",
    "RealityAssessmentRecord",
    "RealityBenchRunRecord",
    "ReasoningTraceStore",
    "RealityStore",
    "SessionBridgeRecord",
    "SessionStore",
    "StorageConfig",
    "StorageFactory",
    "StorageFacade",
    "StoredEvent",
    "TelemetrySnapshotRecord",
    "TelemetryStore",
    "get_storage",
    "reset_storage",
]
