"""Servicio de persistencia de snapshots de telemetría."""

from __future__ import annotations

from typing import Any, Mapping

from runtime.storage import get_storage


class SnapshotService:
    def __init__(self, storage=None):
        self.storage = storage or get_storage()

    def persist_snapshot(
        self, *, cycle: int, metrics: Mapping[str, Any], run_id: str | None = None
    ) -> None:
        payload = {"cycle": cycle, "metrics": dict(metrics)}
        self.storage.write_telemetry_snapshot(metrics=payload, run_id=run_id)
