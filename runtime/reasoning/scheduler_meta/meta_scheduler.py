"""Scheduler META para seleccionar y secuenciar familias de razonamiento."""

from __future__ import annotations

from datetime import datetime, timezone
from importlib import import_module
from typing import Any, Dict, List
from uuid import uuid4

from runtime.reasoning.contracts import ReasoningTraceStep
from runtime.storage.records import ReasoningTraceRecord


class MetaScheduler:
    """Implementación mínima y trazable del scheduler META."""

    DEFAULT_SEQUENCE = ["abd", "ana", "cau", "ctf", "ded", "prob"]

    def __init__(self, sequence: List[str] | None = None, trace_store: object | None = None):
        self.sequence = sequence or list(self.DEFAULT_SEQUENCE)
        self.trace_store = trace_store

    def run(self, context: Dict[str, Any] | None = None) -> Dict[str, Any]:
        state: Dict[str, Any] = dict(context or {})
        run_id = state.get("run_id")
        run_id = run_id if isinstance(run_id, str) else None
        traces: List[ReasoningTraceStep] = []
        for family in self.sequence:
            module = import_module(f"runtime.reasoning.families.{family}")
            result = module.execute(state)
            state.update(result.get("state_delta", {}))
            traces.append(
                ReasoningTraceStep(
                    family=family.upper(),
                    status=result.get("status", "ok"),
                    detail=result,
                )
            )
        self._persist_trace(run_id=run_id, traces=traces)
        return {
            "meta_family": "META",
            "sequence": [f.upper() for f in self.sequence],
            "trace": [t.__dict__ for t in traces],
            "state": state,
        }

    def _persist_trace(
        self, *, run_id: str | None, traces: List[ReasoningTraceStep]
    ) -> None:
        if self.trace_store is None:
            return
        sequence = [step.family for step in traces]
        for step_index, trace_step in enumerate(traces):
            timestamp = datetime.fromtimestamp(
                trace_step.timestamp, tz=timezone.utc
            ).isoformat()
            detail = dict(trace_step.detail or {})
            detail.setdefault("meta_family", "META")
            detail.setdefault("sequence", sequence)
            # Soporta tanto StorageFacade (kwargs) como stores de bajo nivel (dataclass).
            try:
                self.trace_store.append_reasoning_trace(
                    family=trace_step.family,
                    status=trace_step.status,
                    step_index=step_index,
                    detail=detail,
                    timestamp=timestamp,
                    run_id=run_id,
                )
            except TypeError:
                record = ReasoningTraceRecord(
                    trace_id=str(uuid4()),
                    run_id=run_id,
                    step_index=step_index,
                    family=trace_step.family,
                    status=trace_step.status,
                    detail=detail,
                    timestamp=timestamp,
                )
                self.trace_store.append_reasoning_trace(record)
