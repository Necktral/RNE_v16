"""Scheduler META para seleccionar y secuenciar familias de razonamiento."""

from __future__ import annotations

from datetime import datetime, timezone
from importlib import import_module
from typing import Any, Dict, List, Literal
from uuid import uuid4

from runtime.reasoning.contracts import ReasoningTraceStep
from runtime.storage.records import ReasoningTraceRecord
from runtime.reasoning.scheduler_meta.budgeting import compute_budget
from runtime.reasoning.scheduler_meta.context_features import extract_context_features
from runtime.reasoning.scheduler_meta.fallbacks import (
    confidence_from_step,
    cost_from_step,
    should_early_stop,
)
from runtime.reasoning.scheduler_meta.policy import select_sequence
from runtime.reasoning.scheduler_meta.policy import is_eml_experimental_enabled


class MetaScheduler:
    """Implementación mínima y trazable del scheduler META."""

    DEFAULT_SEQUENCE = ["abd", "ana", "cau", "ctf", "ded", "prob"]

    def __init__(
        self,
        sequence: List[str] | None = None,
        trace_store: object | None = None,
        mode: Literal["fixed", "adaptive"] = "fixed",
        max_steps: int | None = None,
    ):
        self.sequence = sequence or list(self.DEFAULT_SEQUENCE)
        self.trace_store = trace_store
        self.mode = mode
        self.max_steps = max_steps

    def run(self, context: Dict[str, Any] | None = None) -> Dict[str, Any]:
        state: Dict[str, Any] = dict(context or {})
        run_id = state.get("run_id")
        run_id = run_id if isinstance(run_id, str) else None
        features = extract_context_features(state)
        budget = compute_budget(features, max_steps_override=self.max_steps)
        if self.mode == "adaptive":
            allow_experimental = is_eml_experimental_enabled()
            selected, scores, recommended = select_sequence(
                features=features,
                budget=budget,
                allow_experimental=allow_experimental,
            )
        else:
            selected = list(self.sequence)
            scores = {fam: 1.0 for fam in selected}
            recommended = selected[-1] if selected else "prob"

        traces: List[ReasoningTraceStep] = []
        executed_sequence: List[str] = []
        for step_index, family in enumerate(selected):
            module = import_module(f"runtime.reasoning.families.{family}")
            state["_meta"] = {
                "mode": self.mode,
                "features": features,
                "budget": budget,
                "step_index": step_index,
                "selected_family": family,
            }
            if family == "eml_sr":
                state["eml_mode"] = "shadow" if is_eml_experimental_enabled() else "disabled"
            result = module.execute(state)
            state.update(result.get("state_delta", {}))
            executed_sequence.append(family.upper())
            detail = dict(result)
            detail["selected_family"] = family.upper()
            detail["selection_reason"] = (
                f"score={scores.get(family, 0.0):.3f}|mode={self.mode}"
            )
            detail["budget_used"] = min(float(step_index + 1), float(budget["cost_budget"]))
            detail["confidence"] = confidence_from_step(result, features=features)
            detail["cost"] = cost_from_step(result)
            detail["recommended_next_family"] = recommended.upper()
            detail["early_stop"] = should_early_stop(
                step_result=result,
                state=state,
                features=features,
                step_index=step_index,
                max_steps=int(budget["max_steps"]),
            )
            traces.append(
                ReasoningTraceStep(
                    family=family.upper(),
                    status=result.get("status", "ok"),
                    detail=detail,
                )
            )
            if detail["early_stop"] and self.mode == "adaptive":
                break
        self._persist_trace(run_id=run_id, traces=traces)
        return {
            "meta_family": "META",
            "sequence": executed_sequence,
            "trace": [t.__dict__ for t in traces],
            "state": state,
            "mode": self.mode,
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
