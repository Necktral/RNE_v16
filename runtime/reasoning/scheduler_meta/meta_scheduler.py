"""Scheduler META para seleccionar y secuenciar familias de razonamiento."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from importlib import import_module
from typing import Any, Dict, List, Literal, Tuple
from uuid import uuid4

from runtime.reasoning.contracts import ReasoningTraceStep, normalize_family_result
from runtime.reasoning.scheduler_meta.budgeting import compute_budget
from runtime.reasoning.scheduler_meta.context_features import extract_context_features
from runtime.reasoning.scheduler_meta.fallbacks import (
    confidence_from_step,
    cost_from_step,
    should_early_stop,
)
from runtime.reasoning.scheduler_meta.policy import is_eml_experimental_enabled, select_sequence
from runtime.storage.records import ReasoningTraceRecord


_SKIP_SNAPSHOT = object()


class MetaScheduler:
    """Implementación mínima y trazable del scheduler META."""

    DEFAULT_SEQUENCE = ["abd", "ana", "cau", "ctf", "ded", "prob"]
    _VALID_MODES = {"fixed", "adaptive"}

    def __init__(
        self,
        sequence: List[str] | None = None,
        trace_store: object | None = None,
        mode: Literal["fixed", "adaptive"] = "fixed",
        max_steps: int | None = None,
        family_profile: str | None = None,
        regime_hint: str | None = None,
    ):
        self.sequence = sequence or list(self.DEFAULT_SEQUENCE)
        self._sequence_override = sequence is not None
        self.trace_store = trace_store
        self.mode = mode
        self.max_steps = max_steps
        self.family_profile = family_profile
        self.regime_hint = regime_hint

    def _resolve_runtime_configuration(self, state: Dict[str, Any]) -> Tuple[str, str | None, str | None]:
        requested_mode = (
            state.get("reasoning_mode")
            or state.get("scheduler_mode")
            or os.environ.get("RNFE_REASONING_MODE")
            or self.mode
        )
        active_mode = str(requested_mode or "fixed").strip().lower()
        if active_mode not in self._VALID_MODES:
            active_mode = self.mode

        profile = (
            state.get("family_profile")
            or state.get("reasoning_family_profile")
            or os.environ.get("RNFE_REASONING_FAMILY_PROFILE")
            or self.family_profile
        )
        regime_hint = (
            state.get("regime_hint")
            or state.get("regime_label")
            or state.get("reasoning_regime_hint")
            or os.environ.get("RNFE_REASONING_REGIME_HINT")
            or self.regime_hint
        )
        return active_mode, profile, regime_hint

    def _resolve_max_steps_override(self, state: Dict[str, Any]) -> int | None:
        raw_value = (
            state.get("reasoning_max_steps")
            or os.environ.get("RNFE_REASONING_MAX_STEPS")
            or self.max_steps
        )
        try:
            return int(raw_value) if raw_value is not None else None
        except (TypeError, ValueError):
            return self.max_steps

    def _snapshot_value(self, value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            snapshot: Dict[str, Any] = {}
            for key, item in value.items():
                item_snapshot = self._snapshot_value(item)
                if item_snapshot is not _SKIP_SNAPSHOT:
                    snapshot[str(key)] = item_snapshot
            return snapshot
        if isinstance(value, (list, tuple)):
            items = []
            for item in value:
                item_snapshot = self._snapshot_value(item)
                if item_snapshot is not _SKIP_SNAPSHOT:
                    items.append(item_snapshot)
            return items
        return _SKIP_SNAPSHOT

    def _snapshot_context(self, state: Dict[str, Any]) -> Dict[str, Any]:
        snapshot: Dict[str, Any] = {}
        for key, value in state.items():
            if key == "_meta":
                continue
            value_snapshot = self._snapshot_value(value)
            if value_snapshot is not _SKIP_SNAPSHOT:
                snapshot[str(key)] = value_snapshot
        return snapshot

    def run(self, context: Dict[str, Any] | None = None) -> Dict[str, Any]:
        state: Dict[str, Any] = dict(context or {})
        run_id = state.get("run_id")
        run_id = run_id if isinstance(run_id, str) else None

        active_mode, requested_profile, regime_hint = self._resolve_runtime_configuration(state)
        state.setdefault("reasoning_mode", active_mode)
        features = extract_context_features(state)
        max_steps_override_int = self._resolve_max_steps_override(state)
        budget = compute_budget(features, max_steps_override=max_steps_override_int)
        input_context = self._snapshot_context(state)

        if (
            active_mode == "fixed"
            and self._sequence_override
            and not requested_profile
            and "RNFE_REASONING_FAMILY_PROFILE" not in os.environ
        ):
            selected = [str(item).strip().lower() for item in self.sequence if str(item).strip()]
            scores = {family: 1.0 for family in selected}
            recommended = selected[-1] if selected else "prob"
            policy_meta = {
                "profile_name": "custom_fixed",
                "regime_label": (regime_hint or "n/a"),
                "primary_regime_label": (regime_hint or "n/a"),
                "cognitive_regime_label": (regime_hint or "n/a"),
                "floor_regime_label": (regime_hint or "n/a"),
                "mandatory_family_floor": [family.upper() for family in selected if family],
                "proposed_sequence": [family.upper() for family in selected if family],
                "validated_sequence": [family.upper() for family in selected if family],
                "effective_max_steps": int(budget["max_steps"]),
                "sequence_validation": {
                    "primary_regime_label": (regime_hint or "n/a"),
                    "cognitive_regime_label": (regime_hint or "n/a"),
                    "floor_regime_label": (regime_hint or "n/a"),
                    "mandatory_family_floor": [family.upper() for family in selected if family],
                    "proposed_sequence": [family.upper() for family in selected if family],
                    "validated_sequence": [family.upper() for family in selected if family],
                    "proposed_passed": True,
                    "validated_passed": True,
                    "missing_floor": [],
                    "missing_core": [],
                    "partial_order_ok": True,
                    "prob_last_ok": True,
                    "length_ok": True,
                    "optional_displacement_detected": False,
                    "autocorrected": False,
                    "fallback_used": False,
                    "budget_overridden_by_floor": False,
                    "effective_max_steps": int(budget["max_steps"]),
                    "unknown_families": [],
                    "admitted_overlays": [],
                    "default_overlays": [],
                    "correction_steps": [],
                    "fallback_profile_name": None,
                },
            }
        else:
            allow_experimental = is_eml_experimental_enabled()
            raw_directives = state.get("overlay_directives")
            overlay_directives = dict(raw_directives) if isinstance(raw_directives, dict) else None
            selected, scores, recommended, policy_meta = select_sequence(
                features=features,
                budget=budget,
                allow_experimental=allow_experimental,
                mode=active_mode,
                profile_name=requested_profile,
                regime_hint=regime_hint,
                return_metadata=True,
                overlay_directives=overlay_directives,
            )

        traces: List[ReasoningTraceStep] = []
        executed_sequence: List[str] = []
        sequence_validation = dict(policy_meta.get("sequence_validation") or {})
        proposed_sequence = list(policy_meta.get("proposed_sequence") or [])
        validated_sequence = list(policy_meta.get("validated_sequence") or [])
        mandatory_floor = list(policy_meta.get("mandatory_family_floor") or [])
        effective_max_steps = int(policy_meta.get("effective_max_steps") or budget["max_steps"])
        # Ejecutar la secuencia VALIDADA: la corrección del validador es el
        # contrato de cierre; ejecutar la propuesta sin corregir rompe el cierre
        # que la certificación luego cobra (p.ej. un overlay expulsando a DED
        # bajo presupuesto corto). Kill-switch forense: RNFE_EXECUTE_PROPOSED_SEQUENCE=1.
        if validated_sequence and os.environ.get("RNFE_EXECUTE_PROPOSED_SEQUENCE") != "1":
            validated_lower = [family.lower() for family in validated_sequence if family]
            if validated_lower != list(selected):
                selected = validated_lower
        for step_index, family in enumerate(selected):
            module = import_module(f"runtime.reasoning.families.{family}")
            state["_meta"] = {
                "mode": active_mode,
                "features": features,
                "budget": budget,
                "step_index": step_index,
                "selected_family": family,
                "family_profile": policy_meta.get("profile_name"),
                "regime_label": policy_meta.get("regime_label"),
                "primary_regime_label": policy_meta.get("primary_regime_label"),
                "cognitive_regime_label": policy_meta.get("cognitive_regime_label"),
                "floor_regime_label": policy_meta.get("floor_regime_label"),
                "mandatory_family_floor": mandatory_floor,
                "proposed_sequence": proposed_sequence,
                "validated_sequence": validated_sequence,
                "sequence_validation": sequence_validation,
                "effective_max_steps": effective_max_steps,
            }
            if family == "eml_sr":
                state["eml_mode"] = "shadow" if is_eml_experimental_enabled() else "disabled"
            result = normalize_family_result(
                module.execute(state),
                family_hint=family.upper(),
            )
            state.update(result.get("state_delta", {}))
            executed_sequence.append(family.upper())

            detail = dict(result)
            detail["selected_family"] = family.upper()
            detail["selection_reason"] = (
                f"score={scores.get(family, 0.0):.3f}|mode={active_mode}|"
                f"profile={policy_meta.get('profile_name')}|regime={policy_meta.get('regime_label')}"
            )
            detail["budget_used"] = min(float(step_index + 1), float(budget["cost_budget"]))
            detail["confidence"] = confidence_from_step(result, features=features)
            detail["cost"] = cost_from_step(result)
            detail["recommended_next_family"] = (
                result.get("recommended_next_family") or recommended.upper()
            )
            detail["family_profile"] = policy_meta.get("profile_name")
            detail["regime_label"] = policy_meta.get("regime_label")
            detail["primary_regime_label"] = policy_meta.get("primary_regime_label")
            detail["cognitive_regime_label"] = policy_meta.get("cognitive_regime_label")
            detail["floor_regime_label"] = policy_meta.get("floor_regime_label")
            detail["mandatory_family_floor"] = mandatory_floor
            detail["proposed_sequence"] = proposed_sequence
            detail["validated_sequence"] = validated_sequence
            detail["sequence_validation"] = sequence_validation
            detail["effective_max_steps"] = effective_max_steps
            if step_index == 0:
                detail["reasoning_context"] = input_context
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
            if detail["early_stop"] and active_mode == "adaptive":
                remaining_sequence = selected[step_index + 1 :]
                if "prob" in remaining_sequence and not state.get("prob_calibrated"):
                    continue
                break

        self._persist_trace(run_id=run_id, traces=traces)
        return {
            "meta_family": "META",
            "sequence": executed_sequence,
            "trace": [t.__dict__ for t in traces],
            "state": state,
            "input_context": input_context,
            "mode": active_mode,
            "family_profile": policy_meta.get("profile_name"),
            "regime_label": policy_meta.get("regime_label"),
            "primary_regime_label": policy_meta.get("primary_regime_label"),
            "cognitive_regime_label": policy_meta.get("cognitive_regime_label"),
            "floor_regime_label": policy_meta.get("floor_regime_label"),
            "mandatory_family_floor": mandatory_floor,
            "proposed_sequence": proposed_sequence,
            "validated_sequence": validated_sequence,
            "sequence_validation": sequence_validation,
            "effective_max_steps": effective_max_steps,
        }

    def _persist_trace(self, *, run_id: str | None, traces: List[ReasoningTraceStep]) -> None:
        if self.trace_store is None:
            return
        sequence = [step.family for step in traces]
        for step_index, trace_step in enumerate(traces):
            timestamp = datetime.fromtimestamp(trace_step.timestamp, tz=timezone.utc).isoformat()
            detail = dict(trace_step.detail or {})
            detail.setdefault("meta_family", "META")
            detail.setdefault("sequence", sequence)
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
