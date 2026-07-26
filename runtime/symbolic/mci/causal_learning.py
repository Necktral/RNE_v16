"""Aprendizaje causal online mediante overlays de parámetros versionados."""

from __future__ import annotations

import statistics
from collections import deque
from dataclasses import dataclass
from typing import Any, Mapping

from runtime.symbolic.schemas import sealed_sha256

from .compiler import TransitionCompiler
from .contracts import CausalOverlay, TransitionSpec
from .specs import with_overlay


@dataclass(frozen=True)
class TransitionEvidence:
    state: Mapping[str, Any]
    action: str
    external_input: float
    observed: Mapping[str, Any]
    predicted: Mapping[str, Any]


class CausalLearningEngine:
    def __init__(self, base_spec: TransitionSpec):
        self.base_spec = base_spec
        self.active_overlay: CausalOverlay | None = None
        self._buffer: deque[TransitionEvidence] = deque(maxlen=32)
        self._regression_streak = 0
        self._overlays: dict[str, CausalOverlay] = {}
        self.last_event: dict[str, Any] | None = None
        self.archived_regimes: list[tuple[TransitionEvidence, ...]] = []

    def segment_regime(self, *, logical_time: int) -> None:
        if self._buffer:
            self.archived_regimes.append(tuple(self._buffer))
        self._buffer.clear()
        self._regression_streak = 0
        self.last_event = {
            "kind": "regime_segmented",
            "logical_time": int(logical_time),
            "archive_index": len(self.archived_regimes) - 1,
        }

    @property
    def active_spec(self) -> TransitionSpec:
        updates = dict(self.active_overlay.parameter_updates) if self.active_overlay else {}
        conditions = (
            dict(self.active_overlay.precondition_updates)
            if self.active_overlay
            else {}
        )
        return with_overlay(self.base_spec, updates, conditions)

    def restore_overlay(self, overlay: CausalOverlay) -> None:
        if overlay.base_spec_sha256 != self.base_spec.sha256:
            raise ValueError("Overlay pertenece a otro Transition IR")
        with_overlay(
            self.base_spec,
            dict(overlay.parameter_updates),
            dict(overlay.precondition_updates),
        )
        self._overlays[overlay.overlay_id] = overlay
        self.active_overlay = overlay

    def observe(self, evidence: TransitionEvidence) -> CausalOverlay | None:
        self.last_event = None
        self._buffer.append(evidence)
        if self.active_overlay is not None:
            parent = self._overlays.get(self.active_overlay.parent_overlay_id or "")
            parent_updates = dict(parent.parameter_updates) if parent is not None else {}
            parent_conditions = dict(parent.precondition_updates) if parent else {}
            parent_spec = with_overlay(
                self.base_spec, parent_updates, parent_conditions
            )
            active_error = self._mae(self.active_spec, [evidence])
            parent_error = self._mae(parent_spec, [evidence])
            self._regression_streak = (
                self._regression_streak + 1 if active_error > parent_error else 0
            )
            if self._regression_streak >= 8:
                rolled_back = self.active_overlay.overlay_id
                self.active_overlay = parent
                self._regression_streak = 0
                self.last_event = {
                    "kind": "rollback",
                    "rolled_back_overlay_id": rolled_back,
                    "restored_overlay_id": parent.overlay_id if parent else None,
                }
                return None
        errors = [
            abs(
                float(item.observed[self.base_spec.main_variable])
                - float(item.predicted[self.base_spec.main_variable])
            )
            for item in self._buffer
        ]
        current_error = errors[-1]
        median = statistics.median(errors)
        mad = statistics.median(abs(item - median) for item in errors)
        if len(self._buffer) < 12 or current_error <= max(1e-6, 2.0 * mad):
            return None
        candidates = [
            item
            for item in (
                self._fit_parameter_overlay(),
                self.generate_structural_hypothesis(),
            )
            if item is not None
        ]
        candidate = (
            min(candidates, key=lambda item: (item.holdout_mae_after, item.overlay_id))
            if candidates
            else None
        )
        if candidate is not None:
            self._overlays[candidate.overlay_id] = candidate
            self.active_overlay = candidate
            self._regression_streak = 0
            self.last_event = {"kind": "promotion", "overlay_id": candidate.overlay_id}
        return candidate

    def generate_structural_hypothesis(
        self,
        context: Mapping[str, Any] | None = None,
        failures: list[TransitionEvidence] | None = None,
    ) -> CausalOverlay | None:
        """Propone una guarda simple cuando un efecto falla según el contexto."""
        del context
        rows = list(self._buffer)
        if len(rows) < 12:
            return None
        error_rows = failures or [
            row
            for row in rows
            if abs(
                float(row.observed[self.base_spec.main_variable])
                - float(row.predicted[self.base_spec.main_variable])
            )
            > 1e-6
        ]
        success_rows = [row for row in rows if row not in error_rows]
        if len(error_rows) < 4 or len(success_rows) < 4:
            return None
        split = max(9, int(len(rows) * 0.75))
        train, holdout = rows[:split], rows[split:]
        before_train = self._mae(self.active_spec, train)
        before_holdout = self._mae(self.active_spec, holdout)
        candidates: list[CausalOverlay] = []
        effects = [
            effect
            for equation in self.active_spec.equations
            for effect in equation.effects
            if effect.precondition is None
        ]
        for effect in effects:
            failed = [row for row in error_rows if row.action == effect.action]
            succeeded = [row for row in success_rows if row.action == effect.action]
            if len(failed) < 4 or len(succeeded) < 4:
                continue
            for variable in sorted(set(failed[0].state) & set(succeeded[0].state)):
                if variable in {
                    self.base_spec.main_variable,
                    self.base_spec.alarm_variable,
                }:
                    continue
                values_failed = [row.state[variable] for row in failed]
                values_success = [row.state[variable] for row in succeeded]
                if any(isinstance(value, bool) for value in values_failed + values_success):
                    continue
                failed_max = max(float(value) for value in values_failed)
                failed_min = min(float(value) for value in values_failed)
                success_max = max(float(value) for value in values_success)
                success_min = min(float(value) for value in values_success)
                if failed_max < success_min:
                    threshold = (failed_max + success_min) / 2.0
                    condition = f"{variable} > {threshold:.9g}"
                elif success_max < failed_min:
                    threshold = (success_max + failed_min) / 2.0
                    condition = f"{variable} < {threshold:.9g}"
                else:
                    continue
                updates = {
                    **(
                        dict(self.active_overlay.precondition_updates)
                        if self.active_overlay
                        else {}
                    ),
                    effect.effect_id: condition,
                }
                candidate_spec = with_overlay(
                    self.active_spec, {}, {effect.effect_id: condition}
                )
                train_mae = self._mae(candidate_spec, train)
                holdout_mae = self._mae(candidate_spec, holdout)
                if not (
                    train_mae <= before_train * 0.80
                    and holdout_mae < before_holdout
                    and self._respects_bounds(candidate_spec, rows)
                ):
                    continue
                payload = {
                    "base": self.base_spec.sha256,
                    "parent": self.active_overlay.overlay_id if self.active_overlay else None,
                    "effect": effect.effect_id,
                    "condition": condition,
                    "n": len(rows),
                }
                candidates.append(
                    CausalOverlay(
                        overlay_id=f"overlay-{sealed_sha256(payload)[:16]}",
                        version=(self.active_overlay.version + 1 if self.active_overlay else 1),
                        base_spec_sha256=self.base_spec.sha256,
                        parameter_updates=(
                            dict(self.active_overlay.parameter_updates)
                            if self.active_overlay
                            else {}
                        ),
                        precondition_updates=updates,
                        edge_updates=(),
                        evidence={
                            "buffer_size": len(rows),
                            "kind": "precondition",
                            "effect_id": effect.effect_id,
                            "condition": condition,
                        },
                        train_mae_before=before_train,
                        train_mae_after=train_mae,
                        holdout_mae_before=before_holdout,
                        holdout_mae_after=holdout_mae,
                        parent_overlay_id=(
                            self.active_overlay.overlay_id if self.active_overlay else None
                        ),
                    )
                )
        return (
            min(candidates, key=lambda item: (item.holdout_mae_after, item.overlay_id))
            if candidates
            else None
        )

    def _fit_parameter_overlay(self) -> CausalOverlay | None:
        rows = list(self._buffer)
        split = max(9, int(len(rows) * 0.75))
        train, holdout = rows[:split], rows[split:]
        before_train = self._mae(self.active_spec, train)
        before_holdout = self._mae(self.active_spec, holdout)
        candidates: list[tuple[tuple[float, str], CausalOverlay]] = []
        for parameter, value in sorted(self.active_spec.parameters.items()):
            if "threshold" in parameter:
                continue
            for factor in (0.5, 0.75, 1.25, 1.5):
                updated = round(float(value) * factor, 9)
                spec = with_overlay(self.active_spec, {parameter: updated})
                train_mae = self._mae(spec, train)
                holdout_mae = self._mae(spec, holdout)
                if (
                    train_mae <= before_train * 0.80
                    and holdout_mae < before_holdout
                    and self._respects_bounds(spec, rows)
                ):
                    payload = {
                        "base": self.base_spec.sha256,
                        "parent": self.active_overlay.overlay_id if self.active_overlay else None,
                        "parameter": parameter,
                        "value": updated,
                        "n": len(rows),
                    }
                    overlay = CausalOverlay(
                        overlay_id=f"overlay-{sealed_sha256(payload)[:16]}",
                        version=(self.active_overlay.version + 1 if self.active_overlay else 1),
                        base_spec_sha256=self.base_spec.sha256,
                        parameter_updates={
                            **(
                                dict(self.active_overlay.parameter_updates)
                                if self.active_overlay
                                else {}
                            ),
                            parameter: updated,
                        },
                        precondition_updates=(
                            dict(self.active_overlay.precondition_updates)
                            if self.active_overlay
                            else {}
                        ),
                        edge_updates=(),
                        evidence={"buffer_size": len(rows), "parameter": parameter},
                        train_mae_before=before_train,
                        train_mae_after=train_mae,
                        holdout_mae_before=before_holdout,
                        holdout_mae_after=holdout_mae,
                        parent_overlay_id=(
                            self.active_overlay.overlay_id if self.active_overlay else None
                        ),
                    )
                    candidates.append(((holdout_mae, parameter), overlay))
        return min(candidates, key=lambda item: item[0])[1] if candidates else None

    def _mae(self, spec: TransitionSpec, rows: list[TransitionEvidence]) -> float:
        if not rows:
            return float("inf")
        compiler = TransitionCompiler(spec)
        variable = spec.main_variable
        return sum(
            abs(
                float(
                    compiler.execute(
                        row.state, action=row.action, external_input=row.external_input
                    )[variable]
                )
                - float(row.observed[variable])
            )
            for row in rows
        ) / len(rows)

    def _respects_bounds(self, spec: TransitionSpec, rows: list[TransitionEvidence]) -> bool:
        compiler = TransitionCompiler(spec)
        return all(
            all(
                variable.kind == "bool"
                or variable.lower
                <= float(
                    compiler.execute(
                        row.state, action=row.action, external_input=row.external_input
                    )[variable.name]
                )
                <= variable.upper
                for variable in spec.variables
            )
            for row in rows
        )
