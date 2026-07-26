"""Generador simbólico determinista de hipótesis causales estructurales."""

from __future__ import annotations

import ast
import statistics
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from runtime.neural.contracts import canonical_sha256
from runtime.symbolic.mci.causal_learning import TransitionEvidence
from runtime.symbolic.mci.compiler import TransitionCompiler
from runtime.symbolic.mci.contracts import TransitionSpec
from runtime.symbolic.mci.specs import with_overlay


@dataclass(frozen=True)
class StructuralCandidate:
    hypothesis_id: str
    kind: str
    target_id: str
    expression: str | None
    proposed_value: float | None
    evidence_refs: tuple[str, ...]
    features: Mapping[str, float]

    def ranking_payload(self) -> dict[str, Any]:
        return {"hypothesis_id": self.hypothesis_id, **dict(self.features)}


class StructuralHypothesisGenerator:
    def __init__(self, *, beam_width: int = 24, max_depth: int = 2) -> None:
        if beam_width < 1 or max_depth not in {1, 2}:
            raise ValueError("invalid_structural_generator_budget")
        self.beam_width = beam_width
        self.max_depth = max_depth

    def generate(
        self,
        *,
        spec: TransitionSpec,
        evidence: Sequence[TransitionEvidence],
    ) -> tuple[StructuralCandidate, ...]:
        rows = tuple(evidence)
        if len(rows) < 8:
            return ()
        candidates = self._parameter_candidates(spec, rows)
        candidates.extend(self._precondition_candidates(spec, rows))
        unique = {self._semantic_key(item): item for item in candidates}
        return tuple(
            sorted(
                unique.values(),
                key=lambda item: (
                    -item.features["empirical_gain"],
                    -item.features["holdout_support"],
                    item.hypothesis_id,
                ),
            )[: self.beam_width]
        )

    def _parameter_candidates(
        self, spec: TransitionSpec, rows: Sequence[TransitionEvidence]
    ) -> list[StructuralCandidate]:
        result: list[StructuralCandidate] = []
        for parameter, current in sorted(spec.parameters.items()):
            if "threshold" in parameter:
                continue
            for factor in (0.5, 0.75, 1.25, 1.5, 2.0):
                proposed = round(float(current) * factor, 9)
                try:
                    candidate_spec = with_overlay(spec, {parameter: proposed})
                except ValueError:
                    continue
                features = self._features(spec, candidate_spec, rows, complexity=1)
                result.append(
                    self._candidate(
                        kind="parameter",
                        target_id=parameter,
                        expression=None,
                        proposed_value=proposed,
                        rows=rows,
                        features=features,
                    )
                )
        return result

    def _precondition_candidates(
        self, spec: TransitionSpec, rows: Sequence[TransitionEvidence]
    ) -> list[StructuralCandidate]:
        result: list[StructuralCandidate] = []
        numeric_variables = [
            variable.name
            for variable in spec.variables
            if variable.kind != "bool"
            and variable.name not in {spec.main_variable, spec.alarm_variable}
        ]
        effects = [
            effect
            for equation in spec.equations
            for effect in equation.effects
            if effect.precondition is None
        ]
        atoms: list[str] = []
        for variable in numeric_variables:
            values = sorted(
                {
                    round(float(row.state[variable]), 9)
                    for row in rows
                    if variable in row.state
                }
            )
            if len(values) < 2:
                continue
            cuts = sorted(
                {
                    (values[index - 1] + values[index]) / 2.0
                    for index in {
                        max(1, len(values) // 4),
                        max(1, len(values) // 2),
                        max(1, (3 * len(values)) // 4),
                    }
                    if index < len(values)
                }
            )
            for cut in cuts:
                atoms.extend(
                    (f"{variable} > {cut:.9g}", f"{variable} < {cut:.9g}")
                )
        expressions = list(atoms)
        if self.max_depth == 2:
            for index, left in enumerate(atoms):
                for right in atoms[index + 1 :]:
                    if left.split()[0] == right.split()[0]:
                        continue
                    expressions.extend((f"{left} and {right}", f"{left} or {right}"))
        for effect in effects:
            action_rows = tuple(row for row in rows if row.action == effect.action)
            if len(action_rows) < 8:
                continue
            for expression in expressions:
                complexity = 1 + int(" and " in expression or " or " in expression)
                try:
                    ast.parse(expression, mode="eval")
                    candidate_spec = with_overlay(
                        spec, {}, {effect.effect_id: expression}
                    )
                except (SyntaxError, ValueError):
                    continue
                features = self._features(
                    spec, candidate_spec, action_rows, complexity=complexity
                )
                result.append(
                    self._candidate(
                        kind="precondition",
                        target_id=effect.effect_id,
                        expression=expression,
                        proposed_value=None,
                        rows=action_rows,
                        features=features,
                    )
                )
        return result

    def _features(
        self,
        base: TransitionSpec,
        candidate: TransitionSpec,
        rows: Sequence[TransitionEvidence],
        *,
        complexity: int,
    ) -> dict[str, float]:
        split = max(1, int(len(rows) * 0.75))
        train, holdout = rows[:split], rows[split:]
        before_train = self._mae(base, train)
        after_train = self._mae(candidate, train)
        before_holdout = self._mae(base, holdout)
        after_holdout = self._mae(candidate, holdout)
        train_gain = max(0.0, (before_train - after_train) / max(before_train, 1e-9))
        holdout_gain = max(
            0.0, (before_holdout - after_holdout) / max(before_holdout, 1e-9)
        )
        errors = [
            abs(
                float(row.observed[base.main_variable])
                - float(row.predicted[base.main_variable])
            )
            for row in rows
        ]
        stability = 1.0 / (1.0 + statistics.pstdev(errors)) if len(errors) > 1 else 1.0
        return {
            "empirical_gain": round(train_gain, 9),
            "holdout_support": round(holdout_gain, 9),
            "coverage": round(min(1.0, len(rows) / 32.0), 9),
            "simplicity": round(1.0 / complexity, 9),
            "stability": round(stability, 9),
            "invariant_safety": 1.0,
        }

    @staticmethod
    def _mae(
        spec: TransitionSpec, rows: Sequence[TransitionEvidence]
    ) -> float:
        if not rows:
            return 0.0
        compiler = TransitionCompiler(spec)
        return sum(
            abs(
                float(
                    compiler.execute(
                        row.state,
                        action=row.action,
                        external_input=row.external_input,
                    )[spec.main_variable]
                )
                - float(row.observed[spec.main_variable])
            )
            for row in rows
        ) / len(rows)

    @staticmethod
    def _candidate(
        *,
        kind: str,
        target_id: str,
        expression: str | None,
        proposed_value: float | None,
        rows: Sequence[TransitionEvidence],
        features: Mapping[str, float],
    ) -> StructuralCandidate:
        evidence_refs = tuple(sorted(row.evidence_id for row in rows))
        payload = {
            "kind": kind,
            "target_id": target_id,
            "expression": expression,
            "proposed_value": proposed_value,
        }
        return StructuralCandidate(
            hypothesis_id=f"n4-candidate-{canonical_sha256(payload)[:24]}",
            kind=kind,
            target_id=target_id,
            expression=expression,
            proposed_value=proposed_value,
            evidence_refs=evidence_refs,
            features=features,
        )

    @staticmethod
    def _semantic_key(item: StructuralCandidate) -> tuple[Any, ...]:
        return (item.kind, item.target_id, item.expression, item.proposed_value)
