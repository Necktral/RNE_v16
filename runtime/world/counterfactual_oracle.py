"""Outcome oracle for P1 SHADOW experiments.

The oracle is intentionally runner-owned and physically separate from reasoning
contexts.  It enumerates interventions through the scenario's non-mutating
counterfactual API and is used only after candidates have been produced.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

from .scenario import CognitiveScenario


ORACLE_SCHEMA_VERSION = "rnfe-counterfactual-oracle-v1"
TIE_EPSILON = 1e-9


@dataclass(frozen=True, slots=True)
class InterventionOutcome:
    intervention: str
    value: float
    delta: float
    state: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CounterfactualOracleResult:
    schema_version: str
    status: str
    main_variable: str
    optimization_direction: str
    outcomes: tuple[InterventionOutcome, ...]
    best_actions: tuple[str, ...]
    snapshot_sha256: str
    outcome_set_sha256: str | None = None
    unavailable_reason: str | None = None
    authority_effect: str = "none"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "main_variable": self.main_variable,
            "optimization_direction": self.optimization_direction,
            "outcomes": [item.to_dict() for item in self.outcomes],
            "best_actions": list(self.best_actions),
            "snapshot_sha256": self.snapshot_sha256,
            "outcome_set_sha256": self.outcome_set_sha256,
            "unavailable_reason": self.unavailable_reason,
            "authority_effect": self.authority_effect,
        }


def enumerate_counterfactual_outcomes(
    *,
    scenario: CognitiveScenario,
    observation: Mapping[str, Any],
    interventions: Sequence[str],
    external_input: float,
    optimization_direction: str,
    tie_epsilon: float = TIE_EPSILON,
) -> CounterfactualOracleResult:
    """Evaluate every unique allowed action from the same pre-action state."""

    ordered = tuple(dict.fromkeys(str(item).strip() for item in interventions if str(item).strip()))
    main_variable = str(scenario.config.main_variable)
    direction = str(optimization_direction or "").strip().lower()
    snapshot = {
        "scenario": scenario.config.name,
        "main_variable": main_variable,
        "observation": dict(observation),
        "interventions": list(ordered),
        "external_input": float(external_input),
        "optimization_direction": direction,
    }
    snapshot_sha256 = hashlib.sha256(
        json.dumps(snapshot, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()

    if not ordered:
        return _unavailable(main_variable, direction, snapshot_sha256, "no_interventions")
    if direction not in {"minimize", "maximize"}:
        return _unavailable(
            main_variable,
            direction,
            snapshot_sha256,
            "unsupported_optimization_direction",
        )
    observed = observation.get(main_variable)
    if not _finite(observed):
        return _unavailable(main_variable, direction, snapshot_sha256, "observation_unmeasured")

    outcomes: list[InterventionOutcome] = []
    try:
        for intervention in ordered:
            transition = scenario.simulate_counterfactual(
                intervention=intervention,
                external_input=float(external_input),
            )
            value = transition.state.get(main_variable)
            if not _finite(value):
                return _unavailable(
                    main_variable,
                    direction,
                    snapshot_sha256,
                    f"nonfinite_outcome:{intervention}",
                )
            numeric = float(value)
            outcomes.append(
                InterventionOutcome(
                    intervention=intervention,
                    value=numeric,
                    delta=numeric - float(observed),
                    state=dict(transition.state),
                )
            )
    except Exception as exc:
        return _unavailable(
            main_variable,
            direction,
            snapshot_sha256,
            f"sandbox_error:{type(exc).__name__}",
        )

    target = (
        min(item.value for item in outcomes)
        if direction == "minimize"
        else max(item.value for item in outcomes)
    )
    epsilon = max(float(tie_epsilon), 0.0)
    best = tuple(item.intervention for item in outcomes if abs(item.value - target) <= epsilon)
    outcome_set_sha256 = hashlib.sha256(
        json.dumps(
            {
                "snapshot_sha256": snapshot_sha256,
                "outcomes": [item.to_dict() for item in outcomes],
                "best_actions": list(best),
            },
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()
    return CounterfactualOracleResult(
        schema_version=ORACLE_SCHEMA_VERSION,
        status="scored",
        main_variable=main_variable,
        optimization_direction=direction,
        outcomes=tuple(outcomes),
        best_actions=best,
        snapshot_sha256=snapshot_sha256,
        outcome_set_sha256=outcome_set_sha256,
    )


def _unavailable(
    main_variable: str,
    direction: str,
    snapshot_sha256: str,
    reason: str,
) -> CounterfactualOracleResult:
    return CounterfactualOracleResult(
        schema_version=ORACLE_SCHEMA_VERSION,
        status="unavailable",
        main_variable=main_variable,
        optimization_direction=direction,
        outcomes=(),
        best_actions=(),
        snapshot_sha256=snapshot_sha256,
        unavailable_reason=reason,
    )


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))
