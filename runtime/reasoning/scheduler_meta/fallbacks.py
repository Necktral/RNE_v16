"""Reglas de early-stop y fallback para META."""

from __future__ import annotations

from typing import Any, Dict


def should_early_stop(
    *,
    step_result: Dict[str, Any],
    state: Dict[str, Any],
    features: Dict[str, float],
    step_index: int,
    max_steps: int,
) -> bool:
    if step_result.get("status") == "critical_fail":
        return True
    if step_index >= max_steps - 1:
        return True
    if features["edge_pressure"] >= 0.85 and state.get("ded_validated"):
        return True
    return False


def confidence_from_step(step_result: Dict[str, Any], *, features: Dict[str, float]) -> float:
    confidence = step_result.get("confidence")
    if isinstance(confidence, (int, float)):
        return max(0.0, min(1.0, float(confidence)))
    baseline = 0.55 + (0.2 * (1.0 - features["uncertainty"])) + (
        0.1 * (1.0 - features["contradiction_signal"])
    )
    return max(0.0, min(1.0, baseline))


def cost_from_step(step_result: Dict[str, Any]) -> float:
    cost = step_result.get("cost")
    if isinstance(cost, (int, float)):
        return max(0.0, float(cost))
    return 1.0
