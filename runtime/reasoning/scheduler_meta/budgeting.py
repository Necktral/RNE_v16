"""Budgeting determinista para META."""

from __future__ import annotations

from typing import Dict


def compute_budget(features: Dict[str, float], *, max_steps_override: int | None = None) -> Dict[str, float]:
    base_steps = 6
    dynamic_bonus = 0
    if features["uncertainty"] >= 0.6:
        dynamic_bonus += 1
    if features["contradiction_signal"] >= 0.5:
        dynamic_bonus += 1
    if features["causal_risk"] >= 0.5:
        dynamic_bonus += 1
    if features["edge_pressure"] >= 0.8:
        dynamic_bonus -= 1
    max_steps = base_steps + dynamic_bonus
    max_steps = max(4, min(10, max_steps))
    if max_steps_override is not None:
        max_steps = max(4, min(10, int(max_steps_override)))

    risk_budget = min(
        1.0,
        0.3
        + (0.4 * features["uncertainty"])
        + (0.2 * features["contradiction_signal"])
        + (0.1 * features["causal_risk"]),
    )
    return {
        "max_steps": float(max_steps),
        "risk_budget": risk_budget,
        "cost_budget": float(max_steps),
    }
