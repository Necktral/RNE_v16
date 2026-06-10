"""Budgeting determinista para META."""

from __future__ import annotations

from typing import Dict


def compute_budget(features: Dict[str, float], *, max_steps_override: int | None = None) -> Dict[str, float]:
    # Robusto ante feature dicts parciales: una feature ausente equivale a "no
    # señalada" (0.0). Evita KeyError si un llamador pasa un subconjunto de features
    # (el camino vivo siempre las trae completas vía context_features).
    uncertainty = float(features.get("uncertainty", 0.0))
    contradiction_signal = float(features.get("contradiction_signal", 0.0))
    causal_risk = float(features.get("causal_risk", 0.0))
    edge_pressure = float(features.get("edge_pressure", 0.0))

    base_steps = 6
    dynamic_bonus = 0
    if uncertainty >= 0.6:
        dynamic_bonus += 1
    if contradiction_signal >= 0.5:
        dynamic_bonus += 1
    if causal_risk >= 0.5:
        dynamic_bonus += 1
    if edge_pressure >= 0.8:
        dynamic_bonus -= 1
    max_steps = base_steps + dynamic_bonus
    max_steps = max(4, min(10, max_steps))
    if max_steps_override is not None:
        max_steps = max(4, min(10, int(max_steps_override)))

    risk_budget = min(
        1.0,
        0.3
        + (0.4 * uncertainty)
        + (0.2 * contradiction_signal)
        + (0.1 * causal_risk),
    )
    return {
        "max_steps": float(max_steps),
        "risk_budget": risk_budget,
        "cost_budget": float(max_steps),
    }
