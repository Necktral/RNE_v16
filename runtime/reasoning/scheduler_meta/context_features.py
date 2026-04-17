"""Extracción determinista de features de contexto para META."""

from __future__ import annotations

from typing import Any, Dict


def _as_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return default


def extract_context_features(context: Dict[str, Any]) -> Dict[str, float]:
    observation = context.get("observation", {})
    uncertainty = _as_float(context.get("uncertainty"), default=0.25)
    contradiction = _as_float(context.get("contradiction_signal"), default=0.0)
    continuity_recent = _as_float(context.get("continuity_recent"), default=1.0)
    edge_pressure = _as_float(context.get("edge_pressure"), default=0.0)
    counterfactual_gap = _as_float(context.get("counterfactual_gap"), default=0.0)
    symbolic_regularity = _as_float(context.get("symbolic_regularity"), default=0.0)
    law_fit_signal = _as_float(context.get("law_fit_signal"), default=0.0)
    if isinstance(observation, dict) and observation.get("alarm") is True:
        contradiction = max(contradiction, 0.4)
        edge_pressure = max(edge_pressure, 0.3)

    causal_risk = min(1.0, max(0.0, abs(counterfactual_gap)))
    uncertainty = min(1.0, max(0.0, uncertainty))
    contradiction = min(1.0, max(0.0, contradiction))
    continuity_recent = min(1.0, max(0.0, continuity_recent))
    edge_pressure = min(1.0, max(0.0, edge_pressure))
    symbolic_regularity = min(1.0, max(0.0, symbolic_regularity))
    law_fit_signal = min(1.0, max(0.0, law_fit_signal))

    return {
        "uncertainty": uncertainty,
        "contradiction_signal": contradiction,
        "continuity_recent": continuity_recent,
        "edge_pressure": edge_pressure,
        "causal_risk": causal_risk,
        "symbolic_regularity": symbolic_regularity,
        "law_fit_signal": law_fit_signal,
    }
