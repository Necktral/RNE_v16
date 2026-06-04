"""Extracción determinista de features de contexto para META."""

from __future__ import annotations

from typing import Any, Dict


def _as_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return min(max(value, lo), hi)


def _safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def extract_context_features(context: Dict[str, Any]) -> Dict[str, float]:
    observation = _safe_dict(context.get("observation"))
    updated_world = _safe_dict(context.get("updated_world"))
    counterfactual_world = _safe_dict(context.get("counterfactual"))

    uncertainty = _as_float(context.get("uncertainty"), default=0.25)
    contradiction = _as_float(context.get("contradiction_signal"), default=0.0)
    continuity_recent = _as_float(context.get("continuity_recent"), default=1.0)
    edge_pressure = _as_float(context.get("edge_pressure"), default=0.0)
    symbolic_regularity = _as_float(context.get("symbolic_regularity"), default=0.0)
    law_fit_signal = _as_float(context.get("law_fit_signal"), default=0.0)
    viability_margin = _as_float(context.get("viability_margin"), default=0.5)
    vram_headroom = _as_float(context.get("vram_headroom"), default=0.0)
    vram_opportunity = _as_float(context.get("vram_opportunity_score"), default=0.0)

    # Riesgo causal por gap factual/contrafactual cuando está disponible.
    counterfactual_gap = _as_float(context.get("counterfactual_gap"), default=0.0)
    if abs(counterfactual_gap) < 1e-9:
        factual_level = _as_float(
            updated_world.get("world_level", updated_world.get("temperature")),
            default=0.0,
        )
        counterfactual_level = _as_float(
            counterfactual_world.get("world_level", counterfactual_world.get("temperature")),
            default=factual_level,
        )
        counterfactual_gap = factual_level - counterfactual_level
    causal_risk = _clamp(abs(counterfactual_gap))

    # Señales espaciales desde observación del mundo.
    temp_std = _as_float(observation.get("temp_std"), default=0.0)
    hotspot_count = _as_float(observation.get("hotspot_count"), default=0.0)
    gradient_strength = _as_float(observation.get("gradient_strength"), default=0.0)
    quadrant_imbalance = _as_float(observation.get("quadrant_imbalance"), default=0.0)
    spatial_entropy = _as_float(observation.get("spatial_entropy"), default=0.0)
    heterogeneity_signal = _clamp(
        0.34 * _clamp(temp_std / 0.18)
        + 0.24 * _clamp(hotspot_count / 4.0)
        + 0.22 * _clamp(gradient_strength / 0.25)
        + 0.10 * _clamp(quadrant_imbalance / 0.2)
        + 0.10 * spatial_entropy
    )

    world_level = _as_float(
        observation.get("world_level", observation.get("temperature")),
        default=0.0,
    )
    world_level_signal = _clamp(world_level)

    alarm_on = bool(observation.get("alarm") is True or context.get("alarm") is True)
    if alarm_on:
        contradiction = max(contradiction, 0.40)
        edge_pressure = max(edge_pressure, 0.30)
        world_level_signal = max(world_level_signal, 0.86)

    viability_edge_signal = _clamp(
        0.40 * edge_pressure
        + 0.35 * world_level_signal
        + 0.20 * (1.0 - _clamp(viability_margin))
        + (0.15 if alarm_on else 0.0)
    )

    ambiguity_signal = _clamp(
        0.45 * uncertainty
        + 0.30 * contradiction
        + 0.25 * _clamp(abs(counterfactual_gap))
    )

    fragility_risk_signal = _clamp(
        0.45 * contradiction
        + 0.35 * (1.0 - _clamp(continuity_recent))
        + (0.20 if alarm_on else 0.0)
    )

    propositions = observation.get("propositions")
    proposition_count = len(propositions) if isinstance(propositions, list) else 0
    structure_missing = 1.0 if proposition_count <= 1 else 0.0
    pattern_without_structure_signal = _clamp(
        0.60 * heterogeneity_signal
        + 0.30 * structure_missing
        + 0.10 * _clamp(spatial_entropy)
    )

    vram_favorable_signal = _clamp(
        0.45 * _clamp(vram_headroom)
        + 0.45 * _clamp(vram_opportunity)
        + 0.10 * _clamp(context.get("vram_favorable_hint", 0.0))
    )

    return {
        "uncertainty": _clamp(uncertainty),
        "contradiction_signal": _clamp(contradiction),
        "continuity_recent": _clamp(continuity_recent),
        "edge_pressure": _clamp(edge_pressure),
        "causal_risk": _clamp(causal_risk),
        "symbolic_regularity": _clamp(symbolic_regularity),
        "law_fit_signal": _clamp(law_fit_signal),
        "heterogeneity_signal": heterogeneity_signal,
        "world_level_signal": world_level_signal,
        "viability_edge_signal": viability_edge_signal,
        "ambiguity_signal": ambiguity_signal,
        "fragility_risk_signal": fragility_risk_signal,
        "pattern_without_structure_signal": pattern_without_structure_signal,
        "vram_favorable_signal": vram_favorable_signal,
    }
