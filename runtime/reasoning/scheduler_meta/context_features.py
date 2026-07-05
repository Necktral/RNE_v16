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


def _first_float(*values: Any, default: float = 0.0) -> float:
    for value in values:
        if isinstance(value, (int, float)):
            return float(value)
    return default


def extract_context_features(context: Dict[str, Any]) -> Dict[str, float]:
    observation = _safe_dict(context.get("observation"))
    telemetry = _safe_dict(context.get("telemetry"))
    vram_snapshot = _safe_dict(context.get("vram_snapshot"))
    updated_world = _safe_dict(context.get("updated_world"))
    counterfactual_world = _safe_dict(context.get("counterfactual"))

    uncertainty = _as_float(context.get("uncertainty"), default=0.25)
    contradiction = _as_float(context.get("contradiction_signal"), default=0.0)
    continuity_recent = _as_float(context.get("continuity_recent"), default=1.0)
    edge_pressure = _as_float(context.get("edge_pressure"), default=0.0)
    symbolic_regularity = _as_float(context.get("symbolic_regularity"), default=0.0)
    law_fit_signal = _as_float(context.get("law_fit_signal"), default=0.0)
    viability_margin = _as_float(context.get("viability_margin"), default=0.5)
    vram_headroom = _first_float(
        context.get("vram_headroom"),
        vram_snapshot.get("vram_headroom"),
        telemetry.get("vram_headroom"),
        default=0.0,
    )
    vram_opportunity = _first_float(
        context.get("vram_opportunity_score"),
        vram_snapshot.get("vram_opportunity_score"),
        telemetry.get("vram_opportunity_score"),
        default=0.0,
    )

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
    causal_attestation = _safe_dict(context.get("causal_attestation"))
    causal_status = str(causal_attestation.get("validation_status") or "").strip().lower()
    if causal_status == "fail":
        contradiction = max(contradiction, 0.75)
        causal_risk = max(causal_risk, 0.85)
        ambiguity_signal = max(ambiguity_signal, 0.70)
        fragility_risk_signal = max(fragility_risk_signal, 0.75)
    elif causal_status == "warn":
        contradiction = max(contradiction, 0.50)
        causal_risk = max(causal_risk, 0.55)
        ambiguity_signal = max(ambiguity_signal, 0.50)
        fragility_risk_signal = max(fragility_risk_signal, 0.55)

    memory_attestation = _safe_dict(context.get("memory_rag_attestation"))
    memory_status = str(memory_attestation.get("validation_status") or "").strip().lower()
    memory_purity = _first_float(memory_attestation.get("retrieval_purity"), default=1.0)
    if memory_status == "fail" or memory_purity < 0.50:
        uncertainty = max(uncertainty, 0.55)
        fragility_risk_signal = max(fragility_risk_signal, 0.70)
    elif memory_status == "warn" or memory_purity < 0.75:
        uncertainty = max(uncertainty, 0.40)
        fragility_risk_signal = max(fragility_risk_signal, 0.50)

    propositions = observation.get("propositions")
    proposition_count = len(propositions) if isinstance(propositions, list) else 0
    structure_missing = 1.0 if proposition_count <= 1 else 0.0
    pattern_without_structure_signal = _clamp(
        0.60 * heterogeneity_signal
        + 0.30 * structure_missing
        + 0.10 * _clamp(spatial_entropy)
    )
    pattern_without_structure_signal = max(
        pattern_without_structure_signal,
        _clamp(_as_float(context.get("pattern_without_structure_signal"), default=0.0)),
    )

    vram_favorable_signal = _clamp(
        0.45 * _clamp(vram_headroom)
        + 0.45 * _clamp(vram_opportunity)
        + 0.10 * _clamp(context.get("vram_favorable_hint", 0.0))
    )
    cpu_pressure = _clamp(_first_float(
        context.get("cpu_pressure"),
        context.get("cpu_load"),
        telemetry.get("cpu_pressure"),
        telemetry.get("cpu_load"),
        observation.get("cpu_pressure"),
        observation.get("cpu_load"),
    ))
    memory_pressure = _clamp(_first_float(
        context.get("memory_pressure"),
        context.get("memory_load"),
        context.get("ram_pressure"),
        telemetry.get("memory_pressure"),
        telemetry.get("memory_load"),
        observation.get("memory_pressure"),
        observation.get("memory_load"),
    ))
    vram_pressure = _clamp(_first_float(
        context.get("vram_pressure"),
        vram_snapshot.get("vram_pressure"),
        telemetry.get("vram_pressure"),
        observation.get("vram_pressure"),
    ))
    gpu_load = _clamp(_first_float(
        context.get("gpu_load"),
        context.get("gpu_utilization"),
        telemetry.get("gpu_load"),
        telemetry.get("gpu_utilization"),
        observation.get("gpu_load"),
        observation.get("gpu_utilization"),
    ))
    gpu_available = bool(
        context.get("gpu_available")
        or context.get("cuda_available")
        or str(context.get("device", "")).strip().lower() in {"cuda", "gpu"}
        or str(telemetry.get("device", "")).strip().lower() in {"cuda", "gpu"}
    )
    thermal_pressure = _clamp(_first_float(
        context.get("thermal_pressure"),
        context.get("temperature_pressure"),
        telemetry.get("thermal_pressure"),
        telemetry.get("temperature_pressure"),
        observation.get("thermal_pressure"),
        observation.get("temperature_pressure"),
    ))
    hardware_pressure_signal = _clamp(max(
        cpu_pressure,
        memory_pressure,
        vram_pressure,
        thermal_pressure,
    ))
    explicit_gpu_opportunity = _first_float(
        context.get("gpu_opportunity_score"),
        context.get("gpu_acceleration_signal"),
        telemetry.get("gpu_opportunity_score"),
        telemetry.get("gpu_acceleration_signal"),
        default=-1.0,
    )
    if explicit_gpu_opportunity >= 0.0:
        gpu_acceleration_signal = _clamp(explicit_gpu_opportunity)
    else:
        gpu_acceleration_signal = _clamp(
            (0.55 * vram_favorable_signal)
            + (0.25 * (1.0 - gpu_load))
            + (0.20 if gpu_available else 0.0)
        )
    if vram_pressure >= 0.88 or thermal_pressure >= 0.85:
        gpu_acceleration_signal = min(gpu_acceleration_signal, 0.25)

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
        "cpu_pressure": cpu_pressure,
        "memory_pressure": memory_pressure,
        "vram_pressure": vram_pressure,
        "gpu_load": gpu_load,
        "gpu_acceleration_signal": gpu_acceleration_signal,
        "thermal_pressure": thermal_pressure,
        "hardware_pressure_signal": hardware_pressure_signal,
    }
