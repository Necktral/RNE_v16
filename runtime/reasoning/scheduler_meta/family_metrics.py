"""Métricas family-sensitive para benchmarking y análisis."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Mapping

from .family_profiles import (
    core_families_upper,
    optional_families_upper,
    profile_optional_families_upper,
)


STATUS_FACTOR = {
    "ok": 1.0,
    "passed": 1.0,
    "certified": 1.0,
    "idle": 0.4,
    "warn": 0.6,
    "warning": 0.6,
    "error": 0.2,
    "failed": 0.2,
}


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_family(family: Any) -> str:
    if not isinstance(family, str):
        return ""
    return family.strip().upper()


def _safe_entropy(counts: Mapping[str, int]) -> float:
    total = sum(max(int(v), 0) for v in counts.values())
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        if count <= 0:
            continue
        p = count / total
        entropy -= p * math.log2(p)
    return entropy


def _normalize_sequence(sequence: Iterable[Any] | None) -> List[str]:
    return [_normalize_family(item) for item in list(sequence or []) if _normalize_family(item)]


def compute_family_activation_metrics(
    *,
    reasoning_sequence: Iterable[Any],
    reasoning_trace: Iterable[Mapping[str, Any]] | None = None,
    profile_name: str | None = None,
    mode: str = "fixed",
) -> Dict[str, Any]:
    sequence = _normalize_sequence(reasoning_sequence)
    traces = list(reasoning_trace or [])

    counts = Counter(sequence)
    first_activation: Dict[str, int] = {}
    last_activation: Dict[str, int] = {}
    for idx, family in enumerate(sequence):
        if family not in first_activation:
            first_activation[family] = idx
        last_activation[family] = idx

    ordered_unique: List[str] = []
    for family in sequence:
        if family not in ordered_unique:
            ordered_unique.append(family)

    trace_by_family: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for idx, trace_step in enumerate(traces):
        family = _normalize_family(trace_step.get("family"))
        if not family:
            family = sequence[idx] if idx < len(sequence) else ""
        if not family:
            continue
        detail = trace_step.get("detail") if isinstance(trace_step, dict) else {}
        trace_by_family[family].append(
            {
                "step_index": idx,
                "status": trace_step.get("status"),
                "confidence": _as_float((detail or {}).get("confidence"), 0.0),
                "cost": _as_float((detail or {}).get("cost"), 0.0),
            }
        )

    core_set = core_families_upper()
    optional_set = profile_optional_families_upper(profile_name, mode=mode)
    if not optional_set:
        optional_set = optional_families_upper()

    active_families = set(counts.keys())
    optional_active = sorted(active_families.intersection(optional_set))
    core_only = bool(active_families) and active_families.issubset(core_set)

    activation_presence = {family: count > 0 for family, count in counts.items()}
    for family in sorted(core_set.union(optional_set)):
        activation_presence.setdefault(family, False)

    activation_counts = {family: int(counts.get(family, 0)) for family in sorted(core_set.union(optional_set))}

    return {
        "family_activation_counts": activation_counts,
        "family_activation_presence": activation_presence,
        "family_activation_order": ordered_unique,
        "family_mix_entropy": _safe_entropy(counts),
        "family_core_only_flag": core_only,
        "family_optional_used_flag": len(optional_active) > 0,
        "family_optional_count": len(optional_active),
        "family_trace_by_family": dict(trace_by_family),
        "family_first_activation_step": {k: int(v) for k, v in first_activation.items()},
        "family_last_activation_step": {k: int(v) for k, v in last_activation.items()},
    }


def _compute_step_weight(trace_step: Mapping[str, Any]) -> float:
    detail = trace_step.get("detail") if isinstance(trace_step, dict) else {}
    status = str((trace_step.get("status") or "")).strip().lower()
    status_factor = STATUS_FACTOR.get(status, 0.7)
    confidence = min(max(_as_float((detail or {}).get("confidence"), 0.5), 0.0), 1.0)
    cost = max(_as_float((detail or {}).get("cost"), 1.0), 0.05)
    return (confidence * status_factor) / cost


def compute_family_impact_metrics(
    *,
    reasoning_sequence: Iterable[Any],
    reasoning_trace: Iterable[Mapping[str, Any]] | None,
    final_metrics: Mapping[str, Any],
) -> Dict[str, Any]:
    sequence = [_normalize_family(item) for item in list(reasoning_sequence or []) if _normalize_family(item)]
    traces = list(reasoning_trace or [])
    counts = Counter(sequence)

    raw_weights: Dict[str, float] = defaultdict(float)
    for idx, family in enumerate(sequence):
        if idx < len(traces) and isinstance(traces[idx], dict):
            raw_weights[family] += _compute_step_weight(traces[idx])
        else:
            raw_weights[family] += 0.5

    total_weight = sum(v for v in raw_weights.values() if v > 0.0)
    contribution_proxy: Dict[str, float] = {}
    if total_weight > 0:
        for family, weight in raw_weights.items():
            contribution_proxy[family] = max(weight, 0.0) / total_weight
    else:
        uniform = 1.0 / len(counts) if counts else 0.0
        for family in counts:
            contribution_proxy[family] = uniform

    ivc_r = _as_float(final_metrics.get("ivc_r"), 0.0)
    intervention_precision = _as_float(final_metrics.get("intervention_precision"), 0.0)
    viability_margin = _as_float(final_metrics.get("viability_margin"), 0.0)
    trace_length = _as_float(final_metrics.get("reasoning_trace_length"), float(len(sequence)))
    success_rate = _as_float(final_metrics.get("success_rate"), 0.0)
    spatial_usage = _as_float(final_metrics.get("spatial_information_usage"), 0.0)

    delta_ivc_r = {}
    delta_precision = {}
    delta_viability = {}
    delta_trace_len = {}
    delta_success = {}
    delta_spatial = {}
    for family in sorted(counts):
        contrib = contribution_proxy.get(family, 0.0)
        delta_ivc_r[family] = contrib * ivc_r
        delta_precision[family] = contrib * intervention_precision
        delta_viability[family] = contrib * viability_margin
        delta_trace_len[family] = float(counts.get(family, 0))
        delta_success[family] = contrib * success_rate
        delta_spatial[family] = contrib * spatial_usage

    return {
        "family_contribution_proxy": contribution_proxy,
        "family_delta_ivc_r": delta_ivc_r,
        "family_delta_intervention_precision": delta_precision,
        "family_delta_viability_margin": delta_viability,
        "family_delta_reasoning_trace_length": delta_trace_len,
        "family_delta_success_rate": delta_success,
        "family_delta_spatial_information_usage": delta_spatial,
    }


def build_family_sensitive_bundle(
    *,
    reasoning_sequence: Iterable[Any],
    reasoning_trace: Iterable[Mapping[str, Any]] | None = None,
    profile_name: str | None = None,
    mode: str = "fixed",
    final_metrics: Mapping[str, Any] | None = None,
    proposed_sequence: Iterable[Any] | None = None,
    validated_sequence: Iterable[Any] | None = None,
    sequence_validation: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    activation = compute_family_activation_metrics(
        reasoning_sequence=reasoning_sequence,
        reasoning_trace=reasoning_trace,
        profile_name=profile_name,
        mode=mode,
    )
    impact = compute_family_impact_metrics(
        reasoning_sequence=reasoning_sequence,
        reasoning_trace=reasoning_trace,
        final_metrics=final_metrics or {},
    )
    validation = dict(sequence_validation or {})
    proposed = _normalize_sequence(proposed_sequence or validation.get("proposed_sequence"))
    validated = _normalize_sequence(validated_sequence or validation.get("validated_sequence") or reasoning_sequence)
    mandatory_floor = _normalize_sequence(validation.get("mandatory_family_floor"))
    primary_regime_label = str(validation.get("primary_regime_label") or "").strip().lower()
    cognitive_regime_label = str(validation.get("cognitive_regime_label") or "").strip().lower()
    floor_regime_label = str(validation.get("floor_regime_label") or "").strip().lower()
    executed = _normalize_sequence(reasoning_sequence)
    backbone_floor_satisfied = bool(executed) and all(family in executed for family in mandatory_floor)
    proposed_passed = bool(validation.get("proposed_passed", True))
    validated_passed = bool(validation.get("validated_passed", True))
    closure_break = (not proposed_passed) or (not validated_passed)

    return {
        **activation,
        **impact,
        "primary_regime_label": primary_regime_label,
        "cognitive_regime_label": cognitive_regime_label,
        "floor_regime_label": floor_regime_label,
        "mandatory_family_floor": [family.upper() for family in mandatory_floor],
        "proposed_sequence": [family.upper() for family in proposed],
        "validated_sequence": [family.upper() for family in validated],
        "sequence_validation_report": validation,
        "admitted_overlays": list(validation.get("admitted_overlays") or []),
        "default_overlays": list(validation.get("default_overlays") or []),
        "correction_steps": list(validation.get("correction_steps") or []),
        "fallback_profile_name": validation.get("fallback_profile_name"),
        "effective_max_steps": int(_as_float(validation.get("effective_max_steps"), 0.0)),
        "backbone_floor_satisfied_flag": backbone_floor_satisfied,
        "sequence_validation_fail_flag": not proposed_passed,
        "fallback_to_safe_sequence_flag": bool(validation.get("fallback_used", False)),
        "optional_displacement_flag": bool(validation.get("optional_displacement_detected", False)),
        "closure_break_flag": closure_break,
        "sequence_autocorrected_flag": bool(validation.get("autocorrected", False)),
        "budget_overridden_by_floor_flag": bool(validation.get("budget_overridden_by_floor", False)),
    }


def aggregate_family_dict_metric(rows: Iterable[Mapping[str, Any]], key: str) -> Dict[str, float]:
    acc: Dict[str, float] = defaultdict(float)
    count = 0
    for row in rows:
        data = row.get(key)
        if not isinstance(data, dict):
            continue
        count += 1
        for family, value in data.items():
            acc[str(family)] += _as_float(value, 0.0)
    if count == 0:
        return {}
    return {family: value / count for family, value in sorted(acc.items())}


def aggregate_family_activation_counts(rows: Iterable[Mapping[str, Any]]) -> Dict[str, int]:
    acc: Dict[str, int] = defaultdict(int)
    for row in rows:
        data = row.get("family_activation_counts")
        if not isinstance(data, dict):
            continue
        for family, value in data.items():
            acc[str(family)] += int(_as_float(value, 0.0))
    return dict(sorted(acc.items()))
