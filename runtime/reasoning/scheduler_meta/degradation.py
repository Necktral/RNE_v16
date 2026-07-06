"""Intelligent degradation planning for META governance."""

from __future__ import annotations

from typing import Any, Dict


def _clamp01(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return min(max(numeric, 0.0), 1.0)


def _status(block: Dict[str, Any] | None) -> str:
    if not isinstance(block, dict):
        return ""
    return str(block.get("validation_status") or "").strip().lower()


def _append_unique(items: list[str], *values: str) -> None:
    for value in values:
        if value and value not in items:
            items.append(value)


def build_degradation_plan(
    *,
    features: Dict[str, float],
    sequence_validation: Dict[str, Any],
    memory_attestation: Dict[str, Any] | None = None,
    causal_attestation: Dict[str, Any] | None = None,
    autonomy_policy: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Return a compact degradation plan from current runtime evidence."""
    severity = 0.0
    reasons: list[str] = []
    actions: list[str] = []
    level = "nominal"
    budget_multiplier = 1.0

    hardware_pressure = _clamp01(features.get("hardware_pressure_signal", 0.0))
    gpu_acceleration = _clamp01(features.get("gpu_acceleration_signal", 0.0))
    if hardware_pressure >= 0.85:
        severity = max(severity, 0.85)
        level = "resource_conservation"
        _append_unique(
            reasons,
            "severe_hardware_pressure",
        )
        _append_unique(
            actions,
            "compress_optional_reasoning_budget",
            "avoid_large_models",
            "prefer_cache_and_local_light_routes",
        )
        budget_multiplier = min(budget_multiplier, 0.50)
    elif hardware_pressure >= 0.70:
        severity = max(severity, 0.65)
        level = "resource_conservation"
        _append_unique(reasons, "hardware_pressure")
        _append_unique(actions, "trim_shadow_families", "prefer_cache")
        budget_multiplier = min(budget_multiplier, 0.75)
    elif gpu_acceleration >= 0.70:
        _append_unique(actions, "allow_gpu_accelerated_optional_reasoning")

    causal_status = _status(causal_attestation)
    if causal_status == "fail":
        severity = max(severity, 0.90)
        level = "causal_recovery"
        _append_unique(reasons, "causal_attestation_failed")
        _append_unique(
            actions,
            "force_causal_counterfactual_recheck",
            "block_actuation_until_causal_revalidated",
        )
        budget_multiplier = min(budget_multiplier, 0.70)
    elif causal_status == "warn":
        severity = max(severity, 0.58)
        if level == "nominal":
            level = "causal_guarded"
        _append_unique(reasons, "causal_attestation_warn")
        _append_unique(actions, "require_causal_evidence_before_strong_recommendation")

    memory_status = _status(memory_attestation)
    memory_purity = None
    if isinstance(memory_attestation, dict):
        purity = memory_attestation.get("retrieval_purity")
        if isinstance(purity, (int, float)):
            memory_purity = _clamp01(purity)
    if memory_status == "fail" or (memory_purity is not None and memory_purity < 0.50):
        severity = max(severity, 0.82)
        level = "memory_isolation"
        _append_unique(reasons, "memory_rag_contamination")
        _append_unique(actions, "strict_same_scenario_memory_only", "drop_analogical_memory")
        budget_multiplier = min(budget_multiplier, 0.80)
    elif memory_status == "warn" or (memory_purity is not None and memory_purity < 0.75):
        severity = max(severity, 0.52)
        if level == "nominal":
            level = "memory_guarded"
        _append_unique(reasons, "memory_rag_uncertain")
        _append_unique(actions, "penalize_analogical_memory", "attach_memory_attestation")

    if sequence_validation.get("fallback_used"):
        severity = max(severity, 0.78)
        level = "safe_profile_fallback"
        _append_unique(reasons, "sequence_fallback_used")
        _append_unique(actions, "fallback_to_safe_family_profile")
        budget_multiplier = min(budget_multiplier, 0.80)
    elif sequence_validation.get("autocorrected"):
        severity = max(severity, 0.45)
        if level == "nominal":
            level = "validator_corrected"
        _append_unique(reasons, "sequence_autocorrected")
        _append_unique(actions, "use_validated_sequence")

    if isinstance(autonomy_policy, dict):
        requested = str(autonomy_policy.get("requested_mode") or "").strip().lower()
        active = str(autonomy_policy.get("active_mode") or "").strip().lower()
        if requested in {"unlimited", "unbounded", "governed_unbounded", "policy_unbounded"}:
            if active != "governed_unbounded" or not autonomy_policy.get("policy_authorized"):
                severity = max(severity, 0.60)
                if level == "nominal":
                    level = "bounded_autonomy"
                _append_unique(reasons, "autonomy_degraded_by_policy")
                _append_unique(actions, "degrade_autonomy_scope")

    if not reasons:
        _append_unique(reasons, "all_degradation_inputs_nominal")
    if not actions:
        _append_unique(actions, "continue_nominal_execution")

    return {
        "schema": "degradation_plan.v1",
        "level": level,
        "severity": round(severity, 4),
        "budget_multiplier": round(budget_multiplier, 4),
        "reasons": reasons,
        "actions": actions,
        "signals": {
            "hardware_pressure": hardware_pressure,
            "gpu_acceleration": gpu_acceleration,
            "causal_status": causal_status or None,
            "memory_status": memory_status or None,
            "memory_purity": memory_purity,
        },
    }
