"""Governance envelope for META reasoning runs."""

from __future__ import annotations

import os
from typing import Any, Dict, Sequence

from runtime.reasoning.scheduler_meta.degradation import build_degradation_plan


def _clamp01(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return min(max(numeric, 0.0), 1.0)


def _memory_count(context: Dict[str, Any]) -> int:
    attestation = context.get("memory_rag_attestation")
    if isinstance(attestation, dict) and isinstance(attestation.get("returned_count"), int):
        return int(attestation["returned_count"])
    retrieved = context.get("retrieved_memory")
    if isinstance(retrieved, list):
        return len(retrieved)
    memory_hits = context.get("memory_hits")
    if isinstance(memory_hits, list):
        return len(memory_hits)
    return 0


def _memory_purity(context: Dict[str, Any]) -> float | None:
    attestation = context.get("memory_rag_attestation")
    if isinstance(attestation, dict):
        purity = attestation.get("retrieval_purity")
        if isinstance(purity, (int, float)):
            return _clamp01(purity)
    explicit = context.get("memory_purity_confidence", context.get("memory_purity"))
    if isinstance(explicit, (int, float)):
        return _clamp01(explicit)
    retrieved = context.get("retrieved_memory")
    scenario = context.get("scenario_name")
    if not isinstance(retrieved, list) or not retrieved or not isinstance(scenario, str):
        return None
    same = 0
    comparable = 0
    for item in retrieved:
        if not isinstance(item, dict):
            continue
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        item_scenario = item.get("scenario_name") or metadata.get("scenario_name")
        if isinstance(item_scenario, str):
            comparable += 1
            same += int(item_scenario == scenario)
    if comparable == 0:
        return None
    return same / comparable


def _degradation_level(
    *,
    features: Dict[str, float],
    sequence_validation: Dict[str, Any],
) -> str:
    hardware_pressure = _clamp01(features.get("hardware_pressure_signal", 0.0))
    if sequence_validation.get("fallback_used"):
        return "fallback_safe_profile"
    if hardware_pressure >= 0.85:
        return "severe_hardware_conservation"
    if hardware_pressure >= 0.70:
        return "hardware_conservation"
    if sequence_validation.get("autocorrected"):
        return "validator_corrected"
    return "nominal"


def _compensations(
    *,
    features: Dict[str, float],
    sequence_validation: Dict[str, Any],
) -> list[str]:
    out: list[str] = []
    hardware_pressure = _clamp01(features.get("hardware_pressure_signal", 0.0))
    if hardware_pressure >= 0.70:
        out.append("compress_optional_reasoning_budget")
    if sequence_validation.get("autocorrected"):
        out.append("restore_validated_core_sequence")
    if sequence_validation.get("fallback_used"):
        out.append("fallback_to_safe_family_profile")
    correction_steps = sequence_validation.get("correction_steps")
    if isinstance(correction_steps, list):
        for step in correction_steps:
            normalized = str(step)
            if normalized.startswith("budget_trim:"):
                out.append("trim_overlays_to_budget")
    if sequence_validation.get("prob_last_ok"):
        out.append("keep_probabilistic_calibration_last")
    return list(dict.fromkeys(out))


def _autonomy_policy(
    *,
    context: Dict[str, Any],
    sequence_validation: Dict[str, Any],
    effective_max_steps: int,
) -> Dict[str, Any]:
    central_policy = context.get("autonomy_policy")
    if isinstance(central_policy, dict):
        requested = str(central_policy.get("requested_mode") or "bounded").strip().lower()
        active = str(central_policy.get("active_mode") or "bounded").strip().lower()
        central_authorized = bool(central_policy.get("policy_authorized"))
        validation_passed = bool(sequence_validation.get("validated_passed"))
        authorized = bool(
            active == "governed_unbounded"
            and central_authorized
            and validation_passed
        )
        return {
            "requested": requested,
            "mode": "governed_unbounded" if authorized else "bounded",
            "policy_authorized": authorized,
            "scope": "scheduler_iteration" if authorized else "single_run",
            "requires_validated_sequence": True,
            "effective_max_steps": effective_max_steps,
            "source": "operational_conjunction",
            "policy": dict(central_policy),
        }
    requested = (
        context.get("autonomy_policy")
        or context.get("autonomy_mode")
        or os.environ.get("RNFE_AUTONOMY_POLICY")
        or "bounded"
    )
    normalized = str(requested).strip().lower()
    unbounded_requested = normalized in {
        "unlimited",
        "unbounded",
        "governed_unbounded",
        "policy_unbounded",
    }
    validation_passed = bool(sequence_validation.get("validated_passed"))
    return {
        "requested": normalized,
        "mode": "governed_unbounded" if unbounded_requested and validation_passed else "bounded",
        "policy_authorized": bool(unbounded_requested and validation_passed),
        "scope": "scheduler_iteration" if unbounded_requested else "single_run",
        "requires_validated_sequence": True,
        "effective_max_steps": effective_max_steps,
    }


def build_governance_envelope(
    *,
    context: Dict[str, Any],
    features: Dict[str, float],
    budget: Dict[str, float],
    selected_sequence: Sequence[str],
    executed_sequence: Sequence[str],
    policy_meta: Dict[str, Any],
    sequence_validation: Dict[str, Any],
    effective_max_steps: int,
) -> Dict[str, Any]:
    hardware_pressure = _clamp01(features.get("hardware_pressure_signal", 0.0))
    gpu_acceleration = _clamp01(features.get("gpu_acceleration_signal", 0.0))
    memory_purity = _memory_purity(context)
    memory_attestation = context.get("memory_rag_attestation")
    if not isinstance(memory_attestation, dict):
        memory_attestation = None
    causal_attestation = context.get("causal_attestation")
    if not isinstance(causal_attestation, dict):
        causal_attestation = None
    central_autonomy_policy = context.get("autonomy_policy")
    central_autonomy_policy = (
        dict(central_autonomy_policy) if isinstance(central_autonomy_policy, dict) else None
    )
    degradation_plan = build_degradation_plan(
        features=features,
        sequence_validation=sequence_validation,
        memory_attestation=memory_attestation,
        causal_attestation=causal_attestation,
        autonomy_policy=central_autonomy_policy,
    )
    return {
        "schema": "reasoning_governance.v1",
        "causality": {
            "causal_risk": _clamp01(features.get("causal_risk", 0.0)),
            "counterfactual_gap": context.get("counterfactual_gap"),
            "regime_label": policy_meta.get("regime_label"),
            "attestation": causal_attestation,
        },
        "memory_rag": {
            "retrieved_count": _memory_count(context),
            "purity": memory_purity,
            "filter_mode": context.get("memory_filter_mode"),
            "attestation": memory_attestation,
        },
        "reasoning": {
            "selected_sequence": [str(item).upper() for item in selected_sequence],
            "executed_sequence": [str(item).upper() for item in executed_sequence],
            "profile": policy_meta.get("profile_name"),
            "mode": policy_meta.get("mode"),
        },
        "agents_governed": {
            "mandatory_family_floor": policy_meta.get("mandatory_family_floor") or [],
            "allowed_families": [
                str(item).upper()
                for item in policy_meta.get("allowed_families", [])
            ],
        },
        "compensations": _compensations(
            features=features,
            sequence_validation=sequence_validation,
        ),
        "hardware": {
            "pressure": hardware_pressure,
            "cpu_pressure": _clamp01(features.get("cpu_pressure", 0.0)),
            "memory_pressure": _clamp01(features.get("memory_pressure", 0.0)),
            "vram_pressure": _clamp01(features.get("vram_pressure", 0.0)),
            "thermal_pressure": _clamp01(features.get("thermal_pressure", 0.0)),
            "gpu_load": _clamp01(features.get("gpu_load", 0.0)),
            "gpu_acceleration": gpu_acceleration,
            "gpu_budget_bonus": bool(gpu_acceleration >= 0.70 and hardware_pressure < 0.70),
        },
        "validation": {
            "passed": bool(sequence_validation.get("validated_passed")),
            "autocorrected": bool(sequence_validation.get("autocorrected")),
            "fallback_used": bool(sequence_validation.get("fallback_used")),
            "correction_steps": list(sequence_validation.get("correction_steps") or []),
        },
        "traceability": {
            "run_id": context.get("run_id"),
            "effective_max_steps": effective_max_steps,
            "cost_budget": budget.get("cost_budget"),
        },
        "graceful_degradation": {
            "level": degradation_plan["level"],
            "hardware_pressure": hardware_pressure,
            "severity": degradation_plan["severity"],
            "budget_multiplier": degradation_plan["budget_multiplier"],
            "reasons": list(degradation_plan["reasons"]),
            "actions": list(degradation_plan["actions"]),
        },
        "degradation_plan": degradation_plan,
        "autonomy": _autonomy_policy(
            context=context,
            sequence_validation=sequence_validation,
            effective_max_steps=effective_max_steps,
        ),
    }
