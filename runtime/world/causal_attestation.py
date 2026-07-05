"""Causal attestation helpers for factual/counterfactual episode evidence."""

from __future__ import annotations

from typing import Any, Dict, Mapping


def _as_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _state_value(state: Mapping[str, Any] | None, main_variable: str) -> float | None:
    data = state if isinstance(state, Mapping) else {}
    for key in (main_variable, "global_temp_mean", "world_level", "temperature", "stock_level"):
        value = _as_float(data.get(key))
        if value is not None:
            return value
    return None


def _direction_from_delta(delta: float | None) -> str | None:
    if delta is None:
        return None
    if delta > 1e-9:
        return "+"
    if delta < -1e-9:
        return "-"
    return "0"


def _supports_choice(
    *,
    factual_value: float | None,
    counterfactual_value: float | None,
    optimization_direction: str,
) -> bool | None:
    if factual_value is None or counterfactual_value is None:
        return None
    if optimization_direction == "maximize":
        return factual_value >= counterfactual_value
    return factual_value <= counterfactual_value


def _expected_direction(signature: Any, intervention: str | None) -> str | None:
    if signature is None or not intervention:
        return None
    for effect in getattr(signature, "intervention_effects", ()) or ():
        if getattr(effect, "intervention_name", None) == intervention:
            return getattr(effect, "expected_direction", None)
    return None


def _signature_summary(signature: Any) -> Dict[str, Any]:
    if signature is None:
        return {
            "signature_present": False,
            "edge_count": 0,
            "intervention_effect_count": 0,
        }
    edges = tuple(getattr(signature, "causal_edges", ()) or ())
    effects = tuple(getattr(signature, "intervention_effects", ()) or ())
    return {
        "signature_present": True,
        "edge_count": len(edges),
        "intervention_effect_count": len(effects),
        "counterfactual_policy": getattr(signature, "counterfactual_policy", None),
        "counterfactual_variable": getattr(signature, "counterfactual_variable", None),
    }


def build_causal_attestation(
    *,
    scenario_name: str | None,
    scenario_version: str | None = None,
    main_variable: str,
    intervention: str | None,
    observation: Mapping[str, Any] | None,
    factual: Mapping[str, Any] | None,
    counterfactual: Mapping[str, Any] | None,
    relation_kind: str | None,
    signature: Any = None,
) -> Dict[str, Any]:
    """Build an auditable causal validation block for an episode.

    The attestation is deterministic and intentionally small: it checks whether
    the factual transition is at least as good as the counterfactual under the
    declared optimization direction, then compares that result with
    ``relation_kind`` and the expected intervention direction from the signature.
    """
    optimization_direction = str(
        getattr(signature, "optimization_direction", None)
        or ("maximize" if "resource" in str(scenario_name or "").lower() else "minimize")
    )
    scenario_version = scenario_version or getattr(signature, "scenario_version", None)
    main_variable = str(main_variable or getattr(signature, "main_variable", "") or "world_level")
    observed_value = _state_value(observation, main_variable)
    factual_value = _state_value(factual, main_variable)
    counterfactual_value = _state_value(counterfactual, main_variable)
    factual_delta = (
        factual_value - observed_value
        if factual_value is not None and observed_value is not None
        else None
    )
    counterfactual_delta = (
        counterfactual_value - observed_value
        if counterfactual_value is not None and observed_value is not None
        else None
    )
    effect = (
        factual_value - counterfactual_value
        if factual_value is not None and counterfactual_value is not None
        else None
    )
    supports_choice = _supports_choice(
        factual_value=factual_value,
        counterfactual_value=counterfactual_value,
        optimization_direction=optimization_direction,
    )
    agreement_with_relation_kind = None
    if supports_choice is not None and relation_kind in {"support", "contradiction"}:
        agreement_with_relation_kind = (relation_kind == "support") == bool(supports_choice)

    expected_direction = _expected_direction(signature, intervention)
    observed_direction = _direction_from_delta(factual_delta)
    direction_match = (
        expected_direction is not None
        and observed_direction is not None
        and observed_direction != "0"
        and expected_direction == observed_direction
    )

    missing = []
    if signature is None:
        missing.append("causal_signature")
    if factual_value is None:
        missing.append("factual_value")
    if counterfactual_value is None:
        missing.append("counterfactual_value")
    if relation_kind not in {"support", "contradiction"}:
        missing.append("relation_kind")
    if expected_direction is None:
        missing.append("intervention_effect")

    if agreement_with_relation_kind is False:
        validation_status = "fail"
        degradation_level = "relation_mismatch"
    elif missing:
        validation_status = "warn"
        degradation_level = "missing_evidence"
    elif direction_match is False:
        validation_status = "warn"
        degradation_level = "effect_direction_mismatch"
    else:
        validation_status = "pass"
        degradation_level = "nominal"

    confidence = 0.45
    confidence += 0.20 if signature is not None else 0.0
    confidence += 0.20 if supports_choice is not None else 0.0
    confidence += 0.10 if agreement_with_relation_kind is True else 0.0
    confidence += 0.05 if direction_match is True else 0.0
    if validation_status == "fail":
        confidence = min(confidence, 0.35)

    return {
        "schema": "causal_attestation.v1",
        "scenario_name": scenario_name,
        "scenario_version": scenario_version,
        "main_variable": main_variable,
        "intervention": intervention,
        "relation_kind": relation_kind,
        "optimization_direction": optimization_direction,
        "observed_value": None if observed_value is None else round(observed_value, 6),
        "factual_value": None if factual_value is None else round(factual_value, 6),
        "counterfactual_value": (
            None if counterfactual_value is None else round(counterfactual_value, 6)
        ),
        "factual_delta": None if factual_delta is None else round(factual_delta, 6),
        "counterfactual_delta": (
            None if counterfactual_delta is None else round(counterfactual_delta, 6)
        ),
        "observed_effect": None if effect is None else round(effect, 6),
        "supports_choice": supports_choice,
        "agreement_with_relation_kind": agreement_with_relation_kind,
        "expected_direction": expected_direction,
        "observed_direction": observed_direction,
        "direction_match": direction_match,
        "validation_status": validation_status,
        "degradation_level": degradation_level,
        "missing_evidence": missing,
        "confidence": round(max(0.0, min(1.0, confidence)), 4),
        "signature": _signature_summary(signature),
    }
