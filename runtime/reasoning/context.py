"""Helpers para construir contexto de razonamiento reusable."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Mapping


_CLOSURE_PROFILE_TO_MODE = {
    "baseline_fixed": "fixed",
    "adaptive_min": "adaptive",
}


def _safe_copy(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if is_dataclass(value):
        return _safe_copy(asdict(value))
    if isinstance(value, dict):
        return {str(key): _safe_copy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_copy(item) for item in value]
    return value


def resolve_reasoning_mode(
    closure_profile: str | None,
    *,
    default: str = "fixed",
) -> str:
    normalized = str(closure_profile or "").strip().lower()
    return _CLOSURE_PROFILE_TO_MODE.get(normalized, default)


def build_reasoning_context(
    *,
    episode_id: str,
    run_id: str,
    observation: Mapping[str, Any],
    intervention: str,
    formula: str | None = None,
    memory_hits: list[dict[str, Any]] | None = None,
    counterfactual: Mapping[str, Any] | None = None,
    updated_world: Mapping[str, Any] | None = None,
    relation_kind: str | None = None,
    scenario: str | None = None,
    scenario_metadata: Mapping[str, Any] | None = None,
    belief_state: Mapping[str, Any] | Any | None = None,
    closure_profile: str | None = None,
    reasoning_mode: str | None = None,
    extra_signals: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Construye el contexto canónico consumido por el scheduler."""
    memory = [_safe_copy(item) for item in list(memory_hits or [])]
    context: Dict[str, Any] = {
        "episode_id": episode_id,
        "run_id": run_id,
        "observation": _safe_copy(dict(observation)),
        "intervention": intervention,
        "memory_hits": memory,
        "retrieved_memory": memory,
        "closure_profile": closure_profile,
        "reasoning_mode": reasoning_mode or resolve_reasoning_mode(closure_profile),
    }

    if formula is not None:
        context["formula"] = formula
    if counterfactual is not None:
        context["counterfactual"] = _safe_copy(dict(counterfactual))
    if updated_world is not None:
        factual = _safe_copy(dict(updated_world))
        context["updated_world"] = factual
        context["factual"] = factual
    if relation_kind is not None:
        context["relation_kind"] = relation_kind
    if scenario is not None:
        context["scenario"] = scenario
    if scenario_metadata is not None:
        context["scenario_metadata"] = _safe_copy(dict(scenario_metadata))
    if belief_state is not None:
        context["belief_state"] = _safe_copy(belief_state)
    if extra_signals:
        for key, value in extra_signals.items():
            if value is not None and key not in context:
                context[str(key)] = _safe_copy(value)
    return context
