"""Evaluación de cierre triádico y trazabilidad por episodio."""

from __future__ import annotations

from typing import Any, Dict

from runtime.lotf import LOTFMin


REQUIRED_META_SEQUENCE = ["ABD", "ANA", "CAU", "CTF", "DED", "PROB"]


def _has_episode_closed_event(storage, *, run_id: str | None, episode_id: str) -> bool:
    events = storage.list_events(run_id=run_id, limit=500)
    for item in events:
        if item.event_type != "episode.closed":
            continue
        payload = item.payload or {}
        if payload.get("episode_id") == episode_id:
            return True
    return False


def evaluate_episode_closure(
    *, storage, run_id: str | None, result: Dict[str, Any]
) -> Dict[str, Any]:
    episode = result.get("episode", {})
    smg_snapshot = result.get("smg_snapshot", {})
    episode_id = episode.get("episode_id", "")
    context = episode.get("context", {})
    output = episode.get("result", {})
    trace = episode.get("trace", [])

    has_observation = isinstance(context.get("observation"), dict)
    has_signs = len(smg_snapshot.get("signs", [])) >= 2

    formula_ok = False
    formula = context.get("formula")
    if isinstance(formula, str) and formula.strip():
        try:
            parsed = LOTFMin().parse(formula)
            LOTFMin().check(parsed, {"TEMP_HIGH": "bool", "ACTIVATE_COOLING": "bool"})
            formula_ok = True
        except Exception:
            formula_ok = False

    has_intervention = isinstance(context.get("intervention"), str) and isinstance(
        context.get("counterfactual"), dict
    )
    has_episode_closed = bool(episode_id) and _has_episode_closed_event(
        storage, run_id=run_id, episode_id=episode_id
    )

    reasoning_sequence = output.get("reasoning_sequence", []) or []
    meta_trace_complete = (
        isinstance(trace, list)
        and len(trace) >= len(REQUIRED_META_SEQUENCE)
        and list(reasoning_sequence) == REQUIRED_META_SEQUENCE
    )
    trace_integrity = has_episode_closed and meta_trace_complete

    checks = {
        "observation_registered": has_observation,
        "signs_persisted": has_signs,
        "lotf_parse_check_ok": formula_ok,
        "factual_and_counterfactual_present": has_intervention,
        "episode_closed_event_present": has_episode_closed,
        "meta_trace_complete": meta_trace_complete,
    }
    closure_passed = all(checks.values())
    return {
        "episode_id": episode_id,
        "checks": checks,
        "closure_passed": closure_passed,
        "trace_integrity": trace_integrity,
    }
