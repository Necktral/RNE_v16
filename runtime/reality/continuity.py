"""Métricas de continuidad entre episodios consecutivos."""

from __future__ import annotations

from typing import Any, Dict, Iterable


def _jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    a = set(left)
    b = set(right)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _sequence_stability(prev: list[str], curr: list[str]) -> float:
    if not prev and not curr:
        return 1.0
    if not prev or not curr:
        return 0.0
    max_len = max(len(prev), len(curr))
    matches = sum(1 for i in range(min(len(prev), len(curr))) if prev[i] == curr[i])
    return matches / max_len


def _causal_consistency(episode: Dict[str, Any]) -> float:
    result = episode.get("result", {})
    ctx = episode.get("context", {})
    factual = result.get("updated_world", {})
    counterfactual = ctx.get("counterfactual", {})
    factual_temp = factual.get("temperature")
    counterfactual_temp = counterfactual.get("temperature")
    if isinstance(factual_temp, (int, float)) and isinstance(
        counterfactual_temp, (int, float)
    ):
        return 1.0 if factual_temp <= counterfactual_temp else 0.0
    relation_kind = result.get("relation_kind")
    if relation_kind == "support":
        return 1.0
    if relation_kind == "contradiction":
        return 0.0
    return 0.0


def continuity_score(
    *,
    previous_episode: Dict[str, Any] | None,
    current_episode: Dict[str, Any],
    previous_smg_snapshot: Dict[str, Any] | None,
    current_smg_snapshot: Dict[str, Any],
    trace_integrity: bool,
) -> float:
    if previous_episode is None or previous_smg_snapshot is None:
        return 1.0

    prev_signs = [
        sign.get("proposition", "")
        for sign in previous_smg_snapshot.get("signs", [])
        if isinstance(sign, dict)
    ]
    curr_signs = [
        sign.get("proposition", "")
        for sign in current_smg_snapshot.get("signs", [])
        if isinstance(sign, dict)
    ]
    overlap = _jaccard(prev_signs, curr_signs)

    prev_seq = previous_episode.get("result", {}).get("reasoning_sequence", []) or []
    curr_seq = current_episode.get("result", {}).get("reasoning_sequence", []) or []
    sequence = _sequence_stability(list(prev_seq), list(curr_seq))

    causal = _causal_consistency(current_episode)
    integrity = 1.0 if trace_integrity else 0.0
    score = (0.4 * overlap) + (0.3 * causal) + (0.2 * sequence) + (0.1 * integrity)
    return max(0.0, min(1.0, score))
