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


def _optimization_direction(scenario_metadata: Dict[str, Any]) -> str:
    """Dirección de optimización del escenario (best-effort vía el registro).

    Default ``"minimize"`` → preserva el comportamiento térmico histórico
    (la temperatura es lower-is-better). Para recursos resuelve ``"maximize"``.
    Import perezoso y tolerante: nunca rompe ni acopla import-time.
    """
    name = scenario_metadata.get("scenario_name")
    if name:
        try:
            from runtime.world.registry import get_scenario

            sig = get_scenario(name).causal_signature
            direction = getattr(sig, "optimization_direction", None)
            if isinstance(direction, str) and direction:
                return direction
        except Exception:
            pass
    low = (name or "").lower()
    if "resource" in low or "stock" in low:
        return "maximize"
    return "minimize"


def _causal_consistency(episode: Dict[str, Any]) -> float:
    """Coherencia causal genérica: compara factual vs contrafactual sobre la
    variable principal del escenario (no hardcodea ``temperature``) según la
    dirección de optimización. Térmico (``minimize``) se mantiene idéntico.
    """
    result = episode.get("result", {})
    ctx = episode.get("context", {})
    factual = result.get("updated_world", {})
    counterfactual = ctx.get("counterfactual", {})
    scenario_metadata = episode.get("scenario_metadata", {}) or {}
    main_var = scenario_metadata.get("main_variable", "temperature")
    factual_val = factual.get(main_var)
    counterfactual_val = counterfactual.get(main_var)
    if isinstance(factual_val, (int, float)) and isinstance(
        counterfactual_val, (int, float)
    ):
        if _optimization_direction(scenario_metadata) == "maximize":
            return 1.0 if factual_val >= counterfactual_val else 0.0
        return 1.0 if factual_val <= counterfactual_val else 0.0
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


def continuity_vector(
    *,
    previous_result: Dict[str, Any],
    current_result: Dict[str, Any],
    compatibility: Any,
    retrieval_metrics: Dict[str, Any] | None = None,
) -> Any:
    """Computa vector de continuidad tensorial (delegación a transition_analysis).

    Esta función es el punto de acceso nuevo desde continuity.py
    que envuelve la implementación vectorial/tensorial.

    Args:
        previous_result: Resultado del episodio anterior.
        current_result: Resultado del episodio actual.
        compatibility: CompatibilityAssessment del grafo.
        retrieval_metrics: Métricas de retrieval de memoria.

    Returns:
        TransitionContinuityVector con todos los componentes.
    """
    from .transition_analysis import compute_transition_vector

    return compute_transition_vector(
        previous_result=previous_result,
        current_result=current_result,
        compatibility=compatibility,
        retrieval_metrics=retrieval_metrics,
    )


def continuity_tensor(
    *,
    vectors: list,
) -> Dict[str, Any]:
    """Construye tensor de continuidad (delegación a transition_analysis).

    Args:
        vectors: Lista de TransitionContinuityVector.

    Returns:
        Dict anidado [source][target] -> ContinuityTensorCell.
    """
    from .transition_analysis import build_continuity_tensor

    return build_continuity_tensor(vectors=vectors)
