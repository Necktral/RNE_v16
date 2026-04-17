"""Análisis de continuidad de transición entre escenarios.

Reemplaza la continuidad escalar por un vector de continuidad por
componentes y por transición, y construye un tensor NxN para
comparaciones matriciales entre escenarios.

Componentes del vector:
- semantic_retention: Jaccard de signos/proposiciones
- trace_stability: estabilidad de secuencia de razonamiento
- causal_stability: coherencia factual vs contrafactual por main_variable
- intervention_policy_stability: estabilidad de la política de intervención
- structural_compatibility: overall_score del grafo de compatibilidad
- memory_purity: (1 - contamination)

Fórmula composite:
  0.20 * semantic + 0.20 * trace + 0.20 * causal
  + 0.15 * intervention + 0.15 * structural + 0.10 * purity
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Literal

from runtime.world.compatibility import CompatibilityAssessment

# ── Data contracts ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TransitionContinuityVector:
    """Vector de continuidad para una transición episodio→episodio."""

    source_scenario: str
    target_scenario: str
    semantic_retention: float
    trace_stability: float
    causal_stability: float
    intervention_policy_stability: float
    structural_compatibility: float
    memory_purity: float
    composite_score: float
    transition_type: Literal["intra", "compatible", "analogical", "incompatible"]


@dataclass(frozen=True)
class ContinuityTensorCell:
    """Celda agregada del tensor de continuidad para un par de escenarios."""

    source_scenario: str
    target_scenario: str
    mean_composite: float
    mean_semantic_retention: float
    mean_causal_stability: float
    mean_memory_purity: float
    sample_count: int


# ── Weights ──────────────────────────────────────────────────────────────────

_W_SEMANTIC = 0.20
_W_TRACE = 0.20
_W_CAUSAL = 0.20
_W_INTERVENTION = 0.15
_W_STRUCTURAL = 0.15
_W_PURITY = 0.10


# ── Helpers ──────────────────────────────────────────────────────────────────

def _jaccard(left, right) -> float:
    a = set(left) if left else set()
    b = set(right) if right else set()
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _sequence_stability(prev: list, curr: list) -> float:
    if not prev and not curr:
        return 1.0
    if not prev or not curr:
        return 0.0
    max_len = max(len(prev), len(curr))
    matches = sum(1 for i in range(min(len(prev), len(curr))) if prev[i] == curr[i])
    return matches / max_len


def _causal_stability_generic(
    current_result: Dict[str, Any],
    main_variable: str | None = None,
) -> float:
    """Coherencia causal genérica usando main_variable del escenario, no hardcode.

    Busca relation_kind primero, luego compara factual vs counterfactual
    usando la variable principal del escenario.
    """
    result = current_result.get("result", {})
    relation_kind = result.get("relation_kind")
    if relation_kind == "support":
        return 1.0
    if relation_kind == "contradiction":
        return 0.0

    # Fallback: comparar valores factual vs counterfactual por main_variable
    if main_variable:
        factual = result.get("updated_world", {})
        ctx = current_result.get("context", {})
        counterfactual = ctx.get("counterfactual", {})
        f_val = factual.get(main_variable)
        c_val = counterfactual.get(main_variable)
        if isinstance(f_val, (int, float)) and isinstance(c_val, (int, float)):
            # Coherencia si están en la misma dirección que el relation_kind esperado
            return 1.0 if abs(f_val - c_val) < 0.15 else 0.5
    return 0.0


def _intervention_stability(
    previous_result: Dict[str, Any],
    current_result: Dict[str, Any],
) -> float:
    """Estabilidad de la política de intervención ante contexto comparable."""
    prev_ctx = previous_result.get("context", {})
    curr_ctx = current_result.get("context", {})
    prev_intervention = prev_ctx.get("intervention")
    curr_intervention = curr_ctx.get("intervention")
    if prev_intervention is None or curr_intervention is None:
        return 0.5  # No comparable
    prev_alarm = prev_ctx.get("observation", {}).get("alarm", False)
    curr_alarm = curr_ctx.get("observation", {}).get("alarm", False)
    # Same alarm state → same intervention expected
    if prev_alarm == curr_alarm:
        return 1.0 if prev_intervention == curr_intervention else 0.3
    # Different alarm state → different intervention is consistent
    return 1.0 if prev_intervention != curr_intervention else 0.6


def _memory_purity(retrieval_metrics: Dict[str, Any] | None) -> float:
    """Pureza de memoria: 1 - contamination."""
    if not retrieval_metrics:
        return 1.0
    cross = retrieval_metrics.get("retrieved_cross_scenario_count", 0)
    same = retrieval_metrics.get("retrieved_same_scenario_count", 0)
    total = same + cross
    if total == 0:
        return 1.0
    return 1.0 - (cross / total)


def _transition_type_from_class(compatibility_class: str) -> str:
    """Map compatibility class to transition type."""
    if compatibility_class == "equivalent":
        return "intra"
    if compatibility_class in ("compatible", "analogical", "incompatible"):
        return compatibility_class
    return "incompatible"


# ── Main functions ───────────────────────────────────────────────────────────

def compute_transition_vector(
    *,
    previous_result: Dict[str, Any],
    current_result: Dict[str, Any],
    compatibility: CompatibilityAssessment,
    retrieval_metrics: Dict[str, Any] | None = None,
) -> TransitionContinuityVector:
    """Computa vector de continuidad para una transición entre episodios.

    Args:
        previous_result: Resultado del episodio anterior (episode payload).
        current_result: Resultado del episodio actual (episode payload).
        compatibility: Evaluación de compatibilidad entre escenarios.
        retrieval_metrics: Métricas de retrieval de memoria del episodio actual.

    Returns:
        TransitionContinuityVector con todos los componentes.
    """
    prev_episode = previous_result.get("episode", previous_result)
    curr_episode = current_result.get("episode", current_result)

    # Semantic retention: Jaccard de proposiciones
    prev_signs = prev_episode.get("context", {}).get("observation", {}).get("propositions", [])
    curr_signs = curr_episode.get("context", {}).get("observation", {}).get("propositions", [])
    semantic = _jaccard(prev_signs, curr_signs)

    # Trace stability
    prev_seq = prev_episode.get("result", {}).get("reasoning_sequence", []) or []
    curr_seq = curr_episode.get("result", {}).get("reasoning_sequence", []) or []
    trace = _sequence_stability(list(prev_seq), list(curr_seq))

    # Causal stability (genérica, por main_variable)
    main_var = curr_episode.get("scenario_metadata", {}).get("main_variable")
    causal = _causal_stability_generic(curr_episode, main_variable=main_var)

    # Intervention policy stability
    intervention = _intervention_stability(prev_episode, curr_episode)

    # Structural compatibility (del grafo)
    structural = compatibility.overall_score

    # Memory purity
    purity = _memory_purity(retrieval_metrics)

    composite = (
        _W_SEMANTIC * semantic
        + _W_TRACE * trace
        + _W_CAUSAL * causal
        + _W_INTERVENTION * intervention
        + _W_STRUCTURAL * structural
        + _W_PURITY * purity
    )
    composite = max(0.0, min(1.0, composite))

    return TransitionContinuityVector(
        source_scenario=compatibility.source_scenario,
        target_scenario=compatibility.target_scenario,
        semantic_retention=round(semantic, 4),
        trace_stability=round(trace, 4),
        causal_stability=round(causal, 4),
        intervention_policy_stability=round(intervention, 4),
        structural_compatibility=round(structural, 4),
        memory_purity=round(purity, 4),
        composite_score=round(composite, 4),
        transition_type=_transition_type_from_class(compatibility.compatibility_class),
    )


def build_continuity_tensor(
    *,
    vectors: list[TransitionContinuityVector],
) -> Dict[str, Dict[str, ContinuityTensorCell]]:
    """Construye tensor NxN de continuidad a partir de vectores individuales.

    Agrega vectores por par (source, target) y computa promedios.

    Args:
        vectors: Lista de TransitionContinuityVector.

    Returns:
        Dict anidado [source][target] -> ContinuityTensorCell con métricas agregadas.
    """
    groups: Dict[tuple[str, str], list[TransitionContinuityVector]] = defaultdict(list)
    for v in vectors:
        groups[(v.source_scenario, v.target_scenario)].append(v)

    tensor: Dict[str, Dict[str, ContinuityTensorCell]] = {}
    for (src, tgt), vecs in groups.items():
        if src not in tensor:
            tensor[src] = {}
        n = len(vecs)
        tensor[src][tgt] = ContinuityTensorCell(
            source_scenario=src,
            target_scenario=tgt,
            mean_composite=round(sum(v.composite_score for v in vecs) / n, 4),
            mean_semantic_retention=round(sum(v.semantic_retention for v in vecs) / n, 4),
            mean_causal_stability=round(sum(v.causal_stability for v in vecs) / n, 4),
            mean_memory_purity=round(sum(v.memory_purity for v in vecs) / n, 4),
            sample_count=n,
        )
    return tensor
