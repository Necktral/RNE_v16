"""Motor de morfismos causales dirigidos entre escenarios.

A diferencia del ScenarioCompatibilityGraph simétrico del RTCME-v1,
este motor computa transformaciones **dirigidas** (source → target)
que capturan:

- Alineamiento semántico de proposiciones
- Alineamiento de intervenciones con efectos esperados
- Alineamiento de efectos causales (polarity + magnitude)
- Consistencia contrafactual
- Penalización por dirección de optimización invertida
- Operador de transporte formal

La transferencia thermal→resource puede diferir de resource→thermal.

Fórmula de costo dirigido:
  C = α·d_semantic + β·d_effect + γ·d_control + δ·d_counterfactual
  con penalización de direccionalidad si optimization direction invierte
  la interpretación de mejora.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Sequence, Tuple

from .causal_signature import ScenarioCausalSignature, CausalEdge
from .alignment import (
    AlignmentResult,
    align_causal_graphs,
    align_interventions,
    align_propositions,
)

# ── Type aliases ─────────────────────────────────────────────────────────────

MorphismClass = Literal[
    "isomorphic",       # Structurally identical causal systems
    "homomorphic",      # Compatible causal structure with minor differences
    "analogical",       # Partially aligned, useful for transfer hints
    "adversarial",      # Inverted or conflicting causal semantics
    "incompatible",     # No useful alignment
]

# ── Weights ──────────────────────────────────────────────────────────────────

_ALPHA = 0.25   # semantic alignment
_BETA = 0.30    # effect alignment (intervention effects)
_GAMMA = 0.25   # control alignment (causal graph structure)
_DELTA = 0.20   # counterfactual consistency

# ── Data contracts ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TransportOperator:
    """Operador formal de transporte entre dos espacios causales.

    Describe cómo traducir conceptos del escenario source al target.
    """

    proposition_map: Tuple[Tuple[str, str], ...]
    intervention_map: Tuple[Tuple[str, str, float], ...]  # (src, tgt, alignment_cost)
    polarity_inversion: bool
    direction_inversion: bool
    estimated_information_loss: float


@dataclass(frozen=True)
class DirectedScenarioMorphism:
    """Resultado del morfismo causal dirigido source → target.

    Attributes:
        source_scenario: Nombre del escenario fuente.
        target_scenario: Nombre del escenario destino.
        semantic_alignment_score: Alineamiento de vocabularios de proposiciones.
        control_alignment_score: Alineamiento de grafos causales.
        effect_alignment_score: Alineamiento de efectos de intervención.
        counterfactual_alignment_score: Consistencia de política contrafactual.
        directionality_penalty: Penalización si la dirección de optimización
            invierte la interpretación de mejora entre escenarios.
        overall_score: Score compuesto tras aplicar penalización.
        morphism_class: Clasificación del morfismo.
        is_transfer_safe_prior: Prior de seguridad de transferencia basado
            en la estructura causal, antes de observar evidencia empírica.
        transport_operator: Operador de transporte formal.
        details: Detalles adicionales del cómputo.
    """

    source_scenario: str
    target_scenario: str
    semantic_alignment_score: float
    control_alignment_score: float
    effect_alignment_score: float
    counterfactual_alignment_score: float
    directionality_penalty: float
    overall_score: float
    morphism_class: MorphismClass
    is_transfer_safe_prior: bool
    transport_operator: TransportOperator
    details: Dict[str, Any] = field(default_factory=dict)


# ── Classification ───────────────────────────────────────────────────────────

def _classify_morphism(
    overall_score: float,
    source: ScenarioCausalSignature,
    target: ScenarioCausalSignature,
    directionality_penalty: float,
) -> MorphismClass:
    """Clasifica el morfismo según score y propiedades estructurales."""
    # Isomorphic: same scenario, same structure
    if (
        overall_score >= 0.95
        and source.scenario_name == target.scenario_name
        and source.scenario_version == target.scenario_version
    ):
        return "isomorphic"

    # Adversarial: significant directionality inversion
    if directionality_penalty >= 0.25 and overall_score < 0.50:
        return "adversarial"

    # Standard classification
    if overall_score >= 0.75:
        return "homomorphic"
    if overall_score >= 0.40:
        return "analogical"
    if overall_score < 0.25:
        return "incompatible"

    # Default for edge cases
    return "analogical"


# ── Counterfactual consistency ───────────────────────────────────────────────

def _counterfactual_consistency(
    source: ScenarioCausalSignature,
    target: ScenarioCausalSignature,
) -> float:
    """Score de consistencia contrafactual [0, 1].

    Evalúa si la forma de razonar contrafactualmente es compatible.
    """
    score = 0.0

    # Same counterfactual policy
    if source.counterfactual_policy == target.counterfactual_policy:
        score += 0.50
    else:
        score += 0.15

    # Same polarity
    if source.causal_polarity == target.causal_polarity:
        score += 0.30
    elif source.causal_polarity != "contextual" and target.causal_polarity != "contextual":
        # Both well-defined but different → partial credit for clarity
        score += 0.10

    # Same alarm semantics (threshold direction)
    if source.alarm_semantics == target.alarm_semantics:
        score += 0.20
    else:
        score += 0.05

    return min(1.0, score)


# ── Directionality penalty ───────────────────────────────────────────────────

def _directionality_penalty(
    source: ScenarioCausalSignature,
    target: ScenarioCausalSignature,
) -> float:
    """Penalización por inversión de dirección de optimización.

    Si source minimiza y target maximiza (o viceversa), la transferencia
    invierte la interpretación de "mejora", lo cual es peligroso.
    """
    if source.optimization_direction == target.optimization_direction:
        return 0.0

    # minimize ↔ maximize is the most dangerous inversion
    pair = {source.optimization_direction, target.optimization_direction}
    if pair == {"minimize", "maximize"}:
        # Further penalize if polarity also inverts
        if source.causal_polarity != target.causal_polarity:
            return 0.30
        return 0.20

    # target_band vs others
    if "target_band" in pair:
        return 0.10

    return 0.15


# ── Transport operator builder ───────────────────────────────────────────────

def _build_transport_operator(
    source: ScenarioCausalSignature,
    target: ScenarioCausalSignature,
    prop_alignment: AlignmentResult,
    int_alignment: AlignmentResult,
    penalty: float,
) -> TransportOperator:
    """Construye operador de transporte formal."""
    prop_map = tuple(
        (p.source_item, p.target_item)
        for p in prop_alignment.pairs
    )
    int_map = tuple(
        (p.source_item, p.target_item, p.cost)
        for p in int_alignment.pairs
    )

    polarity_inv = source.causal_polarity != target.causal_polarity
    direction_inv = source.optimization_direction != target.optimization_direction

    # Information loss estimate
    unmatched_props = len(prop_alignment.source_unmatched) + len(prop_alignment.target_unmatched)
    unmatched_ints = len(int_alignment.source_unmatched) + len(int_alignment.target_unmatched)
    total_elements = max(
        len(source.proposition_vocabulary) + len(target.proposition_vocabulary),
        1,
    )
    info_loss = min(1.0, (unmatched_props + unmatched_ints + penalty) / max(total_elements, 1))

    return TransportOperator(
        proposition_map=prop_map,
        intervention_map=int_map,
        polarity_inversion=polarity_inv,
        direction_inversion=direction_inv,
        estimated_information_loss=round(info_loss, 4),
    )


# ── Main engine ──────────────────────────────────────────────────────────────

class MorphismEngine:
    """Motor de morfismos causales dirigidos entre escenarios.

    Computa transformaciones dirigidas source → target basadas en
    alineamiento semántico, de efectos, causal y contrafactual.
    """

    def __init__(
        self,
        *,
        alpha: float = _ALPHA,
        beta: float = _BETA,
        gamma: float = _GAMMA,
        delta: float = _DELTA,
    ):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

    def compute_morphism(
        self,
        source: ScenarioCausalSignature,
        target: ScenarioCausalSignature,
    ) -> DirectedScenarioMorphism:
        """Computa morfismo dirigido source → target.

        Args:
            source: Firma causal del escenario fuente.
            target: Firma causal del escenario destino.

        Returns:
            DirectedScenarioMorphism con scores, clasificación y operador.
        """
        # 1. Semantic alignment (propositions)
        prop_align = align_propositions(
            source.proposition_vocabulary,
            target.proposition_vocabulary,
        )
        semantic_score = prop_align.normalized_score

        # 2. Effect alignment (interventions)
        int_align = align_interventions(
            source.intervention_effects,
            target.intervention_effects,
        )
        effect_score = int_align.normalized_score

        # 3. Control alignment (causal graphs)
        control_score = align_causal_graphs(
            source.causal_edges,
            target.causal_edges,
        )

        # 4. Counterfactual consistency
        cf_score = _counterfactual_consistency(source, target)

        # 5. Directionality penalty
        penalty = _directionality_penalty(source, target)

        # 6. Composite (before penalty)
        raw_score = (
            self.alpha * semantic_score
            + self.beta * effect_score
            + self.gamma * control_score
            + self.delta * cf_score
        )

        # 7. Apply penalty
        overall = max(0.0, min(1.0, raw_score * (1.0 - penalty)))

        # 8. Classify
        morphism_class = _classify_morphism(overall, source, target, penalty)

        # 9. Transfer safe prior
        is_safe_prior = morphism_class in ("isomorphic", "homomorphic")

        # 10. Transport operator
        transport = _build_transport_operator(
            source, target, prop_align, int_align, penalty,
        )

        return DirectedScenarioMorphism(
            source_scenario=source.scenario_name,
            target_scenario=target.scenario_name,
            semantic_alignment_score=round(semantic_score, 4),
            control_alignment_score=round(control_score, 4),
            effect_alignment_score=round(effect_score, 4),
            counterfactual_alignment_score=round(cf_score, 4),
            directionality_penalty=round(penalty, 4),
            overall_score=round(overall, 4),
            morphism_class=morphism_class,
            is_transfer_safe_prior=is_safe_prior,
            transport_operator=transport,
            details={
                "proposition_alignment": {
                    "score": prop_align.normalized_score,
                    "coverage": prop_align.coverage,
                    "source_unmatched": sorted(prop_align.source_unmatched),
                    "target_unmatched": sorted(prop_align.target_unmatched),
                },
                "intervention_alignment": {
                    "score": int_align.normalized_score,
                    "coverage": int_align.coverage,
                    "pairs": [
                        {"src": p.source_item, "tgt": p.target_item, "cost": p.cost}
                        for p in int_align.pairs
                    ],
                },
                "causal_graph_alignment": control_score,
                "weights": {
                    "alpha": self.alpha,
                    "beta": self.beta,
                    "gamma": self.gamma,
                    "delta": self.delta,
                },
            },
        )

    def compute_morphism_matrix(
        self,
        signatures: Sequence[ScenarioCausalSignature],
    ) -> Dict[str, Dict[str, DirectedScenarioMorphism]]:
        """Computa matriz NxN de morfismos dirigidos.

        La matriz es potencialmente asimétrica: M[A][B] ≠ M[B][A].

        Args:
            signatures: Lista de firmas causales.

        Returns:
            Dict anidado [source][target] → DirectedScenarioMorphism.
        """
        result: Dict[str, Dict[str, DirectedScenarioMorphism]] = {}
        for src in signatures:
            result[src.scenario_name] = {}
            for tgt in signatures:
                result[src.scenario_name][tgt.scenario_name] = self.compute_morphism(src, tgt)
        return result
