"""Grafo de compatibilidad estructural entre escenarios cognitivos.

Define perfiles estructurales para cada escenario y permite evaluar
la compatibilidad entre pares de escenarios según topología de control,
dirección de optimización, semántica de intervención y política contrafactual.

Clasificación:
- equivalent: mismo escenario/versión/hash con score >= 0.95
- compatible: score >= 0.75
- analogical: score >= 0.45
- incompatible: score < 0.45
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

CompatibilityClass = Literal["equivalent", "compatible", "analogical", "incompatible"]
OptimizationDirection = Literal["minimize", "maximize", "target_band"]
ControlTopology = Literal[
    "threshold_single_loop",
    "threshold_recovery_loop",
    "dual_threshold",
    "unknown",
]


@dataclass(frozen=True)
class ScenarioStructuralProfile:
    """Perfil estructural inmutable de un escenario cognitivo.

    Captura la identidad semántica del escenario: cómo controla,
    hacia dónde optimiza, qué intervenciones aplica y cómo evalúa
    su relación factual/contrafactual.
    """

    scenario_name: str
    scenario_version: str
    scenario_config_hash: str
    control_topology: ControlTopology
    optimization_direction: OptimizationDirection
    intervention_semantics: tuple[str, ...]
    counterfactual_policy: str
    relation_polarity: Literal["lower_is_better", "higher_is_better", "contextual"]
    main_variable: str


@dataclass(frozen=True)
class CompatibilityAssessment:
    """Resultado de evaluación de compatibilidad entre dos escenarios.

    Contiene scores por dimensión, clasificación final, penalización
    y flags de transferencia/certificación.
    """

    source_scenario: str
    target_scenario: str
    compatibility_class: CompatibilityClass
    topology_score: float
    objective_score: float
    intervention_score: float
    counterfactual_score: float
    overall_score: float
    penalty_multiplier: float
    transfer_allowed: bool
    certification_allowed: bool


# ── Penalty map ──────────────────────────────────────────────────────────────

_PENALTY_MAP: dict[CompatibilityClass, float] = {
    "equivalent": 1.0,
    "compatible": 0.85,
    "analogical": 0.50,
    "incompatible": 0.0,
}

# ── Scoring weights ──────────────────────────────────────────────────────────

_W_TOPOLOGY = 0.35
_W_OBJECTIVE = 0.25
_W_INTERVENTION = 0.20
_W_COUNTERFACTUAL = 0.20


def _score_topology(source: ScenarioStructuralProfile, target: ScenarioStructuralProfile) -> float:
    """Score de compatibilidad topológica."""
    if source.control_topology == target.control_topology:
        return 1.0
    # Topologías de umbral comparten estructura
    threshold_set = {"threshold_single_loop", "threshold_recovery_loop", "dual_threshold"}
    if source.control_topology in threshold_set and target.control_topology in threshold_set:
        return 0.70
    if source.control_topology == "unknown" or target.control_topology == "unknown":
        return 0.30
    return 0.20


def _score_objective(source: ScenarioStructuralProfile, target: ScenarioStructuralProfile) -> float:
    """Score de compatibilidad de objetivo de optimización."""
    if source.optimization_direction == target.optimization_direction:
        return 1.0
    # minimize y maximize son inversas pero comparten estructura
    opt_pair = {source.optimization_direction, target.optimization_direction}
    if opt_pair == {"minimize", "maximize"}:
        return 0.60
    if "target_band" in opt_pair:
        return 0.40
    return 0.30


def _score_intervention(
    source: ScenarioStructuralProfile, target: ScenarioStructuralProfile,
) -> float:
    """Score de compatibilidad de semántica de intervenciones."""
    src_set = set(source.intervention_semantics)
    tgt_set = set(target.intervention_semantics)
    if not src_set and not tgt_set:
        return 1.0
    if not src_set or not tgt_set:
        return 0.0
    # Jaccard
    intersection = len(src_set & tgt_set)
    union = len(src_set | tgt_set)
    jaccard = intersection / union if union else 0.0
    # Bonus: misma cantidad de intervenciones
    size_match = 1.0 if len(src_set) == len(tgt_set) else 0.80
    return 0.60 * jaccard + 0.40 * size_match


def _score_counterfactual(
    source: ScenarioStructuralProfile, target: ScenarioStructuralProfile,
) -> float:
    """Score de compatibilidad de política contrafactual."""
    if source.counterfactual_policy == target.counterfactual_policy:
        return 1.0
    # Polaridades iguales sugieren evaluación comparable
    if source.relation_polarity == target.relation_polarity:
        return 0.80
    # Polaridades inversas pero bien definidas
    if source.relation_polarity != "contextual" and target.relation_polarity != "contextual":
        return 0.55
    return 0.30


def _classify(overall_score: float, source: ScenarioStructuralProfile,
              target: ScenarioStructuralProfile) -> CompatibilityClass:
    """Clasifica la relación según score y identidad."""
    if (
        overall_score >= 0.95
        and source.scenario_name == target.scenario_name
        and source.scenario_version == target.scenario_version
        and source.scenario_config_hash == target.scenario_config_hash
    ):
        return "equivalent"
    if overall_score >= 0.75:
        return "compatible"
    if overall_score >= 0.45:
        return "analogical"
    return "incompatible"


class ScenarioCompatibilityGraph:
    """Grafo de compatibilidad entre escenarios cognitivos.

    Evalúa pares de escenarios y construye matrices NxN de compatibilidad.
    """

    def assess(
        self,
        source: ScenarioStructuralProfile,
        target: ScenarioStructuralProfile,
    ) -> CompatibilityAssessment:
        """Evalúa compatibilidad entre dos perfiles estructurales.

        Args:
            source: Perfil del escenario origen.
            target: Perfil del escenario destino.

        Returns:
            CompatibilityAssessment con scores, clasificación y flags.
        """
        topology = _score_topology(source, target)
        objective = _score_objective(source, target)
        intervention = _score_intervention(source, target)
        counterfactual = _score_counterfactual(source, target)

        overall = (
            _W_TOPOLOGY * topology
            + _W_OBJECTIVE * objective
            + _W_INTERVENTION * intervention
            + _W_COUNTERFACTUAL * counterfactual
        )
        overall = max(0.0, min(1.0, overall))

        compat_class = _classify(overall, source, target)
        penalty = _PENALTY_MAP[compat_class]

        return CompatibilityAssessment(
            source_scenario=source.scenario_name,
            target_scenario=target.scenario_name,
            compatibility_class=compat_class,
            topology_score=round(topology, 4),
            objective_score=round(objective, 4),
            intervention_score=round(intervention, 4),
            counterfactual_score=round(counterfactual, 4),
            overall_score=round(overall, 4),
            penalty_multiplier=penalty,
            transfer_allowed=compat_class != "incompatible",
            certification_allowed=compat_class in ("equivalent", "compatible"),
        )

    def matrix(
        self,
        profiles: Sequence[ScenarioStructuralProfile],
    ) -> dict[str, dict[str, CompatibilityAssessment]]:
        """Construye matriz NxN de compatibilidad entre escenarios.

        Args:
            profiles: Lista de perfiles a comparar.

        Returns:
            Dict anidado [source_name][target_name] -> CompatibilityAssessment.
        """
        result: dict[str, dict[str, CompatibilityAssessment]] = {}
        for src in profiles:
            result[src.scenario_name] = {}
            for tgt in profiles:
                result[src.scenario_name][tgt.scenario_name] = self.assess(src, tgt)
        return result
