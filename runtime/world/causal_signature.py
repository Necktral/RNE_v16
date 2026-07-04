"""Firma causal rica para cada escenario cognitivo.

Cada escenario expone una ScenarioCausalSignature que captura:
- Variables observables y de control
- Dirección de optimización y polaridad causal
- Semántica de intervención con efectos esperados
- Política contrafactual
- Grafo causal mínimo (DAG dirigido entre variables)

Esta firma es la base para computar morfismos causales dirigidos
y supera el ScenarioStructuralProfile plano del RTCME-v1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Literal, Sequence, Tuple


OptimizationDirection = Literal["minimize", "maximize", "target_band"]
CausalPolarity = Literal["lower_is_better", "higher_is_better", "contextual"]


@dataclass(frozen=True)
class CausalEdge:
    """Arista dirigida en el grafo causal mínimo del escenario.

    Attributes:
        source: Variable fuente.
        target: Variable destino.
        polarity: '+' si un aumento en source causa aumento en target,
                  '-' si causa decremento, '?' si es ambiguo.
        strength: Confianza estimada de la relación [0, 1].
    """

    source: str
    target: str
    polarity: Literal["+", "-", "?"]
    strength: float = 1.0


@dataclass(frozen=True)
class InterventionEffect:
    """Efecto esperado de una intervención sobre la variable principal.

    Attributes:
        intervention_name: Nombre de la intervención.
        target_variable: Variable afectada directamente.
        expected_direction: '+' para incremento, '-' para decremento.
        expected_magnitude: Magnitud estimada del efecto [0, 1].
        semantic_role: Rol semántico ('corrective', 'restorative',
                       'preventive', 'neutral').
    """

    intervention_name: str
    target_variable: str
    expected_direction: Literal["+", "-"]
    expected_magnitude: float
    semantic_role: Literal["corrective", "restorative", "preventive", "neutral"]


@dataclass(frozen=True)
class ScenarioCausalSignature:
    """Firma causal completa de un escenario cognitivo.

    Captura la estructura causal interna: variables, relaciones dirigidas,
    efectos de intervención y política de evaluación.
    """

    scenario_name: str
    scenario_version: str
    # Variables
    observable_variables: FrozenSet[str]
    control_variables: FrozenSet[str]
    main_variable: str
    # Optimization
    optimization_direction: OptimizationDirection
    causal_polarity: CausalPolarity
    alarm_semantics: Literal["threshold_above", "threshold_below"]
    # Interventions
    intervention_effects: Tuple[InterventionEffect, ...]
    # Counterfactual policy
    counterfactual_policy: str
    counterfactual_variable: str
    # Causal graph
    causal_edges: Tuple[CausalEdge, ...]
    # Propositions
    proposition_vocabulary: FrozenSet[str]
    # Extra metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def intervention_names(self) -> FrozenSet[str]:
        return frozenset(e.intervention_name for e in self.intervention_effects)

    @property
    def corrective_interventions(self) -> Tuple[InterventionEffect, ...]:
        return tuple(e for e in self.intervention_effects if e.semantic_role == "corrective")

    @property
    def causal_graph_dict(self) -> Dict[str, list[Dict[str, Any]]]:
        """Returns causal edges grouped by source variable."""
        graph: Dict[str, list[Dict[str, Any]]] = {}
        for edge in self.causal_edges:
            if edge.source not in graph:
                graph[edge.source] = []
            graph[edge.source].append({
                "target": edge.target,
                "polarity": edge.polarity,
                "strength": edge.strength,
            })
        return graph
