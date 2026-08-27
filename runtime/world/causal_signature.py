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


#: FORMA del objetivo. `minimize`/`maximize` son objetivos MONÓTONOS (más frío siempre
#: es mejor, sin punto de satisfacción). `target_band` es un objetivo REGULATORIO: hay una
#: región segura, y estar dentro de ella BASTA.
#:
#: Los cuatro escenarios del organismo son reguladores de umbral: declaran `alarm_semantics`
#: + `alarm_threshold` y su política sólo actúa bajo alarma. Declararlos `minimize` /
#: `maximize` era declarar un objetivo que el organismo NO persigue — y esa firma es la que
#: `causal_attestation` exporta al certificado, o sea lo que el organismo ATESTIGUA A SU
#: CORTE. Ahora declaran `target_band`, que es lo que efectivamente hacen.
OptimizationDirection = Literal["minimize", "maximize", "target_band"]

#: SENTIDO de la mejora: hacia qué lado del eje queda la región segura. Es ORTOGONAL a la
#: forma del objetivo: un regulador de banda igual tiene un lado bueno y uno malo.
CausalPolarity = Literal["lower_is_better", "higher_is_better", "contextual"]


def improvement_direction(signature: Any) -> str:
    """Sentido de mejora de una firma causal: ``'minimize'`` o ``'maximize'``.

    Se deriva de ``causal_polarity`` (hacia dónde queda la región segura), NO de
    ``optimization_direction`` (que declara la forma del objetivo: monótono o banda).
    Tolerante a firmas ausentes/parciales: nunca lanza.

    - ``higher_is_better``            ⇒ ``'maximize'``
    - ``lower_is_better``             ⇒ ``'minimize'``
    - ``contextual`` / sin polaridad  ⇒ se cae al ``optimization_direction`` monótono
      declarado si lo hay; en última instancia ``'minimize'`` (el default histórico).
    """
    polarity = getattr(signature, "causal_polarity", None)
    if polarity == "higher_is_better":
        return "maximize"
    if polarity == "lower_is_better":
        return "minimize"
    declared = getattr(signature, "optimization_direction", None)
    if declared in ("minimize", "maximize"):
        return str(declared)
    return "minimize"


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

    @property
    def improvement_direction(self) -> str:
        """Sentido de MEJORA (``'minimize'`` / ``'maximize'``), derivado de la polaridad.

        SSOT para todo consumidor que necesite saber "¿hacia dónde es mejor?" —
        el effect-model lineal, los rankings de intervención, `supports_choice`.

        `optimization_direction` declara la FORMA del objetivo (monótono vs banda);
        `causal_polarity` declara HACIA DÓNDE queda lo bueno. Un regulador de banda
        (`target_band`) no minimiza sin fin, pero igual tiene un lado seguro: en térmico
        es abajo, en recursos es arriba. Los consumidores monótonos preguntan lo SEGUNDO,
        y por eso leer `optimization_direction` para responderlo era mezclar dos cosas
        distintas — el mismo error que hacía que `evaluate_relation_kind` juzgara los actos
        del organismo contra un objetivo que nadie perseguía.
        """
        return improvement_direction(self)
