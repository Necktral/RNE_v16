"""Modelo de régimen latente.

Un escenario observable es una realización concreta de un régimen latente.
La comparabilidad deja de ser nombre igual / nombre distinto y pasa a ser:
  - mismo régimen
  - régimen compatible
  - régimen transformable
  - régimen no transportable

Cada RegimeModel captura la estructura invariante que subyace a uno
o más escenarios concretos.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Literal, Tuple


# ── Data contracts ───────────────────────────────────────────────────────────

RegimeCompatibility = Literal[
    "same_regime",
    "compatible_regime",
    "transformable_regime",
    "non_transportable",
]


@dataclass(frozen=True)
class RegimeModel:
    """Modelo de régimen latente bajo uno o más escenarios.

    Attributes:
        regime_id: Identificador único del régimen.
        control_topology: Topología de control ('single_loop', 'cascade', 'distributed').
        optimization_geometry: Geometría de optimización ('minimize', 'maximize', 'target_band').
        intervention_algebra: Álgebra de intervenciones ('additive', 'multiplicative', 'switching').
        counterfactual_law: Ley contrafactual ('perturbation', 'withholding', 'substitution').
        causal_polarity: Polaridad causal principal.
        response_sensitivity: Sensibilidad de respuesta [0, 1].
        equilibrium_class: Clase de equilibrio ('stable', 'unstable', 'metastable', 'oscillatory').
        recovery_profile: Perfil de recuperación ('fast', 'moderate', 'slow', 'non_recovering').
        scenario_instances: Escenarios concretos que realizan este régimen.
    """

    regime_id: str
    control_topology: Literal["single_loop", "cascade", "distributed"] = "single_loop"
    optimization_geometry: Literal["minimize", "maximize", "target_band"] = "minimize"
    intervention_algebra: Literal["additive", "multiplicative", "switching"] = "additive"
    counterfactual_law: Literal["perturbation", "withholding", "substitution"] = "perturbation"
    causal_polarity: Literal["lower_is_better", "higher_is_better", "contextual"] = "lower_is_better"
    response_sensitivity: float = 0.5
    equilibrium_class: Literal["stable", "unstable", "metastable", "oscillatory"] = "stable"
    recovery_profile: Literal["fast", "moderate", "slow", "non_recovering"] = "moderate"
    scenario_instances: FrozenSet[str] = frozenset()
    metadata: Dict[str, Any] = field(default_factory=dict)


# ── Regime comparison ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RegimeComparisonResult:
    """Resultado de comparación entre dos regímenes."""

    source_regime: str
    target_regime: str
    compatibility: RegimeCompatibility
    structural_distance: float    # [0, 1]
    topology_match: bool
    geometry_match: bool
    algebra_match: bool
    law_match: bool
    polarity_match: bool
    equilibrium_match: bool
    transport_feasibility: float  # [0, 1]


def compare_regimes(
    source: RegimeModel,
    target: RegimeModel,
) -> RegimeComparisonResult:
    """Compara dos regímenes y determina compatibilidad.

    Args:
        source: Régimen fuente.
        target: Régimen destino.

    Returns:
        RegimeComparisonResult con compatibilidad y distancia.
    """
    topo = source.control_topology == target.control_topology
    geom = source.optimization_geometry == target.optimization_geometry
    alg = source.intervention_algebra == target.intervention_algebra
    law = source.counterfactual_law == target.counterfactual_law
    pol = source.causal_polarity == target.causal_polarity
    eq = source.equilibrium_class == target.equilibrium_class

    matches = [topo, geom, alg, law, pol, eq]
    match_count = sum(matches)
    total = len(matches)

    # Structural distance
    distance = 1.0 - (match_count / total)

    # Sensitivity distance
    sens_diff = abs(source.response_sensitivity - target.response_sensitivity)
    distance = 0.7 * distance + 0.3 * sens_diff

    # Recovery compatibility
    recovery_map = {"fast": 0, "moderate": 1, "slow": 2, "non_recovering": 3}
    recovery_diff = abs(
        recovery_map.get(source.recovery_profile, 1)
        - recovery_map.get(target.recovery_profile, 1)
    ) / 3.0

    # Transport feasibility
    feasibility = max(0.0, 1.0 - distance - 0.2 * recovery_diff)

    # Compatibility classification
    if source.regime_id == target.regime_id:
        compat = "same_regime"
    elif match_count >= 5:
        compat = "compatible_regime"
    elif match_count >= 3 and feasibility > 0.3:
        compat = "transformable_regime"
    else:
        compat = "non_transportable"

    return RegimeComparisonResult(
        source_regime=source.regime_id,
        target_regime=target.regime_id,
        compatibility=compat,
        structural_distance=round(distance, 4),
        topology_match=topo,
        geometry_match=geom,
        algebra_match=alg,
        law_match=law,
        polarity_match=pol,
        equilibrium_match=eq,
        transport_feasibility=round(feasibility, 4),
    )


# ── Built-in regime models ───────────────────────────────────────────────────

THERMAL_REGIME = RegimeModel(
    regime_id="homeostatic_cooling",
    control_topology="single_loop",
    optimization_geometry="minimize",
    intervention_algebra="additive",
    counterfactual_law="perturbation",
    causal_polarity="lower_is_better",
    response_sensitivity=0.6,
    equilibrium_class="stable",
    recovery_profile="fast",
    scenario_instances=frozenset({"thermal_homeostasis"}),
)

RESOURCE_REGIME = RegimeModel(
    regime_id="inventory_maximization",
    control_topology="single_loop",
    optimization_geometry="maximize",
    intervention_algebra="additive",
    counterfactual_law="perturbation",
    causal_polarity="higher_is_better",
    response_sensitivity=0.5,
    equilibrium_class="stable",
    recovery_profile="moderate",
    scenario_instances=frozenset({"resource_management"}),
)

# Registry
REGIME_REGISTRY: Dict[str, RegimeModel] = {
    "homeostatic_cooling": THERMAL_REGIME,
    "inventory_maximization": RESOURCE_REGIME,
}

SCENARIO_TO_REGIME: Dict[str, str] = {
    "thermal_homeostasis": "homeostatic_cooling",
    "resource_management": "inventory_maximization",
}


def get_regime_for_scenario(scenario_name: str) -> RegimeModel | None:
    """Obtiene el régimen latente para un escenario."""
    regime_id = SCENARIO_TO_REGIME.get(scenario_name)
    if regime_id is None:
        return None
    return REGIME_REGISTRY.get(regime_id)
