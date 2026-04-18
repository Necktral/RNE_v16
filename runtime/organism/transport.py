"""Operadores dirigidos de transporte entre regímenes.

Define transporte entre regímenes como operador, no como score:

    T_{i→j}: X_i → X_j

donde X_i, X_j son espacios de estado constitucional.

La transferencia es dirigida: A→B ≠ B→A.

Produce:
  - belief_projection
  - policy_projection
  - sign_inversions
  - scale_transform
  - residual_error
  - transport_uncertainty
  - expected_recovery_cost
  - transport_class
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Tuple

from .state import OrganismBeliefState, OrganismState, PolicyState
from .regime_model import RegimeModel, RegimeComparisonResult, compare_regimes
from .regime_renormalization import RegimeRenormalizationEngine
from .snapshot import OrganismSnapshot


TransportClass = Literal[
    "identity",          # Same regime, no transformation needed
    "isometric",         # Compatible regime, preserves structure
    "projective",        # Transformable regime, some information loss
    "adversarial",       # Inverted semantics, high risk
    "blocked",           # Non-transportable
]


# ── Data contracts ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class BeliefProjection:
    """Proyección de belief state del régimen fuente al destino.

    Attributes:
        projected_alarm: Probabilidad de alarma proyectada.
        projected_efficacy: Eficacia de intervención proyectada.
        projected_causal_support: Soporte causal proyectado.
        sign_inversions: Número de inversiones de signo aplicadas.
        scale_factor: Factor de escala aplicado.
        projection_loss: Pérdida de información por la proyección.
    """

    projected_alarm: float
    projected_efficacy: float
    projected_causal_support: float
    sign_inversions: int
    scale_factor: float
    projection_loss: float


@dataclass(frozen=True)
class PolicyProjection:
    """Proyección de política del régimen fuente al destino."""

    projected_control_class: str
    projected_sensitivity: float
    sensitivity_adjustment: float
    projected_tolerance: float
    compatibility_score: float


@dataclass(frozen=True)
class TransportResult:
    """Resultado completo de un operador de transporte dirigido.

    Attributes:
        source_regime: Régimen fuente.
        target_regime: Régimen destino.
        belief_projection: Proyección de creencias.
        policy_projection: Proyección de política.
        residual_error: Error residual del transporte [0, 1].
        transport_uncertainty: Incertidumbre del transporte [0, 1].
        expected_recovery_cost: Costo de recuperación esperado [0, 1].
        transport_class: Clase de transporte.
        regime_comparison: Comparación de regímenes.
    """

    source_regime: str
    target_regime: str
    belief_projection: BeliefProjection
    policy_projection: PolicyProjection
    residual_error: float
    transport_uncertainty: float
    expected_recovery_cost: float
    transport_class: TransportClass
    regime_comparison: RegimeComparisonResult


# ── Transport operator ───────────────────────────────────────────────────────

class TransportOperatorEngine:
    """Motor de operadores de transporte dirigido entre regímenes.

    Computa transformaciones formales T_{i→j} que proyectan belief state
    y policy state del régimen fuente al destino.
    """

    def __init__(
        self,
        *,
        sensitivity_scale_factor: float = 1.0,
        uncertainty_base: float = 0.10,
    ):
        self.sensitivity_scale_factor = sensitivity_scale_factor
        self.uncertainty_base = uncertainty_base
        self.renorm_engine = RegimeRenormalizationEngine()

    def transport(
        self,
        *,
        source_regime: RegimeModel,
        target_regime: RegimeModel,
        state: OrganismState,
    ) -> TransportResult:
        """Computa transporte dirigido source → target.

        Args:
            source_regime: Régimen de origen.
            target_regime: Régimen de destino.
            state: Estado actual del organismo (en source_regime).

        Returns:
            TransportResult con proyecciones, error y clase.
        """
        comparison = compare_regimes(source_regime, target_regime)
        snapshot = OrganismSnapshot.from_state(state)
        renorm = self.renorm_engine.renormalize(
            source_regime=source_regime,
            target_regime=target_regime,
            snapshot=snapshot,
            constraints={
                "triadic_closure_threshold": 0.50,
                "min_memory_purity": 0.40,
                "min_trace_integrity": 0.30,
                "max_policy_drift": 0.50,
            },
        )

        transport_class = self._classify_transport(comparison)
        if transport_class != "blocked":
            if source_regime.regime_id == target_regime.regime_id:
                transport_class = "identity"
            elif comparison.compatibility == "compatible_regime" and renorm.regime_residual.residual_error < 0.25:
                transport_class = "isometric"
            elif not comparison.polarity_match:
                transport_class = "adversarial"
            else:
                transport_class = "projective"

        if transport_class == "blocked":
            return self._blocked_result(source_regime, target_regime, comparison)

        sign_inversions = 1 if source_regime.causal_polarity != target_regime.causal_polarity else 0
        belief_proj = BeliefProjection(
            projected_alarm=renorm.belief_projection.projected_alarm,
            projected_efficacy=renorm.belief_projection.projected_efficacy,
            projected_causal_support=renorm.belief_projection.projected_causal_support,
            sign_inversions=sign_inversions,
            scale_factor=renorm.renormalization_map.asymmetry_factor,
            projection_loss=min(1.0, renorm.regime_residual.residual_error + 0.20 * renorm.constraint_transform.semantic_shift_cost),
        )
        policy_proj = PolicyProjection(
            projected_control_class=renorm.policy_phase_transform.projected_control_class,
            projected_sensitivity=renorm.policy_phase_transform.projected_sensitivity,
            sensitivity_adjustment=renorm.renormalization_map.asymmetry_factor,
            projected_tolerance=renorm.policy_phase_transform.projected_tolerance,
            compatibility_score=renorm.renormalization_map.feasibility,
        )
        residual = renorm.regime_residual.residual_error
        uncertainty = renorm.uncertainty.transport_uncertainty
        recovery_cost = renorm.regime_residual.expected_recovery_cost

        return TransportResult(
            source_regime=source_regime.regime_id,
            target_regime=target_regime.regime_id,
            belief_projection=belief_proj,
            policy_projection=policy_proj,
            residual_error=round(residual, 4),
            transport_uncertainty=round(uncertainty, 4),
            expected_recovery_cost=round(recovery_cost, 4),
            transport_class=transport_class,
            regime_comparison=comparison,
        )

    def _classify_transport(self, comp: RegimeComparisonResult) -> TransportClass:
        """Clasifica el tipo de transporte."""
        if comp.compatibility == "same_regime":
            return "identity"
        if comp.compatibility == "compatible_regime":
            return "isometric"
        if comp.compatibility == "transformable_regime":
            if not comp.polarity_match:
                return "adversarial"
            return "projective"
        return "blocked"

    def _project_belief(
        self,
        belief: OrganismBeliefState,
        source: RegimeModel,
        target: RegimeModel,
        comp: RegimeComparisonResult,
    ) -> BeliefProjection:
        """Proyecta belief state."""
        sign_inv = 0
        scale = 1.0

        # Polarity inversion
        if not comp.polarity_match:
            sign_inv += 1
            # Invert alarm probability interpretation
            projected_alarm = 1.0 - belief.alarm_probability
        else:
            projected_alarm = belief.alarm_probability

        # Sensitivity scaling
        if source.response_sensitivity > 0 and target.response_sensitivity > 0:
            scale = target.response_sensitivity / source.response_sensitivity
            scale = max(0.5, min(2.0, scale))  # Clamp
        else:
            scale = 1.0

        # Project efficacy with regime compatibility
        compatibility_factor = comp.transport_feasibility
        projected_efficacy = belief.intervention_efficacy * compatibility_factor
        projected_causal = belief.causal_support_confidence * compatibility_factor

        # Projection loss
        loss = comp.structural_distance * 0.5 + (1.0 - compatibility_factor) * 0.3 + (sign_inv * 0.2)
        loss = min(1.0, loss)

        return BeliefProjection(
            projected_alarm=round(max(0.0, min(1.0, projected_alarm)), 4),
            projected_efficacy=round(max(0.0, min(1.0, projected_efficacy)), 4),
            projected_causal_support=round(max(0.0, min(1.0, projected_causal)), 4),
            sign_inversions=sign_inv,
            scale_factor=round(scale, 4),
            projection_loss=round(loss, 4),
        )

    def _project_policy(
        self,
        policy: PolicyState,
        source: RegimeModel,
        target: RegimeModel,
        comp: RegimeComparisonResult,
    ) -> PolicyProjection:
        """Proyecta política."""
        # Control class: keep if topology matches, otherwise default
        if comp.topology_match:
            projected_class = policy.control_class
        else:
            projected_class = "reactive"  # Safe default

        # Sensitivity adjustment
        if source.response_sensitivity > 0:
            sens_ratio = target.response_sensitivity / source.response_sensitivity
        else:
            sens_ratio = 1.0
        sensitivity_adj = sens_ratio * self.sensitivity_scale_factor
        projected_sens = max(0.0, min(1.0, policy.sensitivity * sensitivity_adj))

        # Tolerance
        projected_tol = policy.perturbation_tolerance
        if not comp.equilibrium_match:
            projected_tol *= 0.8  # More conservative if equilibrium differs

        # Compatibility score
        compat = comp.transport_feasibility

        return PolicyProjection(
            projected_control_class=projected_class,
            projected_sensitivity=round(projected_sens, 4),
            sensitivity_adjustment=round(sensitivity_adj, 4),
            projected_tolerance=round(projected_tol, 4),
            compatibility_score=round(compat, 4),
        )

    def _compute_residual(
        self,
        comp: RegimeComparisonResult,
        belief: BeliefProjection,
        policy: PolicyProjection,
    ) -> float:
        """Error residual del transporte."""
        return min(1.0, (
            0.40 * belief.projection_loss
            + 0.30 * comp.structural_distance
            + 0.30 * (1.0 - policy.compatibility_score)
        ))

    def _compute_uncertainty(
        self,
        comp: RegimeComparisonResult,
        transport_class: TransportClass,
    ) -> float:
        """Incertidumbre del transporte."""
        base = self.uncertainty_base
        class_penalty = {
            "identity": 0.0,
            "isometric": 0.05,
            "projective": 0.15,
            "adversarial": 0.30,
            "blocked": 1.0,
        }.get(transport_class, 0.5)
        return min(1.0, base + class_penalty + comp.structural_distance * 0.2)

    def _compute_recovery_cost(
        self,
        comp: RegimeComparisonResult,
        transport_class: TransportClass,
    ) -> float:
        """Costo esperado de recuperación."""
        class_cost = {
            "identity": 0.0,
            "isometric": 0.05,
            "projective": 0.20,
            "adversarial": 0.50,
            "blocked": 1.0,
        }.get(transport_class, 0.5)
        return min(1.0, class_cost + comp.structural_distance * 0.3)

    def _blocked_result(
        self,
        source: RegimeModel,
        target: RegimeModel,
        comp: RegimeComparisonResult,
    ) -> TransportResult:
        """Resultado para transporte bloqueado."""
        return TransportResult(
            source_regime=source.regime_id,
            target_regime=target.regime_id,
            belief_projection=BeliefProjection(
                projected_alarm=0.5,
                projected_efficacy=0.0,
                projected_causal_support=0.0,
                sign_inversions=0,
                scale_factor=0.0,
                projection_loss=1.0,
            ),
            policy_projection=PolicyProjection(
                projected_control_class="reactive",
                projected_sensitivity=0.5,
                sensitivity_adjustment=0.0,
                projected_tolerance=0.0,
                compatibility_score=0.0,
            ),
            residual_error=1.0,
            transport_uncertainty=1.0,
            expected_recovery_cost=1.0,
            transport_class="blocked",
            regime_comparison=comp,
        )
