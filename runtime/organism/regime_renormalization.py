"""Motor de renormalizacion de regimen (RNFE-T4)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal

from .regime_model import RegimeModel, compare_regimes
from .snapshot import OrganismSnapshot


@dataclass(frozen=True)
class RenormalizationMap:
    source_regime: str
    target_regime: str
    asymmetry_factor: float
    structural_distance: float
    feasibility: float


@dataclass(frozen=True)
class ConstraintTransform:
    transformed_constraints: Dict[str, float]
    semantic_shift_cost: float


@dataclass(frozen=True)
class BeliefProjectionField:
    projected_alarm: float
    projected_efficacy: float
    projected_causal_support: float


@dataclass(frozen=True)
class PolicyPhaseTransform:
    projected_control_class: str
    projected_sensitivity: float
    projected_tolerance: float


@dataclass(frozen=True)
class RegimeResidual:
    residual_error: float
    expected_recovery_cost: float


@dataclass(frozen=True)
class RenormalizationUncertainty:
    transport_uncertainty: float


@dataclass(frozen=True)
class RenormalizationResult:
    renormalization_map: RenormalizationMap
    constraint_transform: ConstraintTransform
    belief_projection: BeliefProjectionField
    policy_phase_transform: PolicyPhaseTransform
    regime_residual: RegimeResidual
    uncertainty: RenormalizationUncertainty


class RegimeRenormalizationEngine:
    """Renormaliza (B, Pi, V, C) entre regimenes dirigidos."""

    def renormalize(
        self,
        *,
        source_regime: RegimeModel,
        target_regime: RegimeModel,
        snapshot: OrganismSnapshot,
        constraints: Dict[str, float] | None = None,
    ) -> RenormalizationResult:
        comp = compare_regimes(source_regime, target_regime)

        # Asymmetry by design: order-dependent ratio + polarity mismatch penalty.
        sensitivity_ratio = (target_regime.response_sensitivity + 1e-6) / (source_regime.response_sensitivity + 1e-6)
        polarity_penalty = 0.25 if source_regime.causal_polarity != target_regime.causal_polarity else 0.0
        asymmetry = max(0.25, min(2.5, sensitivity_ratio + polarity_penalty))

        c_in = dict(constraints or {
            "triadic_closure_threshold": 0.50,
            "min_memory_purity": 0.40,
            "min_trace_integrity": 0.30,
            "max_policy_drift": 0.50,
        })

        # Constraint transform is semantic, not only numeric scale.
        transformed_constraints: Dict[str, float] = {}
        for key, value in c_in.items():
            if key.startswith("min_"):
                transformed_constraints[key] = max(0.0, min(1.0, value * (1.0 - 0.10 * comp.structural_distance) / asymmetry))
            elif key.startswith("max_"):
                transformed_constraints[key] = max(0.0, min(1.0, value * asymmetry * (1.0 + 0.15 * comp.structural_distance)))
            else:
                transformed_constraints[key] = max(0.0, min(1.0, value * (1.0 + 0.05 * comp.structural_distance)))

        projected_alarm = snapshot.belief.alarm_probability
        if source_regime.causal_polarity != target_regime.causal_polarity:
            projected_alarm = 1.0 - projected_alarm
        projected_alarm = max(0.0, min(1.0, projected_alarm * (1.0 + 0.20 * (asymmetry - 1.0))))

        projected_efficacy = max(
            0.0,
            min(1.0, snapshot.belief.intervention_efficacy * comp.transport_feasibility / max(0.5, asymmetry)),
        )
        projected_causal = max(
            0.0,
            min(1.0, snapshot.belief.causal_support_confidence * comp.transport_feasibility / max(0.6, asymmetry)),
        )

        projected_control_class = snapshot.policy.control_class if comp.topology_match else "reactive"
        projected_sensitivity = max(0.0, min(1.0, snapshot.policy.sensitivity * asymmetry))
        projected_tolerance = max(
            0.0,
            min(1.0, snapshot.policy.perturbation_tolerance * (1.0 - 0.2 * comp.structural_distance)),
        )

        semantic_shift_cost = min(1.0, 0.35 * comp.structural_distance + 0.35 * polarity_penalty + 0.30 * abs(asymmetry - 1.0))
        residual = min(
            1.0,
            0.40 * semantic_shift_cost
            + 0.30 * (1.0 - comp.transport_feasibility)
            + 0.30 * abs(projected_sensitivity - snapshot.policy.sensitivity),
        )
        recovery_cost = min(1.0, 0.50 * residual + 0.25 * semantic_shift_cost + 0.25 * snapshot.viability.recovery_debt)
        uncertainty = min(1.0, 0.40 * comp.structural_distance + 0.30 * (1.0 - comp.transport_feasibility) + 0.30 * polarity_penalty)

        return RenormalizationResult(
            renormalization_map=RenormalizationMap(
                source_regime=source_regime.regime_id,
                target_regime=target_regime.regime_id,
                asymmetry_factor=round(asymmetry, 4),
                structural_distance=round(comp.structural_distance, 4),
                feasibility=round(comp.transport_feasibility, 4),
            ),
            constraint_transform=ConstraintTransform(
                transformed_constraints={k: round(v, 4) for k, v in transformed_constraints.items()},
                semantic_shift_cost=round(semantic_shift_cost, 4),
            ),
            belief_projection=BeliefProjectionField(
                projected_alarm=round(projected_alarm, 4),
                projected_efficacy=round(projected_efficacy, 4),
                projected_causal_support=round(projected_causal, 4),
            ),
            policy_phase_transform=PolicyPhaseTransform(
                projected_control_class=projected_control_class,
                projected_sensitivity=round(projected_sensitivity, 4),
                projected_tolerance=round(projected_tolerance, 4),
            ),
            regime_residual=RegimeResidual(
                residual_error=round(residual, 4),
                expected_recovery_cost=round(recovery_cost, 4),
            ),
            uncertainty=RenormalizationUncertainty(
                transport_uncertainty=round(uncertainty, 4),
            ),
        )
