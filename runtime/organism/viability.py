"""Kernel de viabilidad del organismo RNFE.

Formaliza la región de estados desde la cual el organismo puede seguir
existiendo sin romper su constitución.

    K = { x | C(x) = valid ∧ M(x) ≥ 0 }

Cada transición evalúa M(x_{t+1}) - M(x_t).  No basta con saber si
cerró; importa si el organismo se acerca peligrosamente al borde.

Produce:
  - is_viable
  - viability_margin
  - distance_to_edge
  - constraint_violations
  - recovery_plan
  - rollback_required
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Tuple

from .state import OrganismState, ViabilityState
from .constitution import ConstitutionalValidation, OrganismConstitution


# ── Data contracts ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RecoveryAction:
    """Acción de recuperación sugerida."""

    action: str
    priority: Literal["critical", "high", "medium", "low"]
    target_component: str
    description: str


@dataclass(frozen=True)
class RecoveryPlan:
    """Plan de recuperación del organismo."""

    actions: Tuple[RecoveryAction, ...]
    estimated_steps: int
    rollback_recommended: bool
    quarantine_recommended: bool

    @property
    def is_empty(self) -> bool:
        return len(self.actions) == 0


@dataclass(frozen=True)
class ViabilityAssessment:
    """Evaluación completa de viabilidad del organismo.

    Attributes:
        is_viable: True si el organismo está dentro de la región viable.
        viability_margin: Margen normalizado [0, 1].  0 = al borde.
        distance_to_edge: Distancia normalizada al borde de inviabilidad.
        constitutional_validation: Resultado de validación constitucional.
        recovery_plan: Plan de recuperación sugerido.
        rollback_required: Si requiere rollback inmediato.
        margin_delta: Cambio en margen respecto al estado previo.
        degradation_rate: Tasa de degradación.
    """

    is_viable: bool
    viability_margin: float
    distance_to_edge: float
    constitutional_validation: ConstitutionalValidation
    recovery_plan: RecoveryPlan
    rollback_required: bool
    margin_delta: float = 0.0
    degradation_rate: float = 0.0


# ── Viability kernel ─────────────────────────────────────────────────────────

class ViabilityKernel:
    """Kernel de viabilidad: evalúa si el organismo permanece en K.

    El kernel combina validación constitucional con métricas de viabilidad
    para producir una evaluación completa.
    """

    def __init__(
        self,
        *,
        constitution: OrganismConstitution | None = None,
        edge_margin_threshold: float = 0.15,
        critical_margin_threshold: float = 0.05,
    ):
        self.constitution = constitution or OrganismConstitution()
        self.edge_margin_threshold = edge_margin_threshold
        self.critical_margin_threshold = critical_margin_threshold

    def assess(
        self,
        state: OrganismState,
        previous_state: OrganismState | None = None,
    ) -> ViabilityAssessment:
        """Evalúa viabilidad del organismo.

        Args:
            state: Estado actual del organismo.
            previous_state: Estado previo (para calcular delta).

        Returns:
            ViabilityAssessment completa.
        """
        # Constitutional validation
        validation = self.constitution.validate(state)

        # Compute viability margin from multiple signals
        belief_margin = state.belief.composite_confidence
        viability_margin_raw = state.viability.viability_margin
        degradation_factor = 1.0 - state.viability.accumulated_degradation
        recovery_factor = 1.0 - state.viability.recovery_debt

        # Combined margin
        margin = (
            0.30 * viability_margin_raw
            + 0.25 * belief_margin
            + 0.25 * degradation_factor
            + 0.20 * recovery_factor
        )
        margin = max(0.0, min(1.0, margin))

        # Penalize by constitutional violations
        if validation.hard_violation_count > 0:
            margin *= max(0.0, 1.0 - 0.30 * validation.hard_violation_count)
        if validation.soft_violation_count > 0:
            margin *= max(0.5, 1.0 - 0.05 * validation.soft_violation_count)

        # Distance to edge
        distance = margin

        # Margin delta
        margin_delta = 0.0
        degradation_rate = 0.0
        if previous_state is not None:
            prev_margin = (
                0.30 * previous_state.viability.viability_margin
                + 0.25 * previous_state.belief.composite_confidence
                + 0.25 * (1.0 - previous_state.viability.accumulated_degradation)
                + 0.20 * (1.0 - previous_state.viability.recovery_debt)
            )
            margin_delta = margin - prev_margin
            degradation_rate = (
                state.viability.accumulated_degradation
                - previous_state.viability.accumulated_degradation
            )

        # Is viable?
        is_viable = (
            validation.is_valid
            and margin > self.critical_margin_threshold
            and state.viability.is_viable
        )

        # Recovery plan
        recovery_plan = self._build_recovery_plan(state, validation, margin)

        # Rollback required?
        rollback_required = (
            validation.verdict == "rollback"
            or margin <= self.critical_margin_threshold
            or not state.viability.is_viable
        )

        return ViabilityAssessment(
            is_viable=is_viable,
            viability_margin=round(margin, 4),
            distance_to_edge=round(distance, 4),
            constitutional_validation=validation,
            recovery_plan=recovery_plan,
            rollback_required=rollback_required,
            margin_delta=round(margin_delta, 4),
            degradation_rate=round(degradation_rate, 4),
        )

    def is_in_kernel(self, state: OrganismState) -> bool:
        """Simple check: is the organism in the viable region?"""
        assessment = self.assess(state)
        return assessment.is_viable

    def margin_trajectory(
        self,
        states: list[OrganismState],
    ) -> list[float]:
        """Computes margin trajectory over a sequence of states."""
        margins = []
        prev = None
        for s in states:
            a = self.assess(s, previous_state=prev)
            margins.append(a.viability_margin)
            prev = s
        return margins

    def _build_recovery_plan(
        self,
        state: OrganismState,
        validation: ConstitutionalValidation,
        margin: float,
    ) -> RecoveryPlan:
        """Construye plan de recuperación basado en violaciones y margen."""
        actions: List[RecoveryAction] = []

        # From hard violations
        for v in validation.violations:
            if v.severity == "hard":
                actions.append(RecoveryAction(
                    action=f"restore_{v.invariant_name}",
                    priority="critical",
                    target_component=v.invariant_name,
                    description=v.description,
                ))

        # From soft violations
        for v in validation.violations:
            if v.severity == "soft":
                actions.append(RecoveryAction(
                    action=f"improve_{v.invariant_name}",
                    priority="medium",
                    target_component=v.invariant_name,
                    description=v.description,
                ))

        # From margin proximity
        if margin < self.edge_margin_threshold:
            actions.append(RecoveryAction(
                action="increase_viability_margin",
                priority="high",
                target_component="viability",
                description=f"Margin {margin:.4f} near edge threshold {self.edge_margin_threshold}",
            ))

        # From recovery debt
        if state.viability.recovery_debt > 0.50:
            actions.append(RecoveryAction(
                action="reduce_recovery_debt",
                priority="high",
                target_component="viability",
                description=f"Recovery debt {state.viability.recovery_debt:.4f} > 0.50",
            ))

        estimated_steps = len(actions) * 2  # rough estimate
        rollback = validation.verdict == "rollback" or margin <= self.critical_margin_threshold
        quarantine = validation.verdict == "quarantine" or (
            margin < self.edge_margin_threshold and not rollback
        )

        return RecoveryPlan(
            actions=tuple(actions),
            estimated_steps=estimated_steps,
            rollback_recommended=rollback,
            quarantine_recommended=quarantine,
        )
