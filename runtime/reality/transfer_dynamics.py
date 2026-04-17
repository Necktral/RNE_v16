"""Dinámica de transferencia con belief state.

Computa transfer_stability como combinación de:
  E = evidence vector composite
  B = 1 - D_belief(prior, posterior)
  transfer_stability = λ·E + (1-λ)·B

Incluye:
- Medición de hysteresis (A→B→A residual distance)
- Recovery steps (cuántos episodios para estabilizar)
- Error de transferencia acumulado
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence

from .belief_state import (
    BeliefShift,
    BeliefState,
    TransitionEvidenceVector,
    compute_belief_shift,
    compute_transition_evidence,
)

# ── Weights ──────────────────────────────────────────────────────────────────

_LAMBDA = 0.60  # Weight for evidence vs belief stability


# ── Data contracts ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TransferStabilityResult:
    """Resultado de evaluación de estabilidad de transferencia.

    Combina evidencia de transición con dinámica de creencia.
    """

    source_scenario: str
    target_scenario: str
    evidence_score: float
    belief_stability_score: float
    transfer_stability: float
    belief_shift: BeliefShift
    evidence_vector: TransitionEvidenceVector
    morphism_score: float
    recovery_needed: bool


@dataclass(frozen=True)
class HysteresisResult:
    """Resultado de evaluación de hysteresis A→B→A.

    Mide cuánta información se pierde en un viaje de ida y vuelta
    entre dos escenarios.
    """

    scenario_a: str
    scenario_b: str
    initial_state: BeliefState
    after_b_state: BeliefState
    return_state: BeliefState
    outbound_shift: BeliefShift
    return_shift: BeliefShift
    hysteresis_gap: float       # Residual distance initial vs return
    round_trip_loss: float      # Information loss in round trip
    full_recovery: bool         # True if gap < threshold


@dataclass(frozen=True)
class RecoveryProfile:
    """Perfil de recuperación tras cambio de dominio.

    Tracks how many steps it takes to stabilize after transition.
    """

    target_scenario: str
    initial_belief: BeliefState
    stabilized_belief: BeliefState | None
    recovery_steps: int
    final_stability: float
    converged: bool


# ── Core functions ───────────────────────────────────────────────────────────

def compute_transfer_stability(
    *,
    prior: BeliefState,
    posterior: BeliefState,
    morphism_score: float = 1.0,
    trace_integrity: bool = True,
    lambda_weight: float = _LAMBDA,
) -> TransferStabilityResult:
    """Computa estabilidad de transferencia combinando evidencia y belief.

    Formula:
        E = composite evidence (from transition evidence vector)
        B = belief stability (1 - KL_approx)
        transfer_stability = λ·E + (1-λ)·B

    Args:
        prior: BeliefState previo al cambio.
        posterior: BeliefState posterior al cambio.
        morphism_score: Score del morfismo dirigido source→target.
        trace_integrity: Integridad de la traza.
        lambda_weight: Peso de evidencia vs belief (default 0.60).

    Returns:
        TransferStabilityResult con scores combinados.
    """
    evidence = compute_transition_evidence(
        prior=prior,
        posterior=posterior,
        morphism_score=morphism_score,
        trace_integrity=trace_integrity,
    )
    shift = compute_belief_shift(prior=prior, posterior=posterior)

    e_score = evidence.composite_evidence
    b_score = shift.stability_score
    stability = lambda_weight * e_score + (1.0 - lambda_weight) * b_score
    stability = max(0.0, min(1.0, stability))

    return TransferStabilityResult(
        source_scenario=prior.scenario_name,
        target_scenario=posterior.scenario_name,
        evidence_score=round(e_score, 4),
        belief_stability_score=round(b_score, 4),
        transfer_stability=round(stability, 4),
        belief_shift=shift,
        evidence_vector=evidence,
        morphism_score=round(morphism_score, 4),
        recovery_needed=shift.recovery_needed,
    )


def compute_hysteresis(
    *,
    initial: BeliefState,
    after_transfer: BeliefState,
    after_return: BeliefState,
) -> HysteresisResult:
    """Mide hysteresis en un viaje A→B→A.

    Compara el estado inicial con el estado tras retorno.
    Un sistema con buena transferencia debería tener bajo hysteresis gap.

    Args:
        initial: BeliefState antes de salir de A.
        after_transfer: BeliefState en B.
        after_return: BeliefState tras retornar a A.

    Returns:
        HysteresisResult con gap, loss y convergencia.
    """
    outbound = compute_belief_shift(prior=initial, posterior=after_transfer)
    return_shift = compute_belief_shift(prior=after_transfer, posterior=after_return)

    # Hysteresis gap: distance between initial and return
    residual = compute_belief_shift(prior=initial, posterior=after_return)
    gap = residual.kl_divergence_approx

    # Round trip loss
    total_movement = outbound.kl_divergence_approx + return_shift.kl_divergence_approx
    loss = min(1.0, gap / max(total_movement, 0.01))

    return HysteresisResult(
        scenario_a=initial.scenario_name,
        scenario_b=after_transfer.scenario_name,
        initial_state=initial,
        after_b_state=after_transfer,
        return_state=after_return,
        outbound_shift=outbound,
        return_shift=return_shift,
        hysteresis_gap=round(gap, 4),
        round_trip_loss=round(loss, 4),
        full_recovery=gap < 0.10,
    )


def compute_recovery_profile(
    *,
    initial_belief: BeliefState,
    subsequent_beliefs: Sequence[BeliefState],
    convergence_threshold: float = 0.05,
) -> RecoveryProfile:
    """Mide cuántos pasos necesita para estabilizarse tras transición.

    Args:
        initial_belief: Primer belief state tras el cambio de dominio.
        subsequent_beliefs: Secuencia de belief states posteriores.
        convergence_threshold: Umbral de cambio para considerar convergencia.

    Returns:
        RecoveryProfile con steps y convergencia.
    """
    if not subsequent_beliefs:
        return RecoveryProfile(
            target_scenario=initial_belief.scenario_name,
            initial_belief=initial_belief,
            stabilized_belief=None,
            recovery_steps=0,
            final_stability=initial_belief.composite_confidence,
            converged=False,
        )

    prev = initial_belief
    for step, current in enumerate(subsequent_beliefs, 1):
        shift = compute_belief_shift(prior=prev, posterior=current)
        if shift.kl_divergence_approx < convergence_threshold:
            return RecoveryProfile(
                target_scenario=current.scenario_name,
                initial_belief=initial_belief,
                stabilized_belief=current,
                recovery_steps=step,
                final_stability=round(current.composite_confidence, 4),
                converged=True,
            )
        prev = current

    last = subsequent_beliefs[-1]
    return RecoveryProfile(
        target_scenario=last.scenario_name,
        initial_belief=initial_belief,
        stabilized_belief=last,
        recovery_steps=len(subsequent_beliefs),
        final_stability=round(last.composite_confidence, 4),
        converged=False,
    )
