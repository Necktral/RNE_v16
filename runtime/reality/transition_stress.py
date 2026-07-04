"""Stress testing de bordes de transición individuales.

Para cada borde dirigido (A → B), ejecuta:
1. Warmup en A (estabilizar creencias)
2. Shock transition a B (medir impacto)
3. Probe de estabilización en B (medir recovery)
4. Retorno A→B→A para medir hysteresis
5. Variación de perturbación externa

Métricas por borde:
- transfer_stability_mean
- recovery_steps_to_baseline
- hysteresis_gap
- policy_drift
- belief_drift
- transfer_posterior_mean
- failure_mode_counts
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Literal
from uuid import uuid4

from runtime.reality.belief_state import BeliefState, build_belief_state, compute_belief_shift
from runtime.reality.transfer_dynamics import (
    compute_hysteresis,
    compute_recovery_profile,
    compute_transfer_stability,
)

EdgeClass = Literal[
    "equivalent_edge",
    "compatible_edge",
    "analogical_edge",
    "adversarial_edge",
]


@dataclass(frozen=True)
class EdgeStressResult:
    """Resultado de stress test para un borde dirigido A → B."""

    source_scenario: str
    target_scenario: str
    edge_class: EdgeClass
    morphism_score: float
    # Transfer stability
    transfer_stability_mean: float
    transfer_stability_min: float
    # Recovery
    recovery_steps: int
    recovery_converged: bool
    # Hysteresis (A→B→A)
    hysteresis_gap: float
    round_trip_loss: float
    full_recovery: bool
    # Drift metrics
    policy_drift: float
    belief_drift: float
    # Posterior
    transfer_posterior_mean: float
    # Failure modes
    failure_mode_counts: Dict[str, int]
    # Samples
    warmup_episodes: int
    probe_episodes: int
    # Raw data
    details: Dict[str, Any]


def classify_edge(
    *,
    morphism_class: str,
    morphism_score: float,
) -> EdgeClass:
    """Clasifica un borde de transición en una de cuatro clases."""
    if morphism_class == "isomorphic":
        return "equivalent_edge"
    if morphism_class == "homomorphic" and morphism_score >= 0.70:
        return "compatible_edge"
    if morphism_class in ("analogical",) or (morphism_class == "homomorphic" and morphism_score < 0.70):
        return "analogical_edge"
    return "adversarial_edge"


def run_edge_stress_test(
    *,
    source_scenario: str,
    target_scenario: str,
    morphism_class: str,
    morphism_score: float,
    warmup_beliefs: List[BeliefState],
    probe_beliefs: List[BeliefState],
    return_beliefs: List[BeliefState] | None = None,
    transfer_posteriors: List[float] | None = None,
    failure_mode_lists: List[Dict[str, int]] | None = None,
    trace_integrity: bool = True,
) -> EdgeStressResult:
    """Ejecuta stress test para un borde dirigido usando beliefs pre-computados.

    This is a pure analysis function that takes pre-computed belief states
    and produces stress metrics. The actual episode execution is done
    by the edge_benchmark module.

    Args:
        source_scenario: Nombre del escenario fuente.
        target_scenario: Nombre del escenario destino.
        morphism_class: Clase del morfismo dirigido.
        morphism_score: Score del morfismo.
        warmup_beliefs: BeliefStates del warmup en source.
        probe_beliefs: BeliefStates del probe en target.
        return_beliefs: BeliefStates del retorno a source (para hysteresis).
        transfer_posteriors: Posteriors de transferencia por episodio probe.
        failure_mode_lists: Failure mode counts por episodio probe.
        trace_integrity: Integridad de traza (default True).

    Returns:
        EdgeStressResult con métricas completas.
    """
    edge_class = classify_edge(morphism_class=morphism_class, morphism_score=morphism_score)

    # Transfer stability across probe episodes
    stabilities = []
    if warmup_beliefs and probe_beliefs:
        prior = warmup_beliefs[-1]
        for posterior in probe_beliefs:
            result = compute_transfer_stability(
                prior=prior,
                posterior=posterior,
                morphism_score=morphism_score,
                trace_integrity=trace_integrity,
            )
            stabilities.append(result.transfer_stability)
            prior = posterior

    stability_mean = sum(stabilities) / len(stabilities) if stabilities else 0.0
    stability_min = min(stabilities) if stabilities else 0.0

    # Recovery profile
    if probe_beliefs and len(probe_beliefs) > 1:
        recovery = compute_recovery_profile(
            initial_belief=probe_beliefs[0],
            subsequent_beliefs=probe_beliefs[1:],
        )
        recovery_steps = recovery.recovery_steps
        recovery_converged = recovery.converged
    else:
        recovery_steps = 0
        recovery_converged = False

    # Hysteresis (if return beliefs available)
    hysteresis_gap = 0.0
    round_trip_loss = 0.0
    full_recovery = True
    if warmup_beliefs and probe_beliefs and return_beliefs:
        hyst = compute_hysteresis(
            initial=warmup_beliefs[-1],
            after_transfer=probe_beliefs[-1] if probe_beliefs else warmup_beliefs[-1],
            after_return=return_beliefs[-1] if return_beliefs else warmup_beliefs[-1],
        )
        hysteresis_gap = hyst.hysteresis_gap
        round_trip_loss = hyst.round_trip_loss
        full_recovery = hyst.full_recovery

    # Policy drift: delta in policy_confidence between last warmup and last probe
    policy_drift = 0.0
    if warmup_beliefs and probe_beliefs:
        policy_drift = abs(
            probe_beliefs[-1].policy_confidence - warmup_beliefs[-1].policy_confidence
        )

    # Belief drift: KL between warmup and probe
    belief_drift = 0.0
    if warmup_beliefs and probe_beliefs:
        shift = compute_belief_shift(prior=warmup_beliefs[-1], posterior=probe_beliefs[-1])
        belief_drift = shift.kl_divergence_approx

    # Posterior mean
    posterior_mean = 0.0
    if transfer_posteriors:
        posterior_mean = sum(transfer_posteriors) / len(transfer_posteriors)

    # Failure mode aggregation
    fm_counts: Dict[str, int] = {}
    if failure_mode_lists:
        for fm_dict in failure_mode_lists:
            for mode, count in fm_dict.items():
                fm_counts[mode] = fm_counts.get(mode, 0) + count

    return EdgeStressResult(
        source_scenario=source_scenario,
        target_scenario=target_scenario,
        edge_class=edge_class,
        morphism_score=round(morphism_score, 4),
        transfer_stability_mean=round(stability_mean, 4),
        transfer_stability_min=round(stability_min, 4),
        recovery_steps=recovery_steps,
        recovery_converged=recovery_converged,
        hysteresis_gap=round(hysteresis_gap, 4),
        round_trip_loss=round(round_trip_loss, 4),
        full_recovery=full_recovery,
        policy_drift=round(policy_drift, 4),
        belief_drift=round(belief_drift, 4),
        transfer_posterior_mean=round(posterior_mean, 4),
        failure_mode_counts=fm_counts,
        warmup_episodes=len(warmup_beliefs),
        probe_episodes=len(probe_beliefs),
        details={
            "stabilities": [round(s, 4) for s in stabilities],
            "posteriors": [round(p, 4) for p in (transfer_posteriors or [])],
        },
    )
