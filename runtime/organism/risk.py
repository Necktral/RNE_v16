"""Certificación bayesiana constitucional del organismo.

Sustituye thresholds locales por riesgo posterior.  Produce
múltiples posteriores:

  P(constitutional_safety | E)
  P(safe_transfer | E)
  P(safe_modification | E)
  P(safe_inheritance | E)

Scopes de certificado:
  - local_safe
  - transfer_safe
  - modification_safe
  - inheritance_safe
  - analogical_hint_only
  - quarantine_only
  - blocked

No se promueve si LCB(P) < τ.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Tuple

from .state import OrganismState
from .constitution import ConstitutionalValidation
from .viability import ViabilityAssessment
from .risk_process import ConstitutionalRiskProcess
from .t5_mode import get_t5_mode


ConstitutionalScope = Literal[
    "local_safe",
    "transfer_safe",
    "modification_safe",
    "inheritance_safe",
    "analogical_hint_only",
    "quarantine_only",
    "blocked",
]


@dataclass(frozen=True)
class ConstitutionalPosterior:
    """Posterior bayesiano constitucional.

    Attributes:
        scope: Alcance del certificado constitucional.
        constitutional_posterior: P(constitutional safety | E).
        transfer_posterior: P(safe transfer | E).
        modification_posterior: P(safe modification | E).
        inheritance_posterior: P(safe inheritance | E).
        lower_confidence_bound: LCB al 95%.
        failure_modes: Lista de modos de fallo detectados.
        quarantine_required: Si requiere cuarentena.
        rollback_required: Si requiere rollback.
        evidence_summary: Resumen de evidencia usada.
    """

    scope: ConstitutionalScope
    constitutional_posterior: float
    transfer_posterior: float
    modification_posterior: float
    inheritance_posterior: float
    lower_confidence_bound: float
    failure_modes: Tuple[str, ...]
    quarantine_required: bool
    rollback_required: bool
    evidence_summary: Dict[str, float]


# ── Beta confidence bounds ───────────────────────────────────────────────────

def _beta_lcb(p: float, n: int, confidence: float = 0.95) -> float:
    """Lower confidence bound via Agresti-Coull."""
    if n <= 0:
        return 0.0
    z = 1.96 if confidence >= 0.95 else 1.645
    n_t = n + z ** 2
    p_t = (p * n + z ** 2 / 2) / n_t
    interval = z * math.sqrt(p_t * (1 - p_t) / n_t)
    return max(0.0, min(1.0, p_t - interval))


# ── Main computation ─────────────────────────────────────────────────────────

def compute_constitutional_posterior(
    *,
    state: OrganismState,
    constitutional_validation: ConstitutionalValidation,
    viability_assessment: ViabilityAssessment,
    morphism_score: float = 1.0,
    transfer_stability: float = 1.0,
    eml_concurrence: float = 0.5,
    lineage_consistency: float = 1.0,
    n_historical: int = 0,
    historical_success_rate: float | None = None,
    threshold: float = 0.60,
) -> ConstitutionalPosterior:
    """Computa posterior bayesiano constitucional.

    Integra todas las fuentes de evidencia para producir:
    - P(constitutional safety | E)
    - P(safe transfer | E)
    - P(safe modification | E)
    - P(safe inheritance | E)

    Args:
        state: Estado actual del organismo.
        constitutional_validation: Validación constitucional.
        viability_assessment: Evaluación de viabilidad.
        morphism_score: Score del morfismo dirigido.
        transfer_stability: Estabilidad de transferencia.
        eml_concurrence: Concordancia EML.
        lineage_consistency: Consistencia de lineage.
        n_historical: Observaciones históricas.
        historical_success_rate: Tasa de éxito histórica.
        threshold: Umbral para LCB.

    Returns:
        ConstitutionalPosterior.
    """
    # Evidence signals
    belief_conf = state.belief.composite_confidence
    policy_stab = state.policy.stability_score
    viability_margin = viability_assessment.viability_margin
    is_constitutionally_valid = constitutional_validation.is_valid
    hard_violations = constitutional_validation.hard_violation_count
    soft_violations = constitutional_validation.soft_violation_count
    memory_purity = state.belief.memory_purity_estimate
    trace_integrity = state.belief.trace_integrity_confidence

    # --- Constitutional posterior ---
    const_prior = 0.80 if is_constitutionally_valid else 0.20
    if historical_success_rate is not None and n_historical > 0:
        pseudo = 5
        const_prior = (pseudo * const_prior + n_historical * historical_success_rate) / (pseudo + n_historical)

    const_likelihood = (
        0.20 * belief_conf
        + 0.20 * viability_margin
        + 0.15 * policy_stab
        + 0.15 * memory_purity
        + 0.15 * trace_integrity
        + 0.10 * eml_concurrence
        + 0.05 * (1.0 - hard_violations * 0.25)
    )
    const_likelihood = max(0.01, min(0.99, const_likelihood))

    p_safe = const_prior * const_likelihood
    p_unsafe = (1 - const_prior) * (1 - const_likelihood)
    const_posterior = p_safe / (p_safe + p_unsafe) if (p_safe + p_unsafe) > 0 else 0.5

    # --- Transfer posterior ---
    transfer_prior = 0.70 * morphism_score
    transfer_likelihood = (
        0.30 * transfer_stability
        + 0.25 * memory_purity
        + 0.20 * trace_integrity
        + 0.15 * eml_concurrence
        + 0.10 * viability_margin
    )
    transfer_likelihood = max(0.01, min(0.99, transfer_likelihood))
    t_safe = transfer_prior * transfer_likelihood
    t_unsafe = (1 - transfer_prior) * (1 - transfer_likelihood)
    transfer_post = t_safe / (t_safe + t_unsafe) if (t_safe + t_unsafe) > 0 else 0.5

    # --- Modification posterior ---
    mod_prior = 0.50 if is_constitutionally_valid and viability_margin > 0.30 else 0.15
    mod_likelihood = (
        0.25 * viability_margin
        + 0.25 * policy_stab
        + 0.20 * (1.0 - state.viability.recovery_debt)
        + 0.15 * belief_conf
        + 0.15 * (1.0 if state.viability.rollback_readiness else 0.0)
    )
    mod_likelihood = max(0.01, min(0.99, mod_likelihood))
    m_safe = mod_prior * mod_likelihood
    m_unsafe = (1 - mod_prior) * (1 - mod_likelihood)
    mod_post = m_safe / (m_safe + m_unsafe) if (m_safe + m_unsafe) > 0 else 0.5

    # --- Inheritance posterior ---
    inh_prior = 0.60 * lineage_consistency
    inh_likelihood = (
        0.30 * const_posterior
        + 0.25 * memory_purity
        + 0.20 * trace_integrity
        + 0.25 * viability_margin
    )
    inh_likelihood = max(0.01, min(0.99, inh_likelihood))
    i_safe = inh_prior * inh_likelihood
    i_unsafe = (1 - inh_prior) * (1 - inh_likelihood)
    inh_post = i_safe / (i_safe + i_unsafe) if (i_safe + i_unsafe) > 0 else 0.5

    # --- T5 sequential adapter ---
    if get_t5_mode() == "on":
        process = ConstitutionalRiskProcess()
        process.update(
            scope_type="organism",
            scope_key=state.identity.lineage_id or "genesis",
            drift_identity=max(0.0, state.identity.identity_distance(OrganismState().identity)),
            drift_policy=max(0.0, state.policy.accumulated_drift),
            delta_viability=max(-1.0, min(1.0, viability_margin - 0.5)),
            delta_purity=max(0.0, 1.0 - memory_purity),
            delta_modification=1.0 if state.modification.lineage_delta_pending else 0.0,
            erosion=max(0.0, 1.0 - float(const_likelihood)),
            renorm_residual=max(0.0, 1.0 - morphism_score),
        )
        seq_state = process.get(
            scope_type="organism",
            scope_key=state.identity.lineage_id or "genesis",
        )
        if seq_state is not None:
            seq_safe = 1.0 - seq_state.risk_score
            const_posterior = max(0.01, min(0.99, 0.70 * const_posterior + 0.30 * seq_safe))
            transfer_post = max(0.01, min(0.99, 0.75 * transfer_post + 0.25 * seq_safe))
            mod_post = max(0.01, min(0.99, 0.75 * mod_post + 0.25 * seq_safe))
            inh_post = max(0.01, min(0.99, 0.75 * inh_post + 0.25 * seq_safe))

    # --- LCB ---
    n_obs = 6 + n_historical
    lcb = _beta_lcb(const_posterior, n_obs)

    # --- Failure modes ---
    failure_modes: list[str] = []
    if hard_violations > 0:
        failure_modes.append("constitutional_violation")
    if memory_purity < 0.50:
        failure_modes.append("memory_contamination")
    if trace_integrity < 0.40:
        failure_modes.append("trace_discontinuity")
    if viability_margin < 0.15:
        failure_modes.append("viability_edge_proximity")
    if state.policy.accumulated_drift > 0.50:
        failure_modes.append("policy_drift")
    if state.viability.recovery_debt > 0.60:
        failure_modes.append("recovery_debt")

    # --- Scope ---
    quarantine = viability_assessment.recovery_plan.quarantine_recommended
    rollback = viability_assessment.rollback_required

    if rollback:
        scope: ConstitutionalScope = "blocked"
    elif quarantine:
        scope = "quarantine_only"
    elif hard_violations > 0:
        scope = "blocked"
    elif lcb >= threshold and morphism_score >= 0.70:
        if lineage_consistency >= 0.80:
            scope = "inheritance_safe"
        else:
            scope = "transfer_safe"
    elif lcb >= threshold * 0.7:
        scope = "analogical_hint_only"
    elif is_constitutionally_valid and lcb >= threshold * 0.5:
        scope = "local_safe"
    else:
        scope = "blocked"

    return ConstitutionalPosterior(
        scope=scope,
        constitutional_posterior=round(const_posterior, 4),
        transfer_posterior=round(transfer_post, 4),
        modification_posterior=round(mod_post, 4),
        inheritance_posterior=round(inh_post, 4),
        lower_confidence_bound=round(lcb, 4),
        failure_modes=tuple(failure_modes),
        quarantine_required=quarantine,
        rollback_required=rollback,
        evidence_summary={
            "belief_confidence": round(belief_conf, 4),
            "policy_stability": round(policy_stab, 4),
            "viability_margin": round(viability_margin, 4),
            "memory_purity": round(memory_purity, 4),
            "trace_integrity": round(trace_integrity, 4),
            "eml_concurrence": round(eml_concurrence, 4),
            "morphism_score": round(morphism_score, 4),
            "transfer_stability": round(transfer_stability, 4),
            "lineage_consistency": round(lineage_consistency, 4),
            "hard_violations": hard_violations,
            "soft_violations": soft_violations,
        },
    )
