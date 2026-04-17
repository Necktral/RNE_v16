"""Certificación Bayesiana de transferencia.

Reemplaza el sistema de verdicts por umbrales fijos con un posterior
Bayesiano de seguridad de transferencia:

  P(safe_transfer | evidence) ~ prior × likelihood

Donde:
  - prior: basado en morphism_class y historial de transferencias
  - likelihood: basado en memory_purity, transfer_stability, trace_integrity,
    EML concordance y failure_modes

La decisión de certificar se basa en:
  LCB_95%(P) >= τ

donde LCB es el lower confidence bound usando una aproximación Beta.

Certificate scope:
  - local_only: sin evidencia cross-scenario
  - compatible_transfer: morphism homomorphic + posterior alto
  - analogical_hint_only: morphism analogical + posterior moderado
  - blocked: failure mode crítico o posterior insuficiente
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Sequence

from .failure_modes import FailureModeAssessment, detect_failure_modes

CertificateScope = Literal[
    "local_only",
    "compatible_transfer",
    "analogical_hint_only",
    "blocked",
]


@dataclass(frozen=True)
class TransferPosterior:
    """Posterior Bayesiano de seguridad de transferencia."""

    source_scenario: str
    target_scenario: str
    transfer_prior: float
    transfer_likelihood: float
    transfer_posterior: float
    lower_confidence_bound: float   # LCB at 95%
    upper_confidence_bound: float   # UCB at 95%
    certificate_scope: CertificateScope
    failure_modes: FailureModeAssessment
    evidence_summary: Dict[str, float]
    n_observations: int


# ── Prior computation ────────────────────────────────────────────────────────

_MORPHISM_PRIORS = {
    "isomorphic": 0.95,
    "homomorphic": 0.80,
    "analogical": 0.45,
    "adversarial": 0.15,
    "incompatible": 0.05,
}


def _compute_prior(
    *,
    morphism_class: str,
    historical_success_rate: float | None = None,
    n_historical: int = 0,
) -> float:
    """Computa prior de seguridad de transferencia.

    Uses morphism class as base prior, then blends with historical
    success rate if available (pseudo-count Bayesian update).

    Args:
        morphism_class: Clase del morfismo dirigido.
        historical_success_rate: Tasa histórica de éxito para este edge.
        n_historical: Número de observaciones históricas.

    Returns:
        Prior en [0.01, 0.99].
    """
    base = _MORPHISM_PRIORS.get(morphism_class, 0.30)

    if historical_success_rate is not None and n_historical > 0:
        # Blend with pseudo-count: more history → more weight to empirical
        pseudo_count = 5  # strength of the prior
        blended = (pseudo_count * base + n_historical * historical_success_rate) / (
            pseudo_count + n_historical
        )
        return max(0.01, min(0.99, blended))

    return max(0.01, min(0.99, base))


# ── Likelihood computation ───────────────────────────────────────────────────

def _compute_likelihood(
    *,
    memory_purity: float,
    transfer_stability: float,
    trace_integrity: bool,
    eml_concurrence: float,
    failure_risk: float,
) -> float:
    """Computa likelihood de evidencia dado transferencia segura.

    Combina múltiples señales de evidencia en una likelihood compuesta.

    Args:
        memory_purity: Pureza de memoria [0, 1].
        transfer_stability: Estabilidad de transferencia [0, 1].
        trace_integrity: Integridad de la traza.
        eml_concurrence: Concordancia EML [0, 1].
        failure_risk: Riesgo agregado de failure modes [0, 1].

    Returns:
        Likelihood en [0.01, 0.99].
    """
    trace_val = 1.0 if trace_integrity else 0.3

    # Weighted combination
    raw = (
        0.25 * memory_purity
        + 0.30 * transfer_stability
        + 0.15 * trace_val
        + 0.10 * eml_concurrence
        + 0.20 * (1.0 - failure_risk)
    )
    return max(0.01, min(0.99, raw))


# ── Beta approximation for confidence bounds ─────────────────────────────────

def _beta_lcb(posterior: float, n: int, confidence: float = 0.95) -> float:
    """Lower confidence bound using Beta distribution approximation.

    For small n, uses Agresti-Coull interval approximation.

    Args:
        posterior: Point estimate of posterior probability.
        n: Number of observations (evidence items).
        confidence: Confidence level (default 95%).

    Returns:
        Lower confidence bound.
    """
    if n <= 0:
        return 0.0

    z = 1.96 if confidence >= 0.95 else 1.645  # z-score

    # Agresti-Coull interval
    n_tilde = n + z ** 2
    p_tilde = (posterior * n + z ** 2 / 2) / n_tilde
    interval = z * math.sqrt(p_tilde * (1 - p_tilde) / n_tilde)

    return max(0.0, min(1.0, p_tilde - interval))


def _beta_ucb(posterior: float, n: int, confidence: float = 0.95) -> float:
    """Upper confidence bound."""
    if n <= 0:
        return 1.0
    z = 1.96 if confidence >= 0.95 else 1.645
    n_tilde = n + z ** 2
    p_tilde = (posterior * n + z ** 2 / 2) / n_tilde
    interval = z * math.sqrt(p_tilde * (1 - p_tilde) / n_tilde)
    return max(0.0, min(1.0, p_tilde + interval))


# ── Scope determination ──────────────────────────────────────────────────────

def _determine_scope(
    *,
    posterior: float,
    lcb: float,
    morphism_class: str,
    has_blocking_failure: bool,
    is_cross_scenario: bool,
    threshold: float = 0.60,
) -> CertificateScope:
    """Determina el scope del certificado basado en posterior y evidencia.

    Args:
        posterior: Posterior point estimate.
        lcb: Lower confidence bound at 95%.
        morphism_class: Clase del morfismo dirigido.
        has_blocking_failure: Si hay failure modes críticos.
        is_cross_scenario: Si hay transferencia cross-scenario.
        threshold: Umbral mínimo para LCB.

    Returns:
        CertificateScope.
    """
    if has_blocking_failure:
        return "blocked"

    if not is_cross_scenario:
        return "local_only"

    if lcb >= threshold and morphism_class in ("isomorphic", "homomorphic"):
        return "compatible_transfer"

    if posterior >= 0.40 and morphism_class in ("isomorphic", "homomorphic", "analogical"):
        return "analogical_hint_only"

    if posterior < 0.25 or lcb < 0.15:
        return "blocked"

    return "analogical_hint_only"


# ── Main function ────────────────────────────────────────────────────────────

def compute_transfer_posterior(
    *,
    source_scenario: str,
    target_scenario: str,
    morphism_class: str,
    morphism_score: float,
    memory_purity: float,
    transfer_stability: float,
    trace_integrity: bool,
    eml_concurrence: float = 0.5,
    polarity_inversion: bool = False,
    policy_confidence: float = 0.5,
    causal_support: float = 0.5,
    belief_shift_kl: float = 0.0,
    historical_success_rate: float | None = None,
    n_historical: int = 0,
    threshold: float = 0.60,
) -> TransferPosterior:
    """Computa posterior Bayesiano de seguridad de transferencia.

    Integra todas las fuentes de evidencia en un posterior probabilístico
    con confidence bounds, en lugar de umbrales fijos.

    Args:
        source_scenario: Escenario fuente.
        target_scenario: Escenario destino.
        morphism_class: Clase del morfismo dirigido.
        morphism_score: Score del morfismo [0, 1].
        memory_purity: Pureza de memoria [0, 1].
        transfer_stability: Estabilidad de transferencia [0, 1].
        trace_integrity: Integridad de la traza.
        eml_concurrence: Concordancia EML [0, 1].
        polarity_inversion: Si hay inversión de polaridad causal.
        policy_confidence: Confianza en la política [0, 1].
        causal_support: Soporte causal [0, 1].
        belief_shift_kl: KL divergence del belief shift [0, 1].
        historical_success_rate: Tasa histórica de éxito.
        n_historical: Número de observaciones históricas.
        threshold: Umbral mínimo para LCB.

    Returns:
        TransferPosterior con prior, likelihood, posterior, bounds y scope.
    """
    # 1. Failure mode detection
    failure_assessment = detect_failure_modes(
        memory_purity=memory_purity,
        morphism_score=morphism_score,
        belief_shift_kl=belief_shift_kl,
        policy_confidence=policy_confidence,
        causal_support=causal_support,
        trace_integrity=trace_integrity,
        polarity_inversion=polarity_inversion,
    )

    # 2. Prior
    prior = _compute_prior(
        morphism_class=morphism_class,
        historical_success_rate=historical_success_rate,
        n_historical=n_historical,
    )

    # 3. Likelihood
    likelihood = _compute_likelihood(
        memory_purity=memory_purity,
        transfer_stability=transfer_stability,
        trace_integrity=trace_integrity,
        eml_concurrence=eml_concurrence,
        failure_risk=failure_assessment.total_risk,
    )

    # 4. Posterior (Bayes rule, normalized)
    p_safe = prior * likelihood
    p_unsafe = (1 - prior) * (1 - likelihood)
    posterior = p_safe / (p_safe + p_unsafe) if (p_safe + p_unsafe) > 0 else 0.5

    # 5. Confidence bounds
    # n_observations: count of evidence signals used
    n_obs = 5 + n_historical  # 5 base evidence signals + historical
    lcb = _beta_lcb(posterior, n_obs)
    ucb = _beta_ucb(posterior, n_obs)

    # 6. Scope
    is_cross = source_scenario != target_scenario
    scope = _determine_scope(
        posterior=posterior,
        lcb=lcb,
        morphism_class=morphism_class,
        has_blocking_failure=failure_assessment.has_blocking_failure,
        is_cross_scenario=is_cross,
        threshold=threshold,
    )

    return TransferPosterior(
        source_scenario=source_scenario,
        target_scenario=target_scenario,
        transfer_prior=round(prior, 4),
        transfer_likelihood=round(likelihood, 4),
        transfer_posterior=round(posterior, 4),
        lower_confidence_bound=round(lcb, 4),
        upper_confidence_bound=round(ucb, 4),
        certificate_scope=scope,
        failure_modes=failure_assessment,
        evidence_summary={
            "memory_purity": memory_purity,
            "transfer_stability": transfer_stability,
            "trace_integrity": 1.0 if trace_integrity else 0.0,
            "eml_concurrence": eml_concurrence,
            "failure_risk": failure_assessment.total_risk,
            "morphism_score": morphism_score,
            "belief_shift_kl": belief_shift_kl,
        },
        n_observations=n_obs,
    )
