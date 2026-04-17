"""Evaluación de transferibilidad de episodios entre escenarios.

RTCME-v2: Integra certificación Bayesiana con posterior de seguridad,
failure modes y certificate scope, manteniendo compatibilidad con v1.

Verdicts:
- certified_local: sin evidencia cross-scenario, scope=local_only
- certified_transfer_safe: LCB >= threshold, scope=compatible_transfer
- certified_analogical_only: posterior moderado, scope=analogical_hint_only
- rejected_for_transfer: posterior insuficiente o blocking failure, scope=blocked
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

TransferVerdict = Literal[
    "certified_local",
    "certified_transfer_safe",
    "certified_analogical_only",
    "rejected_for_transfer",
]


@dataclass(frozen=True)
class TransferAssessment:
    """Resultado de evaluación de transferibilidad de un episodio."""

    episode_id: str
    source_scenario: str
    target_scenario: str
    compatibility_class: str
    closure_profile: str
    memory_mode: str
    cross_scenario_evidence_used: bool
    analogical_source_present: bool
    memory_purity_score: float
    transition_stability_score: float
    transfer_verdict: TransferVerdict
    # RTCME-v2 fields (optional for backward compat)
    transfer_posterior: float = 0.0
    lower_confidence_bound: float = 0.0
    certificate_scope: str = "local_only"
    failure_mode_count: int = 0
    morphism_score: float = 0.0


def assess_transfer(
    *,
    episode_result: dict,
    compatibility: Any | None = None,
    retrieval_metrics: dict | None = None,
    transition_vector: Any | None = None,
    morphism: Any | None = None,
    belief_shift: Any | None = None,
    eml_concurrence: float = 0.5,
    historical_success_rate: float | None = None,
    n_historical: int = 0,
) -> TransferAssessment:
    """Evalúa transferibilidad de un episodio usando Bayesian posterior.

    RTCME-v2: When morphism and belief_shift are provided, uses
    Bayesian posterior for verdict instead of threshold rules.
    Falls back to v1 heuristics when new data is unavailable.

    Args:
        episode_result: Resultado completo del episodio.
        compatibility: Evaluación de compatibilidad (None si intra-escenario).
        retrieval_metrics: Métricas de retrieval de memoria.
        transition_vector: Vector de continuidad (None si primer episodio).
        morphism: DirectedScenarioMorphism (RTCME-v2).
        belief_shift: BeliefShift (RTCME-v2).
        eml_concurrence: EML concordance score.
        historical_success_rate: Historical success rate for this edge.
        n_historical: Number of historical observations.

    Returns:
        TransferAssessment con veredicto y posterior Bayesiano.
    """
    episode = episode_result.get("episode", {})
    episode_id = episode.get("episode_id", "unknown")
    scenario_metadata = episode.get("scenario_metadata", {})
    source_scenario = scenario_metadata.get("scenario_name", "unknown")
    closure_profile = episode.get("closure_profile", "baseline_fixed")

    # Determine target scenario and cross-scenario flags
    cross_evidence = False
    analogical_present = False
    memory_mode = "strict_same_scenario"

    context = episode.get("context", {})
    retrieved = context.get("retrieved_memory", [])
    if isinstance(retrieved, list):
        for hit in retrieved:
            if isinstance(hit, dict):
                if hit.get("analogical_source"):
                    analogical_present = True
                    cross_evidence = True
                metrics = hit.get("retrieval_metrics", {})
                if metrics.get("retrieved_cross_scenario_count", 0) > 0:
                    cross_evidence = True

    # Target scenario
    target_scenario = source_scenario
    if compatibility is not None:
        target_scenario = compatibility.target_scenario

    # Compatibility class
    compat_class = "equivalent"
    if compatibility is not None:
        compat_class = compatibility.compatibility_class

    # Memory purity
    purity = 1.0
    if transition_vector is not None:
        purity = transition_vector.memory_purity

    # Transition stability
    stability = 1.0
    if transition_vector is not None:
        stability = transition_vector.composite_score

    # Extract morphism data
    m_score = 0.0
    m_class = compat_class
    polarity_inv = False
    if morphism is not None:
        m_score = getattr(morphism, "overall_score", 0.0)
        m_class = getattr(morphism, "morphism_class", compat_class)
        op = getattr(morphism, "transport_operator", None)
        if op is not None:
            polarity_inv = getattr(op, "polarity_inversion", False)

    # Extract belief shift data
    shift_kl = 0.0
    policy_conf = 0.5
    causal_supp = 0.5
    if belief_shift is not None:
        shift_kl = getattr(belief_shift, "kl_divergence_approx", 0.0)

    # Try to get belief state for enhanced confidence
    belief_data = episode_result.get("belief_state", {})
    if belief_data and belief_data.get("posterior"):
        posterior_data = belief_data["posterior"]
        policy_conf = float(posterior_data.get("policy_confidence", 0.5))
        causal_supp = float(posterior_data.get("causal_support_confidence", 0.5))

    # Trace integrity (estimate from episode)
    trace = episode.get("trace", [])
    trace_integrity = len(trace) > 0 if trace else True

    # ── Bayesian posterior path (RTCME-v2) ────────────────────────────────
    transfer_post = 0.0
    lcb = 0.0
    cert_scope = "local_only"
    fm_count = 0

    is_cross = source_scenario != target_scenario or cross_evidence

    if is_cross:
        from .transfer_posterior import compute_transfer_posterior

        posterior_result = compute_transfer_posterior(
            source_scenario=source_scenario,
            target_scenario=target_scenario,
            morphism_class=m_class if morphism is not None else _compat_to_morphism_class(compat_class),
            morphism_score=m_score if morphism is not None else (compatibility.overall_score if compatibility else 0.5),
            memory_purity=purity,
            transfer_stability=stability,
            trace_integrity=trace_integrity,
            eml_concurrence=eml_concurrence,
            polarity_inversion=polarity_inv,
            policy_confidence=policy_conf,
            causal_support=causal_supp,
            belief_shift_kl=shift_kl,
            historical_success_rate=historical_success_rate,
            n_historical=n_historical,
        )
        transfer_post = posterior_result.transfer_posterior
        lcb = posterior_result.lower_confidence_bound
        cert_scope = posterior_result.certificate_scope
        fm_count = len(posterior_result.failure_modes.detected_modes)

        # Verdict from posterior
        verdict = _verdict_from_scope(cert_scope)
    else:
        # Local episode — no transfer
        verdict = "certified_local"
        cert_scope = "local_only"

    return TransferAssessment(
        episode_id=episode_id,
        source_scenario=source_scenario,
        target_scenario=target_scenario,
        compatibility_class=compat_class,
        closure_profile=closure_profile,
        memory_mode=memory_mode,
        cross_scenario_evidence_used=cross_evidence,
        analogical_source_present=analogical_present,
        memory_purity_score=round(purity, 4),
        transition_stability_score=round(stability, 4),
        transfer_verdict=verdict,
        transfer_posterior=round(transfer_post, 4),
        lower_confidence_bound=round(lcb, 4),
        certificate_scope=cert_scope,
        failure_mode_count=fm_count,
        morphism_score=round(m_score, 4),
    )


def _compat_to_morphism_class(compat_class: str) -> str:
    """Map old compatibility class to morphism class for backward compat."""
    mapping = {
        "equivalent": "isomorphic",
        "compatible": "homomorphic",
        "analogical": "analogical",
        "incompatible": "incompatible",
    }
    return mapping.get(compat_class, "analogical")


def _verdict_from_scope(scope: str) -> TransferVerdict:
    """Map certificate scope to transfer verdict."""
    scope_to_verdict = {
        "local_only": "certified_local",
        "compatible_transfer": "certified_transfer_safe",
        "analogical_hint_only": "certified_analogical_only",
        "blocked": "rejected_for_transfer",
    }
    return scope_to_verdict.get(scope, "certified_local")
