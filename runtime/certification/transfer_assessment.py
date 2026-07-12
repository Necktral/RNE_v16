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

from dataclasses import dataclass, field
from typing import Any, Literal

from .failure_modes import TransferFailureMode, detect_failure_modes
from .trace_integrity import assess_trace_integrity

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
    canonical_scope: str = "local_safe"
    failure_mode_count: int = 0
    morphism_score: float = 0.0
    # B1: integridad de traza REALMENTE verificada (antes era constante True e
    # invisible para el llamador). `trace_integrity_reason` dice por qué.
    trace_integrity: bool = False
    trace_integrity_reason: str = "trace_missing"
    # P9.6 — las patologías dejan de ser un contador mudo e inalcanzable.
    # `failure_modes`: las detectadas, con nombre y severidad (METADATA: no gatean).
    # `failure_mode_scope`: qué familia se evaluó — `local` (episodio intra-escenario),
    #   `all` (hubo transferencia) o `none`.
    # `detector_checks_applied`: qué detectores SÍ pudieron correr. Los que no corrieron
    #   NO cuentan como aprobados (patrón `trace_integrity.checks_applied`).
    failure_modes: tuple[TransferFailureMode, ...] = ()
    failure_mode_scope: str = "none"
    detector_checks_applied: tuple[str, ...] = ()


def retrieval_metrics_from_hits(hits: Any) -> dict | None:
    """Métricas de retrieval que el episodio registró, o ``None`` si no registró ninguna.

    P9.6: ``None`` significa **NO MEDIDO** (el episodio no dejó evidencia de retrieval),
    NO "retrieval limpio". La diferencia importa: con 0 hits no hubo *oportunidad* de
    contaminarse, que no es lo mismo que haber verificado que no hubo contaminación.
    Quien consume esto debe distinguir los dos casos (ver ``memory_purity_basis``).

    El retriever (``runtime/memory/mfm_lite/retrieval.py``) adjunta el MISMO dict de
    métricas a cada hit devuelto, así que alcanza con el primero que lo traiga.
    """
    if not isinstance(hits, (list, tuple)):
        return None
    for hit in hits:
        if isinstance(hit, dict):
            metrics = hit.get("retrieval_metrics")
            if isinstance(metrics, dict):
                return dict(metrics)
    return None


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

    # Trace integrity — verificación REAL (B1).
    # Antes: `len(trace) > 0 if trace else True` → True en ambas ramas (constante
    # disfrazada de medición): inflaba la likelihood del posterior y volvía
    # inalcanzable el failure mode `trace_discontinuity`. Ahora se verifica
    # presencia, buena formación y continuidad contra la secuencia ejecutada.
    trace_result = assess_trace_integrity(episode)
    trace_integrity = trace_result.integral

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
        fm_assessment = posterior_result.failure_modes
        fm_count = len(fm_assessment.detected_modes)
        fm_scope = "all"

        # Verdict from posterior
        verdict = _verdict_from_scope(cert_scope)
    else:
        # ── Episodio LOCAL (intra-escenario) ─────────────────────────────────
        # P9.6 paso 2 — ABRIR EL GATE. Antes, esta rama era un `pass`: `detect_failure_modes`
        # solo se alcanzaba vía `compute_transfer_posterior`, que corre únicamente dentro de
        # `if is_cross:`. Es decir: en un episodio intra-escenario con memoria limpia —que es
        # DONDE EL ORGANISMO VIVE— no se evaluaba ninguna patología. Contaminación de memoria,
        # deriva de política y colapso de creencias estaban archivadas detrás de un gate de
        # TRANSFERENCIA, y por eso el organismo nunca se veía enfermo: no se miraba.
        #
        # Ahora las patologías LOCALES se evalúan igual. Lo de transferencia NO se mezcla:
        # `causal_inversion` y `morphism_failure` exigen un morfismo dirigido que en un
        # episodio local no existe — pedirlos acá sería inventar transferencia donde no hay.
        #
        # NO GATEA (decisión conservadora explícita de P9.6): el veredicto local sigue siendo
        # `certified_local` aunque se detecten patologías. Los failure modes entran al
        # certificado como METADATA, igual que antes; lo que cambia es que ahora son
        # ALCANZABLES y REALES en vez de inalcanzables. Convertirlos en compuerta es otra
        # decisión, y no es esta.
        fm_assessment = detect_failure_modes(
            memory_purity=purity,
            belief_shift_kl=shift_kl,
            policy_confidence=policy_conf,
            causal_support=causal_supp,
            trace_integrity=trace_integrity,
            morphism_score=None,      # no hay morfismo en un episodio local
            polarity_inversion=False,
            scope="local",
        )
        fm_count = len(fm_assessment.detected_modes)
        fm_scope = "local"
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
        canonical_scope=_canonical_scope_from_legacy(cert_scope),
        failure_mode_count=fm_count,
        morphism_score=round(m_score, 4),
        trace_integrity=trace_integrity,
        trace_integrity_reason=trace_result.reason,
        failure_modes=fm_assessment.detected_modes,
        failure_mode_scope=fm_scope,
        detector_checks_applied=fm_assessment.checks_applied,
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


def _canonical_scope_from_legacy(scope: str) -> str:
    mapping = {
        "local_only": "local_safe",
        "compatible_transfer": "transfer_safe",
        "analogical_hint_only": "quarantine_only",
        "blocked": "blocked",
    }
    return mapping.get(scope, "blocked")
