"""Evaluación de transferibilidad de episodios entre escenarios.

Define TransferVerdict y TransferAssessment, y la función assess_transfer()
que clasifica un episodio como:
- certified_local: sin evidencia cross-scenario
- certified_transfer_safe: compatible + limpio + estable
- certified_analogical_only: evidencia analógica útil, no gobierna promoción
- rejected_for_transfer: contaminación, inestabilidad o incompatible
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


def assess_transfer(
    *,
    episode_result: dict,
    compatibility: Any | None = None,
    retrieval_metrics: dict | None = None,
    transition_vector: Any | None = None,
) -> TransferAssessment:
    """Evalúa transferibilidad de un episodio.

    Args:
        episode_result: Resultado completo del episodio.
        compatibility: Evaluación de compatibilidad (None si intra-escenario).
        retrieval_metrics: Métricas de retrieval de memoria.
        transition_vector: Vector de continuidad (None si primer episodio).

    Returns:
        TransferAssessment con veredicto de transferencia.
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

    # Determine verdict
    verdict: TransferVerdict
    if not cross_evidence and compat_class == "equivalent":
        verdict = "certified_local"
    elif compat_class in ("equivalent", "compatible") and purity >= 0.95 and stability >= 0.70:
        verdict = "certified_transfer_safe"
    elif compat_class == "analogical" and not (purity < 0.50 or stability < 0.30):
        verdict = "certified_analogical_only"
    elif compat_class == "incompatible" or purity < 0.50 or stability < 0.30:
        verdict = "rejected_for_transfer"
    elif cross_evidence and analogical_present:
        verdict = "certified_analogical_only"
    else:
        verdict = "certified_local"

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
    )
