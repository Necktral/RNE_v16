"""Guard del razonador externo — hogar canónico en runtime.

Antes vivía solo en ``scripts/benchmark_external_reasoner_gain.py``; se porta a
runtime para que el conflict resolver externo pueda gobernarse dentro del camino
vivo (bajo perfil admitido + gate + flag), no solo en el benchmark de laboratorio.

El guard rechaza una recomendación del razonador externo cuando: el output no es
válido/schema, la confianza es baja, la intervención recomendada no está permitida
o contradice el texto, o cuando adoptarla regresaría viabilidad/precisión/cierre
respecto del núcleo determinista. Es la última línea antes de que una propuesta
externa toque la decisión.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Mapping


@dataclass(frozen=True, slots=True)
class GuardDecision:
    accepted: bool
    reason: str


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def guard_external_choice(
    *,
    allowed_interventions: List[str],
    core_intervention: str,
    recommended_intervention: str | None,
    core_metrics: Mapping[str, Any],
    candidate_metrics: Mapping[str, Any] | None,
    external_ok: bool = True,
    schema_validated: bool = True,
    confidence_proxy: float = 1.0,
    confidence_threshold: float = 0.55,
    claim: str = "",
    reasoning_summary: str = "",
) -> GuardDecision:
    """Decide si la recomendación externa puede adoptarse sobre el núcleo.

    Contrato idéntico al guard validado en el benchmark del conflict resolver.
    """
    if not external_ok:
        return GuardDecision(False, "external_not_ok")
    if not schema_validated:
        return GuardDecision(False, "schema_not_validated")
    if _safe_float(confidence_proxy, 0.0) < confidence_threshold:
        return GuardDecision(False, "confidence_below_threshold")
    if not recommended_intervention:
        return GuardDecision(False, "no_recommendation")
    if recommended_intervention not in allowed_interventions:
        return GuardDecision(False, "invalid_intervention")
    evidence_text = f"{claim} {reasoning_summary}".strip().lower()
    if recommended_intervention == "deactivate_cooling" and (
        "activate cooling is better" in evidence_text
        or "activating cooling is better" in evidence_text
        or "activating cooling would lower" in evidence_text
    ):
        return GuardDecision(False, "text_recommendation_contradiction")
    if recommended_intervention == "activate_cooling" and (
        "deactivate cooling is better" in evidence_text
        or "deactivating cooling is better" in evidence_text
    ):
        return GuardDecision(False, "text_recommendation_contradiction")
    if candidate_metrics is None:
        return GuardDecision(False, "missing_candidate_metrics")
    if recommended_intervention == core_intervention:
        return GuardDecision(True, "same_as_core")

    core_margin = _safe_float(core_metrics.get("viability_margin"), 0.0)
    candidate_margin = _safe_float(candidate_metrics.get("viability_margin"), 0.0)
    if candidate_margin + 1e-9 < core_margin:
        return GuardDecision(False, "viability_regression")

    core_precision = _safe_float(core_metrics.get("intervention_precision"), 0.0)
    candidate_precision = _safe_float(candidate_metrics.get("intervention_precision"), 0.0)
    if candidate_precision + 1e-9 < core_precision:
        return GuardDecision(False, "precision_regression")

    if candidate_metrics.get("closure_stable") is False and core_metrics.get("closure_stable") is True:
        return GuardDecision(False, "closure_regression")

    return GuardDecision(True, "guard_passed")
