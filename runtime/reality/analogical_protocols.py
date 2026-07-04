"""Protocolos experimentales analógicos v2 con tres regímenes.

Compara tres regímenes de transferencia, no dos:
1. strict_same_scenario: solo memoria del mismo escenario
2. cross_scenario_analogical: incluye transferencia analógica gobernada
3. cross_scenario_adversarial_shadow: incluye transferencia adversarial
   como control negativo para detectar mejoras ilusorias

El tercer régimen es clave: indica si la "mejora" analógica es genuina
o una ilusión por mezcla de memoria contaminante.

Métricas por régimen:
- Ganancia real de continuidad (vs strict baseline)
- Coste en pureza de memoria
- Cambio en posterior de transferencia
- Tasa de falsos positivos de transferencia
- Discordancia EML-runtime
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Sequence

RegimeName = Literal[
    "strict_same_scenario",
    "cross_scenario_analogical",
    "cross_scenario_adversarial_shadow",
]


@dataclass(frozen=True)
class RegimeMetrics:
    """Métricas agregadas de un régimen experimental."""

    regime: RegimeName
    mean_continuity: float
    mean_purity: float
    mean_posterior: float
    transfer_safe_rate: float
    eml_concurrence_mean: float
    collapse_count: int
    total_episodes: int


@dataclass(frozen=True)
class RegimeComparison:
    """Comparación entre dos regímenes."""

    baseline_regime: RegimeName
    experimental_regime: RegimeName
    delta_continuity: float
    delta_purity: float
    delta_posterior: float
    delta_transfer_safe: float
    delta_eml: float
    genuine_improvement: bool   # True only if improvement is real
    assessment: str


@dataclass(frozen=True)
class AnalogicalProtocolResult:
    """Resultado completo del protocolo analógico v2."""

    strict_metrics: RegimeMetrics
    analogical_metrics: RegimeMetrics
    adversarial_metrics: RegimeMetrics
    analogical_vs_strict: RegimeComparison
    adversarial_vs_strict: RegimeComparison
    overall_verdict: str
    details: Dict[str, Any]


# ── Comparison logic ─────────────────────────────────────────────────────────

def compare_regimes(
    *,
    baseline: RegimeMetrics,
    experimental: RegimeMetrics,
    purity_cost_threshold: float = 0.10,
    posterior_degradation_threshold: float = 0.10,
) -> RegimeComparison:
    """Compara un régimen experimental contra el baseline.

    La mejora es genuina solo si:
    - Ganancia en continuidad > 0
    - Coste en pureza < threshold
    - No hay degradación significativa del posterior

    Args:
        baseline: Métricas del régimen baseline (strict).
        experimental: Métricas del régimen experimental.
        purity_cost_threshold: Máxima pérdida de pureza aceptable.
        posterior_degradation_threshold: Máxima pérdida de posterior aceptable.

    Returns:
        RegimeComparison con deltas y juicio de genuinidad.
    """
    d_cont = experimental.mean_continuity - baseline.mean_continuity
    d_purity = experimental.mean_purity - baseline.mean_purity
    d_post = experimental.mean_posterior - baseline.mean_posterior
    d_safe = experimental.transfer_safe_rate - baseline.transfer_safe_rate
    d_eml = experimental.eml_concurrence_mean - baseline.eml_concurrence_mean

    # Genuine improvement check
    continuity_gain = d_cont > 0
    purity_cost_ok = abs(d_purity) < purity_cost_threshold if d_purity < 0 else True
    posterior_ok = d_post >= -posterior_degradation_threshold
    genuine = continuity_gain and purity_cost_ok and posterior_ok

    # Assessment
    if genuine and d_cont > 0.05:
        assessment = "strong_genuine_improvement"
    elif genuine:
        assessment = "marginal_genuine_improvement"
    elif not continuity_gain:
        assessment = "no_improvement"
    elif not purity_cost_ok:
        assessment = "improvement_at_unacceptable_purity_cost"
    elif not posterior_ok:
        assessment = "improvement_degraded_posterior"
    else:
        assessment = "ambiguous"

    return RegimeComparison(
        baseline_regime=baseline.regime,
        experimental_regime=experimental.regime,
        delta_continuity=round(d_cont, 4),
        delta_purity=round(d_purity, 4),
        delta_posterior=round(d_post, 4),
        delta_transfer_safe=round(d_safe, 4),
        delta_eml=round(d_eml, 4),
        genuine_improvement=genuine,
        assessment=assessment,
    )


def run_analogical_protocol(
    *,
    strict_metrics: RegimeMetrics,
    analogical_metrics: RegimeMetrics,
    adversarial_metrics: RegimeMetrics,
) -> AnalogicalProtocolResult:
    """Ejecuta protocolo analógico v2 con tres regímenes.

    Args:
        strict_metrics: Métricas del régimen strict.
        analogical_metrics: Métricas del régimen analógico.
        adversarial_metrics: Métricas del régimen adversarial.

    Returns:
        AnalogicalProtocolResult con comparaciones y veredicto.
    """
    ana_vs_strict = compare_regimes(
        baseline=strict_metrics,
        experimental=analogical_metrics,
    )
    adv_vs_strict = compare_regimes(
        baseline=strict_metrics,
        experimental=adversarial_metrics,
    )

    # Overall verdict
    if ana_vs_strict.genuine_improvement and not adv_vs_strict.genuine_improvement:
        verdict = "analogical_transfer_validated"
    elif ana_vs_strict.genuine_improvement and adv_vs_strict.genuine_improvement:
        # Both show improvement → the "improvement" may be illusory
        verdict = "analogical_benefit_ambiguous_adversarial_also_improves"
    elif not ana_vs_strict.genuine_improvement:
        verdict = "analogical_transfer_not_beneficial"
    else:
        verdict = "inconclusive"

    return AnalogicalProtocolResult(
        strict_metrics=strict_metrics,
        analogical_metrics=analogical_metrics,
        adversarial_metrics=adversarial_metrics,
        analogical_vs_strict=ana_vs_strict,
        adversarial_vs_strict=adv_vs_strict,
        overall_verdict=verdict,
        details={
            "strict_episodes": strict_metrics.total_episodes,
            "analogical_episodes": analogical_metrics.total_episodes,
            "adversarial_episodes": adversarial_metrics.total_episodes,
        },
    )
