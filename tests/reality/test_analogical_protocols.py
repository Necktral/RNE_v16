"""Tests para analogical protocols v2 (RTCME-v2 Program 5).

Valida:
- Three-regime comparison (strict/analogical/adversarial)
- Genuine improvement detection
- Adversarial shadow control
- Overall verdicts
"""

import pytest

from runtime.reality.analogical_protocols import (
    AnalogicalProtocolResult,
    RegimeComparison,
    RegimeMetrics,
    compare_regimes,
    run_analogical_protocol,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _metrics(
    regime,
    continuity=0.70,
    purity=0.95,
    posterior=0.60,
    safe_rate=0.80,
    eml=0.50,
    collapse=0,
    episodes=10,
):
    return RegimeMetrics(
        regime=regime,
        mean_continuity=continuity,
        mean_purity=purity,
        mean_posterior=posterior,
        transfer_safe_rate=safe_rate,
        eml_concurrence_mean=eml,
        collapse_count=collapse,
        total_episodes=episodes,
    )


# ── Comparison tests ─────────────────────────────────────────────────────────

class TestRegimeComparison:
    def test_genuine_improvement(self):
        baseline = _metrics("strict_same_scenario", continuity=0.60, purity=0.95, posterior=0.50)
        experimental = _metrics("cross_scenario_analogical", continuity=0.75, purity=0.92, posterior=0.55)
        comp = compare_regimes(baseline=baseline, experimental=experimental)
        assert comp.genuine_improvement is True
        assert comp.delta_continuity > 0
        assert "genuine_improvement" in comp.assessment

    def test_no_improvement(self):
        baseline = _metrics("strict_same_scenario", continuity=0.70)
        experimental = _metrics("cross_scenario_analogical", continuity=0.65)
        comp = compare_regimes(baseline=baseline, experimental=experimental)
        assert comp.genuine_improvement is False
        assert comp.assessment == "no_improvement"

    def test_improvement_at_purity_cost(self):
        baseline = _metrics("strict_same_scenario", continuity=0.60, purity=0.95)
        experimental = _metrics("cross_scenario_analogical", continuity=0.70, purity=0.80)
        comp = compare_regimes(baseline=baseline, experimental=experimental)
        assert comp.genuine_improvement is False
        assert comp.assessment == "improvement_at_unacceptable_purity_cost"

    def test_improvement_degraded_posterior(self):
        baseline = _metrics("strict_same_scenario", continuity=0.60, posterior=0.70)
        experimental = _metrics("cross_scenario_analogical", continuity=0.65, posterior=0.50)
        comp = compare_regimes(baseline=baseline, experimental=experimental)
        assert comp.genuine_improvement is False
        assert comp.assessment == "improvement_degraded_posterior"


# ── Protocol tests ───────────────────────────────────────────────────────────

class TestAnalogicalProtocol:
    def test_validated_when_analogical_helps_but_adversarial_doesnt(self):
        """Classic case: analogical genuinely helps, adversarial doesn't."""
        strict = _metrics("strict_same_scenario", continuity=0.60, purity=0.95, posterior=0.50)
        analogical = _metrics("cross_scenario_analogical", continuity=0.75, purity=0.92, posterior=0.55)
        adversarial = _metrics("cross_scenario_adversarial_shadow", continuity=0.55, purity=0.70, posterior=0.30)

        result = run_analogical_protocol(
            strict_metrics=strict,
            analogical_metrics=analogical,
            adversarial_metrics=adversarial,
        )
        assert isinstance(result, AnalogicalProtocolResult)
        assert result.overall_verdict == "analogical_transfer_validated"
        assert result.analogical_vs_strict.genuine_improvement is True
        assert result.adversarial_vs_strict.genuine_improvement is False

    def test_ambiguous_when_both_improve(self):
        """Suspicious: both analogical AND adversarial improve."""
        strict = _metrics("strict_same_scenario", continuity=0.60, purity=0.95, posterior=0.50)
        analogical = _metrics("cross_scenario_analogical", continuity=0.70, purity=0.93, posterior=0.55)
        adversarial = _metrics("cross_scenario_adversarial_shadow", continuity=0.68, purity=0.91, posterior=0.53)

        result = run_analogical_protocol(
            strict_metrics=strict,
            analogical_metrics=analogical,
            adversarial_metrics=adversarial,
        )
        assert "ambiguous" in result.overall_verdict

    def test_not_beneficial(self):
        """Analogical doesn't improve."""
        strict = _metrics("strict_same_scenario", continuity=0.70)
        analogical = _metrics("cross_scenario_analogical", continuity=0.65)
        adversarial = _metrics("cross_scenario_adversarial_shadow", continuity=0.55)

        result = run_analogical_protocol(
            strict_metrics=strict,
            analogical_metrics=analogical,
            adversarial_metrics=adversarial,
        )
        assert result.overall_verdict == "analogical_transfer_not_beneficial"
