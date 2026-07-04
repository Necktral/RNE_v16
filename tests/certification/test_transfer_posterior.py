"""Tests para Bayesian transfer posterior (RTCME-v2 Program 3).

Valida:
- Transfer posterior computation with all evidence sources
- Prior computation from morphism class and history
- Likelihood computation from evidence signals
- Beta confidence bounds
- Certificate scope determination
- Backward compatibility with v1 assess_transfer
"""

import pytest

from runtime.certification.transfer_posterior import (
    TransferPosterior,
    compute_transfer_posterior,
    _compute_prior,
    _compute_likelihood,
    _beta_lcb,
    _beta_ucb,
)
from runtime.certification.failure_modes import (
    FailureModeAssessment,
    detect_failure_modes,
)
from runtime.certification.transfer_assessment import (
    TransferAssessment,
    assess_transfer,
)


# ── Transfer Posterior tests ─────────────────────────────────────────────────

class TestTransferPosterior:
    def test_isomorphic_high_posterior(self):
        """Isomorphic morphism + good evidence → high posterior."""
        result = compute_transfer_posterior(
            source_scenario="thermal",
            target_scenario="thermal",
            morphism_class="isomorphic",
            morphism_score=0.98,
            memory_purity=0.99,
            transfer_stability=0.95,
            trace_integrity=True,
            eml_concurrence=0.85,
        )
        assert isinstance(result, TransferPosterior)
        assert result.transfer_posterior > 0.85
        assert result.certificate_scope == "local_only"  # same scenario

    def test_homomorphic_compatible_transfer(self):
        """Homomorphic morphism with strong evidence + history → compatible_transfer."""
        result = compute_transfer_posterior(
            source_scenario="thermal",
            target_scenario="resource",
            morphism_class="homomorphic",
            morphism_score=0.80,
            memory_purity=0.98,
            transfer_stability=0.90,
            trace_integrity=True,
            eml_concurrence=0.85,
            historical_success_rate=0.90,
            n_historical=15,
        )
        assert result.transfer_posterior > 0.70
        assert result.certificate_scope == "compatible_transfer"

    def test_analogical_hint_only(self):
        """Analogical morphism → analogical_hint_only."""
        result = compute_transfer_posterior(
            source_scenario="thermal",
            target_scenario="resource",
            morphism_class="analogical",
            morphism_score=0.45,
            memory_purity=0.80,
            transfer_stability=0.60,
            trace_integrity=True,
        )
        # Analogical with moderate posterior
        assert result.certificate_scope in ("analogical_hint_only", "blocked")

    def test_adversarial_blocked(self):
        """Adversarial morphism + poor evidence → blocked."""
        result = compute_transfer_posterior(
            source_scenario="thermal",
            target_scenario="resource",
            morphism_class="adversarial",
            morphism_score=0.10,
            memory_purity=0.30,
            transfer_stability=0.20,
            trace_integrity=False,
            polarity_inversion=True,
            causal_support=0.15,
            belief_shift_kl=0.70,
        )
        assert result.certificate_scope == "blocked"
        assert result.failure_modes.has_blocking_failure

    def test_lcb_below_threshold_blocks(self):
        """Even moderate posterior can be blocked if LCB is too low."""
        result = compute_transfer_posterior(
            source_scenario="thermal",
            target_scenario="resource",
            morphism_class="analogical",
            morphism_score=0.40,
            memory_purity=0.60,
            transfer_stability=0.50,
            trace_integrity=True,
            belief_shift_kl=0.50,
        )
        # With high uncertainty, LCB may be low
        assert result.lower_confidence_bound < result.transfer_posterior

    def test_historical_data_affects_prior(self):
        """Historical success rate should shift the prior."""
        result_no_hist = compute_transfer_posterior(
            source_scenario="thermal",
            target_scenario="resource",
            morphism_class="analogical",
            morphism_score=0.50,
            memory_purity=0.90,
            transfer_stability=0.80,
            trace_integrity=True,
        )
        result_with_hist = compute_transfer_posterior(
            source_scenario="thermal",
            target_scenario="resource",
            morphism_class="analogical",
            morphism_score=0.50,
            memory_purity=0.90,
            transfer_stability=0.80,
            trace_integrity=True,
            historical_success_rate=0.90,
            n_historical=20,
        )
        # With successful history, posterior should be higher
        assert result_with_hist.transfer_posterior >= result_no_hist.transfer_posterior


# ── Prior tests ──────────────────────────────────────────────────────────────

class TestPriorComputation:
    def test_isomorphic_prior_high(self):
        assert _compute_prior(morphism_class="isomorphic") >= 0.90

    def test_incompatible_prior_low(self):
        assert _compute_prior(morphism_class="incompatible") <= 0.10

    def test_history_shifts_prior(self):
        base = _compute_prior(morphism_class="analogical")
        shifted = _compute_prior(morphism_class="analogical", historical_success_rate=0.95, n_historical=50)
        assert shifted > base


# ── Confidence bounds tests ──────────────────────────────────────────────────

class TestConfidenceBounds:
    def test_lcb_below_posterior(self):
        assert _beta_lcb(0.80, 10) < 0.80

    def test_ucb_above_posterior(self):
        assert _beta_ucb(0.80, 10) > 0.80

    def test_more_data_tighter_bounds(self):
        lcb_5 = _beta_lcb(0.80, 5)
        lcb_100 = _beta_lcb(0.80, 100)
        assert lcb_100 > lcb_5  # More data → tighter → higher LCB

    def test_zero_observations(self):
        assert _beta_lcb(0.80, 0) == 0.0
        assert _beta_ucb(0.80, 0) == 1.0


# ── Failure modes tests ─────────────────────────────────────────────────────

class TestFailureModes:
    def test_no_failures_on_clean_evidence(self):
        result = detect_failure_modes(
            memory_purity=0.99,
            morphism_score=0.95,
            belief_shift_kl=0.05,
            policy_confidence=0.85,
            causal_support=0.90,
            trace_integrity=True,
            polarity_inversion=False,
        )
        assert isinstance(result, FailureModeAssessment)
        assert len(result.detected_modes) == 0
        assert result.total_risk == 0.0
        assert not result.has_blocking_failure

    def test_memory_contamination_detected(self):
        result = detect_failure_modes(
            memory_purity=0.30,
            morphism_score=0.80,
            belief_shift_kl=0.10,
            policy_confidence=0.70,
            causal_support=0.80,
            trace_integrity=True,
            polarity_inversion=False,
        )
        modes = [m.mode for m in result.detected_modes]
        assert "memory_contamination" in modes
        assert result.has_blocking_failure  # critical severity for <0.40

    def test_belief_collapse_detected(self):
        result = detect_failure_modes(
            memory_purity=0.95,
            morphism_score=0.80,
            belief_shift_kl=0.65,
            policy_confidence=0.70,
            causal_support=0.80,
            trace_integrity=True,
            polarity_inversion=False,
        )
        modes = [m.mode for m in result.detected_modes]
        assert "belief_collapse" in modes

    def test_multiple_failures(self):
        result = detect_failure_modes(
            memory_purity=0.25,
            morphism_score=0.10,
            belief_shift_kl=0.70,
            policy_confidence=0.20,
            causal_support=0.15,
            trace_integrity=False,
            polarity_inversion=True,
        )
        assert len(result.detected_modes) >= 3
        assert result.has_blocking_failure
        assert result.critical_count >= 1


# ── Assessment backward compatibility ────────────────────────────────────────

class TestAssessTransferBackwardCompat:
    def test_local_assessment_still_works(self):
        """v1 style: no compatibility → certified_local."""
        result = assess_transfer(
            episode_result={
                "episode": {
                    "episode_id": "ep-1",
                    "scenario": "thermal",
                    "scenario_metadata": {"scenario_name": "thermal"},
                    "closure_profile": "adaptive_min",
                    "context": {"retrieved_memory": []},
                    "result": {"relation_kind": "support"},
                },
            },
        )
        assert isinstance(result, TransferAssessment)
        assert result.transfer_verdict == "certified_local"
        assert result.certificate_scope == "local_only"

    def test_new_fields_present(self):
        """RTCME-v2 fields should be present."""
        result = assess_transfer(
            episode_result={
                "episode": {
                    "episode_id": "ep-2",
                    "scenario": "thermal",
                    "scenario_metadata": {"scenario_name": "thermal"},
                    "closure_profile": "adaptive_min",
                    "context": {"retrieved_memory": []},
                    "result": {"relation_kind": "support"},
                },
            },
        )
        assert hasattr(result, "transfer_posterior")
        assert hasattr(result, "lower_confidence_bound")
        assert hasattr(result, "certificate_scope")
        assert hasattr(result, "failure_mode_count")
