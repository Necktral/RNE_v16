"""Tests para transition stress y edge benchmark (RTCME-v2 Program 4).

Valida:
- EdgeStressResult computation from belief states
- Edge classification by morphism class
- Recovery, hysteresis, and drift metrics
- Graph summary computation
"""

import pytest

from runtime.reality.belief_state import BeliefState
from runtime.reality.transition_stress import (
    EdgeStressResult,
    classify_edge,
    run_edge_stress_test,
)
from runtime.reality.edge_benchmark import _build_graph_summary


# ── Fixtures ─────────────────────────────────────────────────────────────────

def _make_belief(scenario, val=0.85, alarm=0.9, policy=0.7, causal=0.9, trace=0.8, purity=1.0):
    return BeliefState(
        scenario_name=scenario,
        episode_id=f"ep-{scenario}",
        main_variable_estimate=val,
        alarm_probability=alarm,
        policy_confidence=policy,
        causal_support_confidence=causal,
        trace_confidence=trace,
        memory_purity_confidence=purity,
    )


# ── Edge classification tests ────────────────────────────────────────────────

class TestEdgeClassification:
    def test_isomorphic_is_equivalent(self):
        assert classify_edge(morphism_class="isomorphic", morphism_score=0.98) == "equivalent_edge"

    def test_homomorphic_high_is_compatible(self):
        assert classify_edge(morphism_class="homomorphic", morphism_score=0.80) == "compatible_edge"

    def test_homomorphic_low_is_analogical(self):
        assert classify_edge(morphism_class="homomorphic", morphism_score=0.50) == "analogical_edge"

    def test_analogical_is_analogical(self):
        assert classify_edge(morphism_class="analogical", morphism_score=0.45) == "analogical_edge"

    def test_adversarial_is_adversarial(self):
        assert classify_edge(morphism_class="adversarial", morphism_score=0.10) == "adversarial_edge"

    def test_incompatible_is_adversarial(self):
        assert classify_edge(morphism_class="incompatible", morphism_score=0.05) == "adversarial_edge"


# ── Stress test computation ──────────────────────────────────────────────────

class TestEdgeStressTest:
    def test_self_edge_high_stability(self):
        """Same scenario → high stability, no hysteresis."""
        beliefs = [_make_belief("thermal")] * 3
        result = run_edge_stress_test(
            source_scenario="thermal",
            target_scenario="thermal",
            morphism_class="isomorphic",
            morphism_score=0.98,
            warmup_beliefs=beliefs,
            probe_beliefs=beliefs,
            return_beliefs=beliefs,
        )
        assert isinstance(result, EdgeStressResult)
        assert result.edge_class == "equivalent_edge"
        assert result.transfer_stability_mean > 0.5
        assert result.hysteresis_gap == 0.0

    def test_cross_edge_has_drift(self):
        """Different scenarios → measurable drift."""
        warmup = [_make_belief("thermal", val=0.85)]
        probe = [_make_belief("resource", val=0.15, alarm=0.8, policy=0.5, causal=0.6)]
        result = run_edge_stress_test(
            source_scenario="thermal",
            target_scenario="resource",
            morphism_class="analogical",
            morphism_score=0.40,
            warmup_beliefs=warmup,
            probe_beliefs=probe,
        )
        assert result.belief_drift > 0.0
        assert result.policy_drift > 0.0

    def test_hysteresis_measurement(self):
        """Round trip A→B→A should measure hysteresis."""
        warmup = [_make_belief("thermal", val=0.85)]
        probe = [_make_belief("resource", val=0.15, alarm=0.8, policy=0.5)]
        return_beliefs = [_make_belief("thermal", val=0.80, policy=0.65)]

        result = run_edge_stress_test(
            source_scenario="thermal",
            target_scenario="resource",
            morphism_class="analogical",
            morphism_score=0.40,
            warmup_beliefs=warmup,
            probe_beliefs=probe,
            return_beliefs=return_beliefs,
        )
        assert result.hysteresis_gap >= 0.0
        assert result.round_trip_loss >= 0.0

    def test_posterior_aggregation(self):
        """Transfer posteriors should be averaged."""
        warmup = [_make_belief("thermal")]
        probe = [_make_belief("resource", val=0.15)]

        result = run_edge_stress_test(
            source_scenario="thermal",
            target_scenario="resource",
            morphism_class="analogical",
            morphism_score=0.40,
            warmup_beliefs=warmup,
            probe_beliefs=probe,
            transfer_posteriors=[0.60, 0.70, 0.65],
        )
        assert abs(result.transfer_posterior_mean - 0.65) < 0.01

    def test_failure_mode_aggregation(self):
        """Failure mode counts should be summed across episodes."""
        warmup = [_make_belief("thermal")]
        probe = [_make_belief("resource")]

        result = run_edge_stress_test(
            source_scenario="thermal",
            target_scenario="resource",
            morphism_class="analogical",
            morphism_score=0.40,
            warmup_beliefs=warmup,
            probe_beliefs=probe,
            failure_mode_lists=[
                {"memory_contamination": 1, "policy_drift": 1},
                {"memory_contamination": 1},
            ],
        )
        assert result.failure_mode_counts["memory_contamination"] == 2
        assert result.failure_mode_counts["policy_drift"] == 1


# ── Graph summary tests ─────────────────────────────────────────────────────

class TestGraphSummary:
    def test_empty_graph(self):
        summary = _build_graph_summary([])
        assert summary["total_edges"] == 0

    def test_single_edge(self):
        warmup = [_make_belief("thermal")]
        probe = [_make_belief("thermal")]
        edge = run_edge_stress_test(
            source_scenario="thermal",
            target_scenario="thermal",
            morphism_class="isomorphic",
            morphism_score=0.98,
            warmup_beliefs=warmup,
            probe_beliefs=probe,
        )
        summary = _build_graph_summary([edge])
        assert summary["total_edges"] == 1
        assert "equivalent_edge" in summary["edge_class_distribution"]
        assert summary["global_mean_stability"] > 0.0

    def test_mixed_edges(self):
        warmup_t = [_make_belief("thermal")]
        probe_t = [_make_belief("thermal")]
        warmup_r = [_make_belief("resource", val=0.15)]
        probe_r = [_make_belief("resource", val=0.15)]

        edge1 = run_edge_stress_test(
            source_scenario="thermal", target_scenario="thermal",
            morphism_class="isomorphic", morphism_score=0.98,
            warmup_beliefs=warmup_t, probe_beliefs=probe_t,
        )
        edge2 = run_edge_stress_test(
            source_scenario="thermal", target_scenario="resource",
            morphism_class="analogical", morphism_score=0.40,
            warmup_beliefs=warmup_t, probe_beliefs=probe_r,
        )
        summary = _build_graph_summary([edge1, edge2])
        assert summary["total_edges"] == 2
        assert len(summary["class_summaries"]) >= 1
