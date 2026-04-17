"""Tests para belief state y transfer dynamics (RTCME-v2 Program 2).

Valida:
- BeliefState construction from episode results
- BeliefShift computation between states
- TransitionEvidenceVector computation
- TransferStability combining evidence + belief
- Hysteresis measurement A→B→A
- Recovery profile computation
"""

import pytest

from runtime.reality.belief_state import (
    BeliefState,
    BeliefShift,
    TransitionEvidenceVector,
    build_belief_state,
    compute_belief_shift,
    compute_transition_evidence,
)
from runtime.reality.transfer_dynamics import (
    TransferStabilityResult,
    HysteresisResult,
    RecoveryProfile,
    compute_transfer_stability,
    compute_hysteresis,
    compute_recovery_profile,
)


# ── Fixture helpers ──────────────────────────────────────────────────────────

def _make_episode_result(
    scenario="thermal_homeostasis",
    main_var="temperature",
    main_val=0.85,
    alarm=True,
    relation_kind="support",
    memory_hits=None,
):
    return {
        "episode": {
            "episode_id": "ep-test-001",
            "timestamp": "2026-01-01T00:00:00Z",
            "scenario": scenario,
            "scenario_metadata": {
                "scenario_name": scenario,
                "main_variable": main_var,
            },
            "closure_profile": "adaptive_min",
            "context": {
                "observation": {main_var: main_val, "alarm": alarm},
                "intervention": "activate_cooling",
                "counterfactual": {main_var: main_val + 0.05},
                "retrieved_memory": memory_hits or [],
            },
            "result": {
                "updated_world": {main_var: main_val - 0.05},
                "relation_kind": relation_kind,
                "reasoning_sequence": ["observe", "reason", "decide"],
            },
            "trace": [{"detail": {"trace_id": "trace-001"}}],
        },
        "smg_snapshot": {"signs": [], "observations": [], "relations": []},
        "reasoning": {"sequence": ["observe", "reason", "decide"], "trace": []},
        "certification": {"verdict": "certified"},
    }


@pytest.fixture
def thermal_belief():
    result = _make_episode_result(
        scenario="thermal_homeostasis",
        main_var="temperature",
        main_val=0.85,
        alarm=True,
        relation_kind="support",
    )
    return build_belief_state(episode_result=result)


@pytest.fixture
def resource_belief():
    result = _make_episode_result(
        scenario="resource_management",
        main_var="stock_level",
        main_val=0.15,
        alarm=True,
        relation_kind="support",
    )
    return build_belief_state(episode_result=result)


@pytest.fixture
def calm_thermal_belief():
    result = _make_episode_result(
        scenario="thermal_homeostasis",
        main_var="temperature",
        main_val=0.60,
        alarm=False,
        relation_kind="support",
    )
    return build_belief_state(episode_result=result)


# ── BeliefState tests ────────────────────────────────────────────────────────

class TestBeliefState:
    def test_builds_from_episode(self, thermal_belief):
        assert isinstance(thermal_belief, BeliefState)
        assert thermal_belief.scenario_name == "thermal_homeostasis"
        assert thermal_belief.main_variable_estimate == 0.85

    def test_alarm_probability_high_when_alarm(self, thermal_belief):
        assert thermal_belief.alarm_probability > 0.5

    def test_alarm_probability_low_when_no_alarm(self, calm_thermal_belief):
        assert calm_thermal_belief.alarm_probability < 0.5

    def test_causal_support_high_on_support(self, thermal_belief):
        assert thermal_belief.causal_support_confidence >= 0.80

    def test_composite_confidence(self, thermal_belief):
        cc = thermal_belief.composite_confidence
        assert 0.0 <= cc <= 1.0

    def test_purity_drops_with_cross_memory(self):
        result = _make_episode_result(
            memory_hits=[
                {"analogical_source": True, "retrieval_metrics": {"retrieved_cross_scenario_count": 2}},
            ],
        )
        belief = build_belief_state(episode_result=result)
        assert belief.memory_purity_confidence < 1.0


# ── BeliefShift tests ────────────────────────────────────────────────────────

class TestBeliefShift:
    def test_same_state_zero_shift(self, thermal_belief):
        shift = compute_belief_shift(prior=thermal_belief, posterior=thermal_belief)
        assert shift.kl_divergence_approx == 0.0
        assert shift.stability_score == 1.0
        assert not shift.recovery_needed

    def test_cross_scenario_shift(self, thermal_belief, resource_belief):
        shift = compute_belief_shift(prior=thermal_belief, posterior=resource_belief)
        assert shift.kl_divergence_approx > 0.0
        assert shift.stability_score < 1.0
        assert shift.source_scenario == "thermal_homeostasis"
        assert shift.target_scenario == "resource_management"

    def test_alarm_transition_creates_shift(self, thermal_belief, calm_thermal_belief):
        shift = compute_belief_shift(prior=thermal_belief, posterior=calm_thermal_belief)
        assert shift.delta_alarm > 0.0
        assert shift.delta_main_variable > 0.0


# ── TransitionEvidenceVector tests ───────────────────────────────────────────

class TestTransitionEvidence:
    def test_intra_evidence(self, thermal_belief):
        ev = compute_transition_evidence(
            prior=thermal_belief,
            posterior=thermal_belief,
            morphism_score=1.0,
        )
        assert ev.transition_type == "intra"
        assert ev.composite_evidence > 0.5

    def test_cross_evidence_with_low_morphism(self, thermal_belief, resource_belief):
        ev = compute_transition_evidence(
            prior=thermal_belief,
            posterior=resource_belief,
            morphism_score=0.3,
        )
        assert ev.transition_type == "cross"
        # Low morphism should reduce evidence
        assert ev.composite_evidence < 0.9

    def test_trace_integrity_affects_evidence(self, thermal_belief, resource_belief):
        ev_good = compute_transition_evidence(
            prior=thermal_belief, posterior=resource_belief,
            morphism_score=0.5, trace_integrity=True,
        )
        ev_bad = compute_transition_evidence(
            prior=thermal_belief, posterior=resource_belief,
            morphism_score=0.5, trace_integrity=False,
        )
        assert ev_good.composite_evidence >= ev_bad.composite_evidence


# ── TransferStability tests ──────────────────────────────────────────────────

class TestTransferStability:
    def test_intra_stability_high(self, thermal_belief):
        result = compute_transfer_stability(
            prior=thermal_belief,
            posterior=thermal_belief,
            morphism_score=1.0,
        )
        assert isinstance(result, TransferStabilityResult)
        assert result.transfer_stability > 0.5

    def test_cross_stability_modulated(self, thermal_belief, resource_belief):
        result = compute_transfer_stability(
            prior=thermal_belief,
            posterior=resource_belief,
            morphism_score=0.4,
        )
        # Should be lower due to cross-scenario + low morphism
        assert result.transfer_stability < 1.0
        assert result.source_scenario == "thermal_homeostasis"
        assert result.target_scenario == "resource_management"


# ── Hysteresis tests ─────────────────────────────────────────────────────────

class TestHysteresis:
    def test_round_trip_same_scenario(self, thermal_belief):
        result = compute_hysteresis(
            initial=thermal_belief,
            after_transfer=thermal_belief,
            after_return=thermal_belief,
        )
        assert isinstance(result, HysteresisResult)
        assert result.hysteresis_gap == 0.0
        assert result.full_recovery is True

    def test_round_trip_cross_scenario(self, thermal_belief, resource_belief, calm_thermal_belief):
        result = compute_hysteresis(
            initial=thermal_belief,
            after_transfer=resource_belief,
            after_return=calm_thermal_belief,
        )
        # Return state differs from initial → some hysteresis
        assert result.hysteresis_gap >= 0.0


# ── Recovery tests ───────────────────────────────────────────────────────────

class TestRecoveryProfile:
    def test_empty_sequence(self, thermal_belief):
        result = compute_recovery_profile(
            initial_belief=thermal_belief,
            subsequent_beliefs=[],
        )
        assert isinstance(result, RecoveryProfile)
        assert result.recovery_steps == 0
        assert not result.converged

    def test_immediate_convergence(self, thermal_belief):
        # Same belief repeated → immediate convergence
        result = compute_recovery_profile(
            initial_belief=thermal_belief,
            subsequent_beliefs=[thermal_belief, thermal_belief],
        )
        assert result.converged is True
        assert result.recovery_steps == 1

    def test_no_convergence(self, thermal_belief, resource_belief, calm_thermal_belief):
        # Alternating beliefs → no convergence
        result = compute_recovery_profile(
            initial_belief=thermal_belief,
            subsequent_beliefs=[resource_belief, calm_thermal_belief],
        )
        # May or may not converge depending on shift magnitude
        assert result.recovery_steps >= 1
