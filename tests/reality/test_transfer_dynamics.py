"""Tests para transfer dynamics (RTCME-v2 Program 2b).

Valida la integración entre belief state, transfer dynamics
y la extensión a transition_analysis.py con morphism_score.
"""

import pytest

from runtime.reality.belief_state import BeliefState, build_belief_state
from runtime.reality.transfer_dynamics import (
    compute_transfer_stability,
    compute_hysteresis,
    compute_recovery_profile,
)
from runtime.reality.transition_analysis import (
    compute_transition_vector,
    build_continuity_tensor,
)
from runtime.world.compatibility import (
    CompatibilityAssessment,
    ScenarioCompatibilityGraph,
    ScenarioStructuralProfile,
)
from runtime.world.thermal_scenario import ThermalScenario
from runtime.world.resource_scenario import ResourceScenario


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_result(scenario="thermal_homeostasis", main_var="temperature", main_val=0.85, alarm=True):
    return {
        "episode": {
            "episode_id": "ep-dyn-001",
            "scenario": scenario,
            "scenario_metadata": {"scenario_name": scenario, "main_variable": main_var},
            "closure_profile": "adaptive_min",
            "context": {
                "observation": {main_var: main_val, "alarm": alarm, "propositions": ["PROP_A"]},
                "intervention": "activate_cooling",
                "counterfactual": {main_var: main_val + 0.05},
                "retrieved_memory": [],
            },
            "result": {
                "updated_world": {main_var: main_val - 0.05},
                "relation_kind": "support",
                "reasoning_sequence": ["observe", "decide"],
            },
            "trace": [],
        },
        "smg_snapshot": {"signs": [{"proposition": "PROP_A"}], "observations": [], "relations": []},
        "belief_state": {
            "prior": None,
            "posterior": {
                "scenario_name": scenario,
                "episode_id": "ep-dyn-001",
                "main_variable_estimate": main_val,
                "alarm_probability": 0.9 if alarm else 0.1,
                "policy_confidence": 0.7,
                "causal_support_confidence": 0.9,
                "trace_confidence": 0.8,
                "memory_purity_confidence": 1.0,
            },
        },
    }


# ── Tests ────────────────────────────────────────────────────────────────────

class TestTransitionVectorWithMorphism:
    def test_morphism_score_overrides_structural(self):
        """When morphism_score is provided, it should be used for structural_compatibility."""
        prev = _make_result()
        curr = _make_result()
        t = ThermalScenario()
        graph = ScenarioCompatibilityGraph()
        compat = graph.assess(t.structural_profile, t.structural_profile)

        vec_without = compute_transition_vector(
            previous_result=prev,
            current_result=curr,
            compatibility=compat,
        )
        vec_with = compute_transition_vector(
            previous_result=prev,
            current_result=curr,
            compatibility=compat,
            morphism_score=0.3,
        )
        # morphism_score=0.3 should lower structural_compatibility
        assert vec_with.structural_compatibility == 0.3
        assert vec_without.structural_compatibility >= 0.95

    def test_belief_state_enhances_causal(self):
        """Belief state posterior should enhance causal stability."""
        prev = _make_result()
        curr = _make_result()
        t = ThermalScenario()
        graph = ScenarioCompatibilityGraph()
        compat = graph.assess(t.structural_profile, t.structural_profile)

        vec = compute_transition_vector(
            previous_result=prev,
            current_result=curr,
            compatibility=compat,
        )
        # With belief enhancement, causal should be blended
        assert vec.causal_stability > 0.0

    def test_tensor_still_works_with_morphism(self):
        """build_continuity_tensor should work with vectors from morphism-enhanced computation."""
        prev = _make_result()
        curr = _make_result()
        t = ThermalScenario()
        graph = ScenarioCompatibilityGraph()
        compat = graph.assess(t.structural_profile, t.structural_profile)

        vec = compute_transition_vector(
            previous_result=prev,
            current_result=curr,
            compatibility=compat,
            morphism_score=0.5,
        )
        tensor = build_continuity_tensor(vectors=[vec])
        assert "thermal_homeostasis" in tensor
        assert "thermal_homeostasis" in tensor["thermal_homeostasis"]


class TestTransferDynamicsIntegration:
    def test_stability_uses_belief_and_evidence(self):
        prior = BeliefState(
            scenario_name="thermal_homeostasis",
            episode_id="ep-1",
            main_variable_estimate=0.85,
            alarm_probability=0.9,
            policy_confidence=0.7,
            causal_support_confidence=0.9,
            trace_confidence=0.8,
            memory_purity_confidence=1.0,
        )
        posterior = BeliefState(
            scenario_name="resource_management",
            episode_id="ep-2",
            main_variable_estimate=0.15,
            alarm_probability=0.8,
            policy_confidence=0.6,
            causal_support_confidence=0.7,
            trace_confidence=0.7,
            memory_purity_confidence=0.9,
        )
        result = compute_transfer_stability(
            prior=prior,
            posterior=posterior,
            morphism_score=0.5,
        )
        assert 0.0 <= result.transfer_stability <= 1.0
        assert result.evidence_score > 0.0
        assert result.belief_stability_score > 0.0

    def test_hysteresis_integration(self):
        a = BeliefState(
            scenario_name="thermal", episode_id="ep-a",
            main_variable_estimate=0.85, alarm_probability=0.9,
            policy_confidence=0.7, causal_support_confidence=0.9,
            trace_confidence=0.8, memory_purity_confidence=1.0,
        )
        b = BeliefState(
            scenario_name="resource", episode_id="ep-b",
            main_variable_estimate=0.15, alarm_probability=0.8,
            policy_confidence=0.5, causal_support_confidence=0.6,
            trace_confidence=0.6, memory_purity_confidence=0.8,
        )
        a_return = BeliefState(
            scenario_name="thermal", episode_id="ep-c",
            main_variable_estimate=0.80, alarm_probability=0.85,
            policy_confidence=0.65, causal_support_confidence=0.85,
            trace_confidence=0.75, memory_purity_confidence=0.95,
        )
        result = compute_hysteresis(initial=a, after_transfer=b, after_return=a_return)
        assert result.hysteresis_gap > 0.0  # Not perfectly recovered
        assert result.round_trip_loss >= 0.0
