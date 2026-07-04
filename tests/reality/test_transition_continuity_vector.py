"""Tests para TransitionContinuityVector."""

from runtime.reality.transition_analysis import (
    TransitionContinuityVector,
    compute_transition_vector,
)
from runtime.world.compatibility import (
    CompatibilityAssessment,
    ScenarioCompatibilityGraph,
    ScenarioStructuralProfile,
)
from runtime.world.thermal_scenario import ThermalScenario
from runtime.world.resource_scenario import ResourceScenario


def _thermal_profile():
    return ThermalScenario().structural_profile


def _resource_profile():
    return ResourceScenario().structural_profile


def _make_episode(
    scenario_name: str,
    main_variable: str,
    main_value: float,
    alarm: bool,
    propositions: list,
    intervention: str,
    relation_kind: str = "support",
    reasoning_sequence: list | None = None,
    counterfactual_value: float | None = None,
) -> dict:
    """Construye un resultado de episodio mínimo para testing."""
    cf_val = counterfactual_value if counterfactual_value is not None else main_value + 0.05
    return {
        "episode": {
            "episode_id": f"ep-{scenario_name}-test",
            "scenario": scenario_name,
            "scenario_metadata": {
                "scenario_name": scenario_name,
                "main_variable": main_variable,
            },
            "context": {
                "observation": {
                    main_variable: main_value,
                    "alarm": alarm,
                    "propositions": propositions,
                },
                "intervention": intervention,
                "counterfactual": {main_variable: cf_val},
            },
            "result": {
                "updated_world": {main_variable: main_value - 0.05},
                "relation_kind": relation_kind,
                "reasoning_sequence": reasoning_sequence or ["ABD", "ANA", "CAU", "CTF", "DED", "PROB"],
            },
        },
    }


class TestTransitionContinuityVector:
    def test_thermal_to_thermal_intra_high(self):
        """thermal→thermal debe producir continuidad intra alta."""
        graph = ScenarioCompatibilityGraph()
        tp = _thermal_profile()
        compat = graph.assess(tp, tp)
        prev = _make_episode(
            "thermal_homeostasis", "temperature", 0.88, True,
            ["TEMP_HIGH", "COOLING_ACTIVE"], "activate_cooling",
        )
        curr = _make_episode(
            "thermal_homeostasis", "temperature", 0.86, True,
            ["TEMP_HIGH", "COOLING_ACTIVE"], "activate_cooling",
        )
        vec = compute_transition_vector(
            previous_result=prev,
            current_result=curr,
            compatibility=compat,
        )
        assert isinstance(vec, TransitionContinuityVector)
        assert vec.transition_type == "intra"
        assert vec.composite_score >= 0.7
        assert vec.semantic_retention >= 0.5
        assert vec.structural_compatibility >= 0.95

    def test_thermal_to_resource_produces_non_trivial_vector(self):
        """thermal→resource no debe ser un escalar ciego."""
        graph = ScenarioCompatibilityGraph()
        tp = _thermal_profile()
        rp = _resource_profile()
        compat = graph.assess(tp, rp)
        prev = _make_episode(
            "thermal_homeostasis", "temperature", 0.88, True,
            ["TEMP_HIGH"], "activate_cooling",
        )
        curr = _make_episode(
            "resource_management", "stock_level", 0.15, True,
            ["STOCK_LOW"], "start_production",
        )
        vec = compute_transition_vector(
            previous_result=prev,
            current_result=curr,
            compatibility=compat,
        )
        assert vec.transition_type == "analogical"
        # Must be intermediate, not zero or one
        assert 0.1 < vec.composite_score < 0.9
        # Semantic retention should be low (different propositions)
        assert vec.semantic_retention < 0.5
        # Structural compatibility reflects the graph
        assert vec.structural_compatibility == compat.overall_score

    def test_incompatible_penalizes_structural(self):
        """incompatible→otro castiga structural_compatibility."""
        graph = ScenarioCompatibilityGraph()
        tp = _thermal_profile()
        fake = ScenarioStructuralProfile(
            scenario_name="incompatible",
            scenario_version="0.0",
            scenario_config_hash="000000000000",
            control_topology="unknown",
            optimization_direction="target_band",
            intervention_semantics=("nuke",),
            counterfactual_policy="none",
            relation_polarity="contextual",
            main_variable="chaos",
        )
        compat = graph.assess(tp, fake)
        prev = _make_episode(
            "thermal_homeostasis", "temperature", 0.88, True,
            ["TEMP_HIGH"], "activate_cooling",
        )
        curr = _make_episode(
            "incompatible", "chaos", 0.50, False,
            ["CHAOS"], "nuke", relation_kind="contradiction",
        )
        vec = compute_transition_vector(
            previous_result=prev,
            current_result=curr,
            compatibility=compat,
        )
        assert vec.transition_type == "incompatible"
        assert vec.structural_compatibility < 0.45
        assert vec.composite_score < 0.5

    def test_contamination_reduces_memory_purity(self):
        """Contaminación cross-scenario reduce memory_purity."""
        graph = ScenarioCompatibilityGraph()
        tp = _thermal_profile()
        compat = graph.assess(tp, tp)
        prev = _make_episode(
            "thermal_homeostasis", "temperature", 0.88, True,
            ["TEMP_HIGH"], "activate_cooling",
        )
        curr = _make_episode(
            "thermal_homeostasis", "temperature", 0.86, True,
            ["TEMP_HIGH"], "activate_cooling",
        )
        # Sin contaminación
        vec_clean = compute_transition_vector(
            previous_result=prev,
            current_result=curr,
            compatibility=compat,
            retrieval_metrics={"retrieved_same_scenario_count": 5, "retrieved_cross_scenario_count": 0},
        )
        # Con contaminación
        vec_dirty = compute_transition_vector(
            previous_result=prev,
            current_result=curr,
            compatibility=compat,
            retrieval_metrics={"retrieved_same_scenario_count": 2, "retrieved_cross_scenario_count": 3},
        )
        assert vec_clean.memory_purity > vec_dirty.memory_purity
        assert vec_clean.composite_score > vec_dirty.composite_score

    def test_vector_scores_bounded(self):
        """Todos los scores deben estar en [0, 1]."""
        graph = ScenarioCompatibilityGraph()
        tp = _thermal_profile()
        compat = graph.assess(tp, tp)
        prev = _make_episode(
            "thermal_homeostasis", "temperature", 0.88, True,
            ["TEMP_HIGH"], "activate_cooling",
        )
        curr = _make_episode(
            "thermal_homeostasis", "temperature", 0.86, True,
            ["TEMP_HIGH"], "activate_cooling",
        )
        vec = compute_transition_vector(
            previous_result=prev,
            current_result=curr,
            compatibility=compat,
        )
        for val in [
            vec.semantic_retention,
            vec.trace_stability,
            vec.causal_stability,
            vec.intervention_policy_stability,
            vec.structural_compatibility,
            vec.memory_purity,
            vec.composite_score,
        ]:
            assert 0.0 <= val <= 1.0
