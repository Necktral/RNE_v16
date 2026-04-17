"""Tests para el tensor de continuidad (matriz NxN)."""

from runtime.reality.transition_analysis import (
    ContinuityTensorCell,
    TransitionContinuityVector,
    build_continuity_tensor,
    compute_transition_vector,
)
from runtime.world.compatibility import ScenarioCompatibilityGraph
from runtime.world.thermal_scenario import ThermalScenario
from runtime.world.resource_scenario import ResourceScenario


def _make_episode(scenario_name, main_var, main_val, alarm, props, intervention, rk="support"):
    return {
        "episode": {
            "episode_id": f"ep-{scenario_name}",
            "scenario": scenario_name,
            "scenario_metadata": {"scenario_name": scenario_name, "main_variable": main_var},
            "context": {
                "observation": {main_var: main_val, "alarm": alarm, "propositions": props},
                "intervention": intervention,
                "counterfactual": {main_var: main_val + 0.05},
            },
            "result": {
                "updated_world": {main_var: main_val - 0.03},
                "relation_kind": rk,
                "reasoning_sequence": ["ABD", "ANA", "CAU", "CTF", "DED", "PROB"],
            },
        },
    }


class TestContinuityTensor:
    def test_build_tensor_from_vectors(self):
        """Construye tensor a partir de vectores y verifica estructura."""
        graph = ScenarioCompatibilityGraph()
        tp = ThermalScenario().structural_profile
        rp = ResourceScenario().structural_profile

        thermal_ep = _make_episode("thermal_homeostasis", "temperature", 0.88, True, ["TEMP_HIGH"], "activate_cooling")
        resource_ep = _make_episode("resource_management", "stock_level", 0.15, True, ["STOCK_LOW"], "start_production")

        vectors = []
        # thermal → thermal
        compat_tt = graph.assess(tp, tp)
        for _ in range(3):
            v = compute_transition_vector(
                previous_result=thermal_ep, current_result=thermal_ep,
                compatibility=compat_tt,
            )
            vectors.append(v)

        # thermal → resource
        compat_tr = graph.assess(tp, rp)
        for _ in range(2):
            v = compute_transition_vector(
                previous_result=thermal_ep, current_result=resource_ep,
                compatibility=compat_tr,
            )
            vectors.append(v)

        tensor = build_continuity_tensor(vectors=vectors)
        assert "thermal_homeostasis" in tensor
        assert "thermal_homeostasis" in tensor["thermal_homeostasis"]
        assert "resource_management" in tensor["thermal_homeostasis"]

        tt_cell = tensor["thermal_homeostasis"]["thermal_homeostasis"]
        assert isinstance(tt_cell, ContinuityTensorCell)
        assert tt_cell.sample_count == 3
        assert tt_cell.mean_composite >= 0.5

        tr_cell = tensor["thermal_homeostasis"]["resource_management"]
        assert tr_cell.sample_count == 2
        assert tr_cell.mean_composite > 0.0

    def test_tensor_intra_higher_than_cross(self):
        """Intra-scenario mean composite should be > cross-scenario."""
        graph = ScenarioCompatibilityGraph()
        tp = ThermalScenario().structural_profile
        rp = ResourceScenario().structural_profile

        thermal_ep = _make_episode("thermal_homeostasis", "temperature", 0.88, True, ["TEMP_HIGH"], "activate_cooling")
        resource_ep = _make_episode("resource_management", "stock_level", 0.15, True, ["STOCK_LOW"], "start_production")

        vectors = []
        compat_tt = graph.assess(tp, tp)
        compat_tr = graph.assess(tp, rp)
        for _ in range(3):
            vectors.append(compute_transition_vector(
                previous_result=thermal_ep, current_result=thermal_ep,
                compatibility=compat_tt,
            ))
            vectors.append(compute_transition_vector(
                previous_result=thermal_ep, current_result=resource_ep,
                compatibility=compat_tr,
            ))

        tensor = build_continuity_tensor(vectors=vectors)
        intra = tensor["thermal_homeostasis"]["thermal_homeostasis"]
        cross = tensor["thermal_homeostasis"]["resource_management"]
        assert intra.mean_composite > cross.mean_composite

    def test_empty_vectors_produce_empty_tensor(self):
        tensor = build_continuity_tensor(vectors=[])
        assert tensor == {}

    def test_single_vector_tensor(self):
        graph = ScenarioCompatibilityGraph()
        tp = ThermalScenario().structural_profile
        compat = graph.assess(tp, tp)
        ep = _make_episode("thermal_homeostasis", "temperature", 0.88, True, ["TEMP_HIGH"], "activate_cooling")
        v = compute_transition_vector(
            previous_result=ep, current_result=ep, compatibility=compat,
        )
        tensor = build_continuity_tensor(vectors=[v])
        cell = tensor["thermal_homeostasis"]["thermal_homeostasis"]
        assert cell.sample_count == 1
        assert cell.mean_composite == v.composite_score
