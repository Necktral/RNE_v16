"""Tests para el grafo de compatibilidad de escenarios cognitivos."""

from runtime.world.compatibility import (
    CompatibilityAssessment,
    ScenarioCompatibilityGraph,
    ScenarioStructuralProfile,
)
from runtime.world.thermal_scenario import ThermalScenario
from runtime.world.resource_scenario import ResourceScenario


def _thermal_profile() -> ScenarioStructuralProfile:
    return ThermalScenario().structural_profile


def _resource_profile() -> ScenarioStructuralProfile:
    return ResourceScenario().structural_profile


def _incompatible_profile() -> ScenarioStructuralProfile:
    return ScenarioStructuralProfile(
        scenario_name="incompatible_fake",
        scenario_version="0.0",
        scenario_config_hash="000000000000",
        control_topology="unknown",
        optimization_direction="target_band",
        intervention_semantics=("nuke",),
        counterfactual_policy="none",
        relation_polarity="contextual",
        main_variable="chaos_index",
    )


class TestScenarioCompatibilityGraph:
    """Suite de tests para ScenarioCompatibilityGraph."""

    def test_thermal_self_is_equivalent(self):
        graph = ScenarioCompatibilityGraph()
        thermal = _thermal_profile()
        result = graph.assess(thermal, thermal)
        assert result.compatibility_class == "equivalent"
        assert result.overall_score >= 0.95
        assert result.penalty_multiplier == 1.0
        assert result.transfer_allowed is True
        assert result.certification_allowed is True

    def test_resource_self_is_equivalent(self):
        graph = ScenarioCompatibilityGraph()
        resource = _resource_profile()
        result = graph.assess(resource, resource)
        assert result.compatibility_class == "equivalent"
        assert result.overall_score >= 0.95

    def test_thermal_to_resource_is_analogical(self):
        """thermal↔resource debe ser analogical: comparten estructura de
        control de umbral pero difieren en dirección y polaridad."""
        graph = ScenarioCompatibilityGraph()
        thermal = _thermal_profile()
        resource = _resource_profile()
        result = graph.assess(thermal, resource)
        assert result.compatibility_class == "analogical"
        assert 0.45 <= result.overall_score < 0.75
        assert result.transfer_allowed is True
        assert result.certification_allowed is False
        assert result.penalty_multiplier == 0.50

    def test_resource_to_thermal_is_analogical(self):
        """Verificar simetría: resource→thermal debe dar misma clase."""
        graph = ScenarioCompatibilityGraph()
        thermal = _thermal_profile()
        resource = _resource_profile()
        r1 = graph.assess(thermal, resource)
        r2 = graph.assess(resource, thermal)
        assert r1.compatibility_class == r2.compatibility_class
        assert r1.overall_score == r2.overall_score

    def test_incompatible_not_transferable(self):
        graph = ScenarioCompatibilityGraph()
        thermal = _thermal_profile()
        fake = _incompatible_profile()
        result = graph.assess(thermal, fake)
        assert result.compatibility_class == "incompatible"
        assert result.overall_score < 0.45
        assert result.transfer_allowed is False
        assert result.certification_allowed is False
        assert result.penalty_multiplier == 0.0

    def test_matrix_2x2_complete(self):
        graph = ScenarioCompatibilityGraph()
        profiles = [_thermal_profile(), _resource_profile()]
        mat = graph.matrix(profiles)
        assert "thermal_homeostasis" in mat
        assert "resource_management" in mat
        # Diagonal: equivalent
        assert mat["thermal_homeostasis"]["thermal_homeostasis"].compatibility_class == "equivalent"
        assert mat["resource_management"]["resource_management"].compatibility_class == "equivalent"
        # Off-diagonal: analogical
        assert mat["thermal_homeostasis"]["resource_management"].compatibility_class == "analogical"
        assert mat["resource_management"]["thermal_homeostasis"].compatibility_class == "analogical"

    def test_matrix_symmetry_for_symmetric_scoring(self):
        """La función de scoring actual es simétrica: f(a,b) == f(b,a).
        Documentar explícitamente."""
        graph = ScenarioCompatibilityGraph()
        profiles = [_thermal_profile(), _resource_profile()]
        mat = graph.matrix(profiles)
        a_to_b = mat["thermal_homeostasis"]["resource_management"]
        b_to_a = mat["resource_management"]["thermal_homeostasis"]
        assert a_to_b.overall_score == b_to_a.overall_score
        assert a_to_b.compatibility_class == b_to_a.compatibility_class

    def test_matrix_3x3_with_incompatible(self):
        graph = ScenarioCompatibilityGraph()
        profiles = [_thermal_profile(), _resource_profile(), _incompatible_profile()]
        mat = graph.matrix(profiles)
        assert len(mat) == 3
        assert mat["incompatible_fake"]["incompatible_fake"].compatibility_class == "equivalent"
        assert mat["thermal_homeostasis"]["incompatible_fake"].compatibility_class == "incompatible"
        assert mat["resource_management"]["incompatible_fake"].compatibility_class == "incompatible"

    def test_assessment_scores_are_bounded(self):
        graph = ScenarioCompatibilityGraph()
        thermal = _thermal_profile()
        resource = _resource_profile()
        result = graph.assess(thermal, resource)
        for score in [
            result.topology_score,
            result.objective_score,
            result.intervention_score,
            result.counterfactual_score,
            result.overall_score,
        ]:
            assert 0.0 <= score <= 1.0

    def test_structural_profiles_from_scenarios(self):
        """Verificar que los escenarios concretos exponen perfiles válidos."""
        thermal = ThermalScenario()
        resource = ResourceScenario()
        tp = thermal.structural_profile
        rp = resource.structural_profile
        assert tp.scenario_name == "thermal_homeostasis"
        assert tp.optimization_direction == "minimize"
        assert tp.relation_polarity == "lower_is_better"
        assert rp.scenario_name == "resource_management"
        assert rp.optimization_direction == "maximize"
        assert rp.relation_polarity == "higher_is_better"
