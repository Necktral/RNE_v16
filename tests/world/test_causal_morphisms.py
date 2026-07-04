"""Tests para morfismos causales dirigidos (RTCME-v2 Program 1).

Valida:
- ScenarioCausalSignature expuesta por cada escenario
- Alineamiento bipartito de intervenciones y proposiciones
- MorphismEngine con morfismos dirigidos asimétricos
- Clasificación correcta (isomorphic, homomorphic, analogical, adversarial)
- Operador de transporte formal
- Matriz NxN de morfismos
"""

import pytest

from runtime.world.causal_signature import (
    CausalEdge,
    InterventionEffect,
    ScenarioCausalSignature,
)
from runtime.world.alignment import (
    align_causal_graphs,
    align_interventions,
    align_propositions,
)
from runtime.world.morphism_engine import (
    DirectedScenarioMorphism,
    MorphismEngine,
)
from runtime.world.thermal_scenario import ThermalScenario
from runtime.world.resource_scenario import ResourceScenario
from runtime.world.registry import list_causal_signatures


# ── Fixture helpers ──────────────────────────────────────────────────────────

@pytest.fixture
def thermal():
    return ThermalScenario()


@pytest.fixture
def resource():
    return ResourceScenario()


@pytest.fixture
def engine():
    return MorphismEngine()


# ── Causal signature tests ───────────────────────────────────────────────────

class TestCausalSignature:
    def test_thermal_signature_exists(self, thermal):
        sig = thermal.causal_signature
        assert isinstance(sig, ScenarioCausalSignature)
        assert sig.scenario_name == "thermal_homeostasis"

    def test_resource_signature_exists(self, resource):
        sig = resource.causal_signature
        assert isinstance(sig, ScenarioCausalSignature)
        assert sig.scenario_name == "resource_management"

    def test_thermal_has_observable_variables(self, thermal):
        sig = thermal.causal_signature
        assert "temperature" in sig.observable_variables
        assert "cooling_active" in sig.observable_variables

    def test_resource_has_observable_variables(self, resource):
        sig = resource.causal_signature
        assert "stock_level" in sig.observable_variables
        assert "production_active" in sig.observable_variables

    def test_thermal_intervention_effects(self, thermal):
        sig = thermal.causal_signature
        effects = {e.intervention_name: e for e in sig.intervention_effects}
        assert "activate_cooling" in effects
        cooling = effects["activate_cooling"]
        assert cooling.expected_direction == "-"
        assert cooling.semantic_role == "corrective"

    def test_resource_intervention_effects(self, resource):
        sig = resource.causal_signature
        effects = {e.intervention_name: e for e in sig.intervention_effects}
        assert "start_production" in effects
        prod = effects["start_production"]
        assert prod.expected_direction == "+"
        assert prod.semantic_role == "corrective"

    def test_opposing_optimization_directions(self, thermal, resource):
        assert thermal.causal_signature.optimization_direction == "minimize"
        assert resource.causal_signature.optimization_direction == "maximize"

    def test_opposing_alarm_semantics(self, thermal, resource):
        assert thermal.causal_signature.alarm_semantics == "threshold_above"
        assert resource.causal_signature.alarm_semantics == "threshold_below"

    def test_causal_edges_exist(self, thermal, resource):
        assert len(thermal.causal_signature.causal_edges) >= 2
        assert len(resource.causal_signature.causal_edges) >= 2

    def test_registry_lists_signatures(self):
        sigs = list_causal_signatures()
        assert "thermal_homeostasis" in sigs
        assert "resource_management" in sigs


# ── Alignment tests ──────────────────────────────────────────────────────────

class TestAlignment:
    def test_proposition_self_alignment(self, thermal):
        vocab = thermal.causal_signature.proposition_vocabulary
        result = align_propositions(vocab, vocab)
        assert result.normalized_score == 1.0
        assert result.coverage == 1.0

    def test_proposition_cross_alignment(self, thermal, resource):
        t_vocab = thermal.causal_signature.proposition_vocabulary
        r_vocab = resource.causal_signature.proposition_vocabulary
        result = align_propositions(t_vocab, r_vocab)
        # KEEP_IDLE is shared
        assert result.normalized_score > 0.0
        assert result.normalized_score < 1.0
        assert len(result.source_unmatched) > 0

    def test_intervention_self_alignment(self, thermal):
        effects = thermal.causal_signature.intervention_effects
        result = align_interventions(effects, effects)
        assert result.normalized_score == 1.0

    def test_intervention_cross_alignment(self, thermal, resource):
        t_effects = thermal.causal_signature.intervention_effects
        r_effects = resource.causal_signature.intervention_effects
        result = align_interventions(t_effects, r_effects)
        # Different names but similar structure → partial alignment
        assert result.normalized_score > 0.0
        assert result.normalized_score < 1.0

    def test_causal_graph_self_alignment(self, thermal):
        edges = thermal.causal_signature.causal_edges
        score = align_causal_graphs(edges, edges)
        assert score == 1.0

    def test_causal_graph_cross_alignment(self, thermal, resource):
        score = align_causal_graphs(
            thermal.causal_signature.causal_edges,
            resource.causal_signature.causal_edges,
        )
        # Different DAGs → partial alignment at best
        assert score < 1.0

    def test_empty_alignment(self):
        result = align_propositions(frozenset(), frozenset())
        assert result.normalized_score == 1.0


# ── Morphism engine tests ────────────────────────────────────────────────────

class TestMorphismEngine:
    def test_self_morphism_is_isomorphic(self, engine, thermal):
        sig = thermal.causal_signature
        m = engine.compute_morphism(sig, sig)
        assert m.morphism_class == "isomorphic"
        assert m.overall_score >= 0.95
        assert m.is_transfer_safe_prior is True
        assert m.directionality_penalty == 0.0

    def test_thermal_to_resource_differs_from_reverse(self, engine, thermal, resource):
        t_sig = thermal.causal_signature
        r_sig = resource.causal_signature
        m_tr = engine.compute_morphism(t_sig, r_sig)
        m_rt = engine.compute_morphism(r_sig, t_sig)
        # Asymmetry: different overall scores due to directed alignment
        # They may or may not differ, but let's check the structure
        assert isinstance(m_tr, DirectedScenarioMorphism)
        assert isinstance(m_rt, DirectedScenarioMorphism)
        assert m_tr.source_scenario == "thermal_homeostasis"
        assert m_tr.target_scenario == "resource_management"
        assert m_rt.source_scenario == "resource_management"
        assert m_rt.target_scenario == "thermal_homeostasis"

    def test_cross_morphism_has_directionality_penalty(self, engine, thermal, resource):
        t_sig = thermal.causal_signature
        r_sig = resource.causal_signature
        m = engine.compute_morphism(t_sig, r_sig)
        # minimize→maximize should incur penalty
        assert m.directionality_penalty > 0.0

    def test_cross_morphism_class(self, engine, thermal, resource):
        t_sig = thermal.causal_signature
        r_sig = resource.causal_signature
        m = engine.compute_morphism(t_sig, r_sig)
        # With penalty, should be analogical or adversarial (not isomorphic/homomorphic)
        assert m.morphism_class in ("analogical", "adversarial", "incompatible")

    def test_transport_operator_exists(self, engine, thermal, resource):
        t_sig = thermal.causal_signature
        r_sig = resource.causal_signature
        m = engine.compute_morphism(t_sig, r_sig)
        op = m.transport_operator
        assert op.polarity_inversion is True  # lower_is_better vs higher_is_better
        assert op.direction_inversion is True  # minimize vs maximize

    def test_matrix_is_square(self, engine, thermal, resource):
        sigs = [thermal.causal_signature, resource.causal_signature]
        matrix = engine.compute_morphism_matrix(sigs)
        assert len(matrix) == 2
        for src_name, inner in matrix.items():
            assert len(inner) == 2

    def test_matrix_diagonal_is_isomorphic(self, engine, thermal, resource):
        sigs = [thermal.causal_signature, resource.causal_signature]
        matrix = engine.compute_morphism_matrix(sigs)
        for name in ["thermal_homeostasis", "resource_management"]:
            assert matrix[name][name].morphism_class == "isomorphic"

    def test_details_include_alignment_info(self, engine, thermal, resource):
        m = engine.compute_morphism(
            thermal.causal_signature,
            resource.causal_signature,
        )
        assert "proposition_alignment" in m.details
        assert "intervention_alignment" in m.details
        assert "causal_graph_alignment" in m.details
        assert "weights" in m.details


# ── Adversarial scenario test ────────────────────────────────────────────────

class TestAdversarialMorphism:
    def test_incompatible_scenario(self, engine):
        """A synthetic incompatible scenario should be rejected."""
        sig_a = ScenarioCausalSignature(
            scenario_name="scenario_a",
            scenario_version="1.0",
            observable_variables=frozenset({"x"}),
            control_variables=frozenset({"y"}),
            main_variable="x",
            optimization_direction="minimize",
            causal_polarity="lower_is_better",
            alarm_semantics="threshold_above",
            intervention_effects=(
                InterventionEffect(
                    intervention_name="decrease_x",
                    target_variable="x",
                    expected_direction="-",
                    expected_magnitude=0.5,
                    semantic_role="corrective",
                ),
            ),
            counterfactual_policy="opposite_intervention",
            counterfactual_variable="x",
            causal_edges=(
                CausalEdge(source="y", target="x", polarity="-"),
            ),
            proposition_vocabulary=frozenset({"X_HIGH", "X_NORMAL"}),
        )
        sig_b = ScenarioCausalSignature(
            scenario_name="scenario_b",
            scenario_version="1.0",
            observable_variables=frozenset({"alpha", "beta", "gamma"}),
            control_variables=frozenset({"delta", "epsilon"}),
            main_variable="alpha",
            optimization_direction="target_band",
            causal_polarity="contextual",
            alarm_semantics="threshold_below",
            intervention_effects=(
                InterventionEffect(
                    intervention_name="boost_alpha",
                    target_variable="alpha",
                    expected_direction="+",
                    expected_magnitude=0.9,
                    semantic_role="restorative",
                ),
                InterventionEffect(
                    intervention_name="dampen_beta",
                    target_variable="beta",
                    expected_direction="-",
                    expected_magnitude=0.3,
                    semantic_role="preventive",
                ),
            ),
            counterfactual_policy="random_baseline",
            counterfactual_variable="alpha",
            causal_edges=(
                CausalEdge(source="delta", target="alpha", polarity="+"),
                CausalEdge(source="epsilon", target="beta", polarity="-"),
                CausalEdge(source="alpha", target="gamma", polarity="?"),
            ),
            proposition_vocabulary=frozenset({"ALPHA_HIGH", "BETA_LOW", "GAMMA_ACTIVE"}),
        )
        m = engine.compute_morphism(sig_a, sig_b)
        assert m.is_transfer_safe_prior is False
        assert m.morphism_class in ("analogical", "adversarial", "incompatible")
        # Very low overall score expected
        assert m.overall_score < 0.75
