"""Tests para verdicts de transferencia en distintos escenarios."""

from runtime.certification.transfer_assessment import (
    TransferAssessment,
    assess_transfer,
)
from runtime.reality.transition_analysis import TransitionContinuityVector
from runtime.world.compatibility import CompatibilityAssessment


def _ep(scenario="thermal_homeostasis", cross_mem=False):
    mem = []
    if cross_mem:
        mem = [{"analogical_source": True, "retrieval_metrics": {
            "retrieved_same_scenario_count": 1,
            "retrieved_cross_scenario_count": 1,
        }}]
    return {
        "episode": {
            "episode_id": f"ep-{scenario}",
            "scenario": scenario,
            "scenario_metadata": {"scenario_name": scenario, "main_variable": "temperature"},
            "closure_profile": "baseline_fixed",
            "context": {
                "observation": {"temperature": 0.88, "alarm": True, "propositions": ["TEMP_HIGH"]},
                "intervention": "activate_cooling",
                "counterfactual": {"temperature": 0.92},
                "retrieved_memory": mem,
            },
            "result": {
                "updated_world": {"temperature": 0.83},
                "relation_kind": "support",
                "reasoning_sequence": ["ABD", "ANA", "CAU", "CTF", "DED", "PROB"],
            },
        },
    }


def _compat(cls, src="thermal_homeostasis", tgt="thermal_homeostasis", score=1.0):
    return CompatibilityAssessment(
        source_scenario=src,
        target_scenario=tgt,
        compatibility_class=cls,
        topology_score=1.0,
        objective_score=1.0,
        intervention_score=1.0,
        counterfactual_score=1.0,
        overall_score=score,
        penalty_multiplier=1.0,
        transfer_allowed=cls != "incompatible",
        certification_allowed=cls in ("equivalent", "compatible"),
    )


def _vec(purity=1.0, composite=0.85, ttype="intra"):
    return TransitionContinuityVector(
        source_scenario="thermal_homeostasis",
        target_scenario="thermal_homeostasis",
        semantic_retention=0.9,
        trace_stability=0.9,
        causal_stability=1.0,
        intervention_policy_stability=0.9,
        structural_compatibility=0.95,
        memory_purity=purity,
        composite_score=composite,
        transition_type=ttype,
    )


class TestTransferVerdicts:
    """Verifica cada posible verdict."""

    def test_certified_local_when_equivalent_no_cross(self):
        result = assess_transfer(
            episode_result=_ep(),
            compatibility=_compat("equivalent"),
            transition_vector=_vec(),
        )
        assert result.transfer_verdict == "certified_local"

    def test_certified_transfer_safe_compatible_clean(self):
        result = assess_transfer(
            episode_result=_ep(cross_mem=True),
            compatibility=_compat("compatible", tgt="resource_management", score=0.80),
            transition_vector=_vec(purity=0.98, composite=0.80, ttype="compatible"),
        )
        assert result.transfer_verdict == "certified_transfer_safe"

    def test_certified_analogical_only(self):
        result = assess_transfer(
            episode_result=_ep(cross_mem=True),
            compatibility=_compat("analogical", tgt="resource_management", score=0.55),
            transition_vector=_vec(purity=0.80, composite=0.55, ttype="analogical"),
        )
        assert result.transfer_verdict == "certified_analogical_only"

    def test_rejected_incompatible(self):
        result = assess_transfer(
            episode_result=_ep(),
            compatibility=_compat("incompatible", tgt="fake", score=0.20),
            transition_vector=_vec(purity=0.40, composite=0.20, ttype="incompatible"),
        )
        assert result.transfer_verdict == "rejected_for_transfer"

    def test_rejected_low_purity(self):
        """Even compatible, if purity < 0.50 → rejected."""
        result = assess_transfer(
            episode_result=_ep(cross_mem=True),
            compatibility=_compat("compatible", tgt="resource_management"),
            transition_vector=_vec(purity=0.30, composite=0.80, ttype="compatible"),
        )
        assert result.transfer_verdict == "rejected_for_transfer"

    def test_rejected_low_stability(self):
        """Even compatible, if stability < 0.30 → rejected."""
        result = assess_transfer(
            episode_result=_ep(cross_mem=True),
            compatibility=_compat("compatible", tgt="resource_management"),
            transition_vector=_vec(purity=0.98, composite=0.25, ttype="compatible"),
        )
        assert result.transfer_verdict == "rejected_for_transfer"
