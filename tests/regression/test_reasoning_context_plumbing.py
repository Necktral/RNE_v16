"""Regresiones para plumbing de contexto y contrato de familias."""

from runtime.reasoning.context import build_reasoning_context, resolve_reasoning_mode
from runtime.reasoning.families.abd import execute as abd_execute
from runtime.reasoning.families.ana import execute as ana_execute
from runtime.reasoning.families.cau import execute as cau_execute
from runtime.reasoning.families.ctf import execute as ctf_execute
from runtime.reasoning.families.ded import execute as ded_execute
from runtime.reasoning.families.prob import execute as prob_execute


def test_resolve_reasoning_mode_maps_closure_profiles() -> None:
    assert resolve_reasoning_mode("baseline_fixed") == "fixed"
    assert resolve_reasoning_mode("adaptive_min") == "adaptive"


def test_build_reasoning_context_enriches_scheduler_payload() -> None:
    context = build_reasoning_context(
        episode_id="episode-ctx",
        run_id="run-ctx",
        observation={"temperature": 0.82, "alarm": False},
        intervention="activate_cooling",
        formula="TEMP_HIGH -> ACTIVATE_COOLING",
        memory_hits=[{"structure": {"relation_kind": "support"}}],
        counterfactual={"temperature": 0.9, "alarm": True},
        updated_world={"temperature": 0.76, "alarm": False},
        relation_kind="support",
        scenario="thermal_homeostasis",
        scenario_metadata={"scenario_name": "thermal_homeostasis"},
        belief_state={"scenario_name": "thermal_homeostasis", "alarm_probability": 0.1},
        closure_profile="adaptive_min",
        extra_signals={"edge_pressure": 0.2},
    )

    assert context["episode_id"] == "episode-ctx"
    assert context["run_id"] == "run-ctx"
    assert context["formula"] == "TEMP_HIGH -> ACTIVATE_COOLING"
    assert context["memory_hits"] == context["retrieved_memory"]
    assert context["counterfactual"]["temperature"] == 0.9
    assert context["updated_world"]["temperature"] == 0.76
    assert context["factual"]["temperature"] == 0.76
    assert context["relation_kind"] == "support"
    assert context["scenario"] == "thermal_homeostasis"
    assert context["scenario_metadata"]["scenario_name"] == "thermal_homeostasis"
    assert context["belief_state"]["scenario_name"] == "thermal_homeostasis"
    assert context["reasoning_mode"] == "adaptive"
    assert context["edge_pressure"] == 0.2


def test_core_family_contracts_are_homogeneous() -> None:
    core_families = {
        "ABD": abd_execute,
        "ANA": ana_execute,
        "CAU": cau_execute,
        "CTF": ctf_execute,
        "DED": ded_execute,
        "PROB": prob_execute,
    }

    for family_name, execute in core_families.items():
        result = execute({})
        assert result["family"] == family_name
        assert isinstance(result["status"], str)
        assert isinstance(result["state_delta"], dict)
        assert isinstance(result["confidence"], (int, float))
        assert isinstance(result["cost"], (int, float))
