from copy import deepcopy

import pytest

from runtime.world.counterfactual_oracle import enumerate_counterfactual_outcomes
from runtime.world.registry import get_scenario
from runtime.world.thermal_scenario import ThermalScenario


def test_oracle_enumerates_distinct_actions_without_mutating_world() -> None:
    scenario = ThermalScenario(initial_temperature=0.90)
    before = scenario.to_observation_dict(scenario.observe())

    result = enumerate_counterfactual_outcomes(
        scenario=scenario,
        observation=before,
        interventions=scenario.config.interventions,
        external_input=0.04,
        optimization_direction="minimize",
    )

    after = scenario.to_observation_dict(scenario.observe())
    assert after == before
    assert result.status == "scored"
    assert [item.intervention for item in result.outcomes] == scenario.config.interventions
    assert result.best_actions == ("activate_cooling",)
    assert result.authority_effect == "none"
    assert len(result.snapshot_sha256) == 64
    assert len(result.outcome_set_sha256 or "") == 64


def test_oracle_deduplicates_actions_and_fails_closed_for_target_band() -> None:
    scenario = ThermalScenario(initial_temperature=0.90)
    observation = scenario.to_observation_dict(scenario.observe())

    deduped = enumerate_counterfactual_outcomes(
        scenario=scenario,
        observation=observation,
        interventions=["activate_cooling", "activate_cooling", "deactivate_cooling"],
        external_input=0.04,
        optimization_direction="maximize",
    )
    unavailable = enumerate_counterfactual_outcomes(
        scenario=scenario,
        observation=observation,
        interventions=scenario.config.interventions,
        external_input=0.04,
        optimization_direction="target_band",
    )

    assert len(deduped.outcomes) == 2
    assert deduped.best_actions == ("deactivate_cooling",)
    assert unavailable.status == "unavailable"
    assert unavailable.unavailable_reason == "unsupported_optimization_direction"
    assert unavailable.outcomes == ()


def test_oracle_outcome_seal_changes_when_hidden_outcomes_change() -> None:
    first_scenario = ThermalScenario(initial_temperature=0.90)
    second_scenario = ThermalScenario(initial_temperature=0.90)
    observation = first_scenario.to_observation_dict(first_scenario.observe())
    original_simulate = second_scenario.simulate_counterfactual

    def shifted_simulate(*, intervention: str, external_input: float):
        transition = original_simulate(
            intervention=intervention, external_input=external_input
        )
        transition.state["temperature"] += 0.01
        return transition

    second_scenario.simulate_counterfactual = shifted_simulate  # type: ignore[method-assign]

    first = enumerate_counterfactual_outcomes(
        scenario=first_scenario,
        observation=observation,
        interventions=first_scenario.config.interventions,
        external_input=0.04,
        optimization_direction="minimize",
    )
    second = enumerate_counterfactual_outcomes(
        scenario=second_scenario,
        observation=observation,
        interventions=second_scenario.config.interventions,
        external_input=0.04,
        optimization_direction="minimize",
    )

    assert first.snapshot_sha256 == second.snapshot_sha256
    assert first.outcome_set_sha256 != second.outcome_set_sha256


@pytest.mark.parametrize(
    "scenario_name",
    (
        "thermal_homeostasis",
        "resource_management",
        "grid_thermal_5x5",
        "deferred_load_trap",
    ),
)
def test_oracle_never_mutates_any_campaign_scenario(scenario_name: str) -> None:
    scenario = get_scenario(scenario_name)
    observation = scenario.to_observation_dict(scenario.observe())
    before = deepcopy(scenario.__dict__)

    enumerate_counterfactual_outcomes(
        scenario=scenario,
        observation=observation,
        interventions=scenario.config.interventions,
        external_input=0.07,
        optimization_direction=str(scenario.causal_signature.optimization_direction),
    )

    assert scenario.__dict__ == before
