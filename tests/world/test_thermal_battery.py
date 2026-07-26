from __future__ import annotations

from pathlib import Path

import pytest

from runtime.storage import StorageConfig, StorageFactory
from runtime.symbolic.mci import TransitionCompiler, thermal_battery_spec
from runtime.world import ScenarioEpisodeRunner, ThermalBatteryScenario, get_scenario


def _storage(tmp_path: Path):
    return StorageFactory.create_facade(
        StorageConfig(
            mode="sqlite",
            sqlite_db_path=str(tmp_path / "thermal-battery.db"),
            postgres_dsn=None,
            artifact_root=tmp_path / "artifacts",
            prefer_postgres_reads=True,
            strict_dual_write=False,
        )
    )


def test_hidden_guard_only_cools_with_sufficient_battery():
    high = ThermalBatteryScenario(initial_temperature=0.9, initial_battery=0.31)
    low = ThermalBatteryScenario(initial_temperature=0.9, initial_battery=0.3)

    high_result = high.factual_transition(
        intervention="activate_cooling", external_input=0.0
    )
    low_result = low.factual_transition(
        intervention="activate_cooling", external_input=0.0
    )

    assert high_result.state["temperature"] == pytest.approx(0.83)
    assert low_result.state["temperature"] == pytest.approx(0.9)
    assert high_result.state["battery_level"] == pytest.approx(0.25)
    assert low_result.state["battery_level"] == pytest.approx(0.24)


def test_counterfactual_does_not_mutate_and_values_are_clamped():
    scenario = ThermalBatteryScenario(
        initial_temperature=0.99, initial_battery=0.01
    )
    before = scenario.observe()
    simulated = scenario.simulate_counterfactual(
        intervention="deactivate_cooling", external_input=0.5
    )
    after = scenario.observe()

    assert before.state == after.state
    assert simulated.state["temperature"] == 1.0
    assert 0.0 <= simulated.state["battery_level"] <= 1.0


def test_initial_ir_deliberately_omits_hidden_guard():
    spec = thermal_battery_spec()
    cooling = next(
        effect
        for equation in spec.equations
        for effect in equation.effects
        if effect.effect_id == "thermal_battery/cooling"
    )
    assert cooling.precondition is None
    predicted = TransitionCompiler(spec).execute(
        {
            "temperature": 0.9,
            "battery_level": 0.2,
            "cooling_active": False,
            "alarm": True,
        },
        action="activate_cooling",
        external_input=0.0,
    )
    assert predicted["temperature"] == pytest.approx(0.83)


def test_scenario_is_registered_and_runner_builds_mci(tmp_path: Path):
    assert isinstance(get_scenario("thermal_with_battery"), ThermalBatteryScenario)
    runner = ScenarioEpisodeRunner(
        storage=_storage(tmp_path),
        scenario="thermal_with_battery",
        family_profile="mci_integrated_v1",
        run_id="thermal-battery-test",
    )
    assert runner.mci_active
    assert runner._mci_runtime is not None
    assert runner._mci_runtime.base_spec.spec_id == "transition/thermal_with_battery"


def test_battery_dynamics_are_parameterized_and_reflected_in_ir(tmp_path: Path):
    scenario = ThermalBatteryScenario(
        initial_temperature=0.9,
        initial_battery=0.5,
        battery_threshold=0.4,
        battery_discharge_rate=0.1,
        battery_charge_rate=0.03,
    )
    result = scenario.factual_transition(
        intervention="activate_cooling", external_input=0.0
    )
    assert result.state["temperature"] == pytest.approx(0.83)
    assert result.state["battery_level"] == pytest.approx(0.4)
    assert scenario.causal_signature.metadata["hidden_precondition"] == (
        "battery_level > 0.4"
    )

    runner = ScenarioEpisodeRunner(
        storage=_storage(tmp_path),
        scenario=scenario,
        family_profile="mci_integrated_v1",
        run_id="thermal-battery-parameterized",
    )
    assert runner._mci_runtime is not None
    compiled = TransitionCompiler(runner._mci_runtime.base_spec).execute(
        {
            "temperature": 0.9,
            "battery_level": 0.5,
            "cooling_active": False,
            "alarm": True,
        },
        action="activate_cooling",
        external_input=0.0,
    )
    assert compiled["battery_level"] == pytest.approx(0.4)


def test_set_regime_invalidates_cached_causal_signature():
    scenario = ThermalBatteryScenario(cooling_effect=0.07)
    before = scenario.causal_signature
    assert before.intervention_effects[0].expected_magnitude == pytest.approx(0.07)
    scenario.set_regime(0.14)
    after = scenario.causal_signature
    assert after is not before
    assert after.intervention_effects[0].expected_magnitude == pytest.approx(0.14)


@pytest.mark.parametrize(
    "kwargs",
    (
        {"battery_threshold": -0.1},
        {"battery_threshold": 1.1},
        {"battery_discharge_rate": -0.01},
        {"battery_charge_rate": -0.01},
    ),
)
def test_invalid_battery_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        ThermalBatteryScenario(**kwargs)
