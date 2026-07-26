from __future__ import annotations

import random
from pathlib import Path

import pytest

from runtime.storage import StorageConfig, StorageFactory
from runtime.symbolic.mci import (
    TransitionCompiler,
    resource_with_energy_oracle_spec,
    resource_with_energy_spec,
)
from runtime.world import ResourceWithEnergyScenario, ScenarioEpisodeRunner, get_scenario
from runtime.world.resource_with_energy_scenario import ResourceEnergyState


def _storage(tmp_path: Path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    return StorageFactory.create_facade(
        StorageConfig(
            mode="sqlite",
            sqlite_db_path=str(tmp_path / "events.db"),
            postgres_dsn=None,
            artifact_root=tmp_path / "artifacts",
            prefer_postgres_reads=True,
            strict_dual_write=False,
        )
    )


def test_resource_energy_hidden_guard_and_clamps():
    high = ResourceWithEnergyScenario(initial_stock=0.2, initial_energy=0.8)
    low = ResourceWithEnergyScenario(initial_stock=0.2, initial_energy=0.2)
    assert high.factual_transition(
        intervention="start_production", external_input=0.0
    ).state["stock_level"] == pytest.approx(0.28)
    result = low.factual_transition(
        intervention="start_production", external_input=0.0
    )
    assert result.state["stock_level"] == pytest.approx(0.2)
    assert result.state["energy_level"] == pytest.approx(0.14)
    with pytest.raises(ValueError, match="Intervención desconocida"):
        low.simulate_counterfactual(intervention="invalid", external_input=0.0)


def test_resource_energy_world_equals_oracle_ir_10k():
    rng = random.Random(9182)
    scenario = ResourceWithEnergyScenario()
    compiler = TransitionCompiler(resource_with_energy_oracle_spec())
    for index in range(10_000):
        state = ResourceEnergyState(
            stock_level=rng.random(),
            energy_level=rng.random(),
            production_active=bool(rng.getrandbits(1)),
            scarcity_alert=bool(rng.getrandbits(1)),
        )
        action = "start_production" if index % 2 else "stop_production"
        external = rng.uniform(-0.2, 0.2)
        actual = scenario._compute_transition(
            state, intervention=action, external_input=external
        )
        expected = compiler.execute(
            {
                "stock_level": state.stock_level,
                "energy_level": state.energy_level,
                "production_active": state.production_active,
                "scarcity_alert": state.scarcity_alert,
            },
            action=action,
            external_input=external,
        )
        assert actual.stock_level == pytest.approx(expected["stock_level"], abs=1e-12)
        assert actual.energy_level == pytest.approx(expected["energy_level"], abs=1e-12)
        assert actual.production_active is expected["production_active"]
        assert actual.scarcity_alert is expected["scarcity_alert"]


def test_epistemic_spec_deliberately_omits_energy_guard():
    epistemic = resource_with_energy_spec()
    oracle = resource_with_energy_oracle_spec()
    effect = epistemic.equations[0].effects[0]
    oracle_effect = oracle.equations[0].effects[0]
    assert effect.precondition is None
    assert oracle_effect.precondition == "energy_level > 0.3"


def test_resource_alarm_normalization_and_new_runtime(tmp_path: Path):
    for scenario_name in ("resource_management", "resource_with_energy"):
        runner = ScenarioEpisodeRunner(
            storage=_storage(tmp_path / scenario_name),
            run_id=f"alarm-{scenario_name}",
            scenario=scenario_name,
            family_profile="mci_integrated_v1",
        )
        result = runner.run_episode(
            replay_unit_id=f"alarm/{scenario_name}",
            trace_dir=tmp_path / "traces" / scenario_name,
        )
        assert result["mci_outcome"]["status"] == "observed"
        assert runner._mci_runtime is not None
        assert runner._mci_runtime.base_spec.alarm_variable == "scarcity_alert"
    assert isinstance(get_scenario("resource_with_energy"), ResourceWithEnergyScenario)
