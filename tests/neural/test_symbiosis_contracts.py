from __future__ import annotations

import inspect

from runtime.neural.integration import (
    SYMBIOSIS_TRACE_SCHEMA_VERSION,
    SymbioticNeuralCoordinator,
    integration_census,
    validate_active_census,
)


def test_symbiosis_contract_is_versioned_and_active_census_has_no_stub() -> None:
    assert SYMBIOSIS_TRACE_SCHEMA_VERSION == "neural-symbiosis-trace-v1"
    assert validate_active_census() == []
    matrix = {row["organ"]: row for row in integration_census()}
    for organ in ("N0", "N1", "N2", "N3", "N4", "N5", "N6"):
        assert matrix[organ]["caller_count"] > 0
        assert matrix[organ]["consumer_count"] > 0
        assert matrix[organ]["stub_detected"] is False
    assert matrix["N4"]["shadow_consumed"] is True
    assert matrix["N5"]["live"] is True
    assert matrix["EVO_SEARCH"]["reference_only"] is True
    assert matrix["IMAGINATION/A11"]["reference_only"] is True


def test_live_coordinator_source_has_no_stub_control_flow() -> None:
    source = inspect.getsource(SymbioticNeuralCoordinator)
    assert "if False" not in source
    assert "NotImplemented" not in source
    assert "return idle" not in source
    assert "pass\n" not in source


def test_scenario_runner_is_the_non_test_live_caller() -> None:
    from runtime.world import scenario_runner

    source = inspect.getsource(scenario_runner.ScenarioEpisodeRunner.run_episode)
    assert ".begin_episode(" in source
    assert ".consume_reasoning(" in source
    assert ".prepare_certification(" in source
    assert ".finalize_episode(" in source
