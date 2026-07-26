import json
from pathlib import Path

from runtime.storage import StorageConfig, StorageFactory
from runtime.world import ScenarioEpisodeRunner, ThermalScenario
from runtime.world.grid_thermal_scenario import GridThermalScenario


def _storage(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    return StorageFactory.create_facade(
        StorageConfig(
            mode="sqlite",
            sqlite_db_path=str(root / "runtime.db"),
            postgres_dsn=None,
            artifact_root=root / "artifacts",
            prefer_postgres_reads=True,
            strict_dual_write=False,
        )
    )


def _run(root: Path, monkeypatch):
    monkeypatch.setenv("RNFE_REASONING_FAMILY_PROFILE", "core_plus_opt")
    monkeypatch.setenv("RNFE_REASONING_ACTUATES", "1")
    storage = _storage(root)
    runner = ScenarioEpisodeRunner(
        storage=storage,
        run_id="sym0-deterministic",
        scenario=ThermalScenario(
            initial_temperature=0.88,
            alarm_threshold=0.85,
            cooling_effect=0.07,
        ),
    )
    result = runner.run_episode(
        external_input=0.0,
        replay_unit_id="thermal/sym0-test/ep-01",
        trace_dir=root / "traces",
    )
    storage.close()
    return result


def test_sym0_is_bit_deterministic_and_does_not_force_redundant_override(
    tmp_path: Path, monkeypatch
):
    first = _run(tmp_path / "one", monkeypatch)
    second = _run(tmp_path / "two", monkeypatch)

    first_paths = first["acting_trace"]["paths"]
    second_paths = second["acting_trace"]["paths"]
    for key in ("constraints", "core_report", "decision_trace", "outcome_link"):
        assert Path(first_paths[key]).read_bytes() == Path(second_paths[key]).read_bytes()

    assert first["acting_trace"]["trace_status"] == "ok"
    assert first["acting_trace"]["sealed_hash"] == second["acting_trace"]["sealed_hash"]
    assert first["intervention_override"]["fired"] is False
    assert first["intervention_override"]["guard_reason"] == "no_conflict"
    assert first["episode"]["context"]["intervention"] == "activate_cooling"


def test_sym0_ded_and_opt_reports_are_complete(tmp_path: Path, monkeypatch):
    result = _run(tmp_path / "run", monkeypatch)
    trace_path = Path(result["acting_trace"]["paths"]["decision_trace"])
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    assert trace["core_report"]["status"] == "SAT"
    assert any("/kind/formula/" in item for item in trace["core_report"]["all_constraint_ids"])
    assert any("/kind/assumption/" in item for item in trace["core_report"]["all_constraint_ids"])
    assert trace["opt_report"]["candidates"]
    assert trace["opt_report"]["winner"]["intervention"] == "activate_cooling"
    assert trace["guard_report"]["fired"] is False


def test_sym0_records_real_override_in_conflict_regime(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("RNFE_REASONING_MODE", "adaptive")
    monkeypatch.setenv("RNFE_REASONING_FAMILY_PROFILE", "core_plus_opt")
    monkeypatch.setenv("RNFE_REASONING_REGIME_HINT", "causal_counterfactual_conflict")
    monkeypatch.setenv("RNFE_REASONING_MAX_STEPS", "10")
    monkeypatch.setenv("RNFE_REASONING_ACTUATES", "1")
    storage = _storage(tmp_path)
    runner = ScenarioEpisodeRunner(
        storage=storage,
        run_id="sym0-conflict",
        closure_profile="adaptive_min",
        scenario=GridThermalScenario(
            grid_size=5,
            topology="uniform",
            initial_temperature=0.88,
            alarm_threshold=0.85,
            cooling_effect=0.07,
        ),
    )
    result = runner.run_episode(
        external_input=0.04,
        replay_unit_id="thermal/sym0-conflict/ep-01",
        trace_dir=tmp_path / "traces",
    )
    storage.close()

    trace = json.loads(
        Path(result["acting_trace"]["paths"]["decision_trace"]).read_text(encoding="utf-8")
    )
    assert trace["baseline_action"] == "deactivate_cooling"
    assert trace["committed_action"] == "activate_cooling"
    assert trace["guard_report"]["fired"] is True
    assert trace["guard_report"]["guard_reason"] == "guard_passed"
    assert trace["guard_report"]["margin_gain"] == 0.07
