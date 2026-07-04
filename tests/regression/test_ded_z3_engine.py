from pathlib import Path

from runtime.reasoning.families.ded import execute as ded_execute
from runtime.reasoning.scheduler_meta.meta_scheduler import MetaScheduler
from runtime.storage import StorageConfig, StorageFactory
from runtime.world import ScenarioEpisodeRunner, ThermalScenario


def _storage(tmp_path: Path):
    config = StorageConfig(
        mode="sqlite",
        sqlite_db_path=str(tmp_path / "ded_engine.db"),
        postgres_dsn=None,
        artifact_root=tmp_path / "artifacts",
        prefer_postgres_reads=True,
        strict_dual_write=False,
    )
    return StorageFactory.create_facade(config)


def _ded_step(trace):
    return next(step for step in trace if step["family"] == "DED")


def test_ded_sat_simple_implication():
    result = ded_execute(
        {
            "formula": "TEMP_HIGH -> ACTIVATE_COOLING",
            "observation": {"propositions": ["TEMP_HIGH"], "alarm": True},
        }
    )

    assert result["family"] == "DED"
    assert result["status"] == "ok"
    assert result["state_delta"]["ded_status"] == "sat"
    assert result["state_delta"]["ded_consistent"] is True
    assert result["state_delta"]["ded_conclusion"]
    assert "ACTIVATE_COOLING" in result["state_delta"]["ded_conclusion"]
    assert result["artifacts"]["solver_result"] == "sat"
    assert result["artifacts"]["model"]


def test_ded_missing_formula_graceful():
    result = ded_execute({"observation": {"propositions": ["TEMP_HIGH"]}})

    assert result["status"] == "skip"
    assert result["failure_mode"] == "missing_formula"
    assert result["state_delta"]["ded_status"] == "missing_formula"
    assert result["state_delta"]["ded_validated"] is False
    assert result["artifacts"]["solver_result"] == "missing_formula"


def test_ded_unsupported_formula_graceful():
    result = ded_execute({"formula": "TEMP_HIGH XOR ACTIVATE_COOLING"})

    assert result["status"] == "skip"
    assert result["failure_mode"] == "unsupported_formula"
    assert result["state_delta"]["ded_status"] == "unsupported"
    assert result["state_delta"]["ded_validated"] is False
    assert result["artifacts"]["solver_result"] == "unsupported"
    assert result["artifacts"]["parse_error"]


def test_ded_trace_contains_solver_evidence(tmp_path: Path):
    storage = _storage(tmp_path)
    scheduler = MetaScheduler(trace_store=storage)

    result = scheduler.run(
        {
            "episode_id": "episode-ded-trace",
            "run_id": "run-ded-trace",
            "formula": "TEMP_HIGH -> ACTIVATE_COOLING",
            "observation": {"propositions": ["TEMP_HIGH"], "alarm": True},
        }
    )

    ded_step = _ded_step(result["trace"])
    artifacts = ded_step["detail"]["artifacts"]
    assert artifacts["solver_result"] == "sat"
    assert artifacts["z3_expression"]
    assert artifacts["formula_normalized"] == "TEMP_HIGH -> ACTIVATE_COOLING"

    persisted = storage.list_reasoning_traces(run_id="run-ded-trace", limit=20)
    persisted_ded = next(step for step in persisted if step.family == "DED")
    assert persisted_ded.detail["artifacts"]["solver_result"] == "sat"
    assert persisted_ded.detail["artifacts"]["z3_expression"]
    storage.close()


def test_baseline_sequence_unchanged_after_ded_upgrade():
    scheduler = MetaScheduler()
    result = scheduler.run({"episode": "ded-baseline"})
    assert result["sequence"] == ["ABD", "ANA", "CAU", "CTF", "DED", "PROB"]


def test_scenario_runner_still_closes_with_ded_engine(tmp_path: Path):
    storage = _storage(tmp_path)
    runner = ScenarioEpisodeRunner(
        storage=storage,
        run_id="run-ded-scenario",
        scenario=ThermalScenario(initial_temperature=0.9),
    )

    result = runner.run_episode(external_input=0.03)

    assert result["episode"]["result"]["reasoning_sequence"] == [
        "ABD", "ANA", "CAU", "CTF", "DED", "PROB"
    ]
    ded_step = _ded_step(result["episode"]["trace"])
    assert ded_step["detail"]["artifacts"]["solver_result"] == "sat"
    assert ded_step["detail"]["state_delta"]["ded_solver_backend"] == "z3"
    assert ded_step["detail"]["state_delta"]["ded_conclusion"]
    storage.close()
