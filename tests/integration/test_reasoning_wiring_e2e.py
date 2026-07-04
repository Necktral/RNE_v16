"""E2E wiring checks for reasoning families across runtime, benchmark, and reality layers."""

from __future__ import annotations

import json
from pathlib import Path

from runtime.reality.service import RealityValidationService
from runtime.reasoning.scheduler_meta.meta_scheduler import MetaScheduler
from runtime.reality.evaluator import ADAPTIVE_MIN_PROFILE, validate_sequence_with_profile
from runtime.storage import StorageConfig, StorageFactory
from runtime.world.grid_thermal_scenario import GridThermalScenario
from runtime.world.scenario_runner import ScenarioEpisodeRunner
from tests.benchmarks.analysis_report import BenchmarkAnalyzer
from tests.benchmarks.benchmark_runner import (
    BenchmarkConfig,
    BenchmarkRunner,
    adapt_runtime_result_to_benchmark,
)


def _storage_config(tmp_path: Path) -> StorageConfig:
    return StorageConfig(
        mode="sqlite",
        sqlite_db_path=str(tmp_path / "reasoning_wiring_e2e.db"),
        postgres_dsn=None,
        artifact_root=tmp_path / "artifacts",
        prefer_postgres_reads=False,
        strict_dual_write=False,
    )


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def test_reasoning_wiring_e2e_runtime_benchmark_reality(tmp_path: Path) -> None:
    cfg = _storage_config(tmp_path)
    storage = StorageFactory.create_facade(cfg)

    # A) Runtime invocation + scheduler payload propagation
    run_id = "run-reasoning-wiring-e2e"
    runner = ScenarioEpisodeRunner(
        storage=storage,
        run_id=run_id,
        scenario="grid_thermal_5x5",
    )
    result = runner.run_episode(external_input=0.04)

    reasoning_sequence = result["reasoning"]["sequence"]
    episode_sequence = result["episode"]["result"]["reasoning_sequence"]
    episode_trace_families = [step["family"] for step in result["episode"]["trace"]]

    assert reasoning_sequence
    assert episode_sequence == reasoning_sequence
    assert episode_trace_families == reasoning_sequence

    persisted = storage.list_reasoning_traces(run_id=run_id, limit=100)
    assert persisted
    assert [row.family for row in persisted] == reasoning_sequence
    assert [row.step_index for row in persisted] == list(range(len(reasoning_sequence)))
    assert persisted[0].detail.get("sequence") == reasoning_sequence

    # B) Benchmark adaptation + serialized metric bridge
    adapted = adapt_runtime_result_to_benchmark(result)
    assert adapted["reasoning_sequence"] == reasoning_sequence
    assert len(adapted["reasoning_sequence"]) == len(reasoning_sequence)

    bench_root = tmp_path / "bench_output"
    bench_runner = BenchmarkRunner(output_root=bench_root, storage_config=cfg)

    run_1x1_dir = bench_root / "run_1x1"
    run_5x5_dir = bench_root / "run_5x5"

    cfg_1x1 = BenchmarkConfig(
        scenario_name="grid_thermal_1x1",
        scenario_class=GridThermalScenario,
        scenario_params={"grid_size": 1},
        episodes=2,
        base_seed=123,
        max_steps=50,
        output_dir=run_1x1_dir,
        run_id="bench-wiring-1x1",
    )
    cfg_5x5 = BenchmarkConfig(
        scenario_name="grid_thermal_5x5_uniform",
        scenario_class=GridThermalScenario,
        scenario_params={"grid_size": 5, "topology": "uniform"},
        episodes=2,
        base_seed=456,
        max_steps=50,
        output_dir=run_5x5_dir,
        run_id="bench-wiring-5x5",
    )

    bench_runner.run_benchmark(cfg_1x1)
    bench_runner.run_benchmark(cfg_5x5)

    episodes_1x1 = run_1x1_dir / "episodes.jsonl"
    episodes_5x5 = run_5x5_dir / "episodes.jsonl"
    summary_1x1 = run_1x1_dir / "summary.json"
    summary_5x5 = run_5x5_dir / "summary.json"

    assert episodes_1x1.exists()
    assert episodes_5x5.exists()
    assert summary_1x1.exists()
    assert summary_5x5.exists()

    rows_1x1 = _read_jsonl(episodes_1x1)
    rows_5x5 = _read_jsonl(episodes_5x5)
    assert rows_1x1 and rows_5x5
    assert all("reasoning_trace_length" in row for row in rows_1x1)
    assert all("reasoning_trace_length" in row for row in rows_5x5)

    analyzer = BenchmarkAnalyzer()
    analyzer.load_results(run_1x1_dir, run_5x5_dir)
    comparison = analyzer.compute_statistical_comparison()
    assert "reasoning_trace_length" in comparison

    # C) Reality validation bridge with profile-aware sequence validation
    service = RealityValidationService(storage=storage)

    baseline_assessment = service.evaluate_episode_result(
        run_id=run_id,
        bench_run_id="bench-wiring-reality",
        result=result,
        previous_result=None,
        recent_assessments=[],
        scenario_name="grid_thermal_5x5",
        closure_profile="baseline_fixed",
    )
    adaptive_assessment = service.evaluate_episode_result(
        run_id=run_id,
        bench_run_id="bench-wiring-reality",
        result=result,
        previous_result=None,
        recent_assessments=[baseline_assessment],
        scenario_name="grid_thermal_5x5",
        closure_profile="adaptive_min",
    )

    assert baseline_assessment.details["reasoning_sequence"] == reasoning_sequence
    assert baseline_assessment.details["sequence_validation"]["profile"] == "baseline_fixed"
    assert baseline_assessment.details["sequence_validation"]["passed"] is True

    assert adaptive_assessment.details["reasoning_sequence"] == reasoning_sequence
    assert adaptive_assessment.details["sequence_validation"]["profile"] == "adaptive_min"
    assert adaptive_assessment.details["sequence_validation"]["passed"] is True

    storage.close()


def test_reasoning_wiring_adaptive_optional_families_in_controlled_mode(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cfg = _storage_config(tmp_path)
    storage = StorageFactory.create_facade(cfg)

    monkeypatch.setenv("RNFE_EML_MODE", "shadow")
    monkeypatch.setenv("RNFE_META_EXPERIMENTAL_FAMILIES", "eml_sr")

    scheduler = MetaScheduler(mode="adaptive", trace_store=storage, max_steps=10)
    run_id = "run-adaptive-controlled"
    context = {
        "episode_id": "episode-adaptive-controlled",
        "run_id": run_id,
        "storage": storage,
        "observation": {"temperature": 0.92, "alarm": True},
        "updated_world": {"temperature": 0.83},
        "counterfactual": {"temperature": 0.96},
        "uncertainty": 0.86,
        "contradiction_signal": 0.78,
        "counterfactual_gap": 0.71,
        "edge_pressure": 0.83,
        "symbolic_regularity": 0.77,
        "law_fit_signal": 0.74,
    }

    reasoning = scheduler.run(context)
    sequence = reasoning["sequence"]

    core = {"ABD", "ANA", "CAU", "CTF", "DED", "PROB"}
    optional = {"HEUR", "DIA_ADV", "FAL_GUARD", "EML_SR"}

    assert core.issubset(set(sequence))
    assert optional.intersection(set(sequence))

    traces = storage.list_reasoning_traces(run_id=run_id, limit=100)
    assert traces
    assert [row.family for row in traces] == sequence

    validation = validate_sequence_with_profile(sequence, ADAPTIVE_MIN_PROFILE)
    assert validation["passed"] is True

    storage.close()
