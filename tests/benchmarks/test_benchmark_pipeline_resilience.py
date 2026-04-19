"""Tests de robustez para pipeline benchmark_runner -> episodes.jsonl -> analysis_report."""

from __future__ import annotations

import json
import math
from pathlib import Path

from runtime.storage import StorageConfig, StorageFactory
from runtime.world.grid_thermal_scenario import GridThermalScenario

from tests.benchmarks.analysis_report import BenchmarkAnalyzer
from tests.benchmarks.benchmark_runner import (
    BenchmarkConfig,
    BenchmarkRunner,
    EpisodeResult,
    adapt_runtime_result_to_benchmark,
)
from tests.benchmarks.metrics_ivc_r import compute_ivc_r_from_episode


REMOVED_METRICS = [
    "memory_pressure_mb",
    "scheduler_cpu_time_ms",
    "counterfactual_overhead_ratio",
    "factual_cf_divergence",
    "world_level_transitions",
    "spatial_coherence_index",
]


def _storage_config(tmp_path: Path) -> StorageConfig:
    return StorageConfig(
        mode="sqlite",
        sqlite_db_path=str(tmp_path / "bench_resilience.db"),
        postgres_dsn=None,
        artifact_root=tmp_path / "artifacts",
        prefer_postgres_reads=False,
        strict_dual_write=False,
    )


def test_adapter_emite_campos_minimos_requeridos() -> None:
    runtime_result = {
        "run_id": "run-x",
        "episode": {
            "episode_id": "ep-1",
            "scenario": "grid_thermal_5x5",
            "context": {
                "observation": {"world_level": 0.82},
                "counterfactual": {"world_level": 0.79},
            },
            "result": {"updated_world": {"world_level": 0.80}},
        },
        "reasoning": {"sequence": ["ABD", "ANA"]},
        "artifact": {"abs_path": "/tmp/fake_artifact.json"},
        "viability_assessment": {"is_viable": True, "viability_margin": 0.25, "distance_to_edge": 0.75},
        "certification": {"verdict": "passed", "promotion_candidate": True},
        "organism_trajectory": {"points": [{"viability_margin": 0.25}]},
    }

    adapted = adapt_runtime_result_to_benchmark(runtime_result)

    assert adapted["episode_id"] == "ep-1"
    assert adapted["scenario_name"] == "grid_thermal_5x5"
    assert adapted["certification_verdict"] == "passed"
    assert adapted["viability_margin"] == 0.25
    assert adapted["reasoning_sequence"] == ["ABD", "ANA"]


def test_ivc_r_tolera_precision_negativa_sin_math_domain() -> None:
    result = compute_ivc_r_from_episode(
        {
            "cierre_rate": 0.0,
            "continuity_score": 0.30,
            "intervention_precision": -0.40,  # caso real cuando la intervención empeora temperatura
            "proposition_diversity": 1.5,
            "wall_time_ms": 20.0,
        }
    )

    assert result["ivc_r"] >= 0.0
    assert math.isfinite(result["ivc_r"])
    assert math.isfinite(result["ivc_r_log"])


def test_generate_summary_soporta_todos_los_episodios_en_error(tmp_path: Path, monkeypatch) -> None:
    runner = BenchmarkRunner(output_root=tmp_path, storage_config=_storage_config(tmp_path))
    output_dir = tmp_path / "error_run"
    cfg = BenchmarkConfig(
        scenario_name="grid_thermal_1x1_error_smoke",
        scenario_class=GridThermalScenario,
        scenario_params={"grid_size": 1},
        episodes=3,
        base_seed=100,
        max_steps=50,
        output_dir=output_dir,
    )

    def _always_error(self, config: BenchmarkConfig, episode_id: str, seed: int) -> EpisodeResult:
        result = EpisodeResult(episode_id=episode_id, scenario_name=config.scenario_name, seed=seed)
        result.outcome = "error"
        result.error = "simulated failure"
        return result

    monkeypatch.setattr(BenchmarkRunner, "run_single_episode", _always_error)

    summary = runner.run_benchmark(cfg)

    assert summary["total_episodes"] == 3
    assert summary["successful"] == 0
    assert summary["errors"] == 3
    assert summary["success_rate"] == 0.0
    assert (output_dir / "summary.json").exists()
    assert (output_dir / "episodes.jsonl").exists()


def test_benchmark_persiste_reality_bench_run_con_proxies(tmp_path: Path, monkeypatch) -> None:
    cfg_storage = _storage_config(tmp_path)
    runner = BenchmarkRunner(output_root=tmp_path, storage_config=cfg_storage)
    output_dir = tmp_path / "run_db_summary"
    run_id = "bench-db-summary-test"

    cfg = BenchmarkConfig(
        scenario_name="grid_thermal_5x5_uniform_smoke",
        scenario_class=GridThermalScenario,
        scenario_params={"grid_size": 5, "topology": "uniform"},
        episodes=2,
        base_seed=200,
        max_steps=50,
        output_dir=output_dir,
        run_id=run_id,
    )

    def _fake_episode(self, config: BenchmarkConfig, episode_id: str, seed: int) -> EpisodeResult:
        result = EpisodeResult(episode_id=episode_id, scenario_name=config.scenario_name, seed=seed)
        result.metadata = {"grid_size": 5, "topology": "uniform"}
        result.wall_time_ms = 20.0
        result.artifact_size_bytes = 2500
        result.reasoning_trace_length = 6
        result.metrics = {
            "intervention_precision": 0.12,
            "proposition_diversity": 1.4,
            "spatial_information_usage": 0.30,
            "ivc_r": 0.38,
            "failure_primary": None,
            "failure_secondary": [],
        }

        if seed % 2 == 0:
            result.outcome = "success"
            result.certification_verdict = "certified"
            result.is_viable = True
            result.viability_margin = 0.22
        else:
            result.outcome = "failure"
            result.certification_verdict = "failed"
            result.is_viable = False
            result.viability_margin = 0.0
            result.metrics["failure_primary"] = "viability_failed"
        return result

    monkeypatch.setattr(BenchmarkRunner, "run_single_episode", _fake_episode)
    summary = runner.run_benchmark(cfg)

    storage = StorageFactory.create_facade(cfg_storage)
    rows = storage.list_reality_bench_runs(run_id=run_id, limit=5)
    assert rows, "Se esperaba al menos un registro en reality_bench_runs"
    bench_row = rows[0]

    assert bench_row.bench_run_id == run_id
    assert bench_row.run_id == run_id
    assert bench_row.total_episodes == 2
    assert bench_row.closure_rate == summary["success_rate"]
    assert bench_row.continuity_mean == summary["avg_metrics"]["viability_margin"]
    assert bench_row.collapse_count == 1
    assert bench_row.gate_profile == cfg.scenario_name
    assert bench_row.passed is True
    assert bench_row.summary.get("proxy_mapping") == {
        "closure_rate": "success_rate",
        "continuity_mean": "viability_margin",
    }

    serialized_summary = json.dumps(bench_row.summary)
    for metric in REMOVED_METRICS:
        assert metric not in serialized_summary

    storage.close()


def test_analysis_estadistico_funciona_con_muestras_pequenas_validas(tmp_path: Path) -> None:
    dir_1x1 = tmp_path / "r1x1"
    dir_5x5 = tmp_path / "r5x5"
    dir_1x1.mkdir()
    dir_5x5.mkdir()

    rows_1x1 = [
        {
            "success_rate": 1.0,
            "viability_margin": 0.20,
            "intervention_precision": 0.10,
            "proposition_diversity": 1.2,
            "spatial_information_usage": 0.15,
            "wall_time_ms": 18.0,
            "artifact_size_bytes": 1800,
            "reasoning_trace_length": 6,
            "ivc_r": 0.34,
            "failure_primary": None,
        },
        {
            "success_rate": 0.0,
            "viability_margin": 0.05,
            "intervention_precision": 0.02,
            "proposition_diversity": 1.1,
            "spatial_information_usage": 0.10,
            "wall_time_ms": 17.0,
            "artifact_size_bytes": 1750,
            "reasoning_trace_length": 6,
            "ivc_r": 0.25,
            "failure_primary": "certification_failure",
        },
    ]
    rows_5x5 = [
        {
            "success_rate": 1.0,
            "viability_margin": 0.27,
            "intervention_precision": 0.16,
            "proposition_diversity": 1.5,
            "spatial_information_usage": 0.35,
            "wall_time_ms": 26.0,
            "artifact_size_bytes": 3100,
            "reasoning_trace_length": 6,
            "ivc_r": 0.40,
            "failure_primary": None,
        },
        {
            "success_rate": 1.0,
            "viability_margin": 0.24,
            "intervention_precision": 0.11,
            "proposition_diversity": 1.4,
            "spatial_information_usage": 0.30,
            "wall_time_ms": 24.0,
            "artifact_size_bytes": 3000,
            "reasoning_trace_length": 6,
            "ivc_r": 0.37,
            "failure_primary": None,
        },
    ]

    with (dir_1x1 / "episodes.jsonl").open("w") as f:
        for row in rows_1x1:
            f.write(json.dumps(row) + "\n")
    with (dir_5x5 / "episodes.jsonl").open("w") as f:
        for row in rows_5x5:
            f.write(json.dumps(row) + "\n")

    analyzer = BenchmarkAnalyzer()
    analyzer.load_results(dir_1x1, dir_5x5)
    comparison = analyzer.compute_statistical_comparison()

    assert "success_rate" in comparison
    assert "viability_margin" in comparison
    assert "net_cognitive_gain" in comparison
    assert "failure_analysis" in comparison

    net_gain = comparison["net_cognitive_gain"]
    assert "wall_time_ratio" in net_gain
    assert "artifact_ratio" in net_gain
    assert "memory_pressure_mb" not in json.dumps(net_gain)


def test_analysis_estadistico_no_rompe_con_muestra_uno_vs_uno(tmp_path: Path) -> None:
    dir_1x1 = tmp_path / "s1"
    dir_5x5 = tmp_path / "s5"
    dir_1x1.mkdir()
    dir_5x5.mkdir()

    row_1x1 = {
        "success_rate": 1.0,
        "viability_margin": 0.20,
        "intervention_precision": 0.10,
        "proposition_diversity": 1.1,
        "spatial_information_usage": 0.12,
        "wall_time_ms": 18.0,
        "artifact_size_bytes": 1800,
        "reasoning_trace_length": 6,
        "ivc_r": 0.32,
        "failure_primary": None,
    }
    row_5x5 = {
        "success_rate": 1.0,
        "viability_margin": 0.24,
        "intervention_precision": 0.15,
        "proposition_diversity": 1.4,
        "spatial_information_usage": 0.31,
        "wall_time_ms": 24.0,
        "artifact_size_bytes": 2900,
        "reasoning_trace_length": 6,
        "ivc_r": 0.39,
        "failure_primary": None,
    }

    (dir_1x1 / "episodes.jsonl").write_text(json.dumps(row_1x1) + "\n")
    (dir_5x5 / "episodes.jsonl").write_text(json.dumps(row_5x5) + "\n")

    analyzer = BenchmarkAnalyzer()
    analyzer.load_results(dir_1x1, dir_5x5)
    comparison = analyzer.compute_statistical_comparison()

    assert "success_rate" in comparison
    assert "viability_margin" in comparison
    assert comparison["success_rate"].get("cohens_d") == 0.0
