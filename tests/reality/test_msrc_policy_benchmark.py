from pathlib import Path

from runtime.reality.msrc_policy_benchmark import MSRCPolicyBenchmarkRunner
from runtime.reality.service import RealityValidationService
from runtime.storage import StorageConfig, StorageFactory


def _storage(tmp_path: Path):
    config = StorageConfig(
        mode="sqlite",
        sqlite_db_path=str(tmp_path / "msrc_policy.db"),
        postgres_dsn=None,
        artifact_root=tmp_path / "artifacts",
        prefer_postgres_reads=False,
        strict_dual_write=False,
    )
    return StorageFactory.create_facade(config)


def test_policy_benchmark_runner_persists_outputs_and_db(tmp_path: Path):
    storage = _storage(tmp_path)
    runner = MSRCPolicyBenchmarkRunner(storage=storage, output_root=tmp_path / "bench")
    out_dir = tmp_path / "bench" / "always_1x1"

    summary = runner.run_policy(
        run_id="msrc-run-1",
        policy_name="always_1x1",
        episodes=2,
        base_seed=100,
        output_dir=out_dir,
        scenario_params={
            "initial_temperature": 0.82,
            "alarm_threshold": 0.85,
            "cooling_effect": 0.07,
            "grid_size": 1,
        },
        level_label="SAFE",
        topology_label="uniform",
    )

    assert (out_dir / "episodes.jsonl").exists()
    assert (out_dir / "summary.json").exists()
    assert (out_dir / "scale_actions.jsonl").exists()
    assert summary["policy_name"] == "always_1x1"
    assert "msrc_metrics" in summary
    assert summary["proxy_mapping"]["closure_rate"] == "success_rate"
    assert "backbone_floor_satisfied_rate" in summary
    assert "sequence_validation_fail_rate" in summary
    assert "fallback_to_safe_sequence_rate" in summary
    assert "optional_displacement_rate" in summary
    assert "closure_break_rate" in summary

    rows = storage.list_reality_bench_runs(run_id="msrc-run-1", limit=5)
    assert rows
    assert rows[0].summary.get("policy_name") == "always_1x1"
    storage.close()


def test_service_delegates_to_msrc_policy_benchmark(tmp_path: Path):
    storage = _storage(tmp_path)
    service = RealityValidationService(storage=storage)
    out_dir = tmp_path / "bench" / "adaptive"

    result = service.run_msrc_policy_benchmark(
        run_id="msrc-run-2",
        policy_name="adaptive_msrc",
        episodes=3,
        base_seed=200,
        output_dir=out_dir,
        scenario_params={
            "initial_temperature": 0.87,
            "alarm_threshold": 0.85,
            "cooling_effect": 0.07,
            "grid_size": 5,
            "topology": "hotspot_center",
        },
        level_label="WARNING",
        topology_label="hotspot_center",
    )

    assert result["policy_name"] == "adaptive_msrc"
    assert (out_dir / "summary.json").exists()
    # adaptive debe generar auditoría de decisiones
    assert (out_dir / "scale_decisions.jsonl").exists()
    assert (out_dir / "scale_actions.jsonl").exists()

    rows = storage.list_reality_bench_runs(run_id="msrc-run-2", limit=5)
    assert rows
    summary = rows[0].summary
    assert "msrc_metrics" in summary
    assert "vram_headroom_mean" in summary["msrc_metrics"]
    assert "keep_scale_rate" in summary["msrc_metrics"]
    assert "probe_rate" in summary["msrc_metrics"]
    assert "backbone_floor_satisfied_rate" in summary
    assert "closure_break_rate" in summary
    storage.close()


def test_aggressive_policy_is_supported_and_persists_decisions(tmp_path: Path):
    storage = _storage(tmp_path)
    service = RealityValidationService(storage=storage)
    out_dir = tmp_path / "bench" / "aggressive"

    result = service.run_msrc_policy_benchmark(
        run_id="msrc-run-3",
        policy_name="adaptive_msrc_aggressive",
        episodes=4,
        base_seed=300,
        output_dir=out_dir,
        scenario_params={
            "initial_temperature": 0.87,
            "alarm_threshold": 0.85,
            "cooling_effect": 0.07,
            "grid_size": 5,
            "topology": "checkerboard",
        },
        level_label="WARNING",
        topology_label="checkerboard",
    )

    assert result["policy_name"] == "adaptive_msrc_aggressive"
    assert (out_dir / "scale_decisions.jsonl").exists()
    assert (out_dir / "scale_actions.jsonl").exists()

    rows = storage.list_reality_bench_runs(run_id="msrc-run-3", limit=5)
    assert rows
    metrics = rows[0].summary["msrc_metrics"]
    assert "missed_upgrade_regret" in metrics
    assert "probe_commit_rate" in metrics
    storage.close()


def test_regime_v3_policy_persists_regime_metrics_and_decisions(tmp_path: Path):
    storage = _storage(tmp_path)
    service = RealityValidationService(storage=storage)
    out_dir = tmp_path / "bench" / "regime_v3"

    result = service.run_msrc_policy_benchmark(
        run_id="msrc-run-4",
        policy_name="adaptive_msrc_regime_v3",
        episodes=4,
        base_seed=400,
        output_dir=out_dir,
        scenario_params={
            "initial_temperature": 0.87,
            "alarm_threshold": 0.85,
            "cooling_effect": 0.07,
            "grid_size": 5,
            "topology": "hotspot_center",
        },
        level_label="WARNING",
        topology_label="hotspot_center",
    )

    assert result["policy_name"] == "adaptive_msrc_regime_v3"
    assert (out_dir / "scale_decisions.jsonl").exists()
    assert (out_dir / "scale_actions.jsonl").exists()

    rows = storage.list_reality_bench_runs(run_id="msrc-run-4", limit=5)
    assert rows
    metrics = rows[0].summary["msrc_metrics"]
    assert "regime_distribution" in metrics
    assert "regime_probe_rate" in metrics
    storage.close()
