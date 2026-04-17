"""Tests del benchmark heterogéneo mínimo.

Verifica que el benchmark heterogéneo ejecuta, produce métricas
válidas, cierre aceptable y artifact materializado.
"""

from pathlib import Path

from runtime.reality.service import RealityValidationService
from runtime.storage import StorageConfig, StorageFactory


def _storage(tmp_path: Path):
    config = StorageConfig(
        mode="sqlite",
        sqlite_db_path=str(tmp_path / "hetero.db"),
        postgres_dsn=None,
        artifact_root=tmp_path / "artifacts",
        prefer_postgres_reads=True,
        strict_dual_write=False,
    )
    return StorageFactory.create_facade(config)


class TestHeterogeneousBenchmark:
    """Heterogeneous benchmark executes and produces valid results."""

    def test_benchmark_runs_and_returns_results(self, tmp_path: Path):
        """Benchmark executes the default 5-step sequence."""
        storage = _storage(tmp_path)
        service = RealityValidationService(storage=storage)
        result = service.run_heterogeneous_benchmark(
            run_id="run-hetero-1",
            gate_profile="ci",
        )

        assert result["bench_run"]["total_episodes"] == 5
        assert len(result["assessments"]) == 5
        storage.close()

    def test_benchmark_produces_artifact(self, tmp_path: Path):
        """Benchmark materializes an artifact on disk."""
        storage = _storage(tmp_path)
        service = RealityValidationService(storage=storage)
        result = service.run_heterogeneous_benchmark(
            run_id="run-hetero-art",
        )

        assert Path(result["artifact"]["abs_path"]).exists()
        storage.close()

    def test_benchmark_has_transition_metrics(self, tmp_path: Path):
        """Benchmark computes transition metrics between different scenarios."""
        storage = _storage(tmp_path)
        service = RealityValidationService(storage=storage)
        result = service.run_heterogeneous_benchmark(
            run_id="run-hetero-trans",
        )

        tm = result["transition_metrics"]
        assert tm["transition_count"] >= 1
        assert isinstance(tm["transitions"], list)
        assert isinstance(tm["mean_continuity_at_transition"], float)
        storage.close()

    def test_benchmark_has_memory_metrics(self, tmp_path: Path):
        """Benchmark computes memory contamination metrics."""
        storage = _storage(tmp_path)
        service = RealityValidationService(storage=storage)
        result = service.run_heterogeneous_benchmark(
            run_id="run-hetero-mem",
        )

        mm = result["memory_metrics"]
        assert "total_same_scenario_retrievals" in mm
        assert "total_cross_scenario_retrievals" in mm
        assert "pollution_detected" in mm
        storage.close()

    def test_benchmark_has_scenario_metrics(self, tmp_path: Path):
        """Benchmark computes per-scenario metrics."""
        storage = _storage(tmp_path)
        service = RealityValidationService(storage=storage)
        result = service.run_heterogeneous_benchmark(
            run_id="run-hetero-sm",
        )

        summary = result["bench_run"]["summary"]
        sm = summary["scenario_metrics"]
        assert "thermal_homeostasis" in sm
        assert "resource_management" in sm
        storage.close()

    def test_benchmark_with_custom_sequence(self, tmp_path: Path):
        """Benchmark accepts custom scenario sequence."""
        storage = _storage(tmp_path)
        service = RealityValidationService(storage=storage)
        custom_seq = [
            {"scenario": "resource_management", "external_input": 0.03},
            {"scenario": "thermal_homeostasis", "external_input": 0.04},
            {"scenario": "resource_management", "external_input": 0.05},
        ]
        result = service.run_heterogeneous_benchmark(
            run_id="run-hetero-custom",
            scenario_sequence=custom_seq,
        )

        assert result["bench_run"]["total_episodes"] == 3
        storage.close()

    def test_benchmark_emits_event(self, tmp_path: Path):
        """Benchmark emits heterogeneous_validation.completed event."""
        storage = _storage(tmp_path)
        service = RealityValidationService(storage=storage)
        service.run_heterogeneous_benchmark(run_id="run-hetero-evt")

        events = storage.list_events(run_id="run-hetero-evt", limit=100)
        assert any(
            e.event_type == "reality.heterogeneous_validation.completed"
            for e in events
        )
        storage.close()
