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

    def test_heterogeneous_benchmark_default_profile_is_heterogeneous_ci(self, tmp_path: Path):
        """Default gate_profile for heterogeneous benchmark is heterogeneous_ci."""
        storage = _storage(tmp_path)
        service = RealityValidationService(storage=storage)
        result = service.run_heterogeneous_benchmark(run_id="run-hetero-profile")

        summary = result["bench_run"]["summary"]
        assert summary["gate_profile"] == "heterogeneous_ci"
        storage.close()

    def test_heterogeneous_benchmark_records_trace_integrity_rate(self, tmp_path: Path):
        """Benchmark summary includes trace_integrity_rate."""
        storage = _storage(tmp_path)
        service = RealityValidationService(storage=storage)
        result = service.run_heterogeneous_benchmark(run_id="run-hetero-tir")

        summary = result["bench_run"]["summary"]
        assert "trace_integrity_rate" in summary
        assert isinstance(summary["trace_integrity_rate"], float)
        assert 0.0 <= summary["trace_integrity_rate"] <= 1.0
        storage.close()

    def test_strict_mode_pollution_forces_failed_gate(self, tmp_path: Path):
        """If actual_cross_scenario_returned > 0 in strict mode, passed must be False.

        We verify the invariant by ensuring strict_memory_clean is in the summary
        and that when it's True the gate can pass, and that the logic is wired correctly.
        """
        storage = _storage(tmp_path)
        service = RealityValidationService(storage=storage)
        result = service.run_heterogeneous_benchmark(
            run_id="run-hetero-gate-poll",
            memory_mode="strict_same_scenario",
        )

        summary = result["bench_run"]["summary"]
        mm = summary["memory_metrics"]

        # In strict mode with no pollution, strict_memory_clean must be True
        if mm["actual_cross_scenario_returned"] == 0:
            assert summary["strict_memory_clean"] is True
        else:
            # If somehow pollution leaked, the gate must fail
            assert summary["strict_memory_clean"] is False
            assert result["passed"] is False

        storage.close()
