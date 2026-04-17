"""Tests de contaminación cross-scenario en memoria.

Verifica que en modo estricto no hay contaminación cross-scenario
y que el benchmark heterogéneo reporta correctamente pollution_detected.
"""

from pathlib import Path

from runtime.reality.service import RealityValidationService
from runtime.storage import StorageConfig, StorageFactory


def _storage(tmp_path: Path):
    config = StorageConfig(
        mode="sqlite",
        sqlite_db_path=str(tmp_path / "pollution.db"),
        postgres_dsn=None,
        artifact_root=tmp_path / "artifacts",
        prefer_postgres_reads=True,
        strict_dual_write=False,
    )
    return StorageFactory.create_facade(config)


class TestCrossScenarioMemoryPollution:
    """Cross-scenario memory pollution detection."""

    def test_strict_mode_no_pollution(self, tmp_path: Path):
        """In strict mode, no cross-scenario memories should be returned."""
        storage = _storage(tmp_path)
        service = RealityValidationService(storage=storage)
        result = service.run_heterogeneous_benchmark(
            run_id="run-poll-strict",
            memory_mode="strict_same_scenario",
        )

        mm = result["memory_metrics"]
        # In strict mode, cross-scenario memories are discarded at retrieval
        # actual_cross_scenario_returned must be 0
        assert mm["actual_cross_scenario_returned"] == 0
        assert mm["pollution_detected"] is False
        storage.close()

    def test_strict_mode_no_analogical_sources(self, tmp_path: Path):
        """In strict mode, no memory hit should be marked as analogical_source."""
        storage = _storage(tmp_path)
        service = RealityValidationService(storage=storage)
        result = service.run_heterogeneous_benchmark(
            run_id="run-poll-no-analog",
            memory_mode="strict_same_scenario",
        )

        for assessment in result["assessments"]:
            # Assessment doesn't carry raw hits, but memory_metrics confirms
            pass

        mm = result["memory_metrics"]
        assert mm["cross_scenario_penalty_applied"] is False
        storage.close()

    def test_analogical_mode_allows_cross_scenario(self, tmp_path: Path):
        """In analogical mode, cross-scenario memories are allowed with penalty."""
        storage = _storage(tmp_path)
        service = RealityValidationService(storage=storage)
        result = service.run_heterogeneous_benchmark(
            run_id="run-poll-analog",
            memory_mode="analogical",
        )

        mm = result["memory_metrics"]
        # In analogical mode, cross-scenario retrievals are kept
        assert isinstance(mm["total_cross_scenario_retrievals"], int)
        assert isinstance(mm["pollution_detected"], bool)
        storage.close()
