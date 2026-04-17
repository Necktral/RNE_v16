"""Tests de métricas de continuidad en transiciones entre escenarios.

Verifica que las transiciones thermal→resource y resource→thermal
se computan correctamente y mantienen continuidad razonable.
"""

from pathlib import Path

from runtime.reality.service import RealityValidationService
from runtime.storage import StorageConfig, StorageFactory


def _storage(tmp_path: Path):
    config = StorageConfig(
        mode="sqlite",
        sqlite_db_path=str(tmp_path / "transition.db"),
        postgres_dsn=None,
        artifact_root=tmp_path / "artifacts",
        prefer_postgres_reads=True,
        strict_dual_write=False,
    )
    return StorageFactory.create_facade(config)


class TestTransitionContinuityMetrics:
    """Transition continuity metrics are computed correctly."""

    def test_transitions_are_detected(self, tmp_path: Path):
        """Transitions between different scenarios are detected in the sequence."""
        storage = _storage(tmp_path)
        service = RealityValidationService(storage=storage)
        # Default sequence: thermal, thermal, resource, thermal, resource
        # Transitions at index 2 (thermal->resource), 3 (resource->thermal), 4 (thermal->resource)
        result = service.run_heterogeneous_benchmark(
            run_id="run-trans-detect",
        )

        tm = result["transition_metrics"]
        assert tm["transition_count"] == 3
        storage.close()

    def test_transition_continuity_is_computed(self, tmp_path: Path):
        """Continuity at transitions is non-negative."""
        storage = _storage(tmp_path)
        service = RealityValidationService(storage=storage)
        result = service.run_heterogeneous_benchmark(
            run_id="run-trans-cont",
        )

        tm = result["transition_metrics"]
        assert tm["mean_continuity_at_transition"] >= 0.0
        for t in tm["transitions"]:
            assert t["continuity_score"] >= 0.0
            assert isinstance(t["closure_passed"], bool)
        storage.close()

    def test_transition_details_have_scenario_names(self, tmp_path: Path):
        """Each transition record has from_scenario and to_scenario."""
        storage = _storage(tmp_path)
        service = RealityValidationService(storage=storage)
        result = service.run_heterogeneous_benchmark(
            run_id="run-trans-names",
        )

        for t in result["transition_metrics"]["transitions"]:
            assert "from_scenario" in t
            assert "to_scenario" in t
            assert t["from_scenario"] != t["to_scenario"]
        storage.close()

    def test_no_transition_in_homogeneous_sequence(self, tmp_path: Path):
        """A homogeneous sequence has no transitions."""
        storage = _storage(tmp_path)
        service = RealityValidationService(storage=storage)
        seq = [
            {"scenario": "thermal_homeostasis", "external_input": 0.03},
            {"scenario": "thermal_homeostasis", "external_input": 0.05},
            {"scenario": "thermal_homeostasis", "external_input": 0.04},
        ]
        result = service.run_heterogeneous_benchmark(
            run_id="run-trans-homo",
            scenario_sequence=seq,
        )

        tm = result["transition_metrics"]
        assert tm["transition_count"] == 0
        assert tm["transitions"] == []
        storage.close()

    def test_continuity_does_not_collapse_at_transition(self, tmp_path: Path):
        """Continuity at transitions should not be zero (unless real collapse)."""
        storage = _storage(tmp_path)
        service = RealityValidationService(storage=storage)
        result = service.run_heterogeneous_benchmark(
            run_id="run-trans-nozero",
        )

        # At least some transitions should have non-zero continuity
        tm = result["transition_metrics"]
        if tm["transitions"]:
            continuities = [t["continuity_score"] for t in tm["transitions"]]
            assert max(continuities) > 0.0, "All transitions have zero continuity — possible collapse"
        storage.close()
