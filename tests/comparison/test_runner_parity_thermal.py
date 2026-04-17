"""Tests de paridad entre ScenarioEpisodeRunner y MinimalCognitiveEpisodeRunner.

Verifica que el runner basado en escenarios produce resultados comparables
al runner legacy para el caso térmico de referencia.
"""

from pathlib import Path

from runtime.storage import StorageConfig, StorageFactory
from runtime.world.min_cognitive_episode import MinimalCognitiveEpisodeRunner
from runtime.world.scenario_runner import ScenarioEpisodeRunner


def _storage(tmp_path: Path):
    config = StorageConfig(
        mode="sqlite",
        sqlite_db_path=str(tmp_path / "parity.db"),
        postgres_dsn=None,
        artifact_root=tmp_path / "artifacts",
        prefer_postgres_reads=True,
        strict_dual_write=False,
    )
    return StorageFactory.create_facade(config)


class TestRunnerParityThermal:
    """ScenarioEpisodeRunner(thermal) produces parity with legacy runner."""

    def test_both_runners_produce_valid_closure(self, tmp_path: Path):
        """Both runners produce episodes with valid closure."""
        storage = _storage(tmp_path)

        legacy = MinimalCognitiveEpisodeRunner(storage=storage, run_id="run-parity-legacy")
        res_legacy = legacy.run_episode(external_heat=0.05)

        scenario = ScenarioEpisodeRunner(
            storage=storage, run_id="run-parity-scenario",
            scenario="thermal_homeostasis",
        )
        res_scenario = scenario.run_episode(external_input=0.05)

        # Both should have valid reasoning sequences
        assert res_legacy["episode"]["result"]["reasoning_sequence"] == [
            "ABD", "ANA", "CAU", "CTF", "DED", "PROB",
        ]
        assert res_scenario["episode"]["result"]["reasoning_sequence"] == [
            "ABD", "ANA", "CAU", "CTF", "DED", "PROB",
        ]
        storage.close()

    def test_both_runners_produce_comparable_traces(self, tmp_path: Path):
        """Both runners produce traces of equal length."""
        storage = _storage(tmp_path)

        legacy = MinimalCognitiveEpisodeRunner(storage=storage, run_id="run-parity-trace-l")
        res_legacy = legacy.run_episode(external_heat=0.04)

        scenario = ScenarioEpisodeRunner(
            storage=storage, run_id="run-parity-trace-s",
            scenario="thermal_homeostasis",
        )
        res_scenario = scenario.run_episode(external_input=0.04)

        trace_legacy = res_legacy["episode"]["trace"]
        trace_scenario = res_scenario["episode"]["trace"]

        assert len(trace_legacy) == len(trace_scenario)
        storage.close()

    def test_both_runners_materialize_artifact(self, tmp_path: Path):
        """Both runners produce materialized artifacts."""
        storage = _storage(tmp_path)

        legacy = MinimalCognitiveEpisodeRunner(storage=storage, run_id="run-parity-art-l")
        res_legacy = legacy.run_episode(external_heat=0.04)

        scenario = ScenarioEpisodeRunner(
            storage=storage, run_id="run-parity-art-s",
            scenario="thermal_homeostasis",
        )
        res_scenario = scenario.run_episode(external_input=0.04)

        assert Path(res_legacy["artifact"]["abs_path"]).exists()
        assert Path(res_scenario["artifact"]["abs_path"]).exists()
        storage.close()

    def test_both_runners_produce_certification(self, tmp_path: Path):
        """Both runners go through certification without degradation."""
        storage = _storage(tmp_path)

        legacy = MinimalCognitiveEpisodeRunner(storage=storage, run_id="run-parity-cert-l")
        res_legacy = legacy.run_episode(external_heat=0.04)

        scenario = ScenarioEpisodeRunner(
            storage=storage, run_id="run-parity-cert-s",
            scenario="thermal_homeostasis",
        )
        res_scenario = scenario.run_episode(external_input=0.04)

        valid_verdicts = {"certified", "PASSED", "CONDITIONALLY_PASSED"}
        assert res_legacy["certification"]["verdict"] in valid_verdicts
        assert res_scenario["certification"]["verdict"] in valid_verdicts
        storage.close()

    def test_scenario_runner_adds_metadata_legacy_does_not(self, tmp_path: Path):
        """ScenarioEpisodeRunner adds scenario_metadata that legacy doesn't."""
        storage = _storage(tmp_path)

        legacy = MinimalCognitiveEpisodeRunner(storage=storage, run_id="run-parity-meta-l")
        res_legacy = legacy.run_episode(external_heat=0.04)

        scenario = ScenarioEpisodeRunner(
            storage=storage, run_id="run-parity-meta-s",
            scenario="thermal_homeostasis",
        )
        res_scenario = scenario.run_episode(external_input=0.04)

        # Legacy doesn't have scenario_metadata
        assert "scenario_metadata" not in res_legacy["episode"]

        # Scenario runner does
        assert "scenario_metadata" in res_scenario["episode"]
        assert res_scenario["episode"]["scenario_metadata"]["scenario_name"] == "thermal_homeostasis"
        storage.close()
