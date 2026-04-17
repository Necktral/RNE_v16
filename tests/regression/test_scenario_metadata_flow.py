"""Tests de flujo de scenario_metadata a través de todo el pipeline.

Verifica que ScenarioEpisodeRunner produce scenario_metadata y que este
se conserva en assessment, certificate, decision y memory.
"""

from pathlib import Path

from runtime.storage import StorageConfig, StorageFactory
from runtime.world.scenario_runner import ScenarioEpisodeRunner


def _storage(tmp_path: Path):
    config = StorageConfig(
        mode="sqlite",
        sqlite_db_path=str(tmp_path / "metadata.db"),
        postgres_dsn=None,
        artifact_root=tmp_path / "artifacts",
        prefer_postgres_reads=True,
        strict_dual_write=False,
    )
    return StorageFactory.create_facade(config)


class TestScenarioMetadataFlow:
    """scenario_metadata flows through the entire pipeline."""

    def test_episode_payload_has_scenario_metadata(self, tmp_path: Path):
        """Episode payload includes scenario_metadata dict."""
        storage = _storage(tmp_path)
        runner = ScenarioEpisodeRunner(
            storage=storage,
            run_id="run-meta-1",
            scenario="thermal_homeostasis",
        )
        result = runner.run_episode(external_input=0.04)

        sm = result["episode"]["scenario_metadata"]
        assert sm["scenario_name"] == "thermal_homeostasis"
        assert sm["scenario_version"] == "1.0"
        assert isinstance(sm["scenario_config_hash"], str)
        assert sm["main_variable"] == "temperature"
        assert sm["alarm_threshold"] == 0.85
        assert "activate_cooling" in sm["interventions"]
        storage.close()

    def test_episode_still_has_backward_compat_scenario_field(self, tmp_path: Path):
        """Backward-compatible 'scenario' field is still present."""
        storage = _storage(tmp_path)
        runner = ScenarioEpisodeRunner(
            storage=storage, run_id="run-meta-bc", scenario="resource_management",
        )
        result = runner.run_episode(external_input=0.03)
        assert result["episode"]["scenario"] == "resource_management"
        assert result["episode"]["scenario_metadata"]["scenario_name"] == "resource_management"
        storage.close()

    def test_artifact_metadata_has_scenario_metadata(self, tmp_path: Path):
        """Materialized artifact metadata includes scenario_metadata."""
        storage = _storage(tmp_path)
        runner = ScenarioEpisodeRunner(
            storage=storage, run_id="run-meta-art", scenario="thermal_homeostasis",
        )
        result = runner.run_episode(external_input=0.04)
        art_meta = result["artifact"]["metadata"]
        assert "scenario_metadata" in art_meta
        assert art_meta["scenario_metadata"]["scenario_name"] == "thermal_homeostasis"
        storage.close()

    def test_certificate_metadata_has_scenario_metadata(self, tmp_path: Path):
        """Certificate built from episode carries scenario_metadata."""
        storage = _storage(tmp_path)
        runner = ScenarioEpisodeRunner(
            storage=storage, run_id="run-meta-cert", scenario="thermal_homeostasis",
        )
        result = runner.run_episode(external_input=0.04)

        certs = storage.list_episode_certificates(run_id="run-meta-cert", limit=5)
        assert certs
        cert = certs[0]
        assert "scenario_metadata" in cert.metadata
        assert cert.metadata["scenario_metadata"]["scenario_name"] == "thermal_homeostasis"
        storage.close()

    def test_promotion_decision_has_scenario_metadata(self, tmp_path: Path):
        """Promotion decision includes scenario_metadata."""
        storage = _storage(tmp_path)
        runner = ScenarioEpisodeRunner(
            storage=storage, run_id="run-meta-dec", scenario="resource_management",
        )
        result = runner.run_episode(external_input=0.03)

        decisions = storage.list_promotion_decisions(run_id="run-meta-dec", limit=5)
        assert decisions
        dec = decisions[0]
        assert "scenario_metadata" in dec.metadata
        assert dec.metadata["scenario_metadata"]["scenario_name"] == "resource_management"
        storage.close()

    def test_memory_records_have_scenario_metadata(self, tmp_path: Path):
        """Memory records written by promotion carry scenario_metadata."""
        storage = _storage(tmp_path)
        runner = ScenarioEpisodeRunner(
            storage=storage, run_id="run-meta-mem", scenario="thermal_homeostasis",
        )
        result = runner.run_episode(external_input=0.04)

        memories = storage.retrieve_memory_records(
            run_id="run-meta-mem", scales=["micro", "meso", "macro"], limit=20,
        )
        # Should have at least micro + meso if certified
        if memories:
            for mem in memories:
                assert "scenario_metadata" in mem.metadata, (
                    f"Memory {mem.memory_id} (scale={mem.scale}) missing scenario_metadata"
                )
                assert mem.metadata["scenario_metadata"]["scenario_name"] == "thermal_homeostasis"
        storage.close()

    def test_resource_scenario_metadata_differs_from_thermal(self, tmp_path: Path):
        """scenario_metadata is distinct per scenario."""
        storage = _storage(tmp_path)

        runner_t = ScenarioEpisodeRunner(
            storage=storage, run_id="run-meta-diff-t", scenario="thermal_homeostasis",
        )
        res_t = runner_t.run_episode(external_input=0.04)

        runner_r = ScenarioEpisodeRunner(
            storage=storage, run_id="run-meta-diff-r", scenario="resource_management",
        )
        res_r = runner_r.run_episode(external_input=0.03)

        sm_t = res_t["episode"]["scenario_metadata"]
        sm_r = res_r["episode"]["scenario_metadata"]

        assert sm_t["scenario_name"] != sm_r["scenario_name"]
        assert sm_t["main_variable"] != sm_r["main_variable"]
        assert sm_t["scenario_config_hash"] != sm_r["scenario_config_hash"]
        storage.close()

    def test_assessment_details_have_scenario_metadata(self, tmp_path: Path):
        """Reality assessment details includes scenario_metadata."""
        from runtime.reality.service import RealityValidationService

        storage = _storage(tmp_path)
        service = RealityValidationService(storage=storage)
        runner = ScenarioEpisodeRunner(
            storage=storage, run_id="run-meta-assess", scenario="thermal_homeostasis",
        )
        result = runner.run_episode(external_input=0.04)

        assessment = service.evaluate_episode_result(
            run_id="run-meta-assess",
            bench_run_id="bench-meta",
            result=result,
            previous_result=None,
            recent_assessments=[],
            scenario_name="thermal_homeostasis",
        )
        assert "scenario_metadata" in assessment.details
        assert assessment.details["scenario_metadata"]["scenario_name"] == "thermal_homeostasis"
        storage.close()
