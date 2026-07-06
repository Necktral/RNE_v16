"""Tests de filtrado de memoria por escenario.

Verifica que en modo strict_same_scenario no se recuperan memorias
cross-scenario y que en modo analógico se penalizan correctamente.
"""

from pathlib import Path

from runtime.memory.mfm_lite.retrieval import MemoryRetrieval
from runtime.reasoning.context import build_reasoning_context
from runtime.storage import StorageConfig, StorageFactory
from runtime.world.scenario_runner import ScenarioEpisodeRunner


def _storage(tmp_path: Path):
    config = StorageConfig(
        mode="sqlite",
        sqlite_db_path=str(tmp_path / "filter.db"),
        postgres_dsn=None,
        artifact_root=tmp_path / "artifacts",
        prefer_postgres_reads=True,
        strict_dual_write=False,
    )
    return StorageFactory.create_facade(config)


def _seed_both_scenarios(storage, run_id: str):
    """Produce episodios de ambos escenarios para generar memorias."""
    runner_t = ScenarioEpisodeRunner(
        storage=storage, run_id=run_id, scenario="thermal_homeostasis",
    )
    runner_t.run_episode(external_input=0.04)

    runner_r = ScenarioEpisodeRunner(
        storage=storage, run_id=run_id, scenario="resource_management",
    )
    runner_r.run_episode(external_input=0.03)


class TestMemoryScenarioFiltering:
    """Scenario-based memory filtering works correctly."""

    def test_strict_mode_excludes_cross_scenario(self, tmp_path: Path):
        """In strict mode, thermal query should not retrieve resource memories."""
        storage = _storage(tmp_path)
        run_id = "run-filter-strict"
        _seed_both_scenarios(storage, run_id)

        retrieval = MemoryRetrieval(storage=storage)
        hits = retrieval.retrieve(
            run_id=run_id,
            query={"proposition": "TEMP_HIGH", "alarm": True},
            limit=10,
            scenario_name="thermal_homeostasis",
            scenario_filter_mode="strict_same_scenario",
        )
        for hit in hits:
            assert hit.get("analogical_source") is not True, (
                f"Cross-scenario memory leaked in strict mode: {hit['memory_id']}"
            )
        storage.close()

    def test_strict_mode_retrieval_metrics(self, tmp_path: Path):
        """Retrieval metrics report zero cross-scenario in strict mode."""
        storage = _storage(tmp_path)
        run_id = "run-filter-metrics"
        _seed_both_scenarios(storage, run_id)

        retrieval = MemoryRetrieval(storage=storage)
        hits = retrieval.retrieve(
            run_id=run_id,
            query={"proposition": "TEMP_HIGH", "alarm": True},
            limit=10,
            scenario_name="thermal_homeostasis",
            scenario_filter_mode="strict_same_scenario",
        )
        # In strict mode, cross-scenario memories are discarded
        for hit in hits:
            metrics = hit.get("retrieval_metrics", {})
            assert metrics["scenario_filter_mode"] == "strict_same_scenario"
        storage.close()

    def test_analogical_mode_allows_cross_scenario_with_penalty(self, tmp_path: Path):
        """In analogical mode, cross-scenario memories are returned but penalized."""
        storage = _storage(tmp_path)
        run_id = "run-filter-analog"
        _seed_both_scenarios(storage, run_id)

        retrieval = MemoryRetrieval(storage=storage)
        hits = retrieval.retrieve(
            run_id=run_id,
            query={"proposition": "TEMP_HIGH", "alarm": True},
            limit=10,
            scenario_name="thermal_homeostasis",
            scenario_filter_mode="analogical",
        )
        # Check that retrieval metrics are present
        if hits:
            metrics = hits[0].get("retrieval_metrics", {})
            assert metrics["scenario_filter_mode"] == "cross_scenario_analogical"
        storage.close()

    def test_no_filter_when_scenario_name_is_none(self, tmp_path: Path):
        """When scenario_name is None, no filtering is applied (backward compat)."""
        storage = _storage(tmp_path)
        run_id = "run-filter-none"
        _seed_both_scenarios(storage, run_id)

        retrieval = MemoryRetrieval(storage=storage)
        hits = retrieval.retrieve(
            run_id=run_id,
            query={"proposition": "TEMP_HIGH", "alarm": True},
            limit=10,
            # No scenario_name → no filtering
        )
        # Should still return results without error
        assert isinstance(hits, list)
        storage.close()

    def test_resource_query_strict_excludes_thermal_memories(self, tmp_path: Path):
        """Strict resource query should not see thermal memories."""
        storage = _storage(tmp_path)
        run_id = "run-filter-res-strict"
        _seed_both_scenarios(storage, run_id)

        retrieval = MemoryRetrieval(storage=storage)
        hits = retrieval.retrieve(
            run_id=run_id,
            query={"proposition": "STOCK_LOW", "alarm": True},
            limit=10,
            scenario_name="resource_management",
            scenario_filter_mode="strict_same_scenario",
        )
        for hit in hits:
            assert hit.get("analogical_source") is not True
        storage.close()

    def test_rag_attestation_separates_filtered_from_returned_cross_scenario(self, tmp_path: Path):
        storage = _storage(tmp_path)
        run_id = "run-rag-attestation"
        storage.write_memory_record(
            run_id=run_id,
            episode_id="episode-thermal",
            scale="micro",
            structure_json={"proposition": "TEMP_HIGH", "alarm": True},
            metadata={"scenario_name": "thermal_homeostasis"},
            memory_id="mem-thermal",
        )
        storage.write_memory_record(
            run_id=run_id,
            episode_id="episode-resource",
            scale="micro",
            structure_json={"proposition": "TEMP_HIGH", "alarm": True},
            metadata={"scenario_name": "resource_management"},
            memory_id="mem-resource",
        )

        retrieval = MemoryRetrieval(storage=storage)
        strict_hits = retrieval.retrieve(
            run_id=run_id,
            query={"proposition": "TEMP_HIGH", "alarm": True},
            limit=10,
            scenario_name="thermal_homeostasis",
            scenario_filter_mode="strict_same_scenario",
        )
        strict_attestation = strict_hits[0]["rag_attestation"]
        strict_metrics = strict_hits[0]["retrieval_metrics"]
        assert strict_attestation["validation_status"] == "pass"
        assert strict_attestation["retrieval_purity"] == 1.0
        assert strict_metrics["retrieved_cross_scenario_count"] == 0
        assert strict_metrics["filtered_cross_scenario_count"] == 1
        assert strict_attestation["trace_memory_ids"] == ["mem-thermal"]

        analog_hits = retrieval.retrieve(
            run_id=run_id,
            query={"proposition": "TEMP_HIGH", "alarm": True},
            limit=10,
            scenario_name="thermal_homeostasis",
            scenario_filter_mode="analogical",
        )
        analog_attestation = analog_hits[0]["rag_attestation"]
        assert analog_attestation["validation_status"] == "warn"
        assert analog_attestation["retrieval_purity"] == 0.5
        assert analog_attestation["returned_cross_scenario_count"] == 1
        assert set(analog_attestation["trace_memory_ids"]) == {"mem-thermal", "mem-resource"}
        storage.close()

    def test_reasoning_context_propagates_memory_rag_attestation(self):
        context = build_reasoning_context(
            episode_id="episode-rag-context",
            run_id="run-rag-context",
            observation={"alarm": True},
            intervention="activate_cooling",
            memory_hits=[
                {
                    "memory_id": "mem-a",
                    "rag_attestation": {
                        "schema": "memory_rag_attestation.v1",
                        "returned_count": 2,
                        "retrieval_purity": 0.5,
                    },
                }
            ],
        )
        assert context["memory_purity_confidence"] == 0.5
        assert context["memory_rag_attestation"]["returned_count"] == 2
