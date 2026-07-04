from pathlib import Path

from runtime.storage import StorageConfig, StorageFactory
from runtime.world.min_cognitive_episode import MinimalCognitiveEpisodeRunner


def _storage(tmp_path):
    config = StorageConfig(
        mode="sqlite",
        sqlite_db_path=str(tmp_path / "episode.db"),
        postgres_dsn=None,
        artifact_root=tmp_path / "artifacts",
        prefer_postgres_reads=True,
        strict_dual_write=False,
    )
    return StorageFactory.create_facade(config)


def test_min_cognitive_episode_closes_triadic_loop(tmp_path):
    storage = _storage(tmp_path)
    runner = MinimalCognitiveEpisodeRunner(storage=storage, run_id="run-episode-1")
    result = runner.run_episode(external_heat=0.05)

    episode = result["episode"]
    assert episode["episode_id"].startswith("episode-")
    assert episode["closure_profile"] == "baseline_fixed"
    assert episode["context"]["observation"]["temperature"] >= 0.0
    assert episode["result"]["reasoning_sequence"] == ["ABD", "ANA", "CAU", "CTF", "DED", "PROB"]
    assert result["reasoning"]["mode"] == "fixed"
    reasoning_context = episode["trace"][0]["detail"]["reasoning_context"]
    assert reasoning_context["formula"] == "TEMP_HIGH -> ACTIVATE_COOLING"
    assert "memory_hits" in reasoning_context
    assert "counterfactual" in reasoning_context
    assert "updated_world" in reasoning_context
    assert reasoning_context["relation_kind"] in {"support", "contradiction"}
    assert reasoning_context["scenario"] == "thermal_homeostasis"

    traces = storage.list_reasoning_traces(run_id="run-episode-1", limit=10)
    assert len(traces) == 6
    assert traces[0].family == "ABD"
    assert traces[0].detail["reasoning_context"]["episode_id"] == episode["episode_id"]
    ded_trace = next(step for step in traces if step.family == "DED")
    assert ded_trace.detail["artifacts"]["solver_result"] == "sat"
    assert ded_trace.detail["artifacts"]["formula_normalized"] == "TEMP_HIGH -> ACTIVATE_COOLING"
    assert ded_trace.detail["artifacts"]["z3_expression"]

    events = storage.list_events(run_id="run-episode-1", limit=20)
    assert any(evt.event_type == "episode.closed" for evt in events)

    artifact = result["artifact"]
    assert artifact["kind"] == "episode_report"
    assert Path(artifact["abs_path"]).exists()
