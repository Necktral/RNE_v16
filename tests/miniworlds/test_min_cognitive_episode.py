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
    assert episode["context"]["observation"]["temperature"] >= 0.0
    assert episode["result"]["reasoning_sequence"] == ["ABD", "ANA", "CAU", "CTF", "DED", "PROB"]

    traces = storage.list_reasoning_traces(run_id="run-episode-1", limit=10)
    assert len(traces) == 6
    assert traces[0].family == "ABD"

    events = storage.list_events(run_id="run-episode-1", limit=20)
    assert any(evt.event_type == "episode.closed" for evt in events)

    artifact = result["artifact"]
    assert artifact["kind"] == "episode_report"
    assert Path(artifact["abs_path"]).exists()
