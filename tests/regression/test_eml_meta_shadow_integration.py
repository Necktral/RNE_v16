from pathlib import Path

from runtime.reasoning.scheduler_meta.meta_scheduler import MetaScheduler
from runtime.storage import StorageConfig, StorageFactory
from runtime.world.min_cognitive_episode import MinimalCognitiveEpisodeRunner


def _storage(tmp_path: Path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    config = StorageConfig(
        mode="sqlite",
        sqlite_db_path=str(tmp_path / "eml_meta.db"),
        postgres_dsn=None,
        artifact_root=tmp_path / "artifacts",
        prefer_postgres_reads=True,
        strict_dual_write=False,
    )
    return StorageFactory.create_facade(config)


def test_meta_eml_sr_requires_flag_and_allowlist(monkeypatch):
    monkeypatch.delenv("RNFE_EML_MODE", raising=False)
    monkeypatch.delenv("RNFE_META_EXPERIMENTAL_FAMILIES", raising=False)
    scheduler = MetaScheduler(mode="adaptive")
    off = scheduler.run(
        {
            "run_id": "run-eml-off",
            "uncertainty": 0.2,
            "symbolic_regularity": 0.9,
            "law_fit_signal": 0.9,
        }
    )
    assert "EML_SR" not in off["sequence"]

    monkeypatch.setenv("RNFE_EML_MODE", "shadow")
    monkeypatch.setenv("RNFE_META_EXPERIMENTAL_FAMILIES", "eml_sr")
    on = scheduler.run(
        {
            "run_id": "run-eml-on",
            "uncertainty": 0.2,
            "symbolic_regularity": 0.9,
            "law_fit_signal": 0.9,
        }
    )
    assert "EML_SR" in on["sequence"]
    assert any(step["family"] == "EML_SR" for step in on["trace"])


def test_min_episode_shadow_eml_generates_report_without_factual_change(
    tmp_path: Path, monkeypatch
):
    # baseline (EML disabled)
    monkeypatch.setenv("RNFE_EML_MODE", "disabled")
    storage_baseline = _storage(tmp_path / "baseline")
    runner_baseline = MinimalCognitiveEpisodeRunner(
        storage=storage_baseline, run_id="run-eml-baseline"
    )
    result_baseline = runner_baseline.run_episode(external_heat=0.05)

    # shadow on
    monkeypatch.setenv("RNFE_EML_MODE", "shadow")
    storage_shadow = _storage(tmp_path / "shadow")
    runner_shadow = MinimalCognitiveEpisodeRunner(
        storage=storage_shadow, run_id="run-eml-shadow"
    )
    result_shadow = runner_shadow.run_episode(external_heat=0.05)

    baseline_intervention = result_baseline["episode"]["context"]["intervention"]
    shadow_intervention = result_shadow["episode"]["context"]["intervention"]
    assert shadow_intervention == baseline_intervention
    assert (
        result_shadow["episode"]["result"]["updated_world"]
        == result_baseline["episode"]["result"]["updated_world"]
    )
    assert result_shadow["eml_shadow"]["enabled"] is True
    events = storage_shadow.list_events(run_id="run-eml-shadow", limit=100)
    assert any(item.event_type == "eml.run.completed" for item in events)
    artifacts = storage_shadow.list_artifacts(run_id="run-eml-shadow", kind="eml_report", limit=10)
    assert artifacts
    storage_baseline.close()
    storage_shadow.close()
