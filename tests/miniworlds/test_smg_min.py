from runtime.smg import SMGMin
from runtime.storage import StorageConfig, StorageFactory


def _storage(tmp_path):
    config = StorageConfig(
        mode="sqlite",
        sqlite_db_path=str(tmp_path / "smg.db"),
        postgres_dsn=None,
        artifact_root=tmp_path / "artifacts",
        prefer_postgres_reads=True,
        strict_dual_write=False,
    )
    return StorageFactory.create_facade(config)


def test_smg_min_observation_sign_and_relations(tmp_path):
    storage = _storage(tmp_path)
    smg = SMGMin(storage=storage, run_id="run-smg-1")
    obs = smg.add_observation({"temperature": 0.9, "alarm": True})
    sign_a = smg.create_sign(
        proposition="TEMP_HIGH",
        observation_id=obs.observation_id,
    )
    sign_b = smg.create_sign(
        proposition="ACTIVATE_COOLING",
        observation_id=obs.observation_id,
    )
    rel = smg.link_signs(
        source_sign_id=sign_a.sign_id,
        target_sign_id=sign_b.sign_id,
        kind="support",
    )

    snapshot = smg.snapshot()
    assert len(snapshot["observations"]) == 1
    assert len(snapshot["signs"]) == 2
    assert snapshot["relations"][0]["kind"] == "support"
    assert rel.source_sign_id == sign_a.sign_id

    events = storage.list_events(run_id="run-smg-1", limit=10)
    event_types = [ev.event_type for ev in events]
    assert "smg.observation_added" in event_types
    assert "smg.sign_created" in event_types
    assert "smg.relation_created" in event_types
