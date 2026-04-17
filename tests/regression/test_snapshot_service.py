from runtime.storage import StorageConfig, StorageFactory
from runtime.telemetry.snapshot_service import SnapshotService


def test_snapshot_service_persists_metrics(tmp_path):
    config = StorageConfig(
        mode="sqlite",
        sqlite_db_path=str(tmp_path / "snapshot.db"),
        postgres_dsn=None,
        artifact_root=tmp_path / "artifacts",
        prefer_postgres_reads=True,
        strict_dual_write=False,
    )
    storage = StorageFactory.create_facade(config)
    service = SnapshotService(storage=storage)
    service.persist_snapshot(
        cycle=7,
        metrics={"Mem": 0.2, "Temp": 0.1},
        run_id="run-snap-1",
    )
    rows = storage.list_telemetry_snapshots(run_id="run-snap-1", limit=5)
    assert rows and rows[0].metrics["cycle"] == 7
    assert rows[0].metrics["metrics"]["Mem"] == 0.2
