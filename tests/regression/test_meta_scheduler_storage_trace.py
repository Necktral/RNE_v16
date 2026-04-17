from pathlib import Path

from runtime.reasoning.scheduler_meta.meta_scheduler import MetaScheduler
from runtime.storage import StorageConfig, StorageFactory


def test_meta_scheduler_persists_trace_via_storage(tmp_path: Path):
    config = StorageConfig(
        mode="sqlite",
        sqlite_db_path=str(tmp_path / "meta_trace.db"),
        postgres_dsn=None,
        artifact_root=tmp_path / "artifacts",
        prefer_postgres_reads=True,
        strict_dual_write=False,
    )
    storage = StorageFactory.create_facade(config)
    scheduler = MetaScheduler(trace_store=storage)

    result = scheduler.run({"episode": "ep-meta-1", "run_id": "run-meta-1"})
    assert result["meta_family"] == "META"

    traces = storage.list_reasoning_traces(run_id="run-meta-1", limit=20)
    assert [step.family for step in traces] == ["ABD", "ANA", "CAU", "CTF", "DED", "PROB"]
    assert [step.step_index for step in traces] == [0, 1, 2, 3, 4, 5]
    assert traces[0].detail["sequence"] == ["ABD", "ANA", "CAU", "CTF", "DED", "PROB"]

    storage.close()
