from pathlib import Path

from runtime.memory.mfm_lite.retrieval import MemoryRetrieval
from runtime.neural.integration.p2_v2_n3_decision import FrozenRetrieval
from runtime.storage import StorageConfig, StorageFactory


def test_real_retrieval_single_pool_is_discriminative_and_isolated(tmp_path: Path):
    storage = StorageFactory.create_facade(StorageConfig(
        mode="sqlite", sqlite_db_path=str(tmp_path / "p2-v2.db"), postgres_dsn=None,
        artifact_root=tmp_path / "artifacts", prefer_postgres_reads=True,
        strict_dual_write=False,
    ))
    run_id = "isolated-unit"
    for index, (scale, relation, intervention) in enumerate((
        (s, r, i) for i in ("activate", "deactivate")
        for s in ("micro", "meso", "macro")
        for r in ("support", "contradiction")
    )):
        storage.write_memory_record(
            run_id=run_id, episode_id=f"e-{index}", memory_id=f"m-{index}", scale=scale,
            structure_json={"relation_kind": relation, "intervention": intervention,
                            "alarm": index % 3 == 0, "propositions": ["TEMP_HIGH"] if index % 2 else []},
            metadata={"scenario_name": "sandbox"},
        )
    hits = MemoryRetrieval(storage=storage).retrieve(
        run_id=run_id, query={"alarm": True, "proposition": "TEMP_HIGH"},
        limit=12, candidate_pool_size=12, scenario_name="sandbox",
    )
    frozen = FrozenRetrieval.freeze(hits)
    assert len(frozen.hits) == 12
    assert len({x["score"] for x in frozen.hits}) > 1
    assert storage.retrieve_memory_records(run_id="other", limit=20) == []
    storage.close()

