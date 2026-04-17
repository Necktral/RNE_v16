from pathlib import Path

from runtime.storage import StorageConfig, StorageFactory


def _sqlite_config(tmp_path: Path) -> StorageConfig:
    return StorageConfig(
        mode="sqlite",
        sqlite_db_path=str(tmp_path / "storage_contracts.db"),
        postgres_dsn=None,
        artifact_root=tmp_path / "artifacts",
        prefer_postgres_reads=True,
        strict_dual_write=False,
    )


def test_storage_facade_sqlite_roundtrip(tmp_path: Path):
    storage = StorageFactory.create_facade(_sqlite_config(tmp_path))

    event = storage.append_event(
        event_type="contract.event",
        payload={"ok": True, "run_id": "run-ct"},
        run_id="run-ct",
        source="tests",
    )
    assert event.event_type == "contract.event"

    events = storage.list_events(limit=20, run_id="run-ct")
    assert any(item.event_type == "contract.event" for item in events)

    snapshot = storage.write_telemetry_snapshot(
        run_id="run-ct",
        metrics={"latency_ms": 7.5},
    )
    assert snapshot.run_id == "run-ct"
    snapshots = storage.list_telemetry_snapshots(run_id="run-ct")
    assert snapshots and snapshots[0].metrics["latency_ms"] == 7.5

    trace = storage.append_reasoning_trace(
        run_id="run-ct",
        step_index=0,
        family="ABD",
        status="ok",
        detail={"hypothesis": "h1"},
    )
    assert trace.family == "ABD"
    traces = storage.list_reasoning_traces(run_id="run-ct")
    assert traces and traces[0].step_index == 0

    session = storage.upsert_session_bridge(
        session_id="sess-1",
        episode_id="ep-1",
        channel="cli",
        metadata={"origin": "test"},
    )
    assert session.session_id == "sess-1"
    loaded = storage.get_session_bridge("sess-1")
    assert loaded is not None
    assert loaded.metadata["origin"] == "test"

    artifact = storage.materialize_artifact(
        run_id="run-ct",
        kind="trace",
        content=b"artifact payload",
        filename="trace.bin",
        metadata={"scope": "contract"},
    )
    assert Path(artifact.abs_path).exists()
    assert artifact.size_bytes == len(b"artifact payload")
    assert artifact.rel_path
    stored = storage.list_artifacts(run_id="run-ct", kind="trace")
    assert stored and stored[0].sha256 == artifact.sha256

    bench = storage.write_reality_bench_run(
        bench_run_id="bench-ct-1",
        run_id="run-ct",
        total_episodes=10,
        closure_rate=0.9,
        continuity_mean=0.7,
        collapse_count=1,
        gate_profile="ci",
        passed=True,
        summary={"note": "contract"},
    )
    assert bench.gate_profile == "ci"
    benches = storage.list_reality_bench_runs(run_id="run-ct")
    assert benches and benches[0].bench_run_id == "bench-ct-1"

    assess = storage.write_reality_assessment(
        assessment_id="assess-ct-1",
        run_id="run-ct",
        bench_run_id="bench-ct-1",
        episode_id="episode-ct-1",
        closure_passed=True,
        continuity_score=0.8,
        trace_integrity=True,
        collapse_detected=False,
        details={"source": "contract"},
    )
    assert assess.episode_id == "episode-ct-1"
    assessments = storage.list_reality_assessments(run_id="run-ct", bench_run_id="bench-ct-1")
    assert assessments and assessments[0].assessment_id == "assess-ct-1"

    storage.close()
