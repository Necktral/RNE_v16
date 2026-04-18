from __future__ import annotations

from pathlib import Path

from runtime.storage import StorageConfig, StorageFactory


def _facade(tmp_path: Path):
    return StorageFactory.create_facade(
        StorageConfig(
            mode="sqlite",
            sqlite_db_path=str(tmp_path / "t4_storage.db"),
            postgres_dsn=None,
            artifact_root=tmp_path / "artifacts",
            prefer_postgres_reads=True,
            strict_dual_write=False,
        )
    )


def test_t4_storage_roundtrip(tmp_path: Path) -> None:
    storage = _facade(tmp_path)
    run_id = "run-t4"
    traj_id = "traj-run-t4"

    snap = storage.write_organism_snapshot(
        snapshot_id="snap-1",
        run_id=run_id,
        episode_id="ep-1",
        trajectory_id=traj_id,
        regime="thermal_homeostasis",
        snapshot_json={"episode_count": 1, "belief": {"alarm_probability": 0.1}},
    )
    assert snap.snapshot_id == "snap-1"
    assert storage.list_organism_snapshots(run_id=run_id)[0].trajectory_id == traj_id

    window = storage.write_trajectory_window(
        window_id="window-1",
        run_id=run_id,
        trajectory_id=traj_id,
        start_episode=1,
        end_episode=1,
        snapshots_json={"snapshots": [snap.snapshot_json]},
        digest_json={"drift_score": 0.1},
    )
    assert window.window_id == "window-1"
    assert storage.list_trajectory_windows(run_id=run_id)[0].window_id == "window-1"

    flow = storage.write_trajectory_flow_report(
        report_id="flow-1",
        run_id=run_id,
        trajectory_id=traj_id,
        window_id="window-1",
        flow_validity=True,
        erosion=0.1,
        phase_drift=0.05,
        rollback_obligation=False,
        report_json={"violations": []},
    )
    assert flow.flow_validity is True
    assert storage.list_trajectory_flow_reports(run_id=run_id)[0].report_id == "flow-1"

    renorm = storage.write_renormalization_event(
        event_id="renorm-1",
        run_id=run_id,
        trajectory_id=traj_id,
        source_regime="homeostatic_cooling",
        target_regime="inventory_maximization",
        residual_error=0.2,
        transport_uncertainty=0.3,
        expected_recovery_cost=0.4,
        map_json={"asymmetry_factor": 1.2},
    )
    assert renorm.event_id == "renorm-1"
    assert storage.list_renormalization_events(run_id=run_id)[0].event_id == "renorm-1"

    risk = storage.write_constitutional_risk_state(
        state_id="risk-1",
        run_id=run_id,
        trajectory_id=traj_id,
        scope_type="organism",
        scope_key=run_id,
        risk_score=0.33,
        risk_json={"delta_risk": 0.1},
    )
    assert risk.state_id == "risk-1"
    assert storage.list_constitutional_risk_states(run_id=run_id)[0].state_id == "risk-1"

    failure = storage.write_failure_atlas_event(
        event_id="fail-1",
        run_id=run_id,
        trajectory_id=traj_id,
        scope_type="organism",
        scope_key=run_id,
        failure_class="latent_contamination",
        severity="high",
        reversible=True,
        recovery_protocol="memory_isolation_and_lineage_revalidation",
        signature_json={"memory_purity": 0.3},
    )
    assert failure.event_id == "fail-1"
    assert storage.list_failure_atlas_events(run_id=run_id)[0].failure_class == "latent_contamination"

    storage.close()
