from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pytest

from runtime.storage import StorageConfig, StorageFactory


RUN_PG_TESTS = os.environ.get("RNFE_RUN_PG_TESTS") == "1"

pytestmark = [
    pytest.mark.requires_postgres,
    pytest.mark.skipif(
        not RUN_PG_TESTS,
        reason="Set RNFE_RUN_PG_TESTS=1 para ejecutar tests de integracion PostgreSQL.",
    ),
]


def _postgres_dsn() -> str:
    dsn = os.environ.get("RNFE_POSTGRES_DSN")
    if not dsn:
        pytest.skip("RNFE_POSTGRES_DSN no esta definido.")
    return dsn


def _exercise_t4_roundtrip(storage, *, run_id: str, traj_id: str) -> None:
    snap = storage.write_organism_snapshot(
        snapshot_id=f"{run_id}-snap-1",
        run_id=run_id,
        episode_id=f"{run_id}-ep-1",
        trajectory_id=traj_id,
        regime="thermal_homeostasis",
        snapshot_json={"episode_count": 1, "belief": {"alarm_probability": 0.1}},
    )
    assert snap.snapshot_id.endswith("snap-1")

    window = storage.write_trajectory_window(
        window_id=f"{run_id}-window-1",
        run_id=run_id,
        trajectory_id=traj_id,
        start_episode=1,
        end_episode=1,
        snapshots_json={"snapshots": [snap.snapshot_json]},
        digest_json={"drift_score": 0.1},
    )
    assert window.window_id.endswith("window-1")

    flow = storage.write_trajectory_flow_report(
        report_id=f"{run_id}-flow-1",
        run_id=run_id,
        trajectory_id=traj_id,
        window_id=window.window_id,
        flow_validity=True,
        erosion=0.1,
        phase_drift=0.05,
        rollback_obligation=False,
        report_json={"violations": []},
    )
    assert flow.flow_validity is True

    renorm = storage.write_renormalization_event(
        event_id=f"{run_id}-renorm-1",
        run_id=run_id,
        trajectory_id=traj_id,
        source_regime="homeostatic_cooling",
        target_regime="inventory_maximization",
        residual_error=0.2,
        transport_uncertainty=0.3,
        expected_recovery_cost=0.4,
        map_json={"asymmetry_factor": 1.2},
    )
    assert renorm.event_id.endswith("renorm-1")

    risk = storage.write_constitutional_risk_state(
        state_id=f"{run_id}-risk-1",
        run_id=run_id,
        trajectory_id=traj_id,
        scope_type="organism",
        scope_key=run_id,
        risk_score=0.33,
        risk_json={"delta_risk": 0.1},
    )
    assert risk.state_id.endswith("risk-1")

    failure = storage.write_failure_atlas_event(
        event_id=f"{run_id}-failure-1",
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
    assert failure.event_id.endswith("failure-1")

    assert storage.list_organism_snapshots(run_id=run_id, limit=5)
    assert storage.list_trajectory_windows(run_id=run_id, limit=5)
    assert storage.list_trajectory_flow_reports(run_id=run_id, limit=5)
    assert storage.list_renormalization_events(run_id=run_id, limit=5)
    assert storage.list_constitutional_risk_states(run_id=run_id, limit=5)
    assert storage.list_failure_atlas_events(run_id=run_id, limit=5)


def test_t4_storage_roundtrip_postgres(tmp_path: Path) -> None:
    storage = StorageFactory.create_facade(
        StorageConfig(
            mode="postgres",
            sqlite_db_path=str(tmp_path / "unused.db"),
            postgres_dsn=_postgres_dsn(),
            artifact_root=tmp_path / "artifacts_pg",
            prefer_postgres_reads=True,
            strict_dual_write=False,
        )
    )
    try:
        _exercise_t4_roundtrip(
            storage,
            run_id="t4-pg-run",
            traj_id="t4-pg-traj",
        )
    finally:
        storage.close()


def test_t4_storage_roundtrip_hybrid_dualwrite(tmp_path: Path) -> None:
    sqlite_path = tmp_path / "hybrid_t4.db"
    storage = StorageFactory.create_facade(
        StorageConfig(
            mode="hybrid",
            sqlite_db_path=str(sqlite_path),
            postgres_dsn=_postgres_dsn(),
            artifact_root=tmp_path / "artifacts_hybrid",
            prefer_postgres_reads=True,
            strict_dual_write=False,
        )
    )
    try:
        _exercise_t4_roundtrip(
            storage,
            run_id="t4-hybrid-run",
            traj_id="t4-hybrid-traj",
        )
    finally:
        storage.close()

    with sqlite3.connect(sqlite_path) as conn:
        organism_count = conn.execute(
            "SELECT COUNT(*) FROM organism_snapshots WHERE run_id = ?",
            ("t4-hybrid-run",),
        ).fetchone()[0]
        risk_count = conn.execute(
            "SELECT COUNT(*) FROM constitutional_risk_states WHERE run_id = ?",
            ("t4-hybrid-run",),
        ).fetchone()[0]
    assert organism_count >= 1
    assert risk_count >= 1
