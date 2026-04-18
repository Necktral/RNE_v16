from __future__ import annotations

from pathlib import Path

from runtime.storage import StorageConfig, StorageFactory


def _facade(tmp_path: Path):
    return StorageFactory.create_facade(
        StorageConfig(
            mode="sqlite",
            sqlite_db_path=str(tmp_path / "risk_chain.db"),
            postgres_dsn=None,
            artifact_root=tmp_path / "artifacts",
            prefer_postgres_reads=True,
            strict_dual_write=False,
        )
    )


def test_risk_chain_persists_prev_state_and_step_index(tmp_path: Path) -> None:
    storage = _facade(tmp_path)
    try:
        first = storage.write_constitutional_risk_state(
            state_id="risk-1",
            run_id="run-chain",
            trajectory_id="traj-chain",
            scope_type="organism",
            scope_key="run-chain",
            risk_score=0.20,
            risk_json={"state": {"risk_score": 0.20}},
            prev_state_id=None,
            step_index=0,
        )
        second = storage.write_constitutional_risk_state(
            state_id="risk-2",
            run_id="run-chain",
            trajectory_id="traj-chain",
            scope_type="organism",
            scope_key="run-chain",
            risk_score=0.35,
            risk_json={"state": {"risk_score": 0.35}},
            prev_state_id=first.state_id,
            step_index=1,
        )

        rows = storage.list_constitutional_risk_states(
            run_id="run-chain",
            trajectory_id="traj-chain",
            scope_type="organism",
            scope_key="run-chain",
            limit=10,
        )
        assert rows[0].state_id == "risk-2"
        assert rows[0].step_index == 1
        assert rows[0].prev_state_id == "risk-1"
        assert rows[1].state_id == "risk-1"
        assert rows[1].step_index == 0
        assert rows[1].prev_state_id is None
        assert second.step_index == 1
    finally:
        storage.close()

