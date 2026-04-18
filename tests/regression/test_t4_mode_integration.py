from __future__ import annotations

from pathlib import Path

from runtime.storage import StorageConfig, StorageFactory
from runtime.world.min_cognitive_episode import MinimalCognitiveEpisodeRunner


def _storage(tmp_path: Path, db_name: str):
    return StorageFactory.create_facade(
        StorageConfig(
            mode="sqlite",
            sqlite_db_path=str(tmp_path / db_name),
            postgres_dsn=None,
            artifact_root=tmp_path / "artifacts",
            prefer_postgres_reads=True,
            strict_dual_write=False,
        )
    )


def _run_single_episode(*, tmp_path: Path, run_id: str):
    storage = _storage(tmp_path, f"{run_id}.db")
    runner = MinimalCognitiveEpisodeRunner(storage=storage, run_id=run_id)
    runner.run_episode(external_heat=0.05)
    return storage


def test_t4_mode_off_preserves_legacy_and_no_t4_persistence(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("RNFE_T5_MODE", raising=False)
    monkeypatch.setenv("RNFE_T4_MODE", "off")
    storage = _run_single_episode(tmp_path=tmp_path, run_id="run-t4-off")
    try:
        cert = storage.list_episode_certificates(run_id="run-t4-off", limit=1)[0]
        transfer_meta = cert.metadata.get("transfer_assessment", {})

        assert transfer_meta.get("certificate_scope") == "local_only"
        assert transfer_meta.get("transfer_verdict") == "certified_local"
        assert "t4" not in transfer_meta
        assert "t5" not in transfer_meta

        assert storage.list_organism_snapshots(run_id="run-t4-off", limit=5) == []
        assert storage.list_trajectory_windows(run_id="run-t4-off", limit=5) == []
        assert storage.list_trajectory_flow_reports(run_id="run-t4-off", limit=5) == []
        assert storage.list_constitutional_risk_states(run_id="run-t4-off", limit=5) == []
    finally:
        storage.close()


def test_t4_mode_shadow_dual_path_keeps_legacy_scope(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("RNFE_T5_MODE", raising=False)
    monkeypatch.setenv("RNFE_T4_MODE", "shadow")
    storage = _run_single_episode(tmp_path=tmp_path, run_id="run-t4-shadow")
    try:
        cert = storage.list_episode_certificates(run_id="run-t4-shadow", limit=1)[0]
        transfer_meta = cert.metadata.get("transfer_assessment", {})
        t4 = transfer_meta.get("t4", {})
        t5 = transfer_meta.get("t5", {})

        assert transfer_meta.get("certificate_scope") == "local_only"
        assert transfer_meta.get("transfer_verdict") == "certified_local"
        assert t4.get("mode") == "shadow"
        assert t5.get("mode") == "shadow"
        assert t4.get("scope") in {
            "blocked",
            "quarantine_only",
            "local_safe",
            "transfer_safe",
            "modification_safe",
            "inheritance_safe",
        }

        assert len(storage.list_organism_snapshots(run_id="run-t4-shadow", limit=5)) >= 1
        assert len(storage.list_trajectory_windows(run_id="run-t4-shadow", limit=5)) >= 1
        assert len(storage.list_trajectory_flow_reports(run_id="run-t4-shadow", limit=5)) >= 1
        risk_states = storage.list_constitutional_risk_states(run_id="run-t4-shadow", limit=20)
        scope_types = {row.scope_type for row in risk_states}
        assert {"organism", "modification", "inheritance"}.issubset(scope_types)
    finally:
        storage.close()


def test_t4_mode_on_rewires_scope_and_uses_t4_risk(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("RNFE_T5_MODE", raising=False)
    monkeypatch.setenv("RNFE_T4_MODE", "on")
    storage = _run_single_episode(tmp_path=tmp_path, run_id="run-t4-on")
    try:
        cert = storage.list_episode_certificates(run_id="run-t4-on", limit=1)[0]
        transfer_meta = cert.metadata.get("transfer_assessment", {})
        t4 = transfer_meta.get("t4", {})
        t5 = transfer_meta.get("t5", {})

        assert t4.get("mode") == "on"
        assert t5.get("mode") == "on"
        assert transfer_meta.get("certificate_scope") == t4.get("scope")
        assert transfer_meta.get("certificate_scope") in {
            "blocked",
            "quarantine_only",
            "local_safe",
            "transfer_safe",
            "modification_safe",
            "inheritance_safe",
        }
        assert transfer_meta.get("transfer_verdict") in {
            "rejected_for_transfer",
            "certified_local",
            "certified_analogical_only",
            "certified_transfer_safe",
        }
        assert cert.risk_score >= float(t4.get("organism_risk", 0.0))

        risk_states = storage.list_constitutional_risk_states(run_id="run-t4-on", limit=20)
        scope_types = {row.scope_type for row in risk_states}
        assert {"organism", "modification", "inheritance"}.issubset(scope_types)
    finally:
        storage.close()
