from __future__ import annotations

import json
from pathlib import Path

import pytest

from runtime.neural.campaign import (
    CAMPAIGN_SCHEMA_VERSION,
    CampaignError,
    CampaignState,
    build_integral_verdict,
    campaign_database_name,
    load_env_file,
    redact,
    resolve_writable_artifact_root,
    replace_dsn_database,
    seal_holdout_spec,
)
from scripts.stage_neural_lab_artifacts import stage_lab_artifacts
from scripts.run_integral_neural_campaign import (
    _accept_all_skipped_pytest_shard,
    _postgres_test_campaign_id,
)
from tests.neural.test_artifact_staging import _source


def _state(tmp_path: Path) -> CampaignState:
    return CampaignState.create(
        root=tmp_path / "campaign",
        campaign_id="integral-test",
        commit="a" * 40,
        database=campaign_database_name("integral-test"),
        schema_sha256="b" * 64,
        artifact_root=tmp_path / "artifacts",
        blocks=("first", "second"),
        configuration={"postgres_dsn": "postgresql://user:secret@localhost/rnfe"},
    )


def test_campaign_manifest_redacts_secrets_and_requires_postgres(tmp_path: Path) -> None:
    state = _state(tmp_path)
    payload = json.loads(state.manifest_path.read_text())

    assert payload["schema_version"] == CAMPAIGN_SCHEMA_VERSION
    assert payload["storage"]["mode"] == "postgres"
    assert payload["configuration"]["postgres_dsn"] == "<redacted>"
    assert "secret" not in state.manifest_path.read_text()
    assert payload["authority_ceiling"] == "shadow"


def test_campaign_checkpoint_is_hash_bound_and_running_block_restarts(tmp_path: Path) -> None:
    state = _state(tmp_path)
    state.begin("first")
    checkpoint = state.checkpoint(phase="rehearsal", next_block="first")

    assert state.verify_checkpoint(checkpoint["checkpoint_hash"])["phase"] == "rehearsal"
    with pytest.raises(CampaignError, match="checkpoint_hash_mismatch"):
        state.verify_checkpoint("0" * 64)

    assert state.reset_incomplete() == ("first",)
    assert state.manifest["blocks"]["first"]["status"] == "pending"
    with pytest.raises(CampaignError, match="checkpoint_manifest_drift"):
        state.verify_checkpoint(checkpoint["checkpoint_hash"])


def test_campaign_completed_blocks_are_not_reset(tmp_path: Path) -> None:
    state = _state(tmp_path)
    state.begin("first")
    state.complete("first", {"passed": True})
    state.begin("second")

    assert state.reset_incomplete() == ("second",)
    assert state.manifest["blocks"]["first"]["status"] == "completed"


def test_env_loader_returns_keys_not_values_and_redaction_masks_dsn(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / ".env"
    path.write_text(
        "RNFE_STORAGE_MODE=postgres\n"
        "RNFE_POSTGRES_DSN=postgresql://rnfe:very-secret@localhost:5432/rnfe\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("RNFE_STORAGE_MODE", raising=False)
    monkeypatch.delenv("RNFE_POSTGRES_DSN", raising=False)

    keys = load_env_file(path)

    assert keys == ("RNFE_POSTGRES_DSN", "RNFE_STORAGE_MODE")
    assert "very-secret" not in json.dumps(redact({"dsn": "anything"}))
    masked = redact("postgresql://rnfe:very-secret@localhost:5432/rnfe")
    assert "very-secret" not in masked


def test_campaign_database_is_deterministic_and_dsn_only_replaces_database() -> None:
    name = campaign_database_name("integral-test")
    assert name == campaign_database_name("integral-test")
    assert name.startswith("rnfe_campaign_integral_test_")
    replaced = replace_dsn_database(
        "postgresql://rnfe:secret@localhost:5432/rnfe?sslmode=disable", name
    )
    assert replaced.endswith(f"/{name}?sslmode=disable")
    assert "rnfe:secret@localhost:5432" in replaced


def test_stale_native_mount_uses_explicit_writable_fallback(tmp_path: Path) -> None:
    requested = tmp_path / "read-only" / "missing" / "artifacts"
    requested.parent.parent.mkdir()
    requested.parent.parent.chmod(0o500)
    fallback = tmp_path / "native-ext4" / "rnfe_artifacts"
    fallback.parent.mkdir()
    try:
        resolved, remapped = resolve_writable_artifact_root(
            requested, native_fallback=fallback
        )
    finally:
        requested.parent.parent.chmod(0o700)

    assert resolved == fallback.resolve()
    assert remapped is True


def test_holdout_spec_is_immutable_and_hash_verified(tmp_path: Path) -> None:
    path = tmp_path / "holdout.json"
    first = seal_holdout_spec(
        target=path,
        campaign_id="integral-test",
        seed_base=931000,
        contexts_per_generator=20,
        scenarios=("thermal_homeostasis",),
    )
    second = seal_holdout_spec(
        target=path,
        campaign_id="integral-test",
        seed_base=999999,
        contexts_per_generator=1,
        scenarios=("different",),
    )
    assert second == first

    payload = json.loads(path.read_text())
    payload["seed_base"] += 1
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(CampaignError, match="holdout_spec_hash_mismatch"):
        seal_holdout_spec(
            target=path,
            campaign_id="integral-test",
            seed_base=931000,
            contexts_per_generator=20,
            scenarios=("thermal_homeostasis",),
        )


def test_integral_verdict_never_grants_operational_promotion() -> None:
    vector = {
        "closure_rate": 1.0,
        "certification_rate": 1.0,
        "continuity": 1.0,
        "viability": 1.0,
        "latency_ms": 10.0,
        "cpu_pressure": 0.1,
        "memory_pressure": 0.1,
        "vram_gb": 0.1,
        "thermal_pressure": 0.1,
        "safety_violations": 0,
    }
    impact = {
        "promotion_eligible": True,
        "baseline": vector,
        "candidate": vector,
    }
    verdict = build_integral_verdict(
        campaign_id="integral-test",
        regression_passed=True,
        postgres_passed=True,
        organ_reports={f"N{index}": {"safety_violations": 0} for index in range(7)},
        holdout={"calibration_ece": 0.05},
        impact_report=impact,
        artifact_reconciliation={"passed": True},
    )

    assert verdict["staging_authorized"] is True
    assert verdict["shadow_qualification_passed"] is True
    assert verdict["promotion_eligible"] is False
    assert verdict["promotion_authorized"] is False
    assert verdict["training_authorized"] is False


def test_qualified_staging_records_proof_but_keeps_shadow_ceiling(tmp_path: Path) -> None:
    source, target = tmp_path / "source", tmp_path / "target"
    _source(source)
    proof = {
        "campaign_id": "integral-test",
        "verdict_sha256": "a" * 64,
        "checkpoint_hash": "b" * 64,
        "staging_authorized": True,
        "shadow_qualification_passed": True,
    }

    profile = stage_lab_artifacts(
        source_root=source,
        target_root=target,
        organs=("N1",),
        qualification=proof,
    )

    assert profile["classification"] == "qualified_shadow_only"
    assert profile["shadow_qualification_passed"] is True
    assert profile["promotion_authorized"] is False
    assert profile["activation_automatic"] is False
    assert profile["staged"][0]["shadow_qualification_passed"] is True


def test_qualified_staging_rejects_incomplete_proof(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _source(source)
    with pytest.raises(ValueError, match="qualification_proof_invalid"):
        stage_lab_artifacts(
            source_root=source,
            target_root=tmp_path / "target",
            organs=("N1",),
            qualification={"staging_authorized": True},
        )


def test_pytest_shard_accepts_module_level_skip_but_not_empty_file(tmp_path: Path) -> None:
    skipped_log = tmp_path / "skipped.log"
    skipped_log.write_text("1 skipped in 0.02s\n", encoding="utf-8")
    empty_log = tmp_path / "empty.log"
    empty_log.write_text("no tests ran in 0.01s\n", encoding="utf-8")

    skipped = _accept_all_skipped_pytest_shard(
        {"returncode": 5, "passed": False, "log_path": str(skipped_log)}
    )
    empty = _accept_all_skipped_pytest_shard(
        {"returncode": 5, "passed": False, "log_path": str(empty_log)}
    )

    assert skipped["passed"] is True
    assert skipped["all_skipped"] is True
    assert empty["passed"] is False


def test_postgres_contract_tests_get_isolated_database_per_attempt() -> None:
    first = _postgres_test_campaign_id("neural-nightly-20260715-deadbeef", attempt=1)
    retry = _postgres_test_campaign_id("neural-nightly-20260715-deadbeef", attempt=2)

    assert first.endswith("-pgtests-a1")
    assert retry.endswith("-pgtests-a2")
    assert first != retry
    assert len(first) <= 80

    with pytest.raises(CampaignError, match="attempt_must_be_positive"):
        _postgres_test_campaign_id("neural-nightly-20260715-deadbeef", attempt=0)
