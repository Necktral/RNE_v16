from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from runtime.neural import OrganismImpactVector
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
import scripts.run_integral_neural_campaign as integral_runner
from scripts.stage_neural_lab_artifacts import stage_lab_artifacts
from scripts.run_integral_neural_campaign import (
    ABLATION_ORGANS,
    _accept_all_skipped_pytest_shard,
    _ablation_profiles,
    _capture_ablation_resource_snapshot,
    _p1_external_input,
    _p1_profile_summary,
    _p1_profiles,
    _postgres_test_campaign_id,
    _run_ablation_matrix,
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


def test_ablation_profiles_cover_off_shadow_control_single_all_on_and_leave_one_out() -> None:
    profiles = _ablation_profiles()

    assert [profile.profile_id for profile in profiles] == [
        "off",
        "shadow-none",
        "only-n1",
        "only-n2",
        "only-n3",
        "only-n4",
        "only-n5",
        "only-n6",
        "all-on",
        "without-n1",
        "without-n2",
        "without-n3",
        "without-n4",
        "without-n5",
        "without-n6",
    ]
    assert profiles[0].environment == {
        "RNFE_NEURAL_MODE": "off",
        "RNFE_NEURAL_DISABLED_ORGANS": ",".join(ABLATION_ORGANS),
    }
    assert profiles[1].environment == {
        "RNFE_NEURAL_MODE": "shadow",
        "RNFE_NEURAL_DISABLED_ORGANS": ",".join(ABLATION_ORGANS),
    }
    assert profiles[2].enabled_organs == ("N1",)
    assert profiles[2].disabled_organs == ("N2", "N3", "N4", "N5", "N6")
    assert profiles[8].enabled_organs == ABLATION_ORGANS
    assert profiles[8].disabled_organs == ()
    assert profiles[9].enabled_organs == ("N2", "N3", "N4", "N5", "N6")
    assert profiles[9].disabled_organs == ("N1",)
    assert all(profile.to_dict()["promotion_authorized"] is False for profile in profiles)


def test_p1_profiles_cover_control_isolated_trained_all_and_leave_one_out() -> None:
    profiles = _p1_profiles()

    assert [profile.profile_id for profile in profiles] == [
        "off",
        "shadow-none",
        "only-n2",
        "only-n3-reference",
        "only-n3-trained",
        "only-n4-v2",
        "p1-all",
        "p1-without-n2",
        "p1-without-n3",
        "p1-without-n4",
    ]
    assert all(
        profile.environment()["RNFE_EXTERNAL_REASONER_RUNTIME"] == "0"
        and profile.environment()["RNFE_ALLOW_EXTERNAL_REASONER"] == "0"
        and profile.environment()["RNFE_TEACHER"] == "0"
        for profile in profiles
    )
    assert profiles[3].environment()["RNFE_NEURAL_N3_MANIFEST"] == ""
    assert all(profile.to_dict()["promotion_authorized"] is False for profile in profiles)


def test_p1_seed_schedules_are_unique_and_deterministic() -> None:
    seeds = tuple(911001 + index * 101 for index in range(12))
    schedules = {
        tuple(_p1_external_input(seed, step) for step in range(32)) for seed in seeds
    }

    assert len(schedules) == 12
    assert _p1_external_input(seeds[0], 0) == _p1_external_input(seeds[0], 0)


def test_p1_profile_summary_excludes_two_warmup_visits_per_scenario() -> None:
    report = {
                    "n2": {
            "attempt_count": 1,
            "status": "accepted",
            "ground_truth": {
                "scored": True,
                "initial_false_rejection": True,
                "final_false_rejection": False,
                "valid_correction": True,
                "retry_false_accept": False,
            },
                    },
                    "n3": {
            "ground_truth_metrics": {
                "ndcg_delta": 0.2,
                "mrr_delta": 0.1,
                "risk_brier": 0.0,
            },
                    },
                    "n4": {
            "evaluation": {
                "coverage": 1.0,
                "mae_delta": 0.1,
                "pairwise_ranking_accuracy": 1.0,
                "top1_correct": True,
                "regret_delta_vs_canonical": 0.2,
                "regret_delta_vs_prior": 0.1,
                "candidate_hash_preserved": True,
            },
                    },
    }
    rows = [
        {
            "episode_result": {
                "episode": {
                    "scenario": "thermal",
                    "result": {"p1_cognitive_loop": report},
                }
            }
        }
        for _ in range(3)
    ]
    run = {
        "seed": 7,
        "rows": rows,
        "canonical_behavior_sha256": "a" * 64,
        "closure_evidence": {"episode_emission_rate": 1.0},
        "vector": {
            "closure_rate": 1.0,
            "certification_rate": 1.0,
            "safety_violations": 0,
        },
    }

    summary = _p1_profile_summary(_p1_profiles()[6], [run])

    lane = summary["lanes"][0]["summary"]
    assert lane["scored_steps"] == 1
    assert lane["warmup_steps"] == 2
    assert summary["n2_totals"]["valid_corrections"] == 1
    assert summary["metrics"]["n4_coverage"]["mean"] == 1.0


def test_p1_matrix_trains_n4_before_all_rehearsal_lanes(
    tmp_path: Path, monkeypatch
) -> None:
    calls = []
    trained_manifest = tmp_path / "models/n4/manifest.json"
    trained_manifest.parent.mkdir(parents=True)
    trained_manifest.write_text("{}")
    monkeypatch.setattr(
        integral_runner,
        "train_n4_preaction_v2",
        lambda _root: {"manifest_path": str(trained_manifest), "manifest": {}},
    )
    monkeypatch.setattr(
        integral_runner,
        "_capture_ablation_resource_snapshot",
        lambda _ctx: {
            "snapshot": {"available": True},
            "snapshot_sha256": "a" * 64,
            "path": str(tmp_path / "snapshot.json"),
            "artifact_sha256": "b" * 64,
        },
    )
    provenance = {"worktree_snapshot_sha256": "c" * 64}
    monkeypatch.setattr(
        integral_runner, "_materialize_worktree_provenance", lambda _ctx: provenance
    )
    monkeypatch.setattr(
        integral_runner,
        "_worktree_snapshot",
        lambda: (b"", {"worktree_snapshot_sha256": "c" * 64}),
    )
    monkeypatch.setattr(integral_runner, "_register_report", lambda *_a, **_k: None)

    def fake_lane(_ctx, **kwargs):
        calls.append(kwargs)
        return {
            "seed": kwargs["seed"],
            "rows": [],
            "canonical_behavior_sha256": hashlib.sha256(
                f"seed:{kwargs['seed']}".encode()
            ).hexdigest(),
            "closure_evidence": {"episode_emission_rate": 0.0},
            "vector": {
                "closure_rate": 1.0,
                "certification_rate": 1.0,
                "safety_violations": 0,
            },
            "backend_provenance": {
                organ: {"execution_class": "disabled"}
                for organ in ("N2", "N3", "N4")
            },
        }

    monkeypatch.setattr(integral_runner, "_run_life_lane", fake_lane)
    ctx = SimpleNamespace(
        state=SimpleNamespace(
            root=tmp_path,
            campaign_id="p1-matrix-test",
            manifest={"commit": "d" * 40},
        )
    )

    report = integral_runner._run_p1_matrix(
        ctx, steps=8, seeds=(101, 202, 303)
    )

    assert len(calls) == 30
    assert all(
        call["extra_environment"]["RNFE_NEURAL_N4_PREACTION_MANIFEST"]
        == str(trained_manifest)
        for call in calls
    )
    assert report["n4_preaction_training"]["manifest_path"] == str(trained_manifest)
    assert report["gates"]["canonical_behavior_identical"] is True
    assert report["canonical_behavior_parity"]["step_comparison_count"] == 216


def test_ablation_matrix_is_paired_reproducible_and_never_promotable(
    tmp_path: Path, monkeypatch
) -> None:
    calls: list[dict[str, object]] = []
    fixed_snapshot = {
        "available": True,
        "source": "test-host",
        "sample_ts": 1234.5,
        "cpu_pressure": 0.2,
        "memory_pressure": 0.3,
        "thermal_pressure": 0.1,
        "vram_pressure": 0.4,
        "gpu_available": True,
        "hardware_pressure": 0.4,
        "gpu_acceleration": 0.6,
    }
    fixed_snapshot_sha256 = hashlib.sha256(
        json.dumps(fixed_snapshot, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    capture = {
        "schema_version": "rnfe-ablation-resource-snapshot-v1",
        "campaign_id": "ablation-test",
        "snapshot": fixed_snapshot,
        "snapshot_sha256": fixed_snapshot_sha256,
        "path": str(tmp_path / "ablation/resource-snapshot.json"),
        "artifact_sha256": "b" * 64,
    }

    def fake_lane(_ctx, **kwargs):
        calls.append(dict(kwargs))
        enabled = len(ABLATION_ORGANS) - len(kwargs["disabled_organs"])
        seed = int(kwargs["seed"])
        vector = OrganismImpactVector(
            closure_rate=1.0,
            certification_rate=1.0,
            continuity=1.0,
            viability=1.0,
            latency_ms=float(enabled),
            cpu_pressure=0.1,
            memory_pressure=0.1,
            vram_gb=0.0,
            thermal_pressure=0.0,
            safety_violations=0,
        )
        return {
            "run_id": f"run-{kwargs['lane']}-{seed}",
            "path": f"/{kwargs['lane']}-{seed}.json",
            "sha256": "a" * 64,
            "lane": kwargs["lane"],
            "seed": seed,
            "neural_mode": kwargs["neural_mode"],
            "enabled_organs": [
                organ for organ in ABLATION_ORGANS if organ not in kwargs["disabled_organs"]
            ],
            "disabled_organs": list(kwargs["disabled_organs"]),
            "backend_provenance": {
                organ: {
                    "execution_class": (
                        "reference" if organ not in kwargs["disabled_organs"] else "disabled"
                    ),
                    "model_backends": [],
                    "candidate_backends": [],
                    "manifest_sha256": [],
                    "artifact_sha256": [],
                    "reference_ids": (
                        [f"reference:{organ}"]
                        if organ not in kwargs["disabled_organs"]
                        else []
                    ),
                    "model_binding_errors": [],
                }
                for organ in ABLATION_ORGANS
            },
            "primary": float(enabled),
            "vector": vector,
            "rows": [],
        }

    monkeypatch.setattr(integral_runner, "_run_life_lane", fake_lane)
    monkeypatch.setattr(
        integral_runner,
        "_capture_ablation_resource_snapshot",
        lambda _ctx: capture,
    )
    monkeypatch.setattr(integral_runner, "_register_report", lambda *args, **kwargs: None)
    ctx = SimpleNamespace(
        state=SimpleNamespace(
            root=tmp_path,
            campaign_id="ablation-test",
            manifest={"commit": "c" * 40},
        )
    )

    report = _run_ablation_matrix(ctx, steps=2, seeds=(101, 202))

    assert len(calls) == 30
    assert all(call["steps"] == 2 for call in calls)
    assert all(call["resource_snapshot_override"] == fixed_snapshot for call in calls)
    assert all(call["resource_snapshot_provenance"] == capture for call in calls)
    assert report["resource_snapshot_capture"] == capture
    assert report["seed_order"] == [101, 202]
    assert report["profile_order"] == [
        profile.profile_id for profile in _ablation_profiles()
    ]
    summaries = {profile["profile_id"]: profile for profile in report["profiles"]}
    assert summaries["off"]["mean_paired_primary_delta"] == 0.0
    assert summaries["shadow-none"]["mean_paired_primary_delta"] == 0.0
    assert summaries["only-n1"]["mean_paired_primary_delta"] == 1.0
    assert summaries["all-on"]["mean_paired_primary_delta"] == 6.0
    assert summaries["without-n1"]["mean_paired_primary_delta"] == 5.0
    assert summaries["only-n1"]["backend_provenance_summary"]["N1"] == {
        "execution_classes": ["reference"],
        "model_backends": [],
        "candidate_backends": [],
        "manifest_sha256": [],
        "artifact_sha256": [],
        "reference_ids": ["reference:N1"],
        "model_binding_errors": [],
        "stable_across_seeds": True,
    }
    contrasts = {item["organ"]: item for item in report["organ_contrasts"]}
    assert contrasts["N1"]["all-on_minus_without"]["mean_primary_delta"] == 1.0
    assert contrasts["N1"]["only_minus_shadow_none"]["mean_primary_delta"] == 1.0
    assert [
        item["seed"]
        for item in contrasts["N1"]["all-on_minus_without"]["paired_by_seed"]
    ] == [101, 202]
    assert report["schema_version"] == "rnfe-neural-ablation-matrix-v2"
    assert report["metric_contract"]["closure_rate"]["independent_from"] == (
        "certification_rate"
    )
    assert report["training_authorized"] is False
    assert report["staging_authorized"] is False
    assert report["promotion_eligible"] is False
    assert report["promotion_authorized"] is False
    provenance = report["provenance"]
    assert provenance["head_commit"] == integral_runner._git("rev-parse", "HEAD")
    assert provenance["stable_during_run"] is True
    diff_path = Path(provenance["diff_path"])
    assert diff_path.is_file()
    assert hashlib.sha256(diff_path.read_bytes()).hexdigest() == provenance[
        "diff_sha256"
    ]
    assert json.loads((tmp_path / "ablation/matrix.json").read_text())[
        "profile_order"
    ] == report["profile_order"]


def test_ablation_resource_snapshot_is_real_sealed_and_provenanced(
    tmp_path: Path, monkeypatch
) -> None:
    snapshot = {
        "available": True,
        "source": "psutil",
        "sample_ts": 9876.5,
        "cpu_pressure": 0.21,
        "memory_pressure": 0.32,
        "swap_pressure": 0.0,
        "thermal_pressure": 0.12,
        "vram_pressure": 0.22,
        "vram_headroom": 0.78,
        "gpu_available": True,
        "gpu_load": 0.22,
        "hardware_pressure": 0.32,
        "gpu_acceleration": 0.71,
    }
    sampler_inputs: dict[str, object] = {}

    class FakeHostSampler:
        def __init__(self, *, ttl_seconds: float):
            sampler_inputs["host_ttl"] = ttl_seconds

    class FakeVRAMSampler:
        pass

    def fake_build_resource_snapshot(*, host_sampler, vram_sampler):
        sampler_inputs["host"] = host_sampler
        sampler_inputs["vram"] = vram_sampler
        sampler_inputs["host_sensing"] = integral_runner.os.environ.get(
            "RNFE_HOST_SENSING"
        )
        return dict(snapshot)

    registered: list[dict[str, object]] = []
    monkeypatch.setenv("RNFE_HOST_SENSING", "0")
    monkeypatch.setattr(integral_runner, "HostResourceSampler", FakeHostSampler)
    monkeypatch.setattr(integral_runner, "NvidiaVRAMSampler", FakeVRAMSampler)
    monkeypatch.setattr(
        integral_runner,
        "build_resource_snapshot",
        fake_build_resource_snapshot,
    )
    monkeypatch.setattr(
        integral_runner,
        "_register_report",
        lambda _ctx, path, **kwargs: registered.append(
            {"path": path, **kwargs}
        ),
    )
    ctx = SimpleNamespace(
        state=SimpleNamespace(root=tmp_path, campaign_id="ablation-snapshot-test")
    )

    report = _capture_ablation_resource_snapshot(ctx)

    expected_sha256 = hashlib.sha256(
        json.dumps(snapshot, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    persisted = json.loads(
        (tmp_path / "ablation/resource-snapshot.json").read_text()
    )
    assert sampler_inputs["host_ttl"] == 0.0
    assert sampler_inputs["host_sensing"] == "1"
    assert integral_runner.os.environ["RNFE_HOST_SENSING"] == "0"
    assert persisted["snapshot"] == snapshot
    assert persisted["snapshot_sha256"] == expected_sha256
    assert persisted["captured_once"] is True
    assert persisted["reuse_contract"] == "LifeKernelConfig.resource_snapshot_override"
    assert report["snapshot"] == snapshot
    assert report["artifact_sha256"] == hashlib.sha256(
        (tmp_path / "ablation/resource-snapshot.json").read_bytes()
    ).hexdigest()
    assert registered == [
        {
            "path": tmp_path / "ablation/resource-snapshot.json",
            "kind": "neural_ablation_resource_snapshot",
            "run_id": "ablation-snapshot-test",
        }
    ]


def test_life_lane_uses_fixed_snapshot_only_when_explicitly_overridden(
    tmp_path: Path, monkeypatch
) -> None:
    snapshot = {
        "available": True,
        "source": "test-host",
        "sample_ts": 222.0,
        "cpu_pressure": 0.11,
        "memory_pressure": 0.22,
        "thermal_pressure": 0.05,
        "vram_pressure": 0.33,
        "gpu_available": True,
        "hardware_pressure": 0.33,
        "gpu_acceleration": 0.67,
    }
    snapshot_sha256 = hashlib.sha256(
        json.dumps(snapshot, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    configs = []

    class FakeStep:
        def to_dict(self):
            return {}

    class FakeLifeKernel:
        def __init__(self, *, config, storage):
            configs.append(config)

        def step(self, *, external_input):
            return FakeStep()

    vector = OrganismImpactVector(
        closure_rate=1.0,
        certification_rate=1.0,
        continuity=1.0,
        viability=1.0,
        latency_ms=1.0,
        cpu_pressure=0.11,
        memory_pressure=0.22,
        vram_gb=0.0,
        thermal_pressure=0.05,
        safety_violations=0,
    )
    monkeypatch.setattr(integral_runner, "LifeKernel", FakeLifeKernel)
    monkeypatch.setattr(integral_runner, "_model_environment", lambda _ctx: {})
    monkeypatch.setattr(
        integral_runner,
        "_step_vector",
        lambda rows, *, elapsed_s, closed_episode_ids: (1.0, vector),
    )
    monkeypatch.setattr(integral_runner, "_register_report", lambda *args, **kwargs: None)
    storage = SimpleNamespace(list_events=lambda **_kwargs: [])
    ctx = SimpleNamespace(
        state=SimpleNamespace(root=tmp_path, campaign_id="lane-snapshot-test"),
        postgres=SimpleNamespace(dsn="postgresql://example/test"),
        ensure_storage=lambda: storage,
    )
    provenance = {
        "snapshot_sha256": snapshot_sha256,
        "path": str(tmp_path / "ablation/resource-snapshot.json"),
        "artifact_sha256": "c" * 64,
    }

    fixed = integral_runner._run_life_lane(
        ctx,
        phase="ablation",
        lane="off",
        seed=101,
        steps=2,
        neural_mode="off",
        disabled_organs=ABLATION_ORGANS,
        resource_snapshot_override=snapshot,
        resource_snapshot_provenance=provenance,
    )
    live = integral_runner._run_life_lane(
        ctx,
        phase="rehearsal",
        lane="off",
        seed=202,
        steps=1,
        neural_mode="off",
        disabled_organs=ABLATION_ORGANS,
    )

    assert configs[0].resource_snapshot_override == snapshot
    assert configs[1].resource_snapshot_override is None
    assert fixed["resource_snapshot_override"] == {
        "mode": "fixed_ablation_override",
        "snapshot_sha256": snapshot_sha256,
        "capture_path": provenance["path"],
        "capture_artifact_sha256": provenance["artifact_sha256"],
    }
    assert "resource_snapshot_override" not in live
    persisted = json.loads(Path(fixed["path"]).read_text())
    assert persisted["resource_snapshot_override"] == fixed[
        "resource_snapshot_override"
    ]


def test_closure_rate_uses_durable_episode_closed_events_not_certification() -> None:
    rows = [
        {
            "vital_signs": {
                "cognitive_quality": 0.8,
                "certified": certified,
                "identity_continuity": 0.9,
                "viability_margin": 0.7,
            },
            "episode_result": {
                "episode": {"episode_id": episode_id},
                # Deliberately contradictory: this field must not define closure.
                "certification": {"verdict": verdict},
                "neural_symbiosis_trace": {"resource_state": {}, "organs": []},
            },
        }
        for episode_id, certified, verdict in (
            ("episode-a", True, "rejected"),
            ("episode-b", False, "certified"),
            ("episode-c", False, "rejected"),
        )
    ]

    class FakeStorage:
        def list_events(self, **kwargs):
            assert kwargs == {
                "run_id": "closure-run",
                "event_types": ["episode.closed"],
                "limit": 7,
            }
            return [
                SimpleNamespace(payload={"episode_id": "episode-a"}),
                SimpleNamespace(payload={"episode_id": "episode-c"}),
                SimpleNamespace(payload={"episode_id": "episode-c"}),
                SimpleNamespace(payload={"episode_id": "unexpected"}),
            ]

    evidence = integral_runner._collect_episode_closure_evidence(
        FakeStorage(),
        run_id="closure-run",
        rows=rows,
    )
    _primary, vector = integral_runner._step_vector(
        rows,
        elapsed_s=0.03,
        closed_episode_ids=evidence["matched_episode_ids"],
    )

    assert evidence["matched_episode_ids"] == ["episode-a", "episode-c"]
    assert evidence["missing_episode_ids"] == ["episode-b"]
    assert evidence["unexpected_episode_ids"] == ["unexpected"]
    assert evidence["duplicate_event_count"] == 1
    assert vector.closure_rate == pytest.approx(2.0 / 3.0)
    assert vector.certification_rate == pytest.approx(1.0 / 3.0)


def test_non_episode_quarantine_step_is_not_a_missing_closure() -> None:
    rows = [
        {
            "vital_signs": {
                "cognitive_quality": 0.8,
                "certified": True,
                "identity_continuity": 0.9,
                "viability_margin": 0.7,
            },
            "episode_result": {
                "episode": {"episode_id": "episode-a"},
                "neural_symbiosis_trace": {"resource_state": {}, "organs": []},
            },
        },
        {
            "decision": {"action": "quarantine"},
            "vital_signs": {
                "cognitive_quality": 0.8,
                "certified": True,
                "identity_continuity": 0.9,
                "viability_margin": 0.7,
            },
            "episode_result": None,
        },
    ]

    class FakeStorage:
        def list_events(self, **_kwargs):
            return [SimpleNamespace(payload={"episode_id": "episode-a"})]

    evidence = integral_runner._collect_episode_closure_evidence(
        FakeStorage(), run_id="quarantine-run", rows=rows
    )
    _primary, vector = integral_runner._step_vector(
        rows,
        elapsed_s=0.02,
        closed_episode_ids=evidence["matched_episode_ids"],
    )

    assert evidence["denominator"] == "emitted_episode_results"
    assert evidence["episode_result_count"] == 1
    assert evidence["non_episode_step_count"] == 1
    assert evidence["episode_emission_rate"] == 0.5
    assert vector.closure_rate == 1.0


def test_canonical_behavior_hash_excludes_shadow_evidence_but_binds_world() -> None:
    base = {
        "decision": {
            "action": "act",
            "external_input": 0.1,
            "mode": "active",
            "priority": 1,
            "reason": "scheduled",
            "scenario": "thermal",
        },
        "vital_signs": {
            "certified": True,
            "cognitive_quality": 0.8,
            "episode_count": 1,
            "identity_continuity": 0.9,
            "mode": "active",
            "resource_pressure": 0.2,
            "risk_score": 0.1,
            "viability_margin": 0.7,
        },
        "episode_result": {
            "certification": {
                "certificate_id": "random-id",
                "decision_verdict": "promote",
                "promotion_candidate": True,
                "verdict": "certified",
            },
            "episode": {
                "episode_id": "episode-random",
                "scenario": "thermal",
                "context": {"intervention": "cool", "observation": {"temp": 0.8}},
                "result": {
                    "reasoning_sequence": ["CAU", "CTF"],
                    "updated_world": {"temp": 0.7},
                    "p1_cognitive_loop": {"n3": {"risk": 0.1}},
                },
            },
        },
    }
    shadow_changed = json.loads(json.dumps(base))
    shadow_changed["episode_result"]["episode"]["episode_id"] = "episode-other"
    shadow_changed["episode_result"]["episode"]["result"]["p1_cognitive_loop"] = {
        "n3": {"risk": 0.9}
    }
    world_changed = json.loads(json.dumps(base))
    world_changed["episode_result"]["episode"]["result"]["updated_world"]["temp"] = 0.6

    base_hash = integral_runner._canonical_behavior_sha256([base])
    assert integral_runner._canonical_behavior_sha256([shadow_changed]) == base_hash
    assert integral_runner._canonical_behavior_sha256([world_changed]) != base_hash


def test_backend_provenance_distinguishes_model_reference_and_disabled() -> None:
    rows = [
        {
            "episode_result": {
                "neural_symbiosis_trace": {
                    "backend_identities": {
                        "N1": {
                            "backend": "rnfe-compact-mlp-router-v1",
                            "manifest_sha256": "1" * 64,
                            "artifact_sha256": "2" * 64,
                        },
                        "N2": "rnfe:N2:reference",
                        "N5": "rnfe:N5:deterministic_ingestion:chunker-v1",
                    },
                    "organs": [
                        {
                            "organ": "N1",
                            "effective_mode": "shadow",
                            "manifest_sha256": "1" * 64,
                            "artifact_sha256": "2" * 64,
                            "candidate": {"backend": "rnfe-compact-mlp-router-v1"},
                            "cost": {},
                        },
                        {
                            "organ": "N2",
                            "effective_mode": "off",
                            "manifest_sha256": None,
                            "artifact_sha256": None,
                            "cost": {},
                        },
                        {
                            "organ": "N5",
                            "effective_mode": "shadow",
                            "manifest_sha256": None,
                            "artifact_sha256": None,
                            "candidate": {"backend": "deterministic_chunker"},
                            "cost": {
                                "reference_id": (
                                    "rnfe:N5:deterministic_ingestion:chunker-v1"
                                )
                            },
                        },
                    ],
                }
            }
        }
    ]

    provenance = integral_runner._backend_provenance(
        rows,
        enabled_organs=("N1", "N5"),
    )

    assert provenance["N1"]["execution_class"] == "model_bound"
    assert provenance["N1"]["model_backends"] == [
        "rnfe-compact-mlp-router-v1"
    ]
    assert provenance["N5"]["execution_class"] == "reference"
    assert provenance["N5"]["reference_ids"] == [
        "rnfe:N5:deterministic_ingestion:chunker-v1"
    ]
    assert provenance["N2"]["execution_class"] == "disabled"


def test_evaluation_artifacts_reuse_valid_configured_n1_without_overwriting_candidate(
    tmp_path: Path, monkeypatch
) -> None:
    configured = tmp_path / "configured" / "neural"
    _source(configured)
    ctx = SimpleNamespace(state=SimpleNamespace(root=tmp_path / "campaign"))
    monkeypatch.setattr(
        integral_runner,
        "_configured_neural_root",
        lambda: configured,
    )

    base = integral_runner._prepare_evaluation_artifacts(ctx)

    copied_manifest = json.loads(
        (base / "neural/n1/manifest.json").read_text(encoding="utf-8")
    )
    assert copied_manifest["model_id"] == "n1-stage-test"
    assert integral_runner._validated_model_artifact(
        base / "neural", organ="N1"
    ) is not None

    candidate_root = tmp_path / "campaign/n1_recalibration/candidate"
    _source(candidate_root)
    candidate_manifest_path = candidate_root / "n1/manifest.json"
    candidate_manifest = json.loads(candidate_manifest_path.read_text(encoding="utf-8"))
    candidate_manifest["model_id"] = "n1-campaign-candidate"
    candidate_manifest_path.write_text(json.dumps(candidate_manifest), encoding="utf-8")

    integral_runner._prepare_evaluation_artifacts(ctx)

    selected = json.loads(
        (base / "neural/n1/manifest.json").read_text(encoding="utf-8")
    )
    assert selected["model_id"] == "n1-campaign-candidate"


def test_evaluation_artifacts_do_not_copy_configured_n1_with_bad_hash(
    tmp_path: Path, monkeypatch
) -> None:
    configured = tmp_path / "configured" / "neural"
    _source(configured)
    (configured / "n1/model.json").write_text("corrupted", encoding="utf-8")
    ctx = SimpleNamespace(state=SimpleNamespace(root=tmp_path / "campaign"))
    monkeypatch.setattr(
        integral_runner,
        "_configured_neural_root",
        lambda: configured,
    )

    base = integral_runner._prepare_evaluation_artifacts(ctx)

    assert not (base / "neural/n1").exists()
