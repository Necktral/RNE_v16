from __future__ import annotations

from pathlib import Path

import pytest

from runtime.neural.config import NeuralRuntimeConfig
from runtime.neural.contracts import AdmissionDecision, NeuralMode
from runtime.neural.integration import (
    ConsumerVerdictClass,
    SymbiosisIdentity,
    SymbiosisTrace,
    SymbioticNeuralCoordinator,
)
from runtime.storage import StorageConfig, StorageFactory
from runtime.world import ScenarioEpisodeRunner


def _storage(tmp_path: Path, name: str = "symbiosis"):
    return StorageFactory.create_facade(
        StorageConfig(
            mode="sqlite",
            sqlite_db_path=str(tmp_path / f"{name}.db"),
            postgres_dsn=None,
            artifact_root=tmp_path / f"{name}-artifacts",
            prefer_postgres_reads=False,
            strict_dual_write=False,
        )
    )


def _organ(trace: dict, name: str) -> dict:
    return next(item for item in trace["organs"] if item["organ"] == name)


def test_multiple_live_episodes_share_trace_and_close_all_consumption_loops(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("RNFE_NEURAL_MODE", "shadow")
    monkeypatch.setenv("RNFE_EXPERIENCE", "1")
    storage = _storage(tmp_path)
    runner = ScenarioEpisodeRunner(
        storage=storage,
        run_id="run-symbiotic-e2e",
        scenario="thermal_homeostasis",
        closure_profile="adaptive_min",
    )
    runner.set_organism_id("organism-e2e")
    runner.set_causal_context(
        {
            "organism_id": "organism-e2e",
            "lineage_id": runner.lineage.lineage_id,
            "run_id": "run-symbiotic-e2e",
            "trace_group_id": "trace-e2e-1",
            "decision_id": "decision-e2e-1",
        }
    )
    first = runner.run_episode(external_input=0.04)
    runner.set_causal_context(
        {
            "organism_id": "organism-e2e",
            "lineage_id": runner.lineage.lineage_id,
            "run_id": "run-symbiotic-e2e",
            "trace_group_id": "trace-e2e-2",
            "decision_id": "decision-e2e-2",
        }
    )
    second = runner.run_episode(external_input=0.14)

    first_trace = first["neural_symbiosis_trace"]
    second_trace = second["neural_symbiosis_trace"]
    assert first_trace["trace_complete"] is True
    assert second_trace["trace_complete"] is True
    assert first_trace["semantic_complete"] is True
    assert first_trace["durably_complete"] is True
    assert first_trace["persistence_degraded"] is False
    assert second_trace["schema_version"] == "neural-symbiosis-trace-v2"
    receipt_organs = {receipt["organ"] for receipt in second_trace["consumer_receipts"]}
    admitted_organs = {
        entry["organ"]
        for entry in second_trace["organs"]
        if entry["candidate_hash"] is not None
    }
    assert receipt_organs == admitted_organs
    rejected_n4 = _organ(second_trace, "N4")
    if rejected_n4["fallback_reason"]:
        assert rejected_n4["candidate_hash"] is None
        assert rejected_n4["consumer_verdict"].startswith("not_consumed:fallback:")
    candidate_hashes = {entry["organ"]: entry["candidate_hash"] for entry in second_trace["organs"]}
    assert all(
        receipt["candidate_hash"] == candidate_hashes[receipt["organ"]]
        for receipt in second_trace["consumer_receipts"]
    )
    first_transition = first["life_transition"]
    second_transition = second["life_transition"]
    assert first_transition["status"] == "committed"
    assert second_transition["transition_index"] == 2
    assert second_transition["previous_transition_hash"] == first_transition["transition_hash"]
    assert second_trace["life_transition_id"] == second_transition["transition_id"]
    assert second_trace["state_after_hash"] == second["dynamic_state"]["state_hash"]
    dropped = SymbiosisTrace.from_dict(second_trace)
    dropped.trace_health = {**dropped.trace_health, "dropped_events": 1}
    assert dropped.semantic_complete is True
    assert dropped.durably_complete is False
    assert dropped.persistence_degraded is True
    assert {entry["trace_group_id"] for entry in second_trace["organs"]} == {"trace-e2e-2"}
    assert {entry["organ"] for entry in second_trace["organs"]} == {
        "N1", "N2", "N3", "N4", "N5", "N6"
    }
    assert _organ(first_trace, "N3")["candidate"]["uncertainty"] == 0.5

    n5 = _organ(second_trace, "N5")
    assert n5["candidate"]["backend"] == "deterministic_chunker"
    assert n5["candidate"]["hnet_active"] is False
    assert n5["manifest_sha256"] is None
    assert n5["artifact_sha256"] is None
    assert n5["cost"]["reference_sha256"]
    assert second["episode"]["context"]["neural_ingestion"]["smg_sign_ids"]
    assert "SMG+MFM" in n5["consumer_verdict"]

    assert second["episode"]["context"]["retrieved_memory"]
    n1_comparison = second["episode"]["result"]["neural_comparisons"][
        "n1_scheduler_comparison"
    ]
    assert n1_comparison["scheduler_authority_preserved"] is True
    assert n1_comparison["scheduler_selected"] == second["reasoning"]["sequence"]

    n2 = _organ(second_trace, "N2")
    assert set(n2["candidate"]["verification"]) == {"DED", "LOT-F", "NESY"}
    assert n2["consumer"] == "DED+LOT-F+NESY"

    n3 = _organ(second_trace, "N3")["candidate"]
    assert n3["episode_count"] == 2
    assert n3["previous_state"]["episode_count"] == 1
    assert n3["state_key"] == [
        "organism-e2e",
        "thermal_homeostasis@1.0",
        runner.lineage.lineage_id,
    ]
    assert n3["mamba2_active"] is False

    n4 = _organ(second_trace, "N4")
    assert n4["effective_mode"] == "shadow"
    if n4["fallback_reason"]:
        assert n4["candidate"] is None
        assert "certificate_metadata=consumed" not in n4["consumer_verdict"]
    else:
        assert n4["candidate"]["authority"]["may_choose_intervention"] is False
        assert n4["candidate"]["canonical_comparison"]["decision_influence"] == "none"
        assert "certificate_metadata=consumed" in n4["consumer_verdict"]

    n6 = _organ(second_trace, "N6")
    assert n6["candidate"]["applied"] is False
    assert n6["candidate"]["sandbox"]["applied"] is False
    assert second["experience"]["neural_symbiosis"]["trace_group_id"] == "trace-e2e-2"

    certificate = storage.list_episode_certificates(run_id="run-symbiotic-e2e", limit=1)[0]
    neural_metadata = certificate.metadata["neural_symbiosis"]
    assert neural_metadata["verdict_influence"] == "none"
    assert neural_metadata["authority_effective"]["N4"] == "shadow"
    assert neural_metadata["runtime"]["loaded_artifacts"] == 0
    storage.close()


def test_off_ablation_preserves_world_but_removes_downstream_neural_evidence(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("RNFE_NEURAL_MODE", "off")
    off_storage = _storage(tmp_path, "off")
    off = ScenarioEpisodeRunner(
        storage=off_storage, run_id="run-off", scenario="thermal_homeostasis"
    ).run_episode(external_input=0.04)

    monkeypatch.setenv("RNFE_NEURAL_MODE", "shadow")
    shadow_storage = _storage(tmp_path, "shadow")
    shadow = ScenarioEpisodeRunner(
        storage=shadow_storage, run_id="run-shadow", scenario="thermal_homeostasis"
    ).run_episode(external_input=0.04)

    assert off["episode"]["result"]["updated_world"] == shadow["episode"]["result"]["updated_world"]
    assert off["episode"]["context"]["intervention"] == shadow["episode"]["context"]["intervention"]
    assert all(entry["effective_mode"] == "off" for entry in off["neural_symbiosis_trace"]["organs"])
    assert _organ(off["neural_symbiosis_trace"], "N3")["consumer_verdict"] == "disabled"
    assert _organ(off["neural_symbiosis_trace"], "N5")["consumer_verdict"] == "disabled"
    assert not off["episode"]["context"]["neural_ingestion"]["smg_sign_ids"]
    assert _organ(shadow["neural_symbiosis_trace"], "N5")["candidate"]["chunks"]
    assert off_storage.list_episode_certificates(run_id="run-off", limit=1)[0].metadata[
        "neural_symbiosis"
    ]["runtime"]["loaded_artifacts"] == 0
    off_storage.close()
    shadow_storage.close()


def test_resource_pressure_degrades_optional_organs_but_keeps_ingestion_and_continuity(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("RNFE_NEURAL_MODE", "shadow")
    storage = _storage(tmp_path, "pressure")
    runner = ScenarioEpisodeRunner(
        storage=storage, run_id="run-pressure", scenario="thermal_homeostasis"
    )
    runner.set_resource_signals(
        {"cpu_pressure": 0.95, "memory_pressure": 0.93, "thermal_pressure": 0.92}
    )
    result = runner.run_episode(external_input=0.14)
    trace = result["neural_symbiosis_trace"]

    for organ in ("N1", "N2", "N4", "N6"):
        assert _organ(trace, organ)["fallback_reason"] == "reference_degraded_by_resource_pressure"
    assert _organ(trace, "N5")["candidate"]["chunks"]
    assert _organ(trace, "N3")["candidate"]["episode_count"] == 1
    storage.close()


def test_missing_msrc_budget_prevents_all_neural_execution(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("RNFE_NEURAL_MODE", "shadow")
    storage = _storage(tmp_path, "no-msrc-budget")
    runner = ScenarioEpisodeRunner(
        storage=storage, run_id="run-no-budget", scenario="thermal_homeostasis"
    )
    runner.set_resource_signals(
        {"msrc_budget_available": False, "msrc_scale_id": "1x1"}
    )
    result = runner.run_episode(external_input=0.04)

    assert all(
        entry["fallback_reason"] == "msrc_budget_unavailable"
        for entry in result["neural_symbiosis_trace"]["organs"]
    )
    assert result["episode"]["context"]["neural_ingestion"]["smg_sign_ids"] == []
    storage.close()


@pytest.mark.parametrize("disabled", ["N1", "N2", "N3", "N4", "N5", "N6"])
def test_each_organ_ablation_changes_downstream_evidence(
    tmp_path: Path, monkeypatch, disabled: str
) -> None:
    monkeypatch.setenv("RNFE_NEURAL_MODE", "shadow")
    monkeypatch.setenv("RNFE_NEURAL_DISABLED_ORGANS", disabled)
    storage = _storage(tmp_path, f"ablation-{disabled.lower()}")
    result = ScenarioEpisodeRunner(
        storage=storage,
        run_id=f"run-ablation-{disabled.lower()}",
        scenario="thermal_homeostasis",
    ).run_episode(external_input=0.04)
    entry = _organ(result["neural_symbiosis_trace"], disabled)

    assert entry["effective_mode"] == "off"
    assert entry["candidate"] is None
    assert entry["fallback_reason"] == "organ_disabled_by_profile"
    assert entry["consumer_verdict"]
    if disabled in {"N3", "N5"}:
        assert entry["consumer_verdict"] == "disabled"
    if disabled == "N5":
        assert result["episode"]["context"]["neural_ingestion"]["smg_sign_ids"] == []
    elif disabled == "N4":
        comparison = result["episode"]["result"]["neural_comparisons"]["n4_comparison"]
        assert comparison["verdict"] == "disabled_or_no_candidate"
    elif disabled == "N2":
        verification = result["episode"]["result"]["neural_comparisons"]["n2_verification"]
        assert verification["status"] == "disabled"
    storage.close()


def test_rejected_n3_admission_does_not_advance_temporal_state_or_emit_receipts(
    tmp_path: Path,
) -> None:
    storage = _storage(tmp_path, "n3-rejected")
    coordinator = SymbioticNeuralCoordinator(
        storage=storage,
        config=NeuralRuntimeConfig(mode=NeuralMode.PROVISIONAL),
    )
    n3_adapter = coordinator._adapters["N3"]
    n3_adapter.admission_gate = lambda candidate, request: AdmissionDecision(
        False, reason="test_temporal_policy_rejected"
    )
    identity = SymbiosisIdentity(
        trace_group_id="trace-n3-rejected",
        organism_id="organism-n3-rejected",
        lineage_id="lineage-n3-rejected",
        run_id="run-n3-rejected",
        episode_id="episode-n3-rejected",
        scenario_id="thermal_homeostasis@1.0",
        decision_id="decision-n3-rejected",
    )

    signals = coordinator.begin_episode(
        identity=identity,
        observation={"temperature": 0.8},
        formula="temperature > 0.5",
        proposition="temperature high",
        memory_hits=[],
        scenario_metadata={"main_variable": "temperature"},
        causal_attestation={
            "main_variable": "temperature",
            "factual_delta": -0.2,
            "counterfactual_delta": 0.1,
        },
        resources={"gpu_available": False},
    )

    entry = coordinator._session(identity.episode_id).entries["N3"]
    assert signals["n3_temporal"]["status"] == "disabled"
    assert entry.fallback_reason == "test_temporal_policy_rejected"
    assert entry.candidate is None
    assert entry.candidate_hash is None
    assert entry.consumer_verdict.startswith("not_consumed:fallback:")
    assert coordinator.export_temporal_state()["entries"] == []
    assert not any(
        receipt.organ == "N3"
        for receipt in coordinator._session(identity.episode_id).trace.consumer_receipts
    )
    with pytest.raises(ValueError, match="requires_consumable_candidate"):
        coordinator.record_consumer_receipt(
            episode_id=identity.episode_id,
            organ="N3",
            consumer_id="next_episode_state",
            consumer_input={},
            consumer_output={},
            verdict_class=ConsumerVerdictClass.OBSERVED,
            verdict_detail="must_not_be_recorded",
            evidence_refs=("test",),
        )
    storage.close()


def test_rejected_n5_admission_cannot_mutate_smg(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("RNFE_NEURAL_MODE", "provisional")
    storage = _storage(tmp_path, "n5-rejected")
    runner = ScenarioEpisodeRunner(
        storage=storage,
        run_id="run-n5-rejected",
        scenario="thermal_homeostasis",
    )
    runner._neural._adapters["N5"].admission_gate = (
        lambda candidate, request: AdmissionDecision(
            False, reason="test_ingestion_policy_rejected"
        )
    )

    result = runner.run_episode(external_input=0.04)
    trace = result["neural_symbiosis_trace"]
    entry = _organ(trace, "N5")
    n5_signs = [
        sign
        for sign in result["smg_snapshot"]["signs"]
        if (sign.get("metadata") or {}).get("origin") == "N5"
    ]

    assert entry["fallback_reason"] == "test_ingestion_policy_rejected"
    assert entry["candidate"] is None
    assert entry["candidate_hash"] is None
    assert entry["consumer_verdict"].startswith("not_consumed:fallback:")
    assert result["episode"]["context"]["neural_ingestion"]["smg_sign_ids"] == []
    assert n5_signs == []
    assert not any(receipt["organ"] == "N5" for receipt in trace["consumer_receipts"])
    storage.close()


class _FailNeuralEvents:
    def __init__(self, storage):
        self._storage = storage

    def append_event(self, **kwargs):
        if str(kwargs.get("event_type", "")).startswith("neural."):
            raise OSError("simulated neural ledger outage")
        return self._storage.append_event(**kwargs)

    def __getattr__(self, name):
        return getattr(self._storage, name)


def test_neural_trace_storage_failure_is_visible_and_episode_still_finishes(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("RNFE_NEURAL_MODE", "shadow")
    base = _storage(tmp_path, "degraded")
    runner = ScenarioEpisodeRunner(
        storage=_FailNeuralEvents(base),
        run_id="run-degraded",
        scenario="thermal_homeostasis",
    )
    result = runner.run_episode(external_input=0.04)

    assert result["certification"]["verdict"] in {"certified", "rejected"}
    trace = result["neural_symbiosis_trace"]
    assert trace["trace_persisted"] is False
    assert trace["semantic_complete"] is True
    assert trace["durably_complete"] is False
    assert trace["persistence_degraded"] is True
    assert trace["trace_health"]["degraded"] is True
    assert trace["trace_health"]["persistence_failures"] > 0
    assert trace["trace_health"]["pending_events"] > 0
    base.close()


def test_causal_conflict_is_consumed_as_disagreement_without_action_authority(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("RNFE_NEURAL_MODE", "shadow")
    storage = _storage(tmp_path, "conflict")
    coordinator = SymbioticNeuralCoordinator(storage=storage)
    identity = SymbiosisIdentity(
        trace_group_id="trace-conflict",
        organism_id="org-conflict",
        lineage_id="lin-conflict",
        run_id="run-conflict",
        episode_id="episode-conflict",
        scenario_id="scenario@1",
        decision_id="decision-conflict",
    )
    coordinator.begin_episode(
        identity=identity,
        observation={"temperature": 0.9, "alarm": True},
        formula="temperature > alarm_threshold",
        proposition="temperature alarm",
        memory_hits=[],
        scenario_metadata={"main_variable": "temperature"},
            causal_attestation={
                "main_variable": "temperature",
                "optimization_direction": "minimize",
                "factual_delta": -0.1,
            "counterfactual_delta": 0.1,
            "supports_choice": True,
        },
        resources={},
    )
    comparison = coordinator.consume_reasoning(
        episode_id="episode-conflict",
        reasoning={
            "sequence": ["CAU", "CTF", "DED"],
            "state": {
                "cau_link": {"helps_goal": False},
                "ctf_checked": {"supports_choice": False},
                "ded_validated": True,
                "ded_conclusion": "safe",
            },
        },
        lotf_valid=True,
    )["n4_comparison"]
    assert comparison["verdict"] == "disagreement"
    assert comparison["decision_influence"] == "none"
    storage.close()
