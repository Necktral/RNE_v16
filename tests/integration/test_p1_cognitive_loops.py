from __future__ import annotations

from pathlib import Path

from runtime.storage import StorageConfig, StorageFactory
from runtime.world import ScenarioEpisodeRunner
from scripts.train_n4_preaction_v2 import train_to_directory


def _storage(tmp_path: Path, name: str):
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


def _forbidden_keys(value):
    forbidden = {
        "causal_attestation",
        "committed_action",
        "committed_intervention",
        "counterfactual",
        "factual",
        "ground_truth",
        "outcome",
        "outcomes",
        "relation_kind",
        "transition",
    }
    found = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if key in forbidden:
                found.add(key)
            found.update(_forbidden_keys(child))
    elif isinstance(value, list):
        for child in value:
            found.update(_forbidden_keys(child))
    return found


def test_p1_on_off_preserves_canonical_world_action_and_reasoning(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("RNFE_NEURAL_MODE", "shadow")
    monkeypatch.delenv("RNFE_P1_COGNITIVE_LOOPS", raising=False)
    off_storage = _storage(tmp_path, "p1-off")
    off = ScenarioEpisodeRunner(
        storage=off_storage,
        run_id="p1-off",
        scenario="thermal_homeostasis",
        closure_profile="adaptive_min",
    ).run_episode(external_input=0.04)

    monkeypatch.setenv("RNFE_P1_COGNITIVE_LOOPS", "1")
    monkeypatch.setenv("RNFE_N3_SHADOW_COUNTERFACTUAL", "1")
    on_storage = _storage(tmp_path, "p1-on")
    on = ScenarioEpisodeRunner(
        storage=on_storage,
        run_id="p1-on",
        scenario="thermal_homeostasis",
        closure_profile="adaptive_min",
    ).run_episode(external_input=0.04)

    assert "p1_cognitive_loop" not in off["episode"]["result"]
    assert on["episode"]["context"]["intervention"] == off["episode"]["context"]["intervention"]
    assert on["episode"]["result"]["updated_world"] == off["episode"]["result"]["updated_world"]
    assert on["reasoning"]["sequence"] == off["reasoning"]["sequence"]
    assert on["certification"]["verdict"] == off["certification"]["verdict"]
    report = on["episode"]["result"]["p1_cognitive_loop"]
    assert report["authority_effect"] == "none"
    assert report["promotion_authorized"] is False
    assert report["n4"]["status"] == "evaluated"
    assert report["n4"]["evaluation"]["candidate_hash_preserved"] is True
    assert report["n4"]["evaluation"]["oracle_seal_verified"] is True
    assert report["n4"]["evaluation"]["coverage"] == 0.0
    assert _forbidden_keys(report["n4"]["candidate"]) == set()
    n2 = report["n2"]
    assert n2["status"] == "accepted"
    assert n2["attempt_count"] == 1
    assert n2["shadow_reasoning_sequence"].count("IND") == 1
    assert n2["ground_truth"]["valid_correction"] is True
    closed = on_storage.list_events(
        run_id="p1-on", event_types=["episode.closed"], limit=10
    )
    assert len(closed) == 1
    assert "p1_cognitive_loop" not in closed[0].payload["result"]
    off_storage.close()
    on_storage.close()


def test_second_episode_compares_n3_shadow_without_polluting_trace_requirements(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("RNFE_NEURAL_MODE", "shadow")
    monkeypatch.setenv("RNFE_P1_COGNITIVE_LOOPS", "1")
    monkeypatch.setenv("RNFE_N3_SHADOW_COUNTERFACTUAL", "1")
    storage = _storage(tmp_path, "p1-n3")
    runner = ScenarioEpisodeRunner(
        storage=storage,
        run_id="p1-n3",
        scenario="thermal_homeostasis",
        closure_profile="adaptive_min",
    )

    first = runner.run_episode(external_input=0.04)
    second = runner.run_episode(external_input=0.14)

    assert first["episode"]["result"]["p1_cognitive_loop"]["n3"]["status"] == "warmup"
    n3 = second["episode"]["result"]["p1_cognitive_loop"]["n3"]
    assert n3["status"] == "compared"
    assert n3["writes_performed"] is False
    assert n3["snapshot_match"] is True
    assert n3["canonical_snapshot_sha256"] == n3["shadow_snapshot_sha256"]
    assert n3["shadow_scheduler"]["trace_persisted"] is False
    assert second["neural_symbiosis_trace"]["semantic_complete"] is True
    optional = [
        receipt
        for receipt in second["neural_symbiosis_trace"]["consumer_receipts"]
        if receipt["consumer_id"] == "shadow_counterfactual_retrieval_scheduler"
    ]
    assert optional == []
    storage.close()


def test_second_episode_executes_hash_bound_trained_n4_v2(
    tmp_path: Path, monkeypatch
) -> None:
    manifest = train_to_directory(tmp_path / "trained")["manifest_path"]
    monkeypatch.setenv("RNFE_NEURAL_MODE", "shadow")
    monkeypatch.setenv("RNFE_P1_COGNITIVE_LOOPS", "1")
    monkeypatch.setenv("RNFE_NEURAL_N4_PREACTION_MANIFEST", manifest)
    storage = _storage(tmp_path, "p1-n4-trained")
    runner = ScenarioEpisodeRunner(
        storage=storage,
        run_id="p1-n4-trained",
        scenario="thermal_homeostasis",
        closure_profile="adaptive_min",
    )

    runner.run_episode(external_input=0.07)
    second = runner.run_episode(external_input=0.07)
    n4 = second["episode"]["result"]["p1_cognitive_loop"]["n4"]

    assert n4["candidate"]["model"]["execution_class"] == "trained_v2"
    assert len(n4["candidate"]["model"]["artifact_sha256"]) == 64
    assert n4["evaluation"]["coverage"] == 1.0
    assert n4["evaluation"]["oracle_seal_verified"] is True
    storage.close()
