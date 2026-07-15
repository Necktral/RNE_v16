import json

from scripts.benchmark_n1_counterfactual import (
    reconcile_existing_campaign,
    run_n1_counterfactual_campaign,
)


def test_n1_campaign_runs_real_isolated_pairs_and_fails_closed(tmp_path) -> None:
    result = run_n1_counterfactual_campaign(
        output_dir=tmp_path,
        contexts_per_generator=1,
        seed_base=19,
        families=("IND", "PLAN", "OPT"),
    )

    assert result["quality"]["valid_pairs"] == 9
    assert result["quality"]["positive_pairs"] + result["quality"]["negative_pairs"] == 9
    assert result["quality"]["generators"] == 3
    assert result["hash_mismatches"] == 0
    assert result["data_training_ready"] is False
    assert result["promotion_authorized"] is False
    assert sum(item["samples"] for item in result["splits"].values()) == 9
    audit = json.loads((tmp_path / "initial_state_audit.json").read_text())
    assert audit["hash_schema"] == "n1-counterfactual-initial-state-v1"
    assert audit["mismatches"] == []
    manifest = json.loads((tmp_path / "evidence_manifest.json").read_text())
    assert manifest["files"]["paired_records.jsonl"]["sha256"]

    reconciled = reconcile_existing_campaign(tmp_path)
    assert reconciled["evidence_basis"] == "reconciled_paired_records"
    assert reconciled["training_completed"] is False
    assert reconciled["promotion_authorized"] is False
