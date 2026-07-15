from __future__ import annotations

import json

from scripts.benchmark_teacher_advanced import reconcile_stratified_evidence
from scripts.build_teacher_curriculum import select_curriculum_records


def _trials() -> list[dict]:
    rows = []
    for perturbation in ("heldout-low", "heldout-high"):
        for seed in range(10):
            pair = f"thermal-{perturbation}-{seed}"
            rows.append(
                {
                    "evaluation_pair_id": pair,
                    "scenario": "thermal_homeostasis",
                    "perturbation_id": perturbation,
                    "seed": seed,
                    "variant": "no_teacher",
                    "evaluation": {"cumulative_reward": -1.0, "mean_severity": 0.2},
                }
            )
            rows.append(
                {
                    "evaluation_pair_id": pair,
                    "scenario": "thermal_homeostasis",
                    "perturbation_id": perturbation,
                    "external_input": 0.1,
                    "seed": seed,
                    "variant": "codex_frontier",
                    "lesson": {
                        "lesson_id": "lesson-codex",
                        "avoid": "deactivate_cooling",
                        "prefer": "activate_cooling",
                        "lesson": "Activa enfriamiento y verifica el descenso de temperatura.",
                    },
                    "teacher_semantics": {"semantic_pass": True},
                    "evaluation": {"cumulative_reward": -0.5, "mean_severity": 0.1},
                }
            )
    return rows


def test_curriculum_requires_repeated_heldout_support_and_never_authorizes_training() -> None:
    records = select_curriculum_records(_trials(), campaign_id="heldout")

    assert len(records) == 1
    assert records[0]["training_eligible"] is True
    assert records[0]["training_authorized"] is False
    assert records[0]["evidence"]["support_count"] == 20
    assert records[0]["evidence"]["success_rate"] == 1.0


def test_curriculum_rejects_any_heldout_regression() -> None:
    trials = _trials()
    failing = next(row for row in trials if row["variant"] == "codex_frontier")
    failing["evaluation"]["mean_severity"] = 0.3
    records = select_curriculum_records(trials, campaign_id="heldout")

    assert records[0]["training_eligible"] is False
    assert records[0]["evidence"]["regression_count"] == 1


def test_stratified_reconciliation_replaces_stale_aggregate_verdict(tmp_path) -> None:
    campaign = tmp_path / "heldout"
    campaign.mkdir()
    fixtures = {
        "manifest.json": {"campaign_id": "heldout"},
        "trials.json": {"trials": []},
        "summary.json": {
            "trial_count": 0,
            "pair_count": 0,
            "variants": {},
            "scenario_comparisons": {},
            "comparisons": {"codex_cross_scenario_gate_passed": True},
        },
        "verdict.json": {
            "codex_cross_scenario_gate_passed": True,
            "training_authorized": False,
        },
        "stratified_reanalysis.json": {
            "schema_version": "rnfe-teacher-stratified-reanalysis-v1",
            "campaign_id": "heldout",
            "comparisons": {"codex_cross_scenario_gate_passed": False},
            "scenario_comparisons": {},
            "stratum_comparisons": {"thermal:low": {"reward": -0.1}},
            "promotion_authorized": False,
            "training_authorized": False,
            "reason": "stratified_tradeoff_detected",
            "supersedes_aggregate_candidate_claim": True,
            "verdict": "retain_local_7b_as_supervised_student",
        },
    }
    for name, payload in fixtures.items():
        (campaign / name).write_text(json.dumps(payload), encoding="utf-8")
    (campaign / "REPORT.md").write_text("stale", encoding="utf-8")

    manifest_path = reconcile_stratified_evidence(campaign)

    verdict = json.loads((campaign / "verdict.json").read_text(encoding="utf-8"))
    summary = json.loads((campaign / "summary.json").read_text(encoding="utf-8"))
    evidence = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert verdict["codex_cross_scenario_gate_passed"] is False
    assert verdict["codex_teacher_candidate"] is False
    assert verdict["training_authorized"] is False
    assert summary["stratified_reanalysis"]["supersedes_aggregate_candidate_claim"] is True
    assert set(evidence["artifacts"]) == {
        "REPORT.md", "manifest.json", "stratified_reanalysis.json",
        "summary.json", "trials.json", "verdict.json",
    }
