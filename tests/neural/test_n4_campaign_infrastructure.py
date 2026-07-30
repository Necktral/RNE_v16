from __future__ import annotations

import hashlib
import json

import pytest

from runtime.neural.contracts import canonical_sha256
from runtime.neural.training.n4_campaign import (
    load_composite_manifest,
    sample_candidate_sets_for_epoch,
    select_configuration_and_median_run,
)
from runtime.neural.training.n4_ranking import (
    CandidateSetSample,
    N4CandidateRecord,
)
from scripts.orchestrate_n4_safe_training import run_training_grid


def _canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode() + b"\n"


def _row(set_id, hypothesis, seed, *, supervised):
    row = {
        "schema_version": "n4-ranking-sample.v1",
        "candidate_set_id": set_id,
        "hypothesis_id": hypothesis,
        "scenario": "thermal_with_battery",
        "seed": seed,
        "split": "ignored",
        "source": "test",
        "features": {
            "empirical_gain": 0.1,
            "holdout_support": 0.2,
            "coverage": 0.3,
            "simplicity": 0.4,
            "stability": 0.5,
            "invariant_safety": 0.6,
        },
        "valid": hypothesis.endswith("a"),
        "mae_gain": 0.2,
        "invariant_risk": 0.0,
    }
    if supervised:
        risk = 0.1 if hypothesis.endswith("a") else 0.4
        report = {"target_version": "n4-candidate-risk.v1", "risk": risk}
        row.update(
            {
                "risk_label": risk,
                "risk_label_available": True,
                "risk_label_version": "n4-risk-label.multistep.v1",
                "risk_report": report,
                "risk_report_sha256": canonical_sha256(report),
                "safety_contract_version": "thermal_with_battery.safety.v1",
                "rollout_horizon": 3,
            }
        )
    return row


def _write_component(root, name, seeds, supervised):
    rows = [
        _row(f"{name}-{seed}", hypothesis, seed, supervised=supervised)
        for seed in seeds
        for hypothesis in ("a", "b")
    ]
    dataset = root / f"{name}.jsonl"
    source_manifest = root / f"{name}.manifest.json"
    dataset.write_bytes(b"".join(_canonical(row) for row in rows))
    source_manifest.write_bytes(_canonical({"name": name, "seeds": seeds}))
    return {
        "path": dataset.name,
        "manifest_path": source_manifest.name,
        "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256(
            source_manifest.read_bytes()
        ).hexdigest(),
        "seeds": seeds,
    }


def _composite(tmp_path):
    historical = {
        "role": "historical_train",
        "split": "train",
        "schema_version": "n4-ranking-sample.v1",
        "risk_label_coverage": 0.0,
        "risk_label_version": None,
        **_write_component(tmp_path, "historical", [1, 2], False),
    }
    safe = {
        "role": "safe_train_extension",
        "split": "train",
        "schema_version": "n4-ranking-sample.v1",
        "risk_label_coverage": 1.0,
        "risk_label_version": "n4-risk-label.multistep.v1",
        **_write_component(tmp_path, "safe", [3], True),
    }
    validation = {
        "role": "validation",
        "split": "validation",
        "schema_version": "n4-ranking-sample.v1",
        "risk_label_coverage": 1.0,
        "risk_label_version": "n4-risk-label.multistep.v1",
        **_write_component(tmp_path, "validation", [4], True),
    }
    path = tmp_path / "composite.json"
    path.write_bytes(
        _canonical(
            {
                "schema": "n4-composite-training-manifest.v1",
                "campaign_protocol_version": "synthetic",
                "code_checkpoint": "test",
                "feature_order": [
                    "empirical_gain",
                    "holdout_support",
                    "coverage",
                    "simplicity",
                    "stability",
                    "invariant_safety",
                ],
                "components": [historical, safe, validation],
            }
        )
    )
    return path


def test_composite_loader_verifies_hashes_and_external_splits(tmp_path):
    loaded = load_composite_manifest(_composite(tmp_path))
    assert len(loaded.train) == 3
    assert len(loaded.validation) == 1
    assert {item.split for item in loaded.train} == {"train"}
    assert {item.split for item in loaded.validation} == {"validation"}
    assert sum(
        item.candidates[0].risk_label_available for item in loaded.train
    ) == 1


def test_composite_loader_rejects_modified_component_and_seed_overlap(tmp_path):
    path = _composite(tmp_path)
    (tmp_path / "safe.jsonl").write_bytes(b"tampered\n")
    with pytest.raises(ValueError, match="hash_mismatch"):
        load_composite_manifest(path)

    path = _composite(tmp_path)
    payload = json.loads(path.read_bytes())
    payload["components"][1]["seeds"] = [1]
    path.write_bytes(_canonical(payload))
    with pytest.raises(ValueError, match="seed_overlap"):
        load_composite_manifest(path)


def _candidate(name, supervised):
    return N4CandidateRecord(
        hypothesis_id=name,
        candidate_source="test",
        features=(0.1,) * 6,
        valid_label=True,
        mae_gain_label=0.1,
        invariant_risk_label=0.2 if supervised else None,
        risk_label_available=supervised,
        risk_label_version=(
            "n4-risk-label.multistep.v1" if supervised else None
        ),
    )


def _set(name, supervised):
    return CandidateSetSample(
        candidate_set_id=name,
        scenario_id="scenario",
        seed=1,
        split="train",
        candidates=(_candidate(f"{name}-candidate", supervised),),
    )


def test_sampling_is_by_set_balanced_and_deterministic():
    samples = (
        _set("h1", False),
        _set("h2", False),
        _set("h3", False),
        _set("s1", True),
    )
    natural, natural_report = sample_candidate_sets_for_epoch(
        samples, policy="natural", model_seed=41, epoch=1
    )
    assert natural == tuple(sorted(samples, key=lambda item: item.candidate_set_id))
    assert natural_report["historical_candidate_set_count"] == 3
    first, report = sample_candidate_sets_for_epoch(
        samples, policy="stratified_50_50", model_seed=41, epoch=1
    )
    second, _ = sample_candidate_sets_for_epoch(
        samples, policy="stratified_50_50", model_seed=41, epoch=1
    )
    assert first == second
    assert report["historical_candidate_set_count"] == 3
    assert report["supervised_candidate_set_count"] == 3
    assert report["repetitions"]["supervised"] == 2


def test_automatic_selection_uses_configuration_mean_and_median_run():
    runs = []
    for configuration, base in (("J1", 0.6), ("J2", 0.7)):
        for seed, offset in zip((41, 42, 43), (-0.1, 0.0, 0.1)):
            runs.append(
                {
                    "configuration_id": configuration,
                    "model_seed": seed,
                    "artifact_sha256": f"{configuration}-{seed}",
                    "valid": True,
                    "validation_metrics": {
                        "recall_at_2": base + offset,
                        "mrr": base,
                        "ndcg_at_2": base,
                        "recall_at_1": base,
                        "top2_risk": 0.1,
                    },
                }
            )
    decision = select_configuration_and_median_run(runs)
    assert decision["selected_configuration_id"] == "J2"
    assert decision["selected_model_seed"] == 42
    assert decision["selected_artifact_sha256"] == "J2-42"


def test_selection_rejects_configurations_that_fail_reference_gate():
    runs = [
        {
            "configuration_id": "J1",
            "model_seed": seed,
            "artifact_sha256": str(seed),
            "valid": True,
            "validation_metrics": {
                "recall_at_2": 0.5,
                "mrr": 0.5,
                "ndcg_at_2": 0.5,
                "recall_at_1": 0.5,
                "top2_risk": 0.2,
            },
        }
        for seed in (41, 42, 43)
    ]
    with pytest.raises(RuntimeError, match="no_valid_configuration"):
        select_configuration_and_median_run(
            runs,
            reference_metrics={
                "recall_at_2": 0.5,
                "mrr": 0.5,
                "ndcg_at_2": 0.5,
                "recall_at_1": 0.5,
                "top2_risk": 0.2,
            },
        )


def test_synthetic_grid_executes_exactly_twelve_runs_reproducibly(tmp_path):
    manifest = _composite(tmp_path)
    first = run_training_grid(
        composite_manifest_path=manifest,
        output_dir=tmp_path / "grid-first",
        code_checkpoint="test",
        campaign_protocol_version="synthetic",
        epochs=1,
        patience=1,
        synthetic_smoke=True,
    )
    second = run_training_grid(
        composite_manifest_path=manifest,
        output_dir=tmp_path / "grid-second",
        code_checkpoint="test",
        campaign_protocol_version="synthetic",
        epochs=1,
        patience=1,
        synthetic_smoke=True,
    )
    assert first["run_count"] == 12
    assert first["selected_configuration_id"] == second[
        "selected_configuration_id"
    ]
    assert first["selected_model_seed"] == second["selected_model_seed"]
    assert first["selected_artifact_sha256"] == second[
        "selected_artifact_sha256"
    ]
    assert len(list((tmp_path / "grid-first").glob("J*-seed-*"))) == 12
