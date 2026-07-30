from __future__ import annotations

import hashlib
import json

import pytest

from runtime.neural.contracts import canonical_sha256
from scripts.evaluate_n4_holdout import evaluate_holdout_once


def _artifact():
    return {
        "schema": "n4-ranking-artifact.v2",
        "model_kind": "trained_multihead",
        "ranking_objective": "pairwise_constrained_v1",
        "feature_order": [
            "empirical_gain",
            "holdout_support",
            "coverage",
            "simplicity",
            "stability",
            "invariant_safety",
        ],
        "feature_transform": {"kind": "identity"},
        "model": {
            "trunk": {
                "layers": [
                    {
                        "input_dim": 6,
                        "output_dim": 2,
                        "activation": "relu",
                        "weights": [[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]],
                        "bias": [0, 0],
                    }
                ]
            },
            "heads": {
                name: {"weights": [1, 0], "bias": 0}
                for name in ("rank", "validity", "gain", "risk")
            },
        },
        "calibration": {
            "kind": "platt",
            "target": "validity_logit",
            "a": 1,
            "b": 0,
        },
        "training_provenance": {"code_checkpoint": "checkpoint"},
        "gates": {"promotable": False},
    }


def _row(hypothesis, risk):
    report = {"risk": risk}
    return {
        "candidate_set_id": "set",
        "hypothesis_id": hypothesis,
        "scenario": "thermal_with_battery",
        "seed": 9,
        "split": "holdout",
        "source": "test",
        "features": {
            "empirical_gain": risk,
            "holdout_support": 0.2,
            "coverage": 0.3,
            "simplicity": 0.4,
            "stability": 0.5,
            "invariant_safety": 0.6,
        },
        "valid": hypothesis == "a",
        "mae_gain": 0.2,
        "invariant_risk": risk,
        "risk_label": risk,
        "risk_label_available": True,
        "risk_label_version": "n4-risk-label.multistep.v1",
        "risk_report": report,
        "risk_report_sha256": canonical_sha256(report),
        "safety_contract_version": "thermal_with_battery.safety.v1",
        "rollout_horizon": 3,
    }


def _write(tmp_path):
    artifact = tmp_path / "artifact.json"
    dataset = tmp_path / "holdout.jsonl"
    artifact.write_text(
        json.dumps(_artifact(), sort_keys=True, separators=(",", ":")) + "\n"
    )
    dataset.write_text(
        "\n".join(
            json.dumps(_row(name, risk), sort_keys=True, separators=(",", ":"))
            for name, risk in (("a", 0.0), ("b", 0.4))
        )
        + "\n"
    )
    return artifact, dataset


def test_holdout_evaluator_is_single_use_and_hash_guarded(tmp_path):
    artifact, dataset = _write(tmp_path)
    artifact_hash = hashlib.sha256(artifact.read_bytes()).hexdigest()
    dataset_hash = hashlib.sha256(dataset.read_bytes()).hexdigest()
    output = tmp_path / "holdout-result"
    result = evaluate_holdout_once(
        artifact_path=artifact,
        dataset_path=dataset,
        expected_artifact_sha256=artifact_hash,
        expected_dataset_sha256=dataset_hash,
        expected_code_checkpoint="checkpoint",
        output_dir=output,
    )
    assert result["artifact_sha256"] == artifact_hash
    assert result["metrics"]["outputs_finite"]
    assert (output / "holdout_result.json").exists()
    assert (output / "holdout_predictions.jsonl").exists()
    assert (output / "holdout_trace.jsonl").exists()
    assert (output / "holdout_report.md").exists()
    with pytest.raises(FileExistsError, match="already_exists"):
        evaluate_holdout_once(
            artifact_path=artifact,
            dataset_path=dataset,
            expected_artifact_sha256=artifact_hash,
            expected_dataset_sha256=dataset_hash,
            expected_code_checkpoint="checkpoint",
            output_dir=output,
        )
    with pytest.raises(ValueError, match="artifact_hash_mismatch"):
        evaluate_holdout_once(
            artifact_path=artifact,
            dataset_path=dataset,
            expected_artifact_sha256="0" * 64,
            expected_dataset_sha256=dataset_hash,
            expected_code_checkpoint="checkpoint",
            output_dir=tmp_path / "other",
        )
