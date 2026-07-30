"""Consulta única y sellada de un holdout N4."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime.neural.contracts import NeuralInferenceRequest
from runtime.neural.organs import N4CausalRankingBackend
from runtime.neural.training.n4_ranking import FEATURE_NAMES, load_candidate_sets


def evaluate_holdout_once(
    *,
    artifact_path: Path,
    dataset_path: Path,
    expected_artifact_sha256: str,
    expected_dataset_sha256: str,
    expected_code_checkpoint: str,
    output_dir: Path,
) -> dict:
    if output_dir.exists():
        raise FileExistsError("n4_holdout_output_already_exists")
    artifact_raw = artifact_path.read_bytes()
    dataset_raw = dataset_path.read_bytes()
    _require_sha(
        artifact_raw,
        expected_artifact_sha256,
        "n4_holdout_artifact_hash_mismatch",
    )
    _require_sha(
        dataset_raw,
        expected_dataset_sha256,
        "n4_holdout_dataset_hash_mismatch",
    )
    artifact = json.loads(artifact_raw)
    if artifact.get("schema") != "n4-ranking-artifact.v2":
        raise ValueError("n4_holdout_artifact_schema_invalid")
    provenance = artifact.get("training_provenance") or {}
    if provenance.get("code_checkpoint") != expected_code_checkpoint:
        raise ValueError("n4_holdout_code_checkpoint_mismatch")
    rows = [
        json.loads(line) for line in dataset_raw.splitlines() if line.strip()
    ]
    sets = load_candidate_sets([{**row, "split": "holdout"} for row in rows])
    backend = N4CausalRankingBackend(
        artifact, artifact_sha256=expected_artifact_sha256
    )
    predictions = []
    for sample in sets:
        output = backend.infer(
            NeuralInferenceRequest(
                inference_id=f"holdout/{sample.candidate_set_id}",
                run_id="n4-holdout",
                logical_time=1,
                payload={
                    "candidate_set_id": sample.candidate_set_id,
                    "candidates": [
                        {
                            "hypothesis_id": item.hypothesis_id,
                            **dict(zip(FEATURE_NAMES, item.features)),
                        }
                        for item in sample.candidates
                    ],
                },
            )
        )
        by_id = {
            item.hypothesis_id: item for item in sample.candidates
        }
        for ranked in output.candidate_output["rankings"]:
            candidate = by_id[ranked["hypothesis_id"]]
            predictions.append(
                {
                    **ranked,
                    "candidate_set_id": sample.candidate_set_id,
                    "valid_label": candidate.valid_label,
                    "risk_label": candidate.invariant_risk_label,
                }
            )
    metrics = _metrics(predictions)
    result = {
        "schema": "n4-holdout-result.v1",
        "artifact_sha256": expected_artifact_sha256,
        "dataset_sha256": expected_dataset_sha256,
        "code_checkpoint": expected_code_checkpoint,
        "candidate_set_count": len(sets),
        "sample_count": len(predictions),
        "metrics": metrics,
        "predictions_sha256": hashlib.sha256(
            json.dumps(
                predictions,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest(),
    }
    output_dir.mkdir(parents=True)
    predictions_path = output_dir / "holdout_predictions.jsonl"
    predictions_path.write_bytes(
        b"".join(
            json.dumps(
                item,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
            + b"\n"
            for item in predictions
        )
    )
    (output_dir / "holdout_trace.jsonl").write_bytes(
        predictions_path.read_bytes()
    )
    (output_dir / "holdout_result.json").write_bytes(
        json.dumps(
            result,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        + b"\n"
    )
    (output_dir / "holdout_report.md").write_text(
        "# N4 Holdout Result\n\n"
        f"- Artifact SHA-256: `{expected_artifact_sha256}`\n"
        f"- Dataset SHA-256: `{expected_dataset_sha256}`\n"
        f"- Candidate sets: {len(sets)}\n"
        f"- Brier: {metrics['brier']}\n"
        f"- ECE: {metrics['ece']}\n"
        f"- Recall@2: {metrics['recall_at_2']}\n"
        f"- Risk MAE: {metrics['risk_mae']}\n",
        encoding="utf-8",
    )
    return result


def _metrics(predictions):
    probabilities = [
        float(item["validity_probability"]) for item in predictions
    ]
    labels = [float(item["valid_label"]) for item in predictions]
    brier = sum(
        (prediction - label) ** 2
        for prediction, label in zip(probabilities, labels)
    ) / len(labels)
    bins = [[] for _ in range(10)]
    for probability, label in zip(probabilities, labels):
        bins[min(9, int(probability * 10))].append((probability, label))
    ece = sum(
        len(bucket)
        / len(labels)
        * abs(
            sum(item[0] for item in bucket) / len(bucket)
            - sum(item[1] for item in bucket) / len(bucket)
        )
        for bucket in bins
        if bucket
    )
    risk_pairs = [
        (float(item["invariant_risk"]), float(item["risk_label"]))
        for item in predictions
        if item["risk_label"] is not None
    ]
    risk_mae = sum(abs(a - b) for a, b in risk_pairs) / len(risk_pairs)
    constant = sum(item[1] for item in risk_pairs) / len(risk_pairs)
    constant_mae = (
        sum(abs(target - constant) for _, target in risk_pairs)
        / len(risk_pairs)
    )
    by_set = {}
    for item in predictions:
        by_set.setdefault(item["candidate_set_id"], []).append(item)
    recalls = []
    for rows in by_set.values():
        safe_valid = {
            item["hypothesis_id"]
            for item in rows
            if item["valid_label"]
            and item["risk_label"] is not None
            and float(item["risk_label"]) <= 1e-6
        }
        if safe_valid:
            recalls.append(
                float(
                    bool(
                        safe_valid
                        & {
                            item["hypothesis_id"]
                            for item in sorted(
                                rows, key=lambda item: item["rank_position"]
                            )[:2]
                        }
                    )
                )
            )
    return {
        "brier": round(brier, 9),
        "ece": round(ece, 9),
        "recall_at_2": round(sum(recalls) / len(recalls), 9)
        if recalls
        else 0.0,
        "risk_mae": round(risk_mae, 9),
        "risk_constant_predictor_mae": round(constant_mae, 9),
        "outputs_finite": all(
            math.isfinite(float(value))
            for item in predictions
            for key, value in item.items()
            if key
            in {
                "rank_score",
                "validity_probability",
                "expected_mae_gain",
                "invariant_risk",
            }
        ),
    }


def _require_sha(raw, expected, error):
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(error)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--expected-artifact-sha256", required=True)
    parser.add_argument("--expected-dataset-sha256", required=True)
    parser.add_argument("--expected-code-checkpoint", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate_holdout_once(
        artifact_path=args.artifact,
        dataset_path=args.dataset,
        expected_artifact_sha256=args.expected_artifact_sha256,
        expected_dataset_sha256=args.expected_dataset_sha256,
        expected_code_checkpoint=args.expected_code_checkpoint,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
