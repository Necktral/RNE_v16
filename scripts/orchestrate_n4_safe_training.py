"""Orquestador determinista J1–J4 para N4 SAFE-002."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime.neural.contracts import NeuralInferenceRequest
from runtime.neural.organs import N4CausalRankingBackend
from runtime.neural.training.n4_campaign import (
    load_composite_manifest,
    select_configuration_and_median_run,
)
from runtime.neural.training.n4_ranking import (
    FEATURE_NAMES,
    score_n4_artifact_v2,
    train_n4_ranking_v2,
)


CONFIGURATIONS = {
    "J1": {"sampling_policy": "natural", "lambda_risk": 0.2},
    "J2": {"sampling_policy": "natural", "lambda_risk": 0.5},
    "J3": {"sampling_policy": "stratified_50_50", "lambda_risk": 0.2},
    "J4": {"sampling_policy": "stratified_50_50", "lambda_risk": 0.5},
}
MODEL_SEEDS = (41, 42, 43)


def run_training_grid(
    *,
    composite_manifest_path: Path,
    output_dir: Path,
    code_checkpoint: str,
    campaign_protocol_version: str,
    epochs: int,
    patience: int,
    synthetic_smoke: bool = False,
) -> dict:
    if output_dir.exists():
        raise FileExistsError("n4_grid_output_must_be_new")
    composite = load_composite_manifest(composite_manifest_path)
    if composite.manifest.get("code_checkpoint") != code_checkpoint:
        raise ValueError("n4_grid_code_checkpoint_mismatch")
    if (
        composite.manifest.get("campaign_protocol_version")
        != campaign_protocol_version
    ):
        raise ValueError("n4_grid_protocol_version_mismatch")
    output_dir.mkdir(parents=True)
    runs = []
    validation_hashes = set()
    reference_metrics = _reference_validation_metrics(
        composite.validation
    )
    for configuration_id, configuration in CONFIGURATIONS.items():
        for model_seed in MODEL_SEEDS:
            run_dir = output_dir / f"{configuration_id}-seed-{model_seed}"
            run_dir.mkdir()
            artifact_path = run_dir / "artifact.json"
            history_path = run_dir / "epoch_history.jsonl"
            artifact = train_n4_ranking_v2(
                composite.train + composite.validation,
                artifact_path=artifact_path,
                seed=model_seed,
                epochs=epochs,
                patience=patience,
                loss_weights={
                    "rank": 1.0,
                    "valid": 0.5,
                    "gain": 0.3,
                    "risk": configuration["lambda_risk"],
                },
                sampling_policy=configuration["sampling_policy"],
                epoch_history_path=history_path,
                training_metadata={
                    "campaign_protocol_version": campaign_protocol_version,
                    "code_checkpoint": code_checkpoint,
                    "configuration_id": configuration_id,
                    "model_seed": model_seed,
                    "composite_manifest_sha256": (
                        composite.manifest_sha256
                    ),
                    "validation_dataset_sha256": next(
                        item["dataset_sha256"]
                        for item in composite.component_stats
                        if item["role"] == "validation"
                    ),
                    "dataset_lineage": {
                        item["role"]: item["dataset_sha256"]
                        for item in composite.component_stats
                    },
                },
            )
            validation_hashes.add(
                artifact["training_provenance"][
                    "validation_dataset_sha256"
                ]
            )
            parity = _runtime_parity(
                artifact, composite.validation[0].candidates[0]
            )
            metrics = artifact["validation_metrics"]
            valid = (
                parity
                and artifact["gates"]["artifact_quality"]
                and (
                    synthetic_smoke
                    or (
                        artifact["gates"]["calibration"]
                        and artifact["gates"]["risk_target_informative"]
                        and metrics["risk_mae"]
                        < metrics["risk_constant_predictor_mae"]
                        and metrics["risk_spearman"] > 0.0
                    )
                )
            )
            run = {
                "configuration_id": configuration_id,
                "model_seed": model_seed,
                "sampling_policy": configuration["sampling_policy"],
                "loss_weights": artifact["training_provenance"][
                    "loss_weights"
                ],
                "composite_manifest_sha256": composite.manifest_sha256,
                "validation_dataset_sha256": artifact[
                    "training_provenance"
                ]["validation_dataset_sha256"],
                "artifact_path": str(artifact_path.relative_to(output_dir)),
                "artifact_sha256": artifact["artifact_sha256"],
                "epoch_history_sha256": artifact[
                    "training_provenance"
                ]["epoch_history_sha256"],
                "validation_metrics": metrics,
                "runtime_parity_result": parity,
                "valid": valid,
            }
            (run_dir / "run_result.json").write_bytes(
                _canonical(run) + b"\n"
            )
            runs.append(run)
    if len(runs) != 12 or len(validation_hashes) != 1:
        raise RuntimeError("n4_grid_run_or_validation_count_invalid")
    decision = select_configuration_and_median_run(
        runs,
        reference_metrics=None if synthetic_smoke else reference_metrics,
    )
    selected = next(
        item
        for item in runs
        if item["artifact_sha256"]
        == decision["selected_artifact_sha256"]
    )
    source = output_dir / selected["artifact_path"]
    sealed = output_dir / "selected_artifact.json"
    shutil.copyfile(source, sealed)
    sealed_sha256 = hashlib.sha256(sealed.read_bytes()).hexdigest()
    if sealed_sha256 != decision["selected_artifact_sha256"]:
        raise RuntimeError("n4_grid_sealed_artifact_hash_mismatch")
    decision.update(
        {
            "schema": "n4-selection-decision.v1",
            "campaign_protocol_version": campaign_protocol_version,
            "code_checkpoint": code_checkpoint,
            "composite_manifest_sha256": composite.manifest_sha256,
            "run_count": len(runs),
            "validation_dataset_sha256": next(iter(validation_hashes)),
            "reference_validation_metrics": reference_metrics,
            "selected_artifact_path": sealed.name,
        }
    )
    (output_dir / "selection_decision.json").write_bytes(
        _canonical(decision) + b"\n"
    )
    (output_dir / "selected_artifact.sha256").write_text(
        f"{sealed_sha256}  {sealed.name}\n", encoding="utf-8"
    )
    return decision


def _runtime_parity(artifact, candidate):
    expected = score_n4_artifact_v2(candidate.features, artifact)
    backend = N4CausalRankingBackend(artifact)
    observed = backend.infer(
        NeuralInferenceRequest(
            inference_id="grid-parity",
            run_id="grid-parity",
            logical_time=1,
            payload={
                "candidates": [
                    {
                        "hypothesis_id": candidate.hypothesis_id,
                        **dict(zip(FEATURE_NAMES, candidate.features)),
                    }
                ]
            },
        )
    ).candidate_output["rankings"][0]
    return all(
        abs(float(observed[name]) - float(expected[name])) <= 1e-6
        for name in (
            "rank_score",
            "validity_probability",
            "expected_mae_gain",
            "invariant_risk",
        )
    )


def _reference_validation_metrics(samples):
    backend = N4CausalRankingBackend()
    recall1 = []
    recall2 = []
    reciprocal = []
    ndcg = []
    risks = []
    for sample in samples:
        output = backend.infer(
            NeuralInferenceRequest(
                inference_id=f"reference/{sample.candidate_set_id}",
                run_id="reference-validation",
                logical_time=1,
                payload={
                    "candidates": [
                        {
                            "hypothesis_id": item.hypothesis_id,
                            **dict(zip(FEATURE_NAMES, item.features)),
                        }
                        for item in sample.candidates
                    ]
                },
            )
        )
        order = [
            next(
                index
                for index, candidate in enumerate(sample.candidates)
                if candidate.hypothesis_id == ranked["hypothesis_id"]
            )
            for ranked in output.candidate_output["rankings"]
        ]
        safe_valid = {
            index
            for index, candidate in enumerate(sample.candidates)
            if candidate.valid_label
            and candidate.risk_label_available
            and float(candidate.invariant_risk_label) <= 1e-6
        }
        top2 = order[:2]
        risks.append(
            sum(
                float(sample.candidates[index].invariant_risk_label)
                for index in top2
            )
            / len(top2)
        )
        if not safe_valid:
            continue
        recall1.append(float(bool(safe_valid & set(order[:1]))))
        recall2.append(float(bool(safe_valid & set(top2))))
        first = next(
            (
                rank
                for rank, index in enumerate(order, 1)
                if index in safe_valid
            ),
            0,
        )
        reciprocal.append(1.0 / first if first else 0.0)
        gains = [1.0 if index in safe_valid else 0.0 for index in top2]
        dcg = sum(
            value / __import__("math").log2(rank + 1)
            for rank, value in enumerate(gains, 1)
        )
        ideal = sum(
            1.0 / __import__("math").log2(rank + 1)
            for rank in range(1, min(2, len(safe_valid)) + 1)
        )
        ndcg.append(dcg / ideal)
    mean = lambda values: sum(values) / len(values) if values else 0.0
    return {
        "recall_at_1": round(mean(recall1), 9),
        "recall_at_2": round(mean(recall2), 9),
        "mrr": round(mean(reciprocal), 9),
        "ndcg_at_2": round(mean(ndcg), 9),
        "top2_risk": round(mean(risks), 9),
    }


def _canonical(value):
    return json.dumps(
        value,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--composite-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--code-checkpoint", required=True)
    parser.add_argument("--campaign-protocol-version", required=True)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--synthetic-smoke", action="store_true")
    args = parser.parse_args()
    result = run_training_grid(
        composite_manifest_path=args.composite_manifest,
        output_dir=args.output_dir,
        code_checkpoint=args.code_checkpoint,
        campaign_protocol_version=args.campaign_protocol_version,
        epochs=args.epochs,
        patience=args.patience,
        synthetic_smoke=args.synthetic_smoke,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
