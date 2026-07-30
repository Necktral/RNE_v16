"""CLI validada para entrenar y exportar el ranker N4."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime.neural.training.n4_ranking import (
    FEATURE_NAMES,
    N4TrainingSample,
    train_n4_ranking,
)


def load_samples(path: Path) -> tuple[N4TrainingSample, ...]:
    samples = []
    sample_ids: set[str] = set()
    candidate_sets: set[str] = set()
    splits: set[str] = set()
    with path.open("rb") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if row["schema_version"] != "n4-ranking-sample.v1":
                    raise ValueError("schema")
                sample_id = str(row["sample_id"])
                if sample_id in sample_ids:
                    raise ValueError("duplicate_sample_id")
                sample_ids.add(sample_id)
                candidate_sets.add(str(row["candidate_set_id"]))
                splits.add(str(row["split"]))
                features = row["features"]
                samples.append(
                    N4TrainingSample(
                        features=tuple(float(features[name]) for name in FEATURE_NAMES),
                        valid=bool(row["valid"]),
                        mae_gain=float(row["mae_gain"]),
                        invariant_risk=float(row["invariant_risk"]),
                        scenario=str(row["scenario"]),
                        seed=int(row["seed"]),
                        logical_time=int(row["logical_time"]),
                        split=str(row["split"]),
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid_n4_training_line:{line_number}") from exc
    if len(candidate_sets) < 1:
        raise ValueError("n4_dataset_has_no_candidate_sets")
    if splits != {"train", "validation", "holdout"}:
        raise ValueError("n4_dataset_requires_grouped_splits")
    return tuple(samples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    elif args.device == "cuda":
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("n4_cuda_requested_but_unavailable")
    dataset_sha256 = hashlib.sha256(args.dataset.read_bytes()).hexdigest()
    result = train_n4_ranking(
        load_samples(args.dataset),
        artifact_path=args.artifact,
        seed=args.seed,
        epochs=args.epochs,
        patience=args.patience,
        training_metadata={
            "dataset_sha256": dataset_sha256,
            "dataset_path": args.dataset.name,
        },
    )
    print(json.dumps(result, indent=2, sort_keys=True))
