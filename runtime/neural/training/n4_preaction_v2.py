"""Deterministic CPU trainer for the P1 N4 pre-action artifact.

Training consumes labels only here.  Runtime receives the resulting coefficients,
calibration envelope and train-domain ranges; it never receives a label/outcome.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from statistics import fmean
from typing import Any, Iterable, Mapping, Sequence

from runtime.neural.integration.p1_n4 import (
    ARTIFACT_SCHEMA_VERSION,
    FEATURE_NAMES,
    PREACTION_BACKEND,
)


EXPECTED_TRAJECTORY_COUNTS = {"train": 24, "validation": 6, "evaluation": 12}


def _sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"n4_training_{name}_not_finite")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"n4_training_{name}_not_finite")
    return resolved


@dataclass(frozen=True, slots=True)
class N4PreactionTrainingRecord:
    trajectory_id: str
    split: str
    state_hash: str
    scenario_id: str
    intervention: str
    features: Mapping[str, float]
    target_delta: float

    def __post_init__(self) -> None:
        if self.split not in EXPECTED_TRAJECTORY_COUNTS:
            raise ValueError("n4_training_split_invalid")
        if not self.trajectory_id or not self.scenario_id or not self.intervention:
            raise ValueError("n4_training_identity_required")
        if len(self.state_hash) != 64:
            raise ValueError("n4_training_state_hash_invalid")
        if set(self.features) != set(FEATURE_NAMES):
            raise ValueError("n4_training_feature_schema_mismatch")
        for name, value in self.features.items():
            _finite(value, name)
        _finite(self.target_delta, "target_delta")

    @property
    def pair(self) -> str:
        return f"{self.scenario_id}::{self.intervention}"


def _solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    """Small ridge normal-equation solver with deterministic pivoting."""

    size = len(vector)
    augmented = [list(matrix[row]) + [vector[row]] for row in range(size)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            raise ValueError("n4_training_singular_design")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = augmented[column][column]
        augmented[column] = [value / scale for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor == 0.0:
                continue
            augmented[row] = [
                left - factor * right
                for left, right in zip(augmented[row], augmented[column])
            ]
    return [augmented[row][-1] for row in range(size)]


def _predict(payload: Mapping[str, Any], record: N4PreactionTrainingRecord) -> float:
    return float(payload["pair_bias"][record.pair]) + sum(
        float(payload["coefficients"][name]) * float(record.features[name])
        for name in FEATURE_NAMES
    )


def train_n4_preaction_v2(
    records: Iterable[N4PreactionTrainingRecord],
    *,
    model_id: str = "n4-preaction-v2",
) -> dict[str, Any]:
    rows = tuple(records)
    if not rows:
        raise ValueError("n4_training_records_required")
    trajectories = {
        split: {row.trajectory_id for row in rows if row.split == split}
        for split in EXPECTED_TRAJECTORY_COUNTS
    }
    counts = {split: len(values) for split, values in trajectories.items()}
    if counts != EXPECTED_TRAJECTORY_COUNTS:
        raise ValueError(f"n4_training_trajectory_counts_invalid:{counts}")
    if any(
        trajectories[left] & trajectories[right]
        for index, left in enumerate(EXPECTED_TRAJECTORY_COUNTS)
        for right in tuple(EXPECTED_TRAJECTORY_COUNTS)[index + 1 :]
    ):
        raise ValueError("n4_training_trajectory_overlap")
    state_hashes = {
        split: {row.state_hash for row in rows if row.split == split}
        for split in EXPECTED_TRAJECTORY_COUNTS
    }
    if any(
        state_hashes[left] & state_hashes[right]
        for index, left in enumerate(EXPECTED_TRAJECTORY_COUNTS)
        for right in tuple(EXPECTED_TRAJECTORY_COUNTS)[index + 1 :]
    ):
        raise ValueError("n4_training_state_overlap")

    train = [row for row in rows if row.split == "train"]
    validation = [row for row in rows if row.split == "validation"]
    pairs = sorted({row.pair for row in train})
    if any(row.pair not in pairs for row in validation):
        raise ValueError("n4_training_validation_pair_unseen")
    width = len(FEATURE_NAMES) + len(pairs)
    xtx = [[0.0 for _ in range(width)] for _ in range(width)]
    xty = [0.0 for _ in range(width)]
    for row in train:
        vector = [float(row.features[name]) for name in FEATURE_NAMES] + [
            1.0 if row.pair == pair else 0.0 for pair in pairs
        ]
        target = float(row.target_delta)
        for left in range(width):
            xty[left] += vector[left] * target
            for right in range(width):
                xtx[left][right] += vector[left] * vector[right]
    for index in range(width):
        xtx[index][index] += 1e-6
    solution = _solve(xtx, xty)
    payload: dict[str, Any] = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "backend": PREACTION_BACKEND,
        "model_id": str(model_id),
        "feature_names": list(FEATURE_NAMES),
        "coefficients": {
            name: solution[index] for index, name in enumerate(FEATURE_NAMES)
        },
        "pair_bias": {
            pair: solution[len(FEATURE_NAMES) + index]
            for index, pair in enumerate(pairs)
        },
    }
    errors = sorted(abs(_predict(payload, row) - row.target_delta) for row in validation)
    quantile_index = min(len(errors) - 1, int(math.ceil(0.90 * len(errors))) - 1)
    half_width = errors[quantile_index] if errors else 0.0
    validation_mae = fmean(errors) if errors else 0.0
    payload.update(
        {
            "feature_ranges": {
                pair: {
                    name: [
                        min(row.features[name] for row in train if row.pair == pair),
                        max(row.features[name] for row in train if row.pair == pair),
                    ]
                    for name in FEATURE_NAMES
                }
                for pair in pairs
            },
            "calibration_half_width": half_width,
            "confidence": 1.0 / (1.0 + validation_mae),
            "training_provenance": {
                "trajectory_counts": counts,
                "trajectory_hashes": {
                    split: _sha(sorted(values)) for split, values in trajectories.items()
                },
                "state_hashes": {
                    split: _sha(sorted(values)) for split, values in state_hashes.items()
                },
                "split_disjoint": True,
                "split_unit": "complete_trajectory",
                "steps_per_trajectory": 32,
                "scenarios": 4,
                "evaluation_opened": False,
                "train_rows": len(train),
                "validation_rows": len(validation),
                "sealed_evaluation_rows": sum(row.split == "evaluation" for row in rows),
                "validation_mae": validation_mae,
            },
        }
    )
    return payload


__all__ = [
    "EXPECTED_TRAJECTORY_COUNTS",
    "N4PreactionTrainingRecord",
    "train_n4_preaction_v2",
]
