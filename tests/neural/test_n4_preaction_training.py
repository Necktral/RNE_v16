from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

from runtime.neural.integration.p1_n4 import N4PreactionArtifactV2
from runtime.neural.training.n4_preaction_v2 import train_n4_preaction_v2
from scripts.train_n4_preaction_v2 import build_training_records


def _digest(payload) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@pytest.fixture(scope="module")
def records():
    return build_training_records()


def test_dataset_uses_exact_disjoint_24_6_12_complete_trajectories(records) -> None:
    trajectories = {
        split: {row.trajectory_id for row in records if row.split == split}
        for split in ("train", "validation", "evaluation")
    }

    assert {key: len(value) for key, value in trajectories.items()} == {
        "train": 24,
        "validation": 6,
        "evaluation": 12,
    }
    assert not trajectories["train"] & trajectories["validation"]
    assert not trajectories["train"] & trajectories["evaluation"]
    assert not trajectories["validation"] & trajectories["evaluation"]
    assert {len([row for row in records if row.trajectory_id == trajectory]) for trajectory in set().union(*trajectories.values())} == {64}


def test_training_is_deterministic_and_never_opens_sealed_evaluation(records) -> None:
    first = train_n4_preaction_v2(records)
    changed_evaluation = tuple(
        replace(row, target_delta=row.target_delta + 1000.0)
        if row.split == "evaluation"
        else row
        for row in records
    )
    second = train_n4_preaction_v2(changed_evaluation)

    assert first == second
    assert first["training_provenance"]["evaluation_opened"] is False
    artifact = N4PreactionArtifactV2.from_payload(
        first, artifact_sha256=_digest(first)
    )
    assert artifact.backend == "rnfe-n4-preaction-linear-v2"
    assert artifact.training_provenance["trajectory_counts"] == {
        "evaluation": 12,
        "train": 24,
        "validation": 6,
    }


def test_training_rejects_a_state_crossing_splits(records) -> None:
    validation_index = next(
        index for index, row in enumerate(records) if row.split == "validation"
    )
    corrupted = list(records)
    corrupted[validation_index] = replace(
        corrupted[validation_index], state_hash=records[0].state_hash
    )

    with pytest.raises(ValueError, match="state_overlap"):
        train_n4_preaction_v2(corrupted)
