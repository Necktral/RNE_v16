from __future__ import annotations

import copy

import pytest
import torch

from runtime.neural.training.n4_ranking import (
    FEATURE_NAMES,
    collate_candidate_sets,
    load_candidate_sets,
)


def _row(
    set_id: str,
    hypothesis_id: str,
    *,
    split: str = "train",
    source: str = "quantile",
    seed: int = 7,
):
    return {
        "candidate_set_id": set_id,
        "hypothesis_id": hypothesis_id,
        "scenario": "thermal_with_battery",
        "seed": seed,
        "split": split,
        "source": source,
        "features": {
            name: float(index + 1) / 10.0
            for index, name in enumerate(FEATURE_NAMES)
        },
        "valid": hypothesis_id.endswith("valid"),
        "mae_gain": 0.25,
        "invariant_risk": 0.1,
    }


def test_loader_groups_and_orders_candidate_sets_deterministically():
    rows = [
        _row("set-b", "h-2"),
        _row("set-a", "h-z", source="boundary"),
        _row("set-a", "h-a"),
    ]
    first = load_candidate_sets(rows)
    second = load_candidate_sets(reversed(copy.deepcopy(rows)))
    assert [item.candidate_set_id for item in first] == ["set-a", "set-b"]
    assert [item.hypothesis_id for item in first[0].candidates] == ["h-a", "h-z"]
    assert first[0].candidates[1].candidate_source == "boundary"
    assert first == second
    assert [item.logical_hash for item in first] == [
        item.logical_hash for item in second
    ]


def test_collate_masks_padding_and_is_reproducible():
    samples = load_candidate_sets(
        [
            _row("set-b", "h-2"),
            _row("set-a", "h-2"),
            _row("set-a", "h-1"),
        ]
    )
    first = collate_candidate_sets(samples)
    second = collate_candidate_sets(samples)
    assert first.features.shape == (2, 2, 6)
    assert first.valid_labels.shape == (2, 2)
    assert first.gain_labels.shape == (2, 2)
    assert first.risk_labels.shape == (2, 2)
    assert first.candidate_mask.tolist() == [[True, True], [True, False]]
    assert torch.equal(first.features, second.features)
    assert torch.equal(first.candidate_mask, second.candidate_mask)
    assert first.candidate_set_ids == ("set-a", "set-b")


def test_loader_rejects_duplicate_hypothesis_within_set():
    with pytest.raises(ValueError, match="duplicate_hypothesis"):
        load_candidate_sets([_row("set-a", "h-1"), _row("set-a", "h-1")])


def test_loader_rejects_candidate_set_crossing_splits():
    with pytest.raises(ValueError, match="crosses_splits"):
        load_candidate_sets(
            [
                _row("set-a", "h-1", split="train"),
                _row("set-a", "h-2", split="validation"),
            ]
        )


def test_loader_rejects_feature_shape_and_nonfinite_values():
    missing = _row("set-a", "h-1")
    del missing["features"][FEATURE_NAMES[-1]]
    with pytest.raises(ValueError, match="row_invalid"):
        load_candidate_sets([missing])

    nonfinite = _row("set-a", "h-1")
    nonfinite["features"][FEATURE_NAMES[0]] = float("nan")
    with pytest.raises(ValueError, match="row_invalid"):
        load_candidate_sets([nonfinite])


def test_loader_rejects_out_of_range_continuous_labels():
    risk = _row("set-a", "h-1")
    risk["invariant_risk"] = 1.1
    with pytest.raises(ValueError, match="row_invalid"):
        load_candidate_sets([risk])

    gain = _row("set-a", "h-1")
    gain["mae_gain"] = -1.1
    with pytest.raises(ValueError, match="row_invalid"):
        load_candidate_sets([gain])


def test_loader_rejects_metadata_conflicts_inside_set():
    other_seed = _row("set-a", "h-2", seed=8)
    with pytest.raises(ValueError, match="metadata_conflict"):
        load_candidate_sets([_row("set-a", "h-1"), other_seed])


def test_loader_accepts_variable_set_sizes_and_continuous_risk():
    rows = [
        _row("set-a", "h-1"),
        _row("set-a", "h-2"),
        _row("set-a", "h-3"),
        _row("set-b", "h-4"),
    ]
    rows[0]["invariant_risk"] = 0.375
    samples = load_candidate_sets(rows)
    assert [len(item.candidates) for item in samples] == [3, 1]
    assert samples[0].candidates[0].invariant_risk_label == 0.375
