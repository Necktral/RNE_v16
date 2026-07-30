from __future__ import annotations

import pytest

from runtime.neural.contracts import NeuralInferenceRequest
from runtime.neural.organs import N4CausalRankingBackend


def _artifact():
    hidden = 2
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
                        "output_dim": hidden,
                        "activation": "relu",
                        "weights": [[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]],
                        "bias": [0, 0],
                    }
                ]
            },
            "heads": {
                "rank": {"weights": [-1, 2], "bias": 0},
                "validity": {"weights": [2, 0], "bias": 0},
                "gain": {"weights": [1, 0], "bias": 0},
                "risk": {"weights": [0, -2], "bias": 0},
            },
        },
        "calibration": {
            "kind": "platt",
            "target": "validity_logit",
            "a": 2,
            "b": -1,
        },
        "gates": {"promotable": True},
    }


def _infer(artifact, candidates):
    return N4CausalRankingBackend(
        artifact, artifact_sha256="a" * 64
    ).infer(
        NeuralInferenceRequest(
            inference_id="test",
            run_id="test",
            logical_time=1,
            payload={"candidates": candidates},
        )
    ).candidate_output


def test_v2_orders_only_by_rank_score_and_uses_deterministic_tie_break():
    result = _infer(
        _artifact(),
        [
            {"hypothesis_id": "z", "empirical_gain": 1, "holdout_support": 0},
            {"hypothesis_id": "b", "empirical_gain": 0, "holdout_support": 1},
            {"hypothesis_id": "a", "empirical_gain": 0, "holdout_support": 1},
        ],
    )
    rankings = result["rankings"]
    assert [item["hypothesis_id"] for item in rankings] == ["a", "b", "z"]
    assert rankings[0]["priority"] == rankings[0]["rank_score"]
    assert rankings[-1]["validity_probability"] > rankings[0]["validity_probability"]


def test_v2_exposes_trace_and_separately_calibrated_outputs():
    result = _infer(
        _artifact(),
        [{"hypothesis_id": "h", "empirical_gain": 0.5, "holdout_support": 0.25}],
    )
    item = result["rankings"][0]
    assert result["artifact_schema"] == "n4-ranking-artifact.v2"
    assert result["artifact_sha256"] == "a" * 64
    assert item["candidate_set_id"] == "test"
    assert item["candidate_source"] == "unknown"
    assert item["rank_position"] == 1
    assert item["within_top2_budget"] is True
    assert item["validity_logit"] == pytest.approx(1.0)
    assert item["validity_probability"] == pytest.approx(0.731058579)
    assert item["expected_mae_gain"] == pytest.approx(0.462117157)
    assert item["invariant_risk"] == pytest.approx(0.377540669)
    assert len(item["feature_vector"]) == 6


def test_v1_path_remains_available():
    backend = N4CausalRankingBackend(
        {
            "schema": "n4-ranking-artifact.v1",
            "ranking_weights": [1, 0, 0, 0, 0, 0],
            "bias": 0,
            "temperature": 1,
        }
    )
    result = backend.infer(
        NeuralInferenceRequest(
            inference_id="v1",
            run_id="v1",
            logical_time=1,
            payload={"candidates": [{"hypothesis_id": "h", "empirical_gain": 1}]},
        )
    )
    assert result.candidate_output["schema"] == "n4-ranking.v1"
    assert "rank_score" not in result.candidate_output["rankings"][0]


def test_unknown_schema_is_rejected():
    with pytest.raises(ValueError, match="schema_invalid"):
        N4CausalRankingBackend({"schema": "n4-ranking-artifact.v99"})


def test_v2_rejects_nonpositive_platt_scale():
    artifact = _artifact()
    artifact["calibration"]["a"] = 0
    with pytest.raises(ValueError, match="calibration_scale"):
        N4CausalRankingBackend(artifact)
