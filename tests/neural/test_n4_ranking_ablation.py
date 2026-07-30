from __future__ import annotations

from runtime.neural.organs import N4CausalRankingBackend
from scripts.evaluate_n4_ranking_ablation import evaluate_arm


def _row(hypothesis_id, gain, valid, source="quantile"):
    return {
        "candidate_set_id": "set",
        "hypothesis_id": hypothesis_id,
        "logical_time": 10,
        "seed": 1,
        "source": source,
        "valid": valid,
        "mae_gain": gain,
        "invariant_risk": 0.0,
        "evaluation_count": 4,
        "features": {
            "empirical_gain": gain,
            "holdout_support": gain,
            "coverage": 1.0,
            "simplicity": 1.0,
            "stability": 1.0,
            "invariant_safety": 1.0,
        },
    }


def test_ablation_metrics_reward_valid_top_rank():
    metrics = evaluate_arm(
        [[_row("weak", 0.1, False), _row("strong", 0.9, True)]],
        backend=N4CausalRankingBackend(),
        include_boundaries=True,
    )
    assert metrics["recall_at_1"] == 1.0
    assert metrics["mrr"] == 1.0
    assert metrics["top1_mae_gain"] == 0.9


def test_quantile_arm_excludes_change_boundaries():
    metrics = evaluate_arm(
        [[
            _row("quantile", 0.1, False),
            _row("boundary", 0.9, True, "change_boundary"),
        ]],
        backend=N4CausalRankingBackend(),
        include_boundaries=False,
    )
    assert metrics["recall_at_1"] == 0.0
