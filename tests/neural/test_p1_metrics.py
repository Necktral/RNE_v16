from runtime.neural.p1_metrics import bootstrap_mean_ci95, summarize_p1_rows


def _row(scenario: str, report: dict):
    return {
        "episode_result": {
            "episode": {
                "scenario": scenario,
                "result": {"p1_cognitive_loop": report},
            }
        }
    }


def test_summary_excludes_two_warmup_visits_per_scenario() -> None:
    report = {
        "n2": {
            "attempt_count": 1,
            "status": "accepted",
            "ground_truth": {
                "scored": True,
                "initial_false_rejection": True,
                "valid_correction": True,
                "retry_false_accept": False,
                "final_false_rejection": False,
            },
        },
        "n3": {
            "ground_truth_metrics": {
                "ndcg_delta": 0.2,
                "mrr_delta": 0.1,
                "risk_brier": 0.04,
            }
        },
        "n4": {
            "evaluation": {
                "coverage": 1.0,
                "top1_correct": True,
                "mae_delta": 0.1,
                "pairwise_ranking_accuracy": 1.0,
                "regret_delta_vs_canonical": 0.2,
                "regret_delta_vs_prior": 0.1,
                "candidate_hash_preserved": True,
            }
        },
    }
    rows = [_row("thermal", report) for _ in range(3)] + [
        _row("resource", report) for _ in range(3)
    ]

    summary = summarize_p1_rows(rows)

    assert summary["warmup_steps"] == 4
    assert summary["scored_steps"] == 2
    assert summary["n2"]["attempts"] == 2
    assert summary["n2"]["valid_corrections"] == 2
    assert summary["n3"]["mean_ndcg_delta"] == 0.2
    assert summary["n4"]["top1_accuracy"] == 1.0
    assert summary["n4"]["candidate_hash_preserved"] is True


def test_bootstrap_ci_is_reproducible_and_empty_is_unavailable() -> None:
    assert bootstrap_mean_ci95([], seed=7) is None
    assert bootstrap_mean_ci95([0.1, 0.2, 0.3], seed=7) == bootstrap_mean_ci95(
        [0.1, 0.2, 0.3], seed=7
    )
