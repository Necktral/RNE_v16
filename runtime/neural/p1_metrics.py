"""Aggregation for the P1 cognitive-loop SHADOW campaign."""

from __future__ import annotations

import math
import random
from statistics import fmean
from typing import Any, Iterable, Mapping, Sequence


P1_REPORT_SCHEMA_VERSION = "rnfe-p1-cognitive-loop-matrix-v2"


def summarize_p1_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    warmup_visits_per_scenario: int = 2,
) -> dict[str, Any]:
    visits: dict[str, int] = {}
    scored: list[Mapping[str, Any]] = []
    total = 0
    warmup = 0
    unscored = 0
    for row in rows:
        total += 1
        episode_result = row.get("episode_result") if isinstance(row, Mapping) else None
        episode_result = episode_result if isinstance(episode_result, Mapping) else row
        episode = episode_result.get("episode") if isinstance(episode_result, Mapping) else None
        episode = episode if isinstance(episode, Mapping) else {}
        scenario = str(episode.get("scenario") or "unknown")
        visits[scenario] = visits.get(scenario, 0) + 1
        if visits[scenario] <= max(0, int(warmup_visits_per_scenario)):
            warmup += 1
            continue
        result = episode.get("result")
        result = result if isinstance(result, Mapping) else {}
        report = result.get("p1_cognitive_loop")
        if isinstance(report, Mapping):
            scored.append(report)
        else:
            unscored += 1

    n2_ground = [
        item
        for report in scored
        if isinstance((n2 := report.get("n2")), Mapping)
        and isinstance((item := n2.get("ground_truth")), Mapping)
        and item.get("scored") is True
    ]
    n2_attempts = [
        report["n2"]
        for report in scored
        if isinstance(report.get("n2"), Mapping)
        and int(report["n2"].get("attempt_count", 0) or 0) == 1
    ]
    n3_metrics = [
        metrics
        for report in scored
        if isinstance((n3 := report.get("n3")), Mapping)
        and isinstance((metrics := n3.get("ground_truth_metrics")), Mapping)
    ]
    n4_evaluations = [
        evaluation
        for report in scored
        if isinstance((n4 := report.get("n4")), Mapping)
        and isinstance((evaluation := n4.get("evaluation")), Mapping)
    ]
    n4_candidates = [
        candidate
        for report in scored
        if isinstance((n4 := report.get("n4")), Mapping)
        and isinstance((candidate := n4.get("candidate")), Mapping)
    ]
    risk_pairs = [
        (float(metrics["risk_prediction"]), bool(metrics["adverse_outcome"]))
        for metrics in n3_metrics
        if isinstance(metrics.get("risk_prediction"), (int, float))
        and not isinstance(metrics.get("risk_prediction"), bool)
        and isinstance(metrics.get("adverse_outcome"), bool)
    ]
    positive = [prediction >= 0.5 for prediction, label in risk_pairs if label]
    negative = [prediction < 0.5 for prediction, label in risk_pairs if not label]

    return {
        "schema_version": "rnfe-p1-lane-summary-v1",
        "total_steps": total,
        "warmup_steps": warmup,
        "unscored_steps": unscored,
        "scored_steps": len(scored),
        "authority_effect": "none",
        "promotion_authorized": False,
        "n2": {
            "attempts": len(n2_attempts),
            "accepted_retries": sum(item.get("status") == "accepted" for item in n2_attempts),
            "initial_false_rejections": _count_true(n2_ground, "initial_false_rejection"),
            "valid_corrections": _count_true(n2_ground, "valid_correction"),
            "retry_false_accepts": _count_true(n2_ground, "retry_false_accept"),
            "final_false_rejections": _count_true(n2_ground, "final_false_rejection"),
            "scored": len(n2_ground),
        },
        "n3": {
            "compared": len(n3_metrics),
            "mean_ndcg_delta": _mean_present(n3_metrics, "ndcg_delta"),
            "mean_mrr_delta": _mean_present(n3_metrics, "mrr_delta"),
            "mean_risk_brier": _mean_present(n3_metrics, "risk_brier"),
            "balanced_accuracy": (
                (fmean(positive) + fmean(negative)) / 2.0
                if positive and negative
                else None
            ),
        },
        "n4": {
            "evaluated": len(n4_evaluations),
            "mean_coverage": _mean_present(n4_evaluations, "coverage"),
            "top1_accuracy": _mean_bool(n4_evaluations, "top1_correct"),
            "mean_mae_delta": _mean_present(n4_evaluations, "mae_delta"),
            "mean_pairwise_accuracy": _mean_present(
                n4_evaluations, "pairwise_ranking_accuracy"
            ),
            "mean_interval_coverage": _mean_present(
                n4_evaluations, "interval_coverage"
            ),
            "mean_regret_delta_vs_canonical": _mean_present(
                n4_evaluations, "regret_delta_vs_canonical"
            ),
            "mean_regret_delta_vs_prior": _mean_present(
                n4_evaluations, "regret_delta_vs_prior"
            ),
            "candidate_hash_preserved": all(
                item.get("candidate_hash_preserved") is True for item in n4_evaluations
            ) if n4_evaluations else None,
            "trained_v2_rate": (
                fmean(
                    float(
                        isinstance(candidate.get("model"), Mapping)
                        and candidate["model"].get("execution_class") == "trained_v2"
                        and bool(candidate["model"].get("artifact_sha256"))
                    )
                    for candidate in n4_candidates
                )
                if n4_candidates
                else None
            ),
        },
    }


def bootstrap_mean_ci95(
    values: Sequence[float],
    *,
    seed: int = 0,
    samples: int = 2_000,
) -> tuple[float, float] | None:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if not clean:
        return None
    rng = random.Random(seed)
    count = max(200, int(samples))
    means = sorted(
        fmean(clean[rng.randrange(len(clean))] for _ in clean) for _ in range(count)
    )
    return (
        means[int(0.025 * (count - 1))],
        means[int(0.975 * (count - 1))],
    )


def _count_true(rows: Sequence[Mapping[str, Any]], key: str) -> int:
    return sum(item.get(key) is True for item in rows)


def _mean_present(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [
        float(item[key])
        for item in rows
        if isinstance(item.get(key), (int, float))
        and not isinstance(item.get(key), bool)
        and math.isfinite(float(item[key]))
    ]
    return fmean(values) if values else None


def _mean_bool(rows: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [float(item[key]) for item in rows if isinstance(item.get(key), bool)]
    return fmean(values) if values else None


__all__ = [
    "P1_REPORT_SCHEMA_VERSION",
    "bootstrap_mean_ci95",
    "summarize_p1_rows",
]
