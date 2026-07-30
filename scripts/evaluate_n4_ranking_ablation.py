"""Ablación offline pareada A/B/C del generador y ranker N4."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime.neural import NeuralInferenceRequest
from runtime.neural.organs import N4CausalRankingBackend


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _rank(
    rows: list[dict[str, Any]], backend: N4CausalRankingBackend
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    payload = [
        {"hypothesis_id": row["hypothesis_id"], **row["features"]}
        for row in rows
    ]
    result = backend.infer(
        NeuralInferenceRequest(
            inference_id=f"ablation/{rows[0]['candidate_set_id']}",
            run_id="n4-ablation",
            logical_time=int(rows[0]["logical_time"]),
            payload={"candidates": payload},
        )
    )
    by_id = {row["hypothesis_id"]: row for row in rows}
    return [
        (by_id[item["hypothesis_id"]], item)
        for item in result.candidate_output["rankings"]
    ]


def _ece(probabilities: list[float], labels: list[float]) -> float:
    bins = [[] for _ in range(10)]
    for probability, label in zip(probabilities, labels):
        bins[min(9, int(probability * 10))].append((probability, label))
    return sum(
        len(bucket)
        / len(labels)
        * abs(
            sum(item[0] for item in bucket) / len(bucket)
            - sum(item[1] for item in bucket) / len(bucket)
        )
        for bucket in bins
        if bucket
    )


def _ndcg(ranked: list[tuple[dict[str, Any], dict[str, Any]]]) -> float:
    labels = [int(row["valid"]) for row, _ in ranked]
    dcg = sum(label / math.log2(index + 2) for index, label in enumerate(labels))
    ideal = sum(
        1.0 / math.log2(index + 2) for index in range(sum(labels))
    )
    return dcg / ideal if ideal else 0.0


def evaluate_arm(
    candidate_sets: Iterable[list[dict[str, Any]]],
    *,
    backend: N4CausalRankingBackend,
    include_boundaries: bool,
) -> dict[str, Any]:
    recalls = {1: [], 2: [], 4: []}
    reciprocal_ranks: list[float] = []
    ndcgs: list[float] = []
    top_gains: list[float] = []
    top_risks: list[float] = []
    until_valid: list[float] = []
    costs: list[float] = []
    probabilities: list[float] = []
    labels: list[float] = []
    recall2_by_seed: dict[int, list[float]] = defaultdict(list)
    evaluated_sets = 0
    for raw_rows in candidate_sets:
        rows = [
            row
            for row in raw_rows
            if include_boundaries or row["source"] != "change_boundary"
        ]
        if not rows:
            continue
        ranked = _rank(rows, backend)
        evaluated_sets += 1
        valid_ranks = [
            index
            for index, (row, _) in enumerate(ranked, 1)
            if row["valid"]
        ]
        first = valid_ranks[0] if valid_ranks else len(ranked) + 1
        reciprocal_ranks.append(1.0 / first if valid_ranks else 0.0)
        until_valid.append(float(first))
        ndcgs.append(_ndcg(ranked))
        top_gains.append(float(ranked[0][0]["mae_gain"]))
        top_risks.append(float(ranked[0][0]["invariant_risk"]))
        costs.append(sum(float(row["evaluation_count"]) for row, _ in ranked))
        for k in recalls:
            hit = float(any(rank <= k for rank in valid_ranks))
            recalls[k].append(hit)
            if k == 2:
                recall2_by_seed[int(rows[0]["seed"])].append(hit)
        for row, ranking in ranked:
            probabilities.append(float(ranking["probability_valid"]))
            labels.append(float(row["valid"]))
    if not evaluated_sets:
        raise ValueError("n4_ablation_arm_has_no_candidate_sets")
    seed_recall2 = [
        statistics.fmean(values) for values in recall2_by_seed.values()
    ]
    brier = statistics.fmean(
        (probability - label) ** 2
        for probability, label in zip(probabilities, labels)
    )
    return {
        "candidate_sets": evaluated_sets,
        "recall_at_1": statistics.fmean(recalls[1]),
        "recall_at_2": statistics.fmean(recalls[2]),
        "recall_at_4": statistics.fmean(recalls[4]),
        "mrr": statistics.fmean(reciprocal_ranks),
        "ndcg": statistics.fmean(ndcgs),
        "top1_mae_gain": statistics.fmean(top_gains),
        "hypotheses_until_valid": statistics.fmean(until_valid),
        "success_under_budget_2": statistics.fmean(recalls[2]),
        "top1_invariant_risk": statistics.fmean(top_risks),
        "counterfactual_cost": statistics.fmean(costs),
        "brier": brier,
        "ece": _ece(probabilities, labels),
        "recall_at_2_seed_stddev": (
            statistics.pstdev(seed_recall2) if len(seed_recall2) > 1 else 0.0
        ),
    }


def run(*, dataset: Path, artifact: Path, output: Path) -> dict[str, Any]:
    rows = _load_jsonl(dataset)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["candidate_set_id"])].append(row)
    candidate_sets = [grouped[key] for key in sorted(grouped)]
    trained = json.loads(artifact.read_bytes())
    if not trained.get("promotable"):
        raise ValueError("n4_ablation_requires_promotable_artifact")
    reference_backend = N4CausalRankingBackend()
    trained_backend = N4CausalRankingBackend(trained)
    arms = {
        "A": evaluate_arm(
            candidate_sets,
            backend=reference_backend,
            include_boundaries=False,
        ),
        "B": evaluate_arm(
            candidate_sets,
            backend=reference_backend,
            include_boundaries=True,
        ),
        "C": evaluate_arm(
            candidate_sets,
            backend=trained_backend,
            include_boundaries=True,
        ),
    }
    contrast_metrics = (
        "recall_at_1",
        "recall_at_2",
        "mrr",
        "ndcg",
        "top1_mae_gain",
        "success_under_budget_2",
    )
    contrasts = {
        name: {
            metric: arms[right][metric] - arms[left][metric]
            for metric in contrast_metrics
        }
        for name, left, right in (
            ("B-A", "A", "B"),
            ("C-B", "B", "C"),
            ("C-A", "A", "C"),
        )
    }
    payload = {
        "schema_version": "n4-ranking-ablation.v1",
        "dataset": dataset.name,
        "artifact": artifact.name,
        "arms": arms,
        "contrasts": contrasts,
        "success": bool(
            arms["C"]["mrr"] > arms["B"]["mrr"]
            and arms["C"]["success_under_budget_2"]
            >= arms["B"]["success_under_budget_2"]
            and arms["C"]["top1_invariant_risk"]
            <= arms["B"]["top1_invariant_risk"]
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    print(
        json.dumps(
            run(
                dataset=arguments.dataset,
                artifact=arguments.artifact,
                output=arguments.output,
            ),
            indent=2,
            sort_keys=True,
        )
    )
