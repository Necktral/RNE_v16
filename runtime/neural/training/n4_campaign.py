"""Infraestructura reproducible de campaña N4 sin acceso implícito a holdout."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .n4_ranking import CandidateSetSample, FEATURE_NAMES, load_candidate_sets


COMPOSITE_SCHEMA = "n4-composite-training-manifest.v1"
ALLOWED_ROLES = {"historical_train", "safe_train_extension", "validation"}
SAMPLING_POLICIES = {"natural", "stratified_50_50"}


@dataclass(frozen=True)
class LoadedCompositeDataset:
    train: tuple[CandidateSetSample, ...]
    validation: tuple[CandidateSetSample, ...]
    manifest_sha256: str
    manifest: Mapping[str, Any]
    component_stats: tuple[Mapping[str, Any], ...]


def load_composite_manifest(path: Path) -> LoadedCompositeDataset:
    raw_manifest = path.read_bytes()
    manifest_sha256 = hashlib.sha256(raw_manifest).hexdigest()
    manifest = json.loads(raw_manifest)
    if manifest.get("schema") != COMPOSITE_SCHEMA:
        raise ValueError("n4_composite_manifest_schema_invalid")
    if tuple(manifest.get("feature_order", ())) != FEATURE_NAMES:
        raise ValueError("n4_composite_feature_order_invalid")
    components = manifest.get("components")
    if not isinstance(components, list) or not components:
        raise ValueError("n4_composite_manifest_components_required")
    roles = {str(item.get("role")) for item in components}
    if roles != ALLOWED_ROLES:
        raise ValueError("n4_composite_manifest_roles_invalid")
    all_seeds: set[int] = set()
    train = []
    validation = []
    stats = []
    root = path.parent
    for component in components:
        role = str(component["role"])
        expected_split = "validation" if role == "validation" else "train"
        if component.get("split") != expected_split:
            raise ValueError("n4_composite_external_split_invalid")
        if component.get("schema_version") != "n4-ranking-sample.v1":
            raise ValueError("n4_composite_component_schema_invalid")
        dataset_path = _resolved_component_path(root, component["path"])
        manifest_path = _resolved_component_path(
            root, component["manifest_path"]
        )
        dataset_raw = dataset_path.read_bytes()
        source_manifest_raw = manifest_path.read_bytes()
        _require_hash(
            dataset_raw,
            str(component["dataset_sha256"]),
            "n4_composite_dataset_hash_mismatch",
        )
        _require_hash(
            source_manifest_raw,
            str(component["manifest_sha256"]),
            "n4_composite_source_manifest_hash_mismatch",
        )
        expected_seeds = tuple(sorted(int(item) for item in component["seeds"]))
        if not expected_seeds or all_seeds.intersection(expected_seeds):
            raise ValueError("n4_composite_seed_overlap")
        all_seeds.update(expected_seeds)
        rows = [
            json.loads(line)
            for line in dataset_raw.splitlines()
            if line.strip()
        ]
        observed_seeds = tuple(sorted({int(row["seed"]) for row in rows}))
        if observed_seeds != expected_seeds:
            raise ValueError("n4_composite_component_seeds_mismatch")
        split = expected_split
        normalized = [{**row, "split": split} for row in rows]
        samples = load_candidate_sets(normalized)
        expected_available = role != "historical_train"
        expected_coverage = 1.0 if expected_available else 0.0
        if float(component.get("risk_label_coverage", -1.0)) != expected_coverage:
            raise ValueError("n4_composite_declared_risk_coverage_invalid")
        expected_version = (
            "n4-risk-label.multistep.v1" if expected_available else None
        )
        if component.get("risk_label_version") != expected_version:
            raise ValueError("n4_composite_declared_risk_version_invalid")
        if any(
            candidate.risk_label_available != expected_available
            for sample in samples
            for candidate in sample.candidates
        ):
            raise ValueError("n4_composite_risk_coverage_mismatch")
        target = validation if role == "validation" else train
        target.extend(samples)
        supervised = sum(
            candidate.risk_label_available
            for sample in samples
            for candidate in sample.candidates
        )
        sample_count = sum(len(sample.candidates) for sample in samples)
        stats.append(
            {
                "role": role,
                "dataset_sha256": str(component["dataset_sha256"]),
                "manifest_sha256": str(component["manifest_sha256"]),
                "candidate_set_count": len(samples),
                "sample_count": sample_count,
                "risk_label_coverage": round(
                    supervised / max(sample_count, 1), 9
                ),
                "seeds": list(expected_seeds),
            }
        )
    return LoadedCompositeDataset(
        train=tuple(sorted(train, key=lambda item: item.candidate_set_id)),
        validation=tuple(
            sorted(validation, key=lambda item: item.candidate_set_id)
        ),
        manifest_sha256=manifest_sha256,
        manifest=manifest,
        component_stats=tuple(stats),
    )


def sample_candidate_sets_for_epoch(
    samples: Sequence[CandidateSetSample],
    *,
    policy: str,
    model_seed: int,
    epoch: int,
) -> tuple[tuple[CandidateSetSample, ...], dict[str, Any]]:
    if policy not in SAMPLING_POLICIES:
        raise ValueError("n4_sampling_policy_invalid")
    ordered = tuple(sorted(samples, key=lambda item: item.candidate_set_id))
    historical = tuple(
        item for item in ordered if not item.candidates[0].risk_label_available
    )
    supervised = tuple(
        item for item in ordered if item.candidates[0].risk_label_available
    )
    if policy == "natural":
        selected = ordered
        repetitions = {"historical": 0, "supervised": 0}
    else:
        if not historical or not supervised:
            raise ValueError("n4_stratified_sampling_requires_both_strata")
        rng = random.Random(f"{model_seed}:{epoch}:n4-stratified-v1")
        per_stratum = max(len(historical), len(supervised))
        historical_sample = _sample_stratum(historical, per_stratum, rng)
        supervised_sample = _sample_stratum(supervised, per_stratum, rng)
        selected_list = list(historical_sample + supervised_sample)
        rng.shuffle(selected_list)
        selected = tuple(selected_list)
        repetitions = {
            "historical": per_stratum - len(historical),
            "supervised": per_stratum - len(supervised),
        }
    return selected, {
        "policy": policy,
        "epoch": epoch,
        "candidate_set_count": len(selected),
        "historical_candidate_set_count": sum(
            not item.candidates[0].risk_label_available for item in selected
        ),
        "supervised_candidate_set_count": sum(
            item.candidates[0].risk_label_available for item in selected
        ),
        "repetitions": repetitions,
    }


def select_configuration_and_median_run(
    runs: Sequence[Mapping[str, Any]],
    *,
    reference_metrics: Mapping[str, float] | None = None,
    noninferiority_margin: float = 0.01,
) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for run in runs:
        grouped.setdefault(str(run["configuration_id"]), []).append(run)
    if not grouped or any(len(items) != 3 for items in grouped.values()):
        raise ValueError("n4_selection_requires_three_runs_per_configuration")
    summaries = []
    for configuration_id, items in sorted(grouped.items()):
        if any(not bool(item.get("valid")) for item in items):
            continue
        metrics = [item["validation_metrics"] for item in items]
        mean = {
            name: sum(float(item[name]) for item in metrics) / len(metrics)
            for name in (
                "recall_at_2",
                "mrr",
                "ndcg_at_2",
                "recall_at_1",
                "top2_risk",
            )
        }
        if reference_metrics is not None and not (
            mean["recall_at_2"] > float(reference_metrics["recall_at_2"])
            and mean["recall_at_1"]
            > float(reference_metrics["recall_at_1"])
            - noninferiority_margin
            and mean["mrr"]
            > float(reference_metrics["mrr"]) - noninferiority_margin
            and mean["ndcg_at_2"]
            > float(reference_metrics["ndcg_at_2"])
            - noninferiority_margin
            and mean["top2_risk"]
            <= float(reference_metrics["top2_risk"])
        ):
            continue
        summaries.append(
            {
                "configuration_id": configuration_id,
                "mean_validation_metrics": mean,
                "runs": list(items),
            }
        )
    if not summaries:
        raise RuntimeError("n4_selection_has_no_valid_configuration")
    winner = max(
        summaries,
        key=lambda item: (
            item["mean_validation_metrics"]["recall_at_2"],
            item["mean_validation_metrics"]["mrr"],
            item["mean_validation_metrics"]["ndcg_at_2"],
            item["mean_validation_metrics"]["recall_at_1"],
            -item["mean_validation_metrics"]["top2_risk"],
            item["configuration_id"],
        ),
    )
    ordered_runs = sorted(
        winner["runs"],
        key=lambda item: (
            float(item["validation_metrics"]["recall_at_2"]),
            int(item["model_seed"]),
        ),
    )
    selected = ordered_runs[1]
    return {
        "selection_policy": "mean_configuration_then_median_recall_at_2_v1",
        "configuration_summaries": [
            {
                "configuration_id": item["configuration_id"],
                "mean_validation_metrics": item["mean_validation_metrics"],
            }
            for item in summaries
        ],
        "selected_configuration_id": winner["configuration_id"],
        "selected_model_seed": int(selected["model_seed"]),
        "selected_artifact_sha256": str(selected["artifact_sha256"]),
        "reference_metrics": (
            dict(reference_metrics) if reference_metrics is not None else None
        ),
    }


def _sample_stratum(samples, count, rng):
    shuffled = list(samples)
    rng.shuffle(shuffled)
    if count <= len(shuffled):
        return tuple(shuffled[:count])
    return tuple(
        shuffled[index % len(shuffled)] for index in range(count)
    )


def _resolved_component_path(root: Path, value: str) -> Path:
    result = (root / value).resolve()
    if not result.is_file():
        raise ValueError("n4_composite_component_missing")
    return result


def _require_hash(raw: bytes, expected: str, error: str) -> None:
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(error)
