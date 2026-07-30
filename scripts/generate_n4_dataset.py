"""Genera un dataset JSONL temporal y sin leakage para el ranker N4."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime.neural.hypothesis_generator import StructuralHypothesisGenerator
from runtime.neural.training import (
    EvidenceJSONLCollector,
    label_candidate_counterfactually,
)
from scripts.experiments.common import build_isolated_runner


def _canonical(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _resolve_seeds(values: list[str]) -> tuple[int, ...]:
    tokens = [item for value in values for item in value.split(",") if item]
    if len(tokens) == 1 and tokens[0].isdigit():
        return tuple(range(int(tokens[0])))
    return tuple(sorted({int(item) for item in tokens}))


def _existing_dataset_state(path: Path) -> tuple[set[str], Counter[str]]:
    if not path.exists():
        return set(), Counter()
    result = set()
    counts: Counter[str] = Counter()
    with path.open("rb") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                result.add(str(row["sample_id"]))
                counts[f"split:{row['split']}"] += 1
                counts[f"source:{row['source']}"] += 1
                counts[f"seed:{int(row['seed'])}"] += 1
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid_n4_dataset_line:{number}") from exc
    return result, counts


def generate(
    *,
    scenario: str,
    seeds: tuple[int, ...],
    episodes: int,
    output: Path,
    manifest: Path,
    max_candidates: int,
    counterfactuals_per_candidate: int,
) -> dict[str, Any]:
    if len(seeds) < 3:
        raise ValueError("n4_dataset_requires_at_least_three_seeds")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    evidence_path = output.with_name(f"{output.stem}.evidence.jsonl")
    work_root = output.parent / ".n4-work"
    seen, counts = _existing_dataset_state(output)
    split_by_seed = _split_seeds(seeds)
    scenario_name = (
        "thermal_with_battery" if scenario == "battery_boundary" else scenario
    )
    with output.open("ab") as dataset_handle:
        for seed in seeds:
            runner = build_isolated_runner(
                work_root=work_root,
                experiment="n4-dataset",
                profile="mci_integrated_v1",
                seed=seed,
                scenario=scenario_name,
                scenario_kwargs=(
                    {"initial_temperature": 0.9, "initial_battery": 0.75}
                    if scenario_name == "thermal_with_battery"
                    else {}
                ),
            )
            if runner._mci_runtime is None:
                raise RuntimeError(f"scenario_has_no_mci_runtime:{scenario_name}")
            learner = runner._mci_runtime.learner
            spec = runner._mci_runtime.base_spec
            rows = []
            collector = EvidenceJSONLCollector(
                evidence_path,
                run_id=runner.run_id,
                scenario_id=scenario_name,
                seed=seed,
                transition_spec_hash=spec.sha256,
            )
            learner.add_evidence_observer(rows.append)
            learner.add_evidence_observer(collector)
            for episode in range(1, episodes + 1):
                collector.set_context(episode=episode)
                runner.run_episode(
                    external_input=0.04,
                    replay_unit_id=f"n4-dataset/{scenario_name}/{seed}/{episode}",
                )
            generator = StructuralHypothesisGenerator(
                beam_width=max_candidates
            )
            for cutoff in range(8, len(rows) - 1):
                feature_rows = tuple(rows[: cutoff + 1])
                future = tuple(
                    rows[
                        cutoff + 1 : cutoff + 1 + counterfactuals_per_candidate
                    ]
                )
                if not future:
                    continue
                for candidate in generator.generate(
                    spec=spec, evidence=feature_rows
                )[:max_candidates]:
                    label = label_candidate_counterfactually(
                        spec=spec,
                        candidate=candidate,
                        feature_evidence=feature_rows,
                        label_evidence=future,
                    )
                    candidate_set_id = "n4-set-" + hashlib.sha256(
                        (
                            f"{scenario_name}|{seed}|"
                            f"{label.candidate_set_id}"
                        ).encode()
                    ).hexdigest()[:24]
                    sample_id = "n4-sample-" + hashlib.sha256(
                        (
                            f"{candidate_set_id}|"
                            f"{candidate.hypothesis_id}"
                        ).encode()
                    ).hexdigest()[:24]
                    if sample_id in seen:
                        continue
                    record = {
                        "schema_version": "n4-ranking-sample.v1",
                        "sample_id": sample_id,
                        "candidate_set_id": candidate_set_id,
                        "hypothesis_id": candidate.hypothesis_id,
                        "scenario": scenario_name,
                        "seed": seed,
                        "split": split_by_seed[seed],
                        "logical_time": label.cutoff_logical_time,
                        "source": candidate.source,
                        "kind": candidate.kind,
                        "target_id": candidate.target_id,
                        "expression": candidate.expression,
                        "proposed_value": candidate.proposed_value,
                        "features": dict(candidate.features),
                        "valid": label.valid,
                        "mae_gain": label.mae_gain,
                        "invariant_risk": label.invariant_risk,
                        "evaluation_count": label.evaluation_count,
                    }
                    dataset_handle.write(_canonical(record) + b"\n")
                    dataset_handle.flush()
                    seen.add(sample_id)
                    counts[f"split:{split_by_seed[seed]}"] += 1
                    counts[f"source:{candidate.source}"] += 1
                    counts[f"seed:{seed}"] += 1
    dataset_sha256 = hashlib.sha256(output.read_bytes()).hexdigest()
    payload = {
        "schema_version": "n4-dataset-manifest.v1",
        "scenario": scenario_name,
        "seeds": list(seeds),
        "episodes_per_seed": episodes,
        "sample_count": len(seen),
        "counts": dict(sorted(counts.items())),
        "split_by_seed": {str(key): value for key, value in split_by_seed.items()},
        "dataset_path": output.name,
        "evidence_path": evidence_path.name,
        "dataset_sha256": dataset_sha256,
        "anti_leakage": "features_at_or_before_cutoff_labels_after_cutoff",
    }
    manifest.write_bytes(_canonical(payload) + b"\n")
    return payload


def _split_seeds(seeds: tuple[int, ...]) -> dict[int, str]:
    first = max(1, int(len(seeds) * 0.60))
    second = max(first + 1, int(len(seeds) * 0.80))
    second = min(second, len(seeds) - 1)
    return {
        seed: (
            "train"
            if index < first
            else ("validation" if index < second else "holdout")
        )
        for index, seed in enumerate(seeds)
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="battery_boundary")
    parser.add_argument("--seeds", nargs="+", default=["3"])
    parser.add_argument("--episodes", type=int, default=60)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--max-candidates", type=int, default=16)
    parser.add_argument("--counterfactuals-per-candidate", type=int, default=8)
    args = parser.parse_args()
    result = generate(
        scenario=args.scenario,
        seeds=_resolve_seeds(args.seeds),
        episodes=args.episodes,
        output=args.output,
        manifest=args.manifest,
        max_candidates=args.max_candidates,
        counterfactuals_per_candidate=args.counterfactuals_per_candidate,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
