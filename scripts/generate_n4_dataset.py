"""Genera un dataset JSONL temporal y sin leakage para el ranker N4."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
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


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_provenance() -> dict[str, Any]:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=ROOT,
            text=True,
        ).strip()
    )
    try:
        import torch

        torch_version = torch.__version__
        cuda_available = bool(torch.cuda.is_available())
        device = torch.cuda.get_device_name(0) if cuda_available else "cpu"
    except ImportError:
        torch_version = None
        cuda_available = False
        device = "cpu"
    return {
        "git_commit": commit,
        "working_tree_clean": not dirty,
        "python_version": platform.python_version(),
        "torch_version": torch_version,
        "cuda_available": cuda_available,
        "device": device,
    }


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


def _scenario_kwargs(
    scenario_name: str, rng: random.Random
) -> dict[str, float]:
    if scenario_name != "thermal_with_battery":
        return {}
    return {
        "initial_temperature": round(rng.uniform(0.86, 0.94), 9),
        "initial_battery": round(rng.uniform(0.45, 0.95), 9),
    }


def _episode_external_input(rng: random.Random) -> float:
    return round(rng.uniform(0.025, 0.055), 9)


def generate(
    *,
    scenario: str,
    seeds: tuple[int, ...],
    episodes: int,
    output: Path,
    manifest: Path,
    max_candidates: int,
    counterfactuals_per_candidate: int,
    allow_dirty: bool = False,
) -> dict[str, Any]:
    if len(seeds) < 3:
        raise ValueError("n4_dataset_requires_at_least_three_seeds")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    evidence_path = output.with_name(f"{output.stem}.evidence.jsonl")
    work_root = output.parent / ".n4-work" / f"attempt-{os.getpid()}"
    seen, counts = _existing_dataset_state(output)
    split_by_seed = _split_seeds(seeds)
    scenario_name = (
        "thermal_with_battery" if scenario == "battery_boundary" else scenario
    )
    provenance = _git_provenance()
    if not provenance["working_tree_clean"] and not allow_dirty:
        raise RuntimeError("n4_dataset_requires_clean_working_tree")
    probe_rng = random.Random(seeds[0])
    probe = build_isolated_runner(
        work_root=work_root,
        experiment="n4-dataset-probe",
        profile="mci_integrated_v1",
        seed=seeds[0],
        scenario=scenario_name,
        scenario_kwargs=_scenario_kwargs(scenario_name, probe_rng),
    )
    if probe._mci_runtime is None:
        raise RuntimeError(f"scenario_has_no_mci_runtime:{scenario_name}")
    transition_spec_hash = probe._mci_runtime.base_spec.sha256
    configuration = {
        "scenario": scenario_name,
        "seeds": list(seeds),
        "episodes_per_seed": episodes,
        "total_episodes": len(seeds) * episodes,
        "max_candidates": max_candidates,
        "counterfactuals_per_candidate": counterfactuals_per_candidate,
        "split_policy": "whole_seed_groups_60_20_20",
        "anti_leakage": "features_at_or_before_cutoff_labels_after_cutoff",
        "transition_spec_hash": transition_spec_hash,
        "trajectory_policy": (
            "seeded_initial_state_and_external_input_uniform_v1"
        ),
        "schema_version": "n4-dataset-manifest.v1",
        **provenance,
    }
    campaign_id = "n4-campaign-" + hashlib.sha256(
        _canonical(configuration)
    ).hexdigest()[:24]
    manifest_state = _prepare_manifest(
        manifest=manifest,
        configuration=configuration,
        campaign_id=campaign_id,
        output=output,
        evidence_path=evidence_path,
    )
    if manifest_state.get("status") == "complete":
        if not output.exists():
            raise ValueError("n4_complete_manifest_dataset_missing")
        actual_hash = hashlib.sha256(output.read_bytes()).hexdigest()
        if actual_hash != manifest_state.get("dataset_sha256"):
            raise ValueError("n4_complete_manifest_dataset_hash_mismatch")
        return manifest_state
    started_at = str(manifest_state["started_at"])
    with output.open("ab") as dataset_handle:
        for seed in seeds:
            rng = random.Random(seed)
            runner = build_isolated_runner(
                work_root=work_root,
                experiment="n4-dataset",
                profile="mci_integrated_v1",
                seed=seed,
                scenario=scenario_name,
                scenario_kwargs=_scenario_kwargs(scenario_name, rng),
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
                    external_input=_episode_external_input(rng),
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
        "status": "complete",
        "campaign_id": campaign_id,
        "started_at": started_at,
        "completed_at": _utc_now(),
        "configuration": configuration,
        "scenario": scenario_name,
        "seeds": list(seeds),
        "episodes_per_seed": episodes,
        "sample_count": len(seen),
        "counts": dict(sorted(counts.items())),
        "split_by_seed": {str(key): value for key, value in split_by_seed.items()},
        "dataset_path": output.name,
        "evidence_path": evidence_path.name,
        "dataset_sha256": dataset_sha256,
        "transition_spec_hash": transition_spec_hash,
        "provenance": provenance,
        "split_policy": "whole_seed_groups_60_20_20",
        "anti_leakage": "features_at_or_before_cutoff_labels_after_cutoff",
    }
    manifest.write_bytes(_canonical(payload) + b"\n")
    return payload


def _prepare_manifest(
    *,
    manifest: Path,
    configuration: dict[str, Any],
    campaign_id: str,
    output: Path,
    evidence_path: Path,
) -> dict[str, Any]:
    if manifest.exists():
        try:
            previous = json.loads(manifest.read_bytes())
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid_n4_existing_manifest") from exc
        if previous.get("campaign_id") != campaign_id:
            raise ValueError("n4_resume_configuration_mismatch")
        return previous
    started_at = _utc_now()
    planned = {
        "schema_version": "n4-dataset-manifest.v1",
        "status": "running",
        "campaign_id": campaign_id,
        "started_at": started_at,
        "configuration": configuration,
        "dataset_path": output.name,
        "evidence_path": evidence_path.name,
    }
    manifest.write_bytes(_canonical(planned) + b"\n")
    return planned


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
    parser = argparse.ArgumentParser(
        description="Generate a leakage-safe N4 dataset; episodes are per seed."
    )
    parser.add_argument("--scenario", default="battery_boundary")
    parser.add_argument(
        "--seeds",
        nargs="+",
        default=["3"],
        help="One integer means seeds range(0, N); comma/space values are explicit seeds.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=60,
        help="Episodes per seed, not total episodes.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--max-candidates", type=int, default=16)
    parser.add_argument("--counterfactuals-per-candidate", type=int, default=8)
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Allow generation from modified tracked files (not for scientific runs).",
    )
    args = parser.parse_args()
    result = generate(
        scenario=args.scenario,
        seeds=_resolve_seeds(args.seeds),
        episodes=args.episodes,
        output=args.output,
        manifest=args.manifest,
        max_candidates=args.max_candidates,
        counterfactuals_per_candidate=args.counterfactuals_per_candidate,
        allow_dirty=args.allow_dirty,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
