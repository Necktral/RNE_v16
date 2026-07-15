#!/usr/bin/env python3
"""Generate real paired family ablations for the N1 router.

Every context is executed once with ``core_only`` and once with each selected
single-family profile. Branches use empty isolated stores. The generator seed
changes physical scenario parameters; it is not a decorative label.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from time import perf_counter
from typing import Any, Iterator, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime.neural.training import (
    CounterfactualDatasetBuilder,
    counterfactual_initial_state_hash,
    train_n1_router,
)
from runtime.storage import StorageConfig, StorageFactory
from runtime.world.scenario_runner import ScenarioEpisodeRunner


FAMILY_PROFILES = {
    "HEUR": "core_plus_heur",
    "DIA_ADV": "core_plus_dialectic",
    "FAL_GUARD": "core_plus_guard",
    "IND": "core_plus_ind",
    "PLAN": "core_plus_plan",
    "OPT": "core_plus_opt",
}
GENERATORS = ("thermal_homeostasis", "resource_management", "grid_thermal_5x5")
_TOPOLOGIES = (
    "uniform",
    "hotspot_center",
    "hotspot_corner",
    "gradient_ns",
    "gradient_ew",
    "checkerboard",
    "quadrants",
)


@dataclass(frozen=True, slots=True)
class ContextSpec:
    context_key: str
    scenario_generator: str
    seed: int
    scenario_kwargs: Mapping[str, Any]
    external_input: float
    features: Mapping[str, float]


def _phase(seed: int, salt: int) -> float:
    return ((seed * salt + 97) % 1009) / 1008.0


def build_context_spec(generator: str, seed: int) -> ContextSpec:
    """Map a seed to deterministic, materially different world parameters."""
    if generator not in GENERATORS:
        raise ValueError(f"unsupported_n1_generator:{generator}")
    p1, p2, p3 = _phase(seed, 37), _phase(seed, 71), _phase(seed, 113)
    features = {
        "generator_thermal": float(generator == "thermal_homeostasis"),
        "generator_resource": float(generator == "resource_management"),
        "generator_grid": float(generator == "grid_thermal_5x5"),
        "spatial": float(generator == "grid_thermal_5x5"),
    }
    if generator == "resource_management":
        threshold = 0.18 + 0.05 * p2
        effect = 0.05 + 0.06 * p3
        cohort = seed % 4
        if cohort == 0:  # adequate and stable
            initial, external = threshold + 0.14 + 0.03 * p1, 0.02 + 0.01 * p3
        elif cohort == 1:  # adequate now, scarcity after consumption unless reasoning acts
            gap = 0.015 + 0.025 * p1
            initial, external = threshold + gap, gap + 0.035 + 0.015 * p3
        elif cohort == 2:  # already scarce; greedy controller has enough evidence
            initial, external = threshold - 0.04 - 0.02 * p1, 0.025 + 0.01 * p3
        else:  # near threshold but non-crossing
            initial, external = threshold + 0.06 + 0.02 * p1, 0.01 + 0.01 * p3
        kwargs = {
            "initial_stock": initial,
            "scarcity_threshold": threshold,
            "production_rate": effect,
        }
        direction = 1.0
        topology_code = 0.0
        initial_alarm = initial <= threshold
    else:
        threshold = 0.82 + 0.08 * p2
        effect = 0.04 + 0.07 * p3
        cohort = seed % 4
        if cohort == 0:  # cool and stable
            initial, external = threshold - 0.14 - 0.03 * p1, 0.02 + 0.01 * p3
        elif cohort == 1:  # safe now, overheats unless a reasoning family acts
            gap = 0.015 + 0.025 * p1
            initial, external = threshold - gap, gap + 0.035 + 0.015 * p3
        elif cohort == 2:  # already hot; greedy controller has enough evidence
            initial, external = threshold + 0.04 + 0.02 * p1, 0.025 + 0.01 * p3
        else:  # near threshold but non-crossing
            initial, external = threshold - 0.06 - 0.02 * p1, 0.01 + 0.01 * p3
        kwargs = {
            "initial_temperature": initial,
            "alarm_threshold": threshold,
            "cooling_effect": effect,
        }
        direction = -1.0
        topology_code = 0.0
        if generator == "grid_thermal_5x5":
            topology_index = seed % len(_TOPOLOGIES)
            kwargs["topology"] = _TOPOLOGIES[topology_index]
            topology_code = topology_index / (len(_TOPOLOGIES) - 1)
        initial_alarm = initial >= threshold
    features.update(
        initial_value=initial,
        alarm_threshold=threshold,
        actuation_effect=effect,
        external_input=external,
        optimization_direction=direction,
        distance_to_threshold=abs(initial - threshold),
        initial_alarm=float(initial_alarm),
        topology_code=topology_code,
    )
    material = json.dumps(
        {
            "generator": generator,
            "seed": seed,
            "scenario_kwargs": kwargs,
            "external_input": external,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    key = hashlib.sha256(material.encode("utf-8")).hexdigest()
    return ContextSpec(key, generator, seed, kwargs, external, features)


@contextmanager
def _runtime_environment(profile: str) -> Iterator[None]:
    values = {
        "RNFE_NEURAL_MODE": "off",
        "RNFE_REASONING_FAMILY_PROFILE": profile,
        "RNFE_REASONING_ACTUATES": "1",
        "RNFE_REASONING_MAX_STEPS": "10",
        "RNFE_EXPERIENCE": "0",
        "RNFE_EXTERNAL_REASONER_RUNTIME": "0",
    }
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _run_branch(spec: ContextSpec, profile: str, db_path: Path, artifact_root: Path) -> dict[str, Any]:
    config = StorageConfig(
        mode="sqlite",
        sqlite_db_path=str(db_path),
        postgres_dsn=None,
        artifact_root=artifact_root,
        prefer_postgres_reads=False,
        strict_dual_write=False,
    )
    storage = StorageFactory.create_facade(config)
    started = perf_counter()
    try:
        with _runtime_environment(profile):
            result = ScenarioEpisodeRunner(
                storage=storage,
                run_id=f"n1-{spec.scenario_generator}-{spec.seed}-{profile}",
                scenario=spec.scenario_generator,
                scenario_kwargs=dict(spec.scenario_kwargs),
                closure_profile="adaptive_min",
            ).run_episode(external_input=spec.external_input)
    finally:
        storage.close()
    elapsed_ms = (perf_counter() - started) * 1000.0
    transition = result["life_transition"]
    before, after = transition["state_before"], transition["state_after"]
    closure = after["organism"]["closure"].get("value")
    certification = result.get("certification") or {}
    constitutional = result.get("constitutional_validation") or {}
    return {
        "initial_state_hash": counterfactual_initial_state_hash(before),
        "reward": float(result["reasoning_reward"]["reward"]),
        "effectiveness": float(result["reasoning_reward"]["effectiveness"]),
        "closure": float(closure == "certified"),
        "certified": float(certification.get("verdict") == "certified"),
        "continuity": float(after["organism"]["continuity"]["value"]),
        "viability": float(after["organism"]["viability"]["value"]),
        "latency_ms": elapsed_ms,
        "hard_violations": int(constitutional.get("hard_violation_count") or 0),
        "decision_verdict": certification.get("decision_verdict"),
        "active_action": after["organism"].get("active_action"),
    }


def _record(spec: ContextSpec, family: str, enabled: bool, outcome: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "context_key": spec.context_key,
        "scenario_generator": spec.scenario_generator,
        "seed": spec.seed,
        "family": family,
        "family_enabled": enabled,
        "initial_state_hash": outcome["initial_state_hash"],
        "features": dict(spec.features),
        "reward": outcome["reward"],
        "effectiveness": outcome["effectiveness"],
        "closure": outcome["closure"],
        "certified": outcome["certified"],
        "continuity": outcome["continuity"],
        "viability": outcome["viability"],
        "observed": {
            key: outcome[key]
            for key in (
                "latency_ms",
                "hard_violations",
                "decision_verdict",
                "active_action",
            )
        },
    }


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def _heldout_calibration_gate(evidence: Mapping[str, Any] | None) -> bool:
    if not evidence or evidence.get("heldout_evaluated") is not True:
        return False
    metrics = dict(evidence.get("split_metrics") or {})
    try:
        return all(
            bool((metrics.get(name) or {}).get("evaluated"))
            and float((metrics.get(name) or {})["calibration_ece"]) <= 0.10
            for name in ("validation", "test")
        )
    except (KeyError, TypeError, ValueError):
        return False


def _write_evidence_manifest(root: Path) -> None:
    names = (
        "paired_records.jsonl",
        "counterfactual_samples.jsonl",
        "dataset_quality.json",
        "split_summary.json",
        "initial_state_audit.json",
        "verdict.json",
        "artifacts/n1/router-lab-v1.json",
        "artifacts/n1/manifest.json",
        "artifacts/n1/model_card.json",
    )
    files = {}
    for name in names:
        path = root / name
        if path.is_file():
            files[name] = {
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
    _write_json(root / "evidence_manifest.json", {
        "schema": "n1-counterfactual-evidence-manifest-v1",
        "files": files,
    })


def run_n1_counterfactual_campaign(
    *,
    output_dir: str | Path,
    contexts_per_generator: int = 20,
    seed_base: int = 731_000,
    families: Sequence[str] = tuple(FAMILY_PROFILES),
    train: bool = False,
    epochs: int = 80,
) -> dict[str, Any]:
    if contexts_per_generator <= 0:
        raise ValueError("contexts_per_generator_must_be_positive")
    selected = tuple(str(name).upper() for name in families)
    if len(set(selected)) != len(selected) or any(name not in FAMILY_PROFILES for name in selected):
        raise ValueError("n1_families_must_be_unique_supported_single_family_profiles")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    hash_mismatches: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="rnfe-n1-pairs-") as scratch_name:
        scratch = Path(scratch_name)
        branch_index = 0
        for generator_index, generator in enumerate(GENERATORS):
            for index in range(contexts_per_generator):
                seed = seed_base + generator_index * 100_000 + index
                spec = build_context_spec(generator, seed)
                baseline = _run_branch(
                    spec,
                    "core_only",
                    scratch / f"branch-{branch_index:05d}.sqlite3",
                    scratch / "artifacts",
                )
                branch_index += 1
                for family in selected:
                    candidate = _run_branch(
                        spec,
                        FAMILY_PROFILES[family],
                        scratch / f"branch-{branch_index:05d}.sqlite3",
                        scratch / "artifacts",
                    )
                    branch_index += 1
                    if baseline["initial_state_hash"] != candidate["initial_state_hash"]:
                        hash_mismatches.append(
                            {"context_key": spec.context_key, "family": family, "seed": seed}
                        )
                    records.append(_record(spec, family, False, baseline))
                    records.append(_record(spec, family, True, candidate))

    builder = CounterfactualDatasetBuilder()
    samples, quality = builder.build(records)
    splits = builder.split(samples)
    split_summary = {
        name: {
            "samples": len(rows),
            "contexts": len({row.context_key for row in rows}),
            "groups": len({(row.scenario_generator, row.seed) for row in rows}),
        }
        for name, rows in splits.items()
    }
    data_ready = quality.training_ready() and all(splits.values()) and not hash_mismatches
    training_evidence = None
    manifest = None
    if train and data_ready:
        manifest, training_evidence = train_n1_router(
            splits["train"],
            quality,
            artifact_root=root / "artifacts",
            epochs=epochs,
            dataset_classification="runtime_paired_family_ablation_v1",
            validation_samples=splits["validation"],
            test_samples=splits["test"],
        )

    _write_jsonl(root / "paired_records.jsonl", records)
    _write_jsonl(root / "counterfactual_samples.jsonl", [asdict(sample) for sample in samples])
    _write_json(root / "dataset_quality.json", asdict(quality))
    _write_json(root / "split_summary.json", split_summary)
    _write_json(root / "initial_state_audit.json", {
        "schema": "n1-initial-state-audit-v1",
        "hash_schema": "n1-counterfactual-initial-state-v1",
        "pairs_checked": len(samples),
        "mismatches": hash_mismatches,
    })
    calibration_gate_passed = _heldout_calibration_gate(training_evidence)
    verdict = {
        "schema": "n1-counterfactual-campaign-verdict-v1",
        "classification": "experimental_shadow_only",
        "data_training_ready": data_ready,
        "training_requested": train,
        "training_completed": training_evidence is not None,
        "heldout_calibration_gate_passed": calibration_gate_passed,
        "promotion_authorized": False,
        "reason": (
            "heldout_calibration_gate_failed"
            if training_evidence is not None and not calibration_gate_passed
            else "artifact_requires_scheduler_shadow_comparison_and_a_m0_impact_report"
            if training_evidence is not None
            else "dataset_class_diversity_or_quality_gate_failed"
            if train and not data_ready
            else "dataset_ready_training_not_requested"
            if data_ready
            else "dataset_quality_or_split_gate_failed"
        ),
        "quality": asdict(quality),
        "splits": split_summary,
        "hash_mismatches": len(hash_mismatches),
        "training_evidence": training_evidence,
        "manifest": manifest.to_dict() if manifest is not None else None,
    }
    _write_json(root / "verdict.json", verdict)
    _write_evidence_manifest(root)
    return {"output_dir": str(root.resolve()), **verdict}


def reconcile_existing_campaign(output_dir: str | Path) -> dict[str, Any]:
    """Reapply current dataset gates without rerunning physical episodes."""
    root = Path(output_dir)
    records_path = root / "paired_records.jsonl"
    if not records_path.is_file():
        raise ValueError("n1_existing_campaign_records_missing")
    records = [
        json.loads(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    samples, quality = CounterfactualDatasetBuilder().build(records)
    splits = CounterfactualDatasetBuilder().split(samples)
    split_summary = {
        name: {
            "samples": len(rows),
            "contexts": len({row.context_key for row in rows}),
            "groups": len({(row.scenario_generator, row.seed) for row in rows}),
        }
        for name, rows in splits.items()
    }
    audit_path = root / "initial_state_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.is_file() else {}
    mismatches = list(audit.get("mismatches") or [])
    data_ready = quality.training_ready() and all(splits.values()) and not mismatches
    artifact_path = root / "artifacts" / "n1" / "router-lab-v1.json"
    previous_path = root / "verdict.json"
    previous = json.loads(previous_path.read_text(encoding="utf-8")) if previous_path.is_file() else {}
    _write_jsonl(root / "counterfactual_samples.jsonl", [asdict(sample) for sample in samples])
    _write_json(root / "dataset_quality.json", asdict(quality))
    _write_json(root / "split_summary.json", split_summary)
    previous_evidence = previous.get("training_evidence")
    training_completed = bool(
        data_ready
        and previous.get("training_completed") is True
        and isinstance(previous_evidence, Mapping)
        and artifact_path.is_file()
    )
    calibration_gate_passed = _heldout_calibration_gate(
        previous_evidence if training_completed else None
    )
    quarantined = bool(artifact_path.is_file() and (not data_ready or not calibration_gate_passed))
    verdict = {
        "schema": "n1-counterfactual-campaign-verdict-v1",
        "classification": "experimental_shadow_only",
        "evidence_basis": "reconciled_paired_records",
        "data_training_ready": data_ready,
        "training_requested": bool(previous.get("training_requested")),
        "training_completed": training_completed,
        "heldout_calibration_gate_passed": calibration_gate_passed,
        "promotion_authorized": False,
        "reason": (
            "dataset_class_diversity_or_quality_gate_failed"
            if not data_ready
            else "dataset_ready_requires_fresh_training_under_current_gate"
            if not training_completed
            else "heldout_calibration_gate_failed"
            if not calibration_gate_passed
            else "artifact_requires_scheduler_shadow_comparison_and_a_m0_impact_report"
        ),
        "quality": asdict(quality),
        "splits": split_summary,
        "hash_mismatches": len(mismatches),
        "training_evidence": previous_evidence if training_completed else None,
        "manifest": previous.get("manifest") if training_completed else None,
        "quarantined_artifact": str(artifact_path) if quarantined else None,
    }
    _write_json(root / "verdict.json", verdict)
    _write_evidence_manifest(root)
    return {"output_dir": str(root.resolve()), **verdict}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--contexts-per-generator", type=int, default=20)
    parser.add_argument("--seed-base", type=int, default=731_000)
    parser.add_argument("--families", nargs="+", default=list(FAMILY_PROFILES))
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--reconcile-existing", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.reconcile_existing:
        result = reconcile_existing_campaign(args.output_dir)
    else:
        result = run_n1_counterfactual_campaign(
            output_dir=args.output_dir,
            contexts_per_generator=args.contexts_per_generator,
            seed_base=args.seed_base,
            families=args.families,
            train=args.train,
            epochs=args.epochs,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
