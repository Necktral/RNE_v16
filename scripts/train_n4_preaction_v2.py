#!/usr/bin/env python3
"""Train the P1 N4-v2 artifact on 24/6/12 complete sealed trajectories."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime.neural.campaign import atomic_write_json, file_sha256
from runtime.neural.integration.p1_n4 import (
    MANIFEST_SCHEMA_VERSION,
    N4PreactionInterventionSet,
    PREACTION_BACKEND,
    causal_signature_prior_evidence,
    preaction_feature_row,
)
from runtime.neural.training.n4_preaction_v2 import (
    N4PreactionTrainingRecord,
    train_n4_preaction_v2,
)
from runtime.world.registry import get_scenario


SCENARIOS = (
    "thermal_homeostasis",
    "resource_management",
    "grid_thermal_5x5",
    "deferred_load_trap",
)
TRAJECTORY_SEEDS = tuple(731_003 + index * 10_007 for index in range(42))


def _sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _input(seed: int, step: int) -> float:
    digest = hashlib.sha256(f"n4-v2:{seed}:{step}".encode()).digest()
    return 0.035 + int.from_bytes(digest[:2], "big") / 65535.0 * 0.11


def build_training_records() -> tuple[N4PreactionTrainingRecord, ...]:
    records: list[N4PreactionTrainingRecord] = []
    for seed_index, seed in enumerate(TRAJECTORY_SEEDS):
        split = "train" if seed_index < 24 else "validation" if seed_index < 30 else "evaluation"
        trajectory_id = f"n4-v2-trajectory-{seed}"
        global_step = 0
        for scenario_name in SCENARIOS:
            scenario = get_scenario(scenario_name)
            history: dict[str, list[float]] = {
                action: [] for action in scenario.config.interventions
            }
            previous_value: float | None = None
            for scenario_step in range(8):
                observation = scenario.observe()
                observation_dict = scenario.to_observation_dict(observation)
                main_variable = scenario.config.main_variable
                observed = float(observation_dict[main_variable])
                direction = str(scenario.causal_signature.optimization_direction)
                trend = observed - previous_value if previous_value is not None else 0.0
                threshold = abs(float(scenario.config.alarm_threshold or 0.0)) or max(abs(observed), 1.0)
                adverse = max(trend, 0.0) if direction == "minimize" else max(-trend, 0.0)
                importance = min(abs(trend) / threshold, 1.0)
                n3 = {
                    "trend": trend,
                    "uncertainty": 1.0 / (scenario_step + 1.0),
                    "risk": min(adverse / threshold, 1.0),
                    "importance": importance,
                    "continuity": 1.0 - importance,
                    "evidence_ref": f"training-lag:{trajectory_id}:{global_step}",
                }
                lagged = {
                    action: {
                        "mean_delta": sum(values) / len(values),
                        "sample_count": len(values),
                        "confidence": len(values) / (len(values) + 4.0),
                        "evidence_ref": f"training-history:{trajectory_id}:{action}",
                    }
                    for action, values in history.items()
                    if values
                }
                request = N4PreactionInterventionSet(
                    scenario_id=scenario_name,
                    main_variable=main_variable,
                    optimization_direction=direction,
                    observation=observation_dict,
                    interventions=tuple(scenario.config.interventions),
                    canonical_intervention=scenario.select_intervention(observation),
                    prior_evidence=causal_signature_prior_evidence(
                        scenario.causal_signature,
                        interventions=scenario.config.interventions,
                    ),
                    lagged_evidence=lagged,
                    n3_signals=n3,
                )
                external_input = _input(seed, global_step)
                for action in scenario.config.interventions:
                    transition = scenario.simulate_counterfactual(
                        intervention=action, external_input=external_input
                    )
                    target_delta = float(transition.state[main_variable]) - observed
                    state_hash = _sha(
                        {
                            "trajectory_id": trajectory_id,
                            "scenario": scenario_name,
                            "step": scenario_step,
                            "request_hash": request.input_hash,
                        }
                    )
                    records.append(
                        N4PreactionTrainingRecord(
                            trajectory_id=trajectory_id,
                            split=split,
                            state_hash=state_hash,
                            scenario_id=scenario_name,
                            intervention=action,
                            features=preaction_feature_row(request, action),
                            target_delta=target_delta,
                        )
                    )
                    history[action].append(target_delta)
                committed = request.canonical_intervention
                scenario.factual_transition(
                    intervention=committed, external_input=external_input
                )
                previous_value = observed
                global_step += 1
    return tuple(records)


def train_to_directory(output_root: Path) -> dict[str, Any]:
    records = build_training_records()
    artifact = train_n4_preaction_v2(records)
    model_dir = output_root / "n4_preaction_v2"
    artifact_path = model_dir / "artifact.json"
    manifest_path = model_dir / "manifest.json"
    atomic_write_json(artifact_path, artifact)
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "organ": "N4",
        "capability": "preaction_intervention_scoring",
        "backend": PREACTION_BACKEND,
        "model_id": artifact["model_id"],
        "artifact_path": "artifact.json",
        "artifact_sha256": file_sha256(artifact_path),
        "input_schema_version": "n4-preaction-intervention-set-v1",
        "output_schema_version": "n4-intervention-score-set-v1",
        "authority_effect": "none",
        "promotion_authorized": False,
        "dataset": {
            "trajectory_seeds_sha256": _sha(list(TRAJECTORY_SEEDS)),
            "record_count": len(records),
            "steps_per_trajectory": 32,
            "trajectory_counts": {"train": 24, "validation": 6, "evaluation": 12},
            "evaluation_opened": False,
        },
    }
    atomic_write_json(manifest_path, manifest)
    return {"manifest_path": str(manifest_path), "manifest": manifest}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(train_to_directory(args.output_root.resolve()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
