from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from runtime.symbolic.mci import MCIPlanningConfig

from .common import (
    EXPERIMENT_VERSION,
    SEEDS,
    bootstrap_paired_diff,
    build_isolated_runner,
    experiment_work_root,
    generate_external_inputs,
    observed_outcome,
    run_paired_campaign,
    write_canonical_json,
)


def run(*, root: Path, quick: bool = False) -> dict[str, Any]:
    os.environ["RNFE_REASONING_ACTUATES"] = "1"
    seeds = SEEDS[:2] if quick else SEEDS
    episodes = 5 if quick else 50

    def factory(*, profile: str, seed: int):
        planning_config = (
            MCIPlanningConfig(
                horizon=3,
                exact_horizon=True,
                objective_mode="trajectory_loss",
            )
            if profile == "mci_integrated_v1"
            else None
        )
        return build_isolated_runner(
            work_root=experiment_work_root(root),
            experiment="exp1",
            profile=profile,
            seed=seed,
            scenario="deferred_load_trap",
            mci_planning_config=planning_config,
        )

    def utility(result: dict[str, Any]) -> float:
        outcome = observed_outcome(result)
        load = float(outcome["load"])
        alarm = bool(outcome.get("alarm", load >= 0.85))
        return (0.85 - load) - float(alarm)

    campaign = run_paired_campaign(
        factory,
        seeds=seeds,
        n_episodes_per_seed=episodes,
        external_input_gen=generate_external_inputs,
        metric=utility,
        replay_prefix="exp1",
    )
    mci = [
        float(item["mean"])
        for item in campaign["arms"]["mci_integrated_v1"]
    ]
    baseline = [
        float(item["mean"]) for item in campaign["arms"]["core_plus_opt"]
    ]
    interval = bootstrap_paired_diff(
        mci,
        baseline,
        n_resamples=500 if quick else 10_000,
    )
    payload = {
        "experiment": "exp1_regret",
        "version": EXPERIMENT_VERSION,
        "quick": quick,
        "campaign": campaign,
        "paired_mean_difference": sum(a - b for a, b in zip(mci, baseline))
        / len(mci),
        "bootstrap_95": interval,
        "gate": {"criterion": "bootstrap_lower_gt_zero", "passed": interval[0] > 0.0},
    }
    write_canonical_json(root / "results" / "exp1_regret.json", payload)
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    run(root=args.root, quick=args.quick)
