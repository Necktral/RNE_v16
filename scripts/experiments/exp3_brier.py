from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import Any

from .common import (
    EXPERIMENT_VERSION,
    build_isolated_runner,
    experiment_work_root,
    write_canonical_json,
)


def run(*, root: Path, quick: bool = False) -> dict[str, Any]:
    os.environ["RNFE_REASONING_ACTUATES"] = "1"
    episodes = 20 if quick else 200
    runner = build_isolated_runner(
        work_root=experiment_work_root(root),
        experiment="exp3",
        profile="mci_integrated_v1",
        seed=42,
        scenario="thermal_homeostasis",
        scenario_kwargs={"initial_temperature": 0.9},
    )
    rng = random.Random(42)
    scored: list[tuple[float, bool]] = []
    abstentions = 0
    for episode in range(1, episodes + 1):
        result = runner.run_episode(
            external_input=rng.uniform(0.15, 0.25),
            replay_unit_id=f"exp3/mci/ep-{episode}",
        )
        model = (result.get("reasoning") or {}).get("state", {}).get(
            "mci_self_model"
        ) or {}
        outcome = result.get("mci_outcome") or {}
        if model.get("decision") == "abstain":
            abstentions += 1
        if outcome.get("recommendation_committed") is True:
            scored.append(
                (
                    float(model["success_probability"]),
                    bool(outcome["success"]),
                )
            )
    brier = (
        sum((probability - float(success)) ** 2 for probability, success in scored)
        / len(scored)
        if scored
        else None
    )
    passed = brier is not None and brier < 0.20 and abstentions > 0
    payload = {
        "experiment": "exp3_brier",
        "version": EXPERIMENT_VERSION,
        "quick": quick,
        "episodes": episodes,
        "scored_outcomes": len(scored),
        "brier_score": brier,
        "abstentions": abstentions,
        "gate": {
            "criterion": "brier_lt_0.20_and_abstentions_gt_zero",
            "passed": passed,
        },
    }
    write_canonical_json(root / "results" / "exp3_brier.json", payload)
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    run(root=args.root, quick=args.quick)
