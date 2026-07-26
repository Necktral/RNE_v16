from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from runtime.symbolic.mci import ExternalHypothesis

from .common import (
    EXPERIMENT_VERSION,
    build_isolated_runner,
    experiment_work_root,
    write_canonical_json,
)
from .synthetic_provider import ScheduledHypothesisProvider


def _run_arm(*, root: Path, assisted: bool, episodes: int) -> dict[str, Any]:
    arm = "assisted" if assisted else "autonomous"
    runner = build_isolated_runner(
        work_root=experiment_work_root(root),
        experiment=f"exp2-{arm}",
        profile="mci_integrated_v1",
        seed=42,
        scenario="thermal_with_battery",
        scenario_kwargs={
            "initial_temperature": 0.9,
            "initial_battery": 0.75,
        },
    )
    provider = None
    if assisted:
        proposal = ExternalHypothesis(
            kind="precondition",
            target_id="thermal_battery/cooling",
            expression="battery_level > 0.3",
            confidence=0.9,
            provider="synthetic",
            model_ref="thermal-battery-v1",
        )
        provider = ScheduledHypothesisProvider({20: (proposal,)})
        runner.set_hypothesis_provider(provider)
    promotions: list[dict[str, Any]] = []
    for episode in range(1, episodes + 1):
        result = runner.run_episode(
            external_input=0.04,
            replay_unit_id=f"exp2/{arm}/ep-{episode}",
        )
        overlay = (result.get("mci_outcome") or {}).get("promoted_overlay")
        if isinstance(overlay, dict):
            promotions.append({"episode": episode, "overlay": overlay})
    structural = [
        item
        for item in promotions
        if "thermal_battery/cooling"
        in (item["overlay"].get("precondition_updates") or {})
    ]
    return {
        "arm": arm,
        "promotions": promotions,
        "structural_promotion_episode": (
            structural[0]["episode"] if structural else None
        ),
        "feedback_count": len(provider.feedback) if provider else 0,
        "feedback": (
            [evaluation.to_dict() for evaluation in provider.feedback]
            if provider
            else []
        ),
        "passed": bool(structural and structural[0]["episode"] < 50),
    }


def run(*, root: Path, quick: bool = False) -> dict[str, Any]:
    os.environ["RNFE_REASONING_ACTUATES"] = "1"
    episodes = 24 if quick else 60
    autonomous = _run_arm(root=root, assisted=False, episodes=episodes)
    assisted = _run_arm(root=root, assisted=True, episodes=episodes)
    payload = {
        "experiment": "exp2_learning",
        "version": EXPERIMENT_VERSION,
        "quick": quick,
        "autonomous": autonomous,
        "assisted": assisted,
        "gate": {
            "criterion": "structural_promotion_before_episode_50",
            "autonomous_passed": autonomous["passed"],
            "assisted_passed": assisted["passed"],
        },
    }
    write_canonical_json(root / "results" / "exp2_learning.json", payload)
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    run(root=args.root, quick=args.quick)
