from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from runtime.world import ThermalBatteryScenario

from .common import (
    EXPERIMENT_VERSION,
    build_isolated_runner,
    experiment_work_root,
    write_canonical_json,
)


def run(*, root: Path, quick: bool = False) -> dict[str, Any]:
    os.environ["RNFE_REASONING_ACTUATES"] = "1"
    phase_length = 8 if quick else 30
    runner = build_isolated_runner(
        work_root=experiment_work_root(root),
        experiment="exp4",
        profile="mci_integrated_v1",
        seed=42,
        scenario="thermal_with_battery",
        scenario_kwargs={
            "initial_temperature": 0.9,
            "initial_battery": 1.0,
            "cooling_effect": 0.07,
            "battery_discharge_rate": 0.0,
            "battery_charge_rate": 0.0,
        },
    )
    scenario = runner.scenario
    if not isinstance(scenario, ThermalBatteryScenario):
        raise RuntimeError("El experimento requiere ThermalBatteryScenario")
    events: list[dict[str, Any]] = []
    adaptation_episode = None
    for episode in range(1, phase_length * 2 + 1):
        if episode == phase_length + 1:
            scenario.set_regime(0.14)
        result = runner.run_episode(
            external_input=0.10,
            replay_unit_id=f"exp4/mci/ep-{episode}",
        )
        outcome = result.get("mci_outcome") or {}
        overlay = outcome.get("promoted_overlay")
        event = {
            "episode": episode,
            "prediction_error": outcome.get("prediction_error"),
            "regime_change": outcome.get("regime_change"),
            "belief_revision": outcome.get("belief_revision"),
            "promoted_overlay": overlay,
        }
        events.append(event)
        updates = overlay.get("parameter_updates", {}) if isinstance(overlay, dict) else {}
        if (
            episode > phase_length
            and abs(float(updates.get("cooling_effect", 0.0)) - 0.14) <= 0.02
            and adaptation_episode is None
        ):
            adaptation_episode = episode
    latency = (
        adaptation_episode - phase_length
        if adaptation_episode is not None
        else None
    )
    detected = any(
        item["regime_change"] is not None for item in events[phase_length:]
    )
    revised = any(
        bool((item["belief_revision"] or {}).get("revisable"))
        or bool((item["belief_revision"] or {}).get("out"))
        for item in events[phase_length:]
    )
    payload = {
        "experiment": "exp4_regime",
        "version": EXPERIMENT_VERSION,
        "quick": quick,
        "phase_length": phase_length,
        "events": events,
        "regime_detected": detected,
        "old_belief_revised": revised,
        "adaptation_episode": adaptation_episode,
        "adaptation_latency": latency,
        "gate": {
            "criterion": "detected_revised_and_adapted_within_15",
            "passed": bool(
                detected and revised and latency is not None and latency <= 15
            ),
        },
    }
    write_canonical_json(root / "results" / "exp4_regime.json", payload)
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    run(root=args.root, quick=args.quick)
