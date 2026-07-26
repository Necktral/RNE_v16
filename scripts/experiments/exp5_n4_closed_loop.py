"""Campaña pareada MCI vs generador heurístico vs ranking N4."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from runtime.neural.hypothesis_generator import StructuralHypothesisGenerator
from runtime.symbolic.mci import ExternalHypothesis, NeuralHypothesisEvaluation
from runtime.symbolic.mci.conditions import evaluate_condition

from .common import (
    EXPERIMENT_VERSION,
    SEEDS,
    bootstrap_paired_diff,
    build_isolated_runner,
    experiment_work_root,
    write_canonical_json,
)


class HeuristicStructuralProvider:
    def __init__(self, spec, *, budget: int = 2) -> None:
        self.spec = spec
        self.budget = budget
        self.generator = StructuralHypothesisGenerator()
        self.feedback: list[NeuralHypothesisEvaluation] = []
        self.emitted: set[str] = set()

    def infer_hypotheses(
        self, context: Mapping[str, Any]
    ) -> Sequence[ExternalHypothesis]:
        candidates = self.generator.generate(
            spec=self.spec,
            evidence=tuple(context.get("recent_evidence") or ()),
        )
        proposals = []
        for item in candidates:
            if item.hypothesis_id in self.emitted:
                continue
            proposals.append(ExternalHypothesis(
                hypothesis_id=item.hypothesis_id,
                kind=item.kind,
                target_id=item.target_id,
                expression=item.expression,
                proposed_value=item.proposed_value,
                confidence=0.5,
                evidence_refs=item.evidence_refs,
                provider="structural-heuristic",
                model_ref="heuristic-v1",
            ))
            self.emitted.add(item.hypothesis_id)
            if len(proposals) >= self.budget:
                break
        return tuple(proposals)

    def receive_feedback(
        self, evaluations: Sequence[NeuralHypothesisEvaluation]
    ) -> None:
        self.feedback.extend(evaluations)


def _run_arm(
    *,
    root: Path,
    arm: str,
    seed: int,
    episodes: int,
    n4_artifact_path: Path | None,
) -> dict[str, Any]:
    profile = (
        "mci_n4_closed_loop_v1"
        if arm == "n4"
        else "mci_integrated_v1"
    )
    runner = build_isolated_runner(
        work_root=experiment_work_root(root),
        experiment=f"exp5-{arm}",
        profile=profile,
        seed=seed,
        scenario="thermal_with_battery",
        scenario_kwargs={"initial_temperature": 0.9, "initial_battery": 0.75},
        n4_artifact_path=n4_artifact_path if arm == "n4" else None,
    )
    if arm == "heuristic":
        assert runner._mci_runtime is not None
        runner.set_hypothesis_provider(
            HeuristicStructuralProvider(runner._mci_runtime.base_spec)
        )
    discovery = None
    false_promotions = 0
    violations = 0
    final_mae = None
    for episode in range(1, episodes + 1):
        result = runner.run_episode(
            external_input=0.04,
            replay_unit_id=f"exp5/{arm}/seed-{seed}/ep-{episode}",
        )
        outcome = result.get("mci_outcome") or {}
        violations += int(not bool(outcome.get("success", True)))
        overlay = outcome.get("promoted_overlay")
        if isinstance(overlay, dict):
            conditions = overlay.get("precondition_updates") or {}
            expression = str(conditions.get("thermal_battery/cooling") or "")
            structurally_correct = False
            if "battery_level" in expression:
                try:
                    structurally_correct = (
                        not evaluate_condition(expression, {"battery_level": 0.2})
                        and evaluate_condition(expression, {"battery_level": 0.4})
                    )
                except ValueError:
                    structurally_correct = False
            if structurally_correct:
                discovery = discovery or episode
            elif conditions or overlay.get("parameter_updates"):
                false_promotions += 1
            final_mae = overlay.get("holdout_mae_after")
    return {
        "seed": seed,
        "discovery_episode": discovery,
        "censored": discovery is None,
        "event_time": discovery or episodes,
        "false_promotions": false_promotions,
        "violations": violations,
        "final_holdout_mae": final_mae,
    }


def _logrank(left: Sequence[dict[str, Any]], right: Sequence[dict[str, Any]]) -> dict[str, float]:
    event_times = sorted(
        {
            int(row["event_time"])
            for row in (*left, *right)
            if not row["censored"]
        }
    )
    observed = expected = variance = 0.0
    for time in event_times:
        risk_left = sum(int(row["event_time"]) >= time for row in left)
        risk_right = sum(int(row["event_time"]) >= time for row in right)
        events_left = sum(
            int(row["event_time"]) == time and not row["censored"] for row in left
        )
        events_right = sum(
            int(row["event_time"]) == time and not row["censored"] for row in right
        )
        risk = risk_left + risk_right
        events = events_left + events_right
        if risk <= 1 or events == 0:
            continue
        observed += events_left
        expected += events * risk_left / risk
        variance += (
            risk_left
            * risk_right
            * events
            * (risk - events)
            / (risk * risk * (risk - 1))
        )
    statistic = ((observed - expected) ** 2 / variance) if variance else 0.0
    # Chi-square df=1 survival function.
    p_value = math.erfc(math.sqrt(statistic / 2.0))
    return {"chi_square": round(statistic, 9), "p_value": round(p_value, 9)}


def run(
    *,
    root: Path,
    quick: bool = False,
    n4_artifact_path: Path | None = None,
) -> dict[str, Any]:
    os.environ["RNFE_REASONING_ACTUATES"] = "1"
    seeds = SEEDS[:2] if quick else tuple(range(30))
    episodes = 16 if quick else 60
    arms = {
        arm: [
            _run_arm(
                root=root,
                arm=arm,
                seed=seed,
                episodes=episodes,
                n4_artifact_path=n4_artifact_path,
            )
            for seed in seeds
        ]
        for arm in ("mci", "heuristic", "n4")
    }
    heuristic_times = [float(row["event_time"]) for row in arms["heuristic"]]
    n4_times = [float(row["event_time"]) for row in arms["n4"]]
    acceleration = bootstrap_paired_diff(
        heuristic_times,
        n4_times,
        n_resamples=500 if quick else 10_000,
    )
    n4_discovered_before_40 = sum(
        row["discovery_episode"] is not None
        and int(row["discovery_episode"]) < 40
        for row in arms["n4"]
    ) / len(seeds)
    n4_false = sum(row["false_promotions"] for row in arms["n4"])
    n4_total = n4_false + sum(not row["censored"] for row in arms["n4"])
    fdr = n4_false / n4_total if n4_total else 0.0
    logrank = _logrank(arms["n4"], arms["heuristic"])
    payload = {
        "experiment": "exp5_n4_closed_loop",
        "version": EXPERIMENT_VERSION,
        "quick": quick,
        "backend_classification": (
            "trained" if n4_artifact_path is not None else "reference"
        ),
        "scientific_gate_eligible": n4_artifact_path is not None,
        "arms": arms,
        "metrics": {
            "n4_discovery_before_40": n4_discovered_before_40,
            "n4_false_discovery_rate": fdr,
            "paired_acceleration_bootstrap_95": acceleration,
            "logrank_n4_vs_heuristic": logrank,
        },
        "technical_gate": {
            "passed": True,
            "criterion": "closed_loop_completes_with_reference_backend",
        },
        "scientific_gate": (
            {
                "passed": bool(
                    n4_discovered_before_40 >= 0.70
                    and acceleration[0] > 0.0
                    and logrank["p_value"] < 0.05
                    and fdr <= 0.05
                ),
                "status": "evaluated_trained_backend",
            }
            if n4_artifact_path is not None
            else {
                "passed": False,
                "status": "not_applicable_reference_backend",
            }
        ),
    }
    write_canonical_json(root / "results" / "exp5_n4_closed_loop.json", payload)
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--n4-artifact", type=Path)
    arguments = parser.parse_args()
    run(
        root=arguments.root,
        quick=arguments.quick,
        n4_artifact_path=arguments.n4_artifact,
    )
