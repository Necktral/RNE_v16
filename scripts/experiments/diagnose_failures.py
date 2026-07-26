"""Captura evidencia causal de los experimentos que no superaron sus gates."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from runtime.symbolic.mci import ExternalHypothesis
from runtime.symbolic.mci.causal_learning import TransitionEvidence
from runtime.world import ThermalBatteryScenario

from .common import (
    build_isolated_runner,
    experiment_work_root,
    generate_external_inputs,
    write_canonical_json,
)
from .synthetic_provider import ScheduledHypothesisProvider


def _evidence_dict(evidence: TransitionEvidence) -> dict[str, Any]:
    return {
        "evidence_id": evidence.evidence_id,
        "replay_unit_id": evidence.replay_unit_id,
        "logical_time": evidence.logical_time,
        "decision_trace_sha256": evidence.decision_trace_sha256,
        "state": dict(evidence.state),
        "action": evidence.action,
        "external_input": evidence.external_input,
        "predicted": dict(evidence.predicted),
        "observed": dict(evidence.observed),
    }


def diagnose_exp1(root: Path) -> dict[str, Any]:
    inputs = generate_external_inputs(42, 20)
    arms: dict[str, list[dict[str, Any]]] = {}
    for profile in ("mci_integrated_v1", "core_plus_opt"):
        runner = build_isolated_runner(
            work_root=experiment_work_root(root),
            experiment="diagnostic-exp1",
            profile=profile,
            seed=42,
            scenario="deferred_load_trap",
        )
        episodes: list[dict[str, Any]] = []
        for episode, external_input in enumerate(inputs, 1):
            result = runner.run_episode(
                external_input=external_input,
                replay_unit_id=f"diagnostic/exp1/{profile}/ep-{episode}",
            )
            state = (result.get("reasoning") or {}).get("state") or {}
            outcome = (
                ((result.get("acting_trace") or {}).get("outcome_link") or {}).get(
                    "outcome_observed"
                )
                or {}
            )
            override = result.get("intervention_override") or {}
            committed_action = (result.get("episode") or {}).get(
                "context", {}
            ).get("intervention")
            baseline_action = (
                override.get("from_intervention")
                if override.get("fired")
                else committed_action
            )
            episodes.append(
                {
                    "episode": episode,
                    "external_input": external_input,
                    "baseline_action": baseline_action,
                    "committed_action": committed_action,
                    "intervention_override": override,
                    "mci_first_action": state.get("mci_first_action"),
                    "mci_plan_report": state.get("mci_plan_report"),
                    "opt_choice": state.get("opt_choice"),
                    "outcome": outcome,
                }
            )
        arms[profile] = episodes
    paired = []
    for mci, baseline in zip(
        arms["mci_integrated_v1"], arms["core_plus_opt"]
    ):
        paired.append(
            {
                "episode": mci["episode"],
                "external_input": mci["external_input"],
                "same_committed_action": (
                    mci["committed_action"] == baseline["committed_action"]
                ),
                "mci_committed_action": mci["committed_action"],
                "baseline_committed_action": baseline["committed_action"],
                "mci_planner_differs_from_baseline": (
                    mci["mci_first_action"] != mci["baseline_action"]
                ),
            }
        )
    payload = {
        "experiment": "diagnostic_exp1",
        "seed": 42,
        "episodes": 20,
        "arms": arms,
        "paired_summary": paired,
    }
    write_canonical_json(root / "results" / "diagnostic_exp1_actions.json", payload)
    return payload


def diagnose_exp2(root: Path) -> dict[str, Any]:
    runner = build_isolated_runner(
        work_root=experiment_work_root(root),
        experiment="diagnostic-exp2",
        profile="mci_integrated_v1",
        seed=42,
        scenario="thermal_with_battery",
        scenario_kwargs={
            "initial_temperature": 0.9,
            "initial_battery": 0.75,
        },
    )
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
    episodes: list[dict[str, Any]] = []
    buffer_at_29: list[dict[str, Any]] = []
    for episode in range(1, 30):
        result = runner.run_episode(
            external_input=0.10,
            replay_unit_id=f"diagnostic/exp2/ep-{episode}",
        )
        outcome = result.get("mci_outcome") or {}
        episodes.append(
            {
                "episode": episode,
                "prediction_error": outcome.get("prediction_error"),
                "promoted_overlay": outcome.get("promoted_overlay"),
                "neural_hypotheses_evaluated": outcome.get(
                    "neural_hypotheses_evaluated"
                ),
                "regime_change": outcome.get("regime_change"),
            }
        )
        if episode == 29 and runner._mci_runtime is not None:
            buffer_at_29 = [
                _evidence_dict(item)
                for item in runner._mci_runtime.learner.get_recent_evidence(10)
            ]
    internal_overlay = next(
        (
            item["promoted_overlay"]
            for item in episodes
            if isinstance(item["promoted_overlay"], dict)
            and item["promoted_overlay"]
            .get("parameter_updates", {})
            .get("cooling_effect")
            == 0.035
        ),
        None,
    )
    payload = {
        "experiment": "diagnostic_exp2",
        "injected_hypothesis": {
            **proposal.__dict__,
            "evidence_refs": list(proposal.evidence_refs),
        },
        "provider_feedback": [
            evaluation.to_dict() for evaluation in provider.feedback
        ],
        "promoted_internal_overlay": internal_overlay,
        "buffer_at_episode_29": buffer_at_29,
        "episodes": episodes,
    }
    write_canonical_json(
        root / "results" / "diagnostic_exp2_validation.json", payload
    )
    return payload


def diagnose_exp4(root: Path) -> dict[str, Any]:
    runner = build_isolated_runner(
        work_root=experiment_work_root(root),
        experiment="diagnostic-exp4",
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
        raise RuntimeError("Se esperaba ThermalBatteryScenario")
    window: list[dict[str, Any]] = []
    for episode in range(1, 61):
        if episode == 31:
            scenario.set_regime(0.14)
        result = runner.run_episode(
            external_input=0.04,
            replay_unit_id=f"diagnostic/exp4/ep-{episode}",
        )
        if 30 <= episode <= 45:
            outcome = result.get("mci_outcome") or {}
            window.append(
                {
                    "episode": episode,
                    "prediction_error": outcome.get("prediction_error"),
                    "regime_change": outcome.get("regime_change"),
                    "belief_revision": outcome.get("belief_revision"),
                    "promoted_overlay": outcome.get("promoted_overlay"),
                }
            )
    if runner._mci_runtime is None:
        raise RuntimeError("MCI no disponible al finalizar el diagnóstico")
    detector = runner._mci_runtime.regime_detector
    spec = runner._mci_runtime.learner.active_spec
    payload = {
        "experiment": "diagnostic_exp4",
        "episodes_30_to_45": window,
        "final_active_spec": {
            "spec_id": spec.spec_id,
            "sha256": spec.sha256,
            "parameters": dict(spec.parameters),
            "active_overlay": (
                runner._mci_runtime.learner.active_overlay.to_dict()
                if runner._mci_runtime.learner.active_overlay
                else None
            ),
        },
        "regime_detector": {
            "window": detector._errors.maxlen,
            "consecutive_required": detector.consecutive_required,
            "recent_absolute_errors": list(detector._errors),
            "last_event": (
                detector.last_event.to_dict() if detector.last_event else None
            ),
        },
    }
    write_canonical_json(root / "results" / "diagnostic_exp4_regime.json", payload)
    return payload


def run(root: Path) -> dict[str, Any]:
    os.environ["RNFE_REASONING_ACTUATES"] = "1"
    return {
        "exp1": diagnose_exp1(root),
        "exp2": diagnose_exp2(root),
        "exp4": diagnose_exp4(root),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    run(args.root)
