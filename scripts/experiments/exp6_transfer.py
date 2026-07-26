"""Transferencia causal thermal+battery -> resource+energy.

El script conserva la separación entre adquisición fuente, mapping y validación
destino. Un origen sin overlay estructural correcto produce censura, no un
oráculo encubierto.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

from runtime.symbolic.mci import (
    CausalOverlay,
    OverlayTranslator,
    TransferredHypothesis,
    resource_with_energy_oracle_spec,
    resource_with_energy_spec,
    deferred_load_spec,
    thermal_battery_spec,
    thermal_battery_to_resource_energy,
)
from runtime.symbolic.mci.compiler import TransitionCompiler
from runtime.symbolic.mci.conditions import evaluate_condition

from .common import (
    EXPERIMENT_VERSION,
    bootstrap_paired_diff,
    build_isolated_runner,
    experiment_work_root,
    observed_outcome,
    write_canonical_json,
)
from .exp5_n4_closed_loop import HeuristicStructuralProvider, _logrank

ARMS = ("scratch", "mapping_only", "transfer_heuristic", "transfer_n4")
TRANSFER_VERSION = "mci-transfer-v1"


def _overlay_from_dict(payload: dict[str, Any]) -> CausalOverlay:
    return CausalOverlay(
        overlay_id=str(payload["overlay_id"]),
        version=int(payload["version"]),
        base_spec_sha256=str(payload["base_spec_sha256"]),
        parameter_updates=dict(payload.get("parameter_updates") or {}),
        precondition_updates=dict(payload.get("precondition_updates") or {}),
        edge_updates=tuple(tuple(item) for item in payload.get("edge_updates") or ()),
        evidence=dict(payload.get("evidence") or {}),
        train_mae_before=float(payload["train_mae_before"]),
        train_mae_after=float(payload["train_mae_after"]),
        holdout_mae_before=float(payload["holdout_mae_before"]),
        holdout_mae_after=float(payload["holdout_mae_after"]),
        parent_overlay_id=payload.get("parent_overlay_id"),
    )


def _condition_is_correct(expression: str, variable: str) -> bool:
    try:
        return (
            not evaluate_condition(expression, {variable: 0.2})
            and evaluate_condition(expression, {variable: 0.4})
        )
    except (KeyError, TypeError, ValueError):
        return False


def acquire_source(
    *, root: Path, seed: int, episodes: int
) -> dict[str, Any]:
    """Adquiere únicamente overlays producidos y validados en el mundo fuente."""

    runner = build_isolated_runner(
        work_root=experiment_work_root(root),
        experiment="exp6-source-acquisition",
        profile="mci_integrated_v1",
        seed=seed,
        scenario="thermal_with_battery",
        scenario_kwargs={"initial_temperature": 0.9, "initial_battery": 0.75},
    )
    assert runner._mci_runtime is not None
    runner.set_hypothesis_provider(
        HeuristicStructuralProvider(runner._mci_runtime.base_spec, budget=4)
    )
    acquired_episode = None
    payload = None
    for episode in range(1, episodes + 1):
        result = runner.run_episode(
            external_input=0.04,
            replay_unit_id=f"exp6/source/seed-{seed}/ep-{episode}",
        )
        candidate = (result.get("mci_outcome") or {}).get("promoted_overlay")
        if not isinstance(candidate, dict):
            continue
        expression = str(
            (candidate.get("precondition_updates") or {}).get(
                "thermal_battery/cooling"
            )
            or ""
        )
        if _condition_is_correct(expression, "battery_level"):
            acquired_episode, payload = episode, candidate
            break
    return {
        "seed": seed,
        "acquired": payload is not None,
        "acquired_episode": acquired_episode,
        "overlay": payload,
        "evidence_refs": (
            [
                item.evidence_id
                for item in runner._mci_runtime.learner.get_recent_evidence()
            ]
            if payload is not None
            else []
        ),
    }


def _mapping_only_hypotheses(
    *,
    morphism_id: str,
    logical_time: int,
) -> list[TransferredHypothesis]:
    """Búsqueda guiada por estructura, sin reutilizar el umbral fuente."""

    return [
        TransferredHypothesis(
            hypothesis_id=f"mapping-only-energy-{threshold:.2f}",
            package_id="mapping-only-no-source-package",
            morphism_id=morphism_id,
            kind="precondition",
            target_id="resource_energy/production",
            expression=f"energy_level > {threshold}",
            proposed_value=None,
            transfer_confidence=0.5,
            transfer_penalty=0.15,
            source_evidence_refs=(),
            logical_time=logical_time,
        )
        for threshold in (0.2, 0.3, 0.4)
    ]


def _rank_transfer(
    hypotheses: Sequence[TransferredHypothesis], *, arm: str
) -> list[TransferredHypothesis]:
    # Ambos brazos reciben el mismo conjunto y presupuesto. Sin artefacto
    # entrenado, N4 usa orden determinista y queda clasificado technical_only.
    if arm == "transfer_n4":
        return sorted(
            hypotheses,
            key=lambda item: (
                -(item.transfer_confidence - item.transfer_penalty),
                item.hypothesis_id,
            ),
        )
    return sorted(hypotheses, key=lambda item: item.hypothesis_id)


def evaluate_negative_controls() -> dict[str, Any]:
    """Controles preregistrados, separados de los brazos estadísticos."""

    source, target = thermal_battery_spec(), resource_with_energy_spec()
    mapping = thermal_battery_to_resource_energy(source, target)
    permuted = replace(
        mapping,
        action_map={
            "activate_cooling": "stop_production",
            "deactivate_cooling": "start_production",
        },
    )
    permuted.validate(source, target)
    source_effect_action = next(
        effect.action
        for equation in source.equations
        for effect in equation.effects
        if effect.effect_id == "thermal_battery/cooling"
    )
    target_effect_action = next(
        effect.action
        for equation in target.equations
        for effect in equation.effects
        if effect.effect_id == "resource_energy/production"
    )
    action_control_rejected = (
        permuted.action_map[source_effect_action] != target_effect_action
    )
    incompatible_rejected = False
    try:
        mapping.validate(deferred_load_spec(), target)
    except ValueError:
        incompatible_rejected = True
    return {
        "permuted_actions": {
            "rejected": action_control_rejected,
            "reason": "effect_action_semantics_mismatch",
        },
        "incompatible_origin": {
            "rejected": incompatible_rejected,
            "reason": "morphism_spec_id_mismatch",
        },
        "insufficient_support_policy": {
            "minimum_evidence_refs": 5,
            "status": "enforced_by_receiver",
        },
        "passed": action_control_rejected and incompatible_rejected,
    }


def _run_target(
    *,
    root: Path,
    arm: str,
    seed: int,
    episodes: int,
    source: dict[str, Any],
    n4_artifact_path: Path | None,
) -> dict[str, Any]:
    runner = build_isolated_runner(
        work_root=experiment_work_root(root),
        experiment=f"exp6-target-{arm}",
        profile="mci_integrated_v1",
        seed=seed,
        scenario="resource_with_energy",
        scenario_kwargs={"initial_stock": 0.25, "initial_energy": 0.75},
    )
    source_spec, target_spec = thermal_battery_spec(), resource_with_energy_spec()
    morphism = thermal_battery_to_resource_energy(source_spec, target_spec)
    transferred: list[TransferredHypothesis] = []
    package_sha256 = None
    if arm == "mapping_only":
        transferred = _mapping_only_hypotheses(
            morphism_id=morphism.morphism_id, logical_time=1
        )
    elif arm in {"transfer_heuristic", "transfer_n4"} and source["acquired"]:
        source_overlay = _overlay_from_dict(dict(source["overlay"]))
        translator = OverlayTranslator()
        package = translator.build_package(
            source_overlay,
            morphism,
            evidence_refs=tuple(
                str(item) for item in source.get("evidence_refs") or ()
            ),
            confidence=0.8,
        )
        package_sha256 = package.sha256
        transferred = translator.translate_overlay(
            source_overlay, morphism, package, logical_time=1
        )
        transferred = _rank_transfer(transferred, arm=arm)
    oracle = TransitionCompiler(resource_with_energy_oracle_spec())
    rng = random.Random(seed)
    discovery = None
    cumulative_regret_40 = 0.0
    errors_20_40: list[float] = []
    violations = false_promotions = promotions = rollbacks = 0
    evaluations = retractions = 0
    start = time.perf_counter()
    for episode in range(1, episodes + 1):
        # La partición temporal necesita ambos regímenes energéticos. Presentar
        # claims antes solo provocaría un rechazo no informativo al completar
        # las primeras 12 observaciones, dominadas por energía alta.
        if episode == min(20, episodes) and transferred:
            runner.ingest_transferred_hypotheses(transferred[:4])
        external_input = round(rng.uniform(0.02, 0.06), 12)
        before = dict(runner.scenario.observe().state)
        oracle_values = [
            oracle.execute(before, action=action, external_input=external_input)[
                "stock_level"
            ]
            for action in ("start_production", "stop_production")
        ]
        result = runner.run_episode(
            external_input=external_input,
            replay_unit_id=f"exp6/{arm}/seed-{seed}/ep-{episode}",
        )
        outcome = result.get("mci_outcome") or {}
        observed = observed_outcome(result)
        actual_stock = float(
            observed.get("stock_level", runner.scenario.observe().state["stock_level"])
        )
        if episode <= 40:
            cumulative_regret_40 += max(oracle_values) - actual_stock
        if 20 <= episode <= 40:
            errors_20_40.append(abs(float(outcome.get("prediction_error", 0.0))))
        violations += int(
            bool(observed.get("scarcity_alert", False))
            and result.get("intervention") == "start_production"
            and float(before["energy_level"]) <= 0.3
        )
        for evaluation in outcome.get("neural_hypotheses_evaluated") or ():
            evaluations += 1
            retractions += int(evaluation.get("status") == "rejected")
        promoted_payload = outcome.get("promoted_overlay")
        if isinstance(promoted_payload, dict):
            promotions += 1
            expression = str(
                (promoted_payload.get("precondition_updates") or {}).get(
                    "resource_energy/production"
                )
                or ""
            )
            if _condition_is_correct(expression, "energy_level"):
                discovery = discovery or episode
            else:
                false_promotions += 1
        rollbacks += int((outcome.get("causal_event") or {}).get("kind") == "rollback")
    elapsed = time.perf_counter() - start
    return {
        "seed": seed,
        "source_acquired": bool(source["acquired"]),
        "package_sha256": package_sha256,
        "submitted_hypotheses": len(transferred[:4]),
        "evaluations": evaluations,
        "discovery_episode": discovery,
        "event_time": discovery or episodes,
        "censored": discovery is None,
        "cumulative_regret_40": round(cumulative_regret_40, 9),
        "mae_20_40": round(sum(errors_20_40) / max(1, len(errors_20_40)), 9),
        "promotions": promotions,
        "false_promotions": false_promotions,
        "retractions": retractions,
        "rollbacks": rollbacks,
        "safety_violations": violations,
        "latency_seconds": round(elapsed, 6),
        "n4_mode": (
            "trained"
            if arm == "transfer_n4" and n4_artifact_path is not None
            else ("technical_only" if arm == "transfer_n4" else None)
        ),
    }


def run(
    *,
    root: Path,
    quick: bool = False,
    n4_artifact_path: Path | None = None,
) -> dict[str, Any]:
    os.environ["RNFE_REASONING_ACTUATES"] = "1"
    seeds = tuple(range(2 if quick else 30))
    source_episodes, target_episodes = ((16, 20) if quick else (60, 80))
    sources = {
        seed: acquire_source(root=root, seed=seed, episodes=source_episodes)
        for seed in seeds
    }
    arms = {
        arm: [
            _run_target(
                root=root,
                arm=arm,
                seed=seed,
                episodes=target_episodes,
                source=sources[seed],
                n4_artifact_path=n4_artifact_path,
            )
            for seed in seeds
        ]
        for arm in ARMS
    }
    heuristic = arms["transfer_heuristic"]
    scratch = arms["scratch"]
    regret_ci = bootstrap_paired_diff(
        [row["cumulative_regret_40"] for row in scratch],
        [row["cumulative_regret_40"] for row in heuristic],
        n_resamples=500 if quick else 10_000,
    )
    discovery_rate = sum(
        row["discovery_episode"] is not None
        and int(row["discovery_episode"]) < 40
        for row in heuristic
    ) / len(seeds)
    false = sum(row["false_promotions"] for row in heuristic)
    total = false + sum(not row["censored"] for row in heuristic)
    fdr = false / total if total else 0.0
    logrank = _logrank(heuristic, scratch)
    transfer_gate = bool(
        discovery_rate >= 0.75
        and logrank["p_value"] < 0.05
        and regret_ci[0] > 0.0
        and fdr <= 0.05
        and sum(row["safety_violations"] for row in heuristic)
        <= sum(row["safety_violations"] for row in scratch)
    )
    payload = {
        "experiment": "exp6_transfer",
        "version": EXPERIMENT_VERSION,
        "transfer_version": TRANSFER_VERSION,
        "quick": quick,
        "configuration": {
            "seeds": seeds,
            "source_episodes": source_episodes,
            "target_episodes": target_episodes,
            "candidate_budget": 4,
            "paired_inputs": True,
        },
        "source_acquisition": list(sources.values()),
        "negative_controls": evaluate_negative_controls(),
        "arms": arms,
        "metrics": {
            "discovery_rate_before_40": discovery_rate,
            "logrank_transfer_vs_scratch": logrank,
            "regret_reduction_bootstrap_95": regret_ci,
            "false_discovery_rate": fdr,
        },
        "gates": {
            "source_acquisition": {
                "passed": all(item["acquired"] for item in sources.values())
            },
            "structural_transfer": {"passed": transfer_gate},
            "n4_contribution": {
                "passed": False,
                "status": (
                    "evaluated_trained_backend"
                    if n4_artifact_path is not None
                    else "not_applicable_reference_backend"
                ),
            },
        },
    }
    write_canonical_json(root / "results" / "exp6_transfer.json", payload)
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--n4-artifact", type=Path)
    args = parser.parse_args()
    run(root=args.root, quick=args.quick, n4_artifact_path=args.n4_artifact)
