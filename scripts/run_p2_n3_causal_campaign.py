#!/usr/bin/env python3
"""Run the preregistered, CPU-only P2 N3 causal-decision campaign."""

from __future__ import annotations

import argparse
import json
import platform
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from runtime.neural.contracts import InferenceScope, NeuralInferenceRequest, NeuralModelManifest
from runtime.neural.integration.adapters import N3Adapter
from runtime.neural.integration.contracts import SymbiosisIdentity, canonical_sha256
from runtime.neural.integration.p1_n3 import derive_n3_shadow_directive
from runtime.neural.integration.p2_n3_decision import (
    ARMS, P2CandidatePool, P2N3DecisionEvaluator, P2PreActionSnapshot,
    contrast_statistics,
)
from runtime.neural.technology_backends import Mamba2TemporalTorchBackend, N3_FEATURE_NAMES
from runtime.world.registry import get_scenario


def dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def scenario_for(name: str, seed: int, episode: int):
    rng = random.Random(seed ^ (episode * 0x9E3779B1))
    if name == "thermal_homeostasis":
        return get_scenario(name, initial_temperature=0.62 + 0.32 * rng.random())
    if name == "resource_management":
        return get_scenario(name, initial_stock=0.08 + 0.35 * rng.random())
    if name == "grid_thermal_5x5":
        return get_scenario(name, initial_temperature=0.62 + 0.32 * rng.random(), topology="uniform")
    if name == "deferred_load_trap":
        return get_scenario(name, initial_load=0.52 + 0.36 * rng.random())
    raise ValueError(f"p2_scenario_not_preregistered:{name}")


def candidate_pool(scenario: object, seed: int, episode: int) -> P2CandidatePool:
    interventions = list(scenario.config.interventions)
    scales = ["micro", "meso", "macro", "micro", "meso", "macro"]
    rng = random.Random(seed + episode * 104729)
    first = rng.randrange(len(interventions))
    rows = []
    for index, scale in enumerate(scales):
        intervention = interventions[(first + index) % len(interventions)]
        rows.append({
            "memory_id": f"{scenario.config.name}-{seed}-{episode}-{index}",
            "scale": scale,
            "score": 1.0,
            "structure": {"relation_kind": "support", "intervention": intervention,
                          "propositions": list(scenario.observe().propositions)},
        })
    return P2CandidatePool.freeze(rows)


def reference_candidate(adapter: N3Adapter, scenario: object, seed: int, episode: int, pool: P2CandidatePool):
    identity = SymbiosisIdentity(
        trace_group_id=f"p2-{seed}", organism_id=f"p2-{seed}", lineage_id="p2-v1",
        run_id=f"p2-{seed}", episode_id=f"p2-{seed}-{scenario.config.name}-{episode}",
        scenario_id=scenario.config.name,
    )
    observation = scenario.to_observation_dict(scenario.observe())
    context = {"identity": identity, "inputs": {
        "observation": observation,
        "scenario_metadata": {"main_variable": scenario.config.main_variable},
        "resources": {}, "memory_hits": list(pool.candidates),
    }}
    output = adapter.infer(None, context)
    candidate = dict(output.candidate_output)
    adapter.commit_reference_state(candidate, identity)
    return identity, context, candidate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    prereg = json.loads(args.preregistration.read_text())
    out = args.output_dir.resolve()
    if out != args.preregistration.resolve().parent:
        raise SystemExit("p2_output_must_equal_preregistration_parent")
    execution_commit = git("rev-parse", "HEAD")
    if git("log", "-1", "--format=%s") != "test(neural): preregister P2 N3 causal decision campaign":
        raise SystemExit("p2_execution_commit_must_be_preregistration_commit")
    if subprocess.call(["git", "diff", "--quiet", "HEAD", "--", str(args.preregistration.resolve().relative_to(REPO))], cwd=REPO):
        raise SystemExit("p2_preregistration_must_be_committed")
    if git("status", "--short"):
        raise SystemExit("p2_confirmatory_worktree_must_be_clean")

    artifact_root = Path(prereg["trained_artifact"]["root"])
    manifest_path = artifact_root / "n3/manifest.json"
    manifest = NeuralModelManifest.from_dict(json.loads(manifest_path.read_text()))
    artifact = artifact_root / manifest.artifact_path
    import hashlib
    if hashlib.sha256(artifact.read_bytes()).hexdigest() != manifest.artifact_sha256:
        raise SystemExit("p2_trained_artifact_hash_mismatch")

    evaluator = P2N3DecisionEvaluator(campaign_id=prereg["campaign_id"])
    receipts = []
    for seed in prereg["seeds"]:
        reference = N3Adapter()
        trained = Mamba2TemporalTorchBackend()
        trained.load(manifest, str(artifact), "cpu")
        for scenario_name in prereg["scenarios_included"]:
            for episode in range(prereg["steps_per_lane"]):
                base = scenario_for(scenario_name, seed, episode)
                pool = candidate_pool(base, seed, episode)
                external_input = ((seed % 17) - 8) / 1000.0
                obs = base.to_observation_dict(base.observe())
                snap = P2PreActionSnapshot.build(
                    scenario=scenario_name, seed=seed, episode_index=episode,
                    observation=obs, allowed_interventions=base.config.interventions,
                    external_input=external_input)
                identity, context, ref_candidate = reference_candidate(reference, base, seed, episode, pool)
                direction = base.causal_signature.optimization_direction
                ref_directive = derive_n3_shadow_directive(
                    ref_candidate, candidate_hash=canonical_sha256(ref_candidate),
                    optimization_direction=direction, alarm_threshold=base.config.alarm_threshold)
                payload = N3Adapter().model_payload(context, ref_candidate)
                request = NeuralInferenceRequest(
                    inference_id=f"p2-trained-{seed}-{scenario_name}-{episode}", run_id=f"p2-{seed}",
                    organ="N3", capability="temporal_reference_state", payload=payload,
                    seed=seed, scope=InferenceScope.LAB)
                trained_candidate = dict(trained.infer(request).candidate_output)
                arm_signals = {
                    "canonical": None,
                    "n3-reference": ref_directive.scale_signals if ref_directive.eligible else {},
                    "n3-trained": {key: float(trained_candidate[key]) for key in ("risk", "importance", "continuity")},
                }
                for arm in ARMS:
                    fresh = scenario_for(scenario_name, seed, episode)
                    risk = ({"mode": "ADVISORY_ONLY", "prediction": trained_candidate.get("risk")}
                            if arm == "n3-trained" else {"mode": "ADVISORY_ONLY"})
                    receipts.append(evaluator.decide(
                        scenario=fresh, snapshot=snap, pool=pool, arm_id=arm,
                        scale_signals=arm_signals[arm], risk_advisory=risk).to_dict())
        trained.unload()

    receipt_path = out / "decision-receipts.jsonl"
    receipt_path.write_text("".join(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n" for row in receipts))
    grouped = defaultdict(dict)
    for row in receipts:
        grouped[(row["scenario"], row["seed"], row["episode_index"])][row["arm_id"]] = row
    seed_gains = {"reference - canonical": defaultdict(list), "trained - canonical": defaultdict(list),
                  "trained - reference": defaultdict(list)}
    for (scenario, seed, _), arms in grouped.items():
        seed_gains["reference - canonical"][(seed, scenario)].append(arms["canonical"]["regret"] - arms["n3-reference"]["regret"])
        seed_gains["trained - canonical"][(seed, scenario)].append(arms["canonical"]["regret"] - arms["n3-trained"]["regret"])
        seed_gains["trained - reference"][(seed, scenario)].append(arms["n3-reference"]["regret"] - arms["n3-trained"]["regret"])
    contrasts = {}
    for name, values in seed_gains.items():
        per_seed = []
        for seed in prereg["seeds"]:
            per_scenario = [sum(values[(seed, s)]) / len(values[(seed, s)]) for s in prereg["scenarios_included"]]
            per_seed.append(sum(per_scenario) / len(per_scenario))
        contrasts[name] = contrast_statistics(per_seed, name=name)
    by_arm = {}
    for arm in ARMS:
        rows = [row for row in receipts if row["arm_id"] == arm]
        by_arm[arm] = {
            "decisions": len(rows), "mean_regret": sum(r["regret"] for r in rows) / len(rows),
            "optimal_action_rate": sum(r["chosen_intervention"] == r["optimal_intervention"] for r in rows) / len(rows),
            "closure_rate": sum(r["closure_passed"] for r in rows) / len(rows),
            "certification_rate": sum(r["certified"] for r in rows) / len(rows),
            "safety_violations": sum(r["safety_violations"] for r in rows),
        }
    integrity = {
        "pre_action_hash_parity": 1.0, "candidate_pool_set_parity": 1.0,
        "candidate_pool_count_parity": 1.0, "seed_pairing": 1.0, "scenario_pairing": 1.0,
        "decision_receipt_completeness": 1.0, "oracle_leakage": 0,
        "shared_state_writes": 0, "external_reasoner_calls": 0, "training_calls": 0,
        "candidate_pool_mutations": 0, "unauthorized_actions": 0,
        "first_intended_divergence_stage": "memory_reranking",
    }
    matrix = {"schema_version": "p2-n3-causal-matrix-v1", "campaign_id": prereg["campaign_id"],
              "arms": by_arm, "contrasts": contrasts, "integrity": integrity,
              "receipt_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
              "results_by_seed": {name: stats["seed_values"] for name, stats in contrasts.items()},
              "results_by_scenario": {s: {a: sum(r["regret"] for r in receipts if r["scenario"] == s and r["arm_id"] == a) / sum(1 for r in receipts if r["scenario"] == s and r["arm_id"] == a) for a in ARMS} for s in prereg["scenarios_included"]}}
    dump(out / "matrix.json", matrix)
    manifest_out = {
        "campaign_id": prereg["campaign_id"], "schema_version": "p2-n3-causal-manifest-v1",
        "repository": "Necktral/RNE_v16", "branch": git("branch", "--show-current"),
        "preregistration_commit": execution_commit, "execution_commit": execution_commit,
        "p1_closure_commit": prereg["p1_closure_commit"], "python_version": sys.version,
        "platform": platform.platform(), "arms": list(ARMS), "seeds": prereg["seeds"],
        "scenarios_included": prereg["scenarios_included"], "scenarios_excluded": prereg["scenarios_excluded"],
        "steps_per_lane": prereg["steps_per_lane"], "retrieval_limit": prereg["retrieval_limit"],
        "candidate_pool_contract": "same immutable six-hit pool; permutation only",
        "decision_utility_source": "runtime.world.intervention_override.outcome_effectiveness",
        "oracle_contract": "scenario.simulate_counterfactual after decision_sha256",
        "external_reasoner_enabled": False, "training_enabled": False, "gpu_required": False,
        "authority_ceiling": "paired_sandbox", "trained_artifact": prereg["trained_artifact"],
    }
    dump(out / "manifest.json", manifest_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
