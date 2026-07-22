#!/usr/bin/env python3
"""Execute the preregistered CPU-only P2-v2 paired sandbox."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from runtime.memory.mfm_lite.retrieval import MemoryRetrieval
from runtime.neural.contracts import InferenceScope, NeuralInferenceRequest, NeuralModelManifest
from runtime.neural.integration.adapters import N3Adapter
from runtime.neural.integration.contracts import SymbiosisIdentity, canonical_sha256
from runtime.neural.integration.p1_n3 import derive_n3_shadow_directive
from runtime.neural.integration.p2_v2_n3_decision import (
    ARMS, FrozenRetrieval, PreActionSnapshot, evaluate_arm,
)
from runtime.neural.technology_backends import Mamba2TemporalTorchBackend
from runtime.storage import StorageConfig, StorageFactory
from runtime.world.registry import get_scenario
from runtime.reasoning.families.core_inference import induce


def dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False,
                               allow_nan=False) + "\n")


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scenario_for(name: str, seed: int, episode: int):
    rng = random.Random(seed ^ (episode * 0x9E3779B1))
    if name == "thermal_homeostasis":
        return get_scenario(name, initial_temperature=.62 + .32 * rng.random())
    if name == "resource_management":
        return get_scenario(name, initial_stock=.08 + .35 * rng.random())
    if name == "grid_thermal_5x5":
        return get_scenario(name, initial_temperature=.62 + .32 * rng.random(), topology="uniform")
    if name == "deferred_load_trap":
        return get_scenario(name, initial_load=.52 + .36 * rng.random())
    raise ValueError(f"p2_v2_scenario_not_preregistered:{name}")


def seed_balanced_bank(storage, *, run_id: str, scenario: object, seed: int,
                       episode: int) -> None:
    interventions = list(scenario.config.interventions)
    if len(interventions) != 2:
        raise ValueError("p2_v2_requires_two_interventions")
    cells = [(iv, scale, relation) for iv in interventions
             for scale in ("micro", "meso", "macro")
             for relation in ("support", "contradiction")]
    random.Random(seed ^ episode ^ 0xC0FFEE).shuffle(cells)
    propositions = list(scenario.observe().propositions)
    for index, (intervention, scale, relation) in enumerate(cells):
        # Vary query overlap orthogonally to treatment cells; retrieval computes score.
        selected_props = propositions if (index + seed + episode) % 3 else [f"DISTRACTOR_{index}"]
        storage.write_memory_record(
            run_id=run_id, episode_id=f"{run_id}-e{index}",
            memory_id=f"{run_id}-m{index}", scale=scale,
            structure_json={"relation_kind": relation, "intervention": intervention,
                            "alarm": bool((index + episode) % 2),
                            "propositions": selected_props},
            metadata={"scenario_name": scenario.config.name,
                      "scenario_version": getattr(scenario.config, "version", None),
                      "p2_v2_cell": [intervention, scale, relation]},
        )


def context_for(adapter: N3Adapter, scenario: object, seed: int, episode: int,
                raw: FrozenRetrieval):
    identity = SymbiosisIdentity(
        trace_group_id=f"p2-v2-{seed}", organism_id=f"p2-v2-{seed}",
        lineage_id="p2-v2", run_id=f"p2-v2-{seed}",
        episode_id=f"p2-v2-{seed}-{scenario.config.name}-{episode}",
        scenario_id=scenario.config.name,
    )
    context = {"identity": identity, "inputs": {
        "observation": scenario.to_observation_dict(scenario.observe()),
        "scenario_metadata": {"main_variable": scenario.config.main_variable},
        "resources": {}, "memory_hits": list(raw.hits),
    }}
    candidate = dict(adapter.infer(None, context).candidate_output)
    adapter.commit_reference_state(candidate, identity)
    return identity, context, candidate


def verify_ind_memory_sensitivity(scenario: object) -> bool:
    interventions = list(scenario.config.interventions)
    if len(interventions) != 2:
        return False
    def choose(target: str) -> str | None:
        memory = [
            {"memory_id": f"control-{target}-{i}", "score": 1.0,
             "structure": {"relation_kind": "support" if i < 3 else "contradiction",
                           "intervention": target}}
            for i in range(4)
        ]
        state = {
            "observation": {**dict(scenario.observe().state),
                            "alarm": scenario.observe().alarm},
            "retrieved_memory": memory,
            "scenario_metadata": {
                "scenario_name": scenario.config.name,
                "main_variable": scenario.config.main_variable,
                "alarm_threshold": scenario.config.alarm_threshold,
                "interventions": interventions,
                "optimization_direction": scenario.causal_signature.optimization_direction,
                "causal_signature": scenario.causal_signature,
            },
        }
        return induce(state)["state_delta"].get("ind_best_intervention")
    return choose(interventions[0]) == interventions[0] and choose(interventions[1]) == interventions[1]


def verify_execution(prereg_path: Path, prereg: dict) -> str:
    if git("branch", "--show-current") != "codex/p2-n3-causal-decision-v2":
        raise SystemExit("p2_v2_wrong_branch")
    if git("status", "--short"):
        raise SystemExit("p2_v2_confirmatory_worktree_must_be_clean")
    head = git("rev-parse", "HEAD")
    prereg_commit = prereg["preregistration_commit"]
    if subprocess.call(["git", "merge-base", "--is-ancestor", prereg_commit, head], cwd=REPO):
        raise SystemExit("p2_v2_preregistration_commit_not_ancestor")
    if subprocess.call(["git", "diff", "--quiet", "HEAD", "--",
                        str(prereg_path.resolve().relative_to(REPO))], cwd=REPO):
        raise SystemExit("p2_v2_preregistration_not_committed")
    return head


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    prereg = json.loads(args.preregistration.read_text())
    out = args.output_dir.resolve()
    if out != args.preregistration.resolve().parent:
        raise SystemExit("p2_v2_output_must_equal_preregistration_parent")
    execution_commit = verify_execution(args.preregistration, prereg)
    controls = {name: verify_ind_memory_sensitivity(scenario_for(name, prereg["seeds"][0], 0))
                for name in prereg["scenarios"]}
    if not all(controls.values()):
        raise SystemExit("P2_V2_BLOCKED_BY_INSENSITIVE_DECISION_SEAM")

    manifest_path = args.artifact_root.resolve() / "n3" / "manifest.json"
    if file_sha(manifest_path) != prereg["trained_artifact"]["manifest_sha256"]:
        raise SystemExit("P2_V2_TRAINED_MANIFEST_HASH_MISMATCH")
    model_manifest = NeuralModelManifest.from_dict(json.loads(manifest_path.read_text()))
    artifact = args.artifact_root.resolve() / model_manifest.artifact_path
    if (model_manifest.model_id != prereg["trained_artifact"]["model_id"]
            or model_manifest.backend != prereg["trained_artifact"]["backend"]
            or file_sha(artifact) != prereg["trained_artifact"]["artifact_sha256"]):
        raise SystemExit("P2_V2_TRAINED_ARTIFACT_HASH_MISMATCH")

    storage = StorageFactory.create_facade(StorageConfig(
        mode="sqlite", sqlite_db_path=str(out / "p2-v2-sandbox.db"), postgres_dsn=None,
        artifact_root=out / "sandbox-artifacts", prefer_postgres_reads=True,
        strict_dual_write=False,
    ))
    retrieval = MemoryRetrieval(storage=storage)
    trained = Mamba2TemporalTorchBackend()
    trained.load(model_manifest, str(artifact), "cpu")
    receipts = []
    started = time.time()
    try:
        for seed in prereg["seeds"]:
            reference = N3Adapter()
            for scenario_name in prereg["scenarios"]:
                for episode in range(prereg["episodes_per_scenario"]):
                    base = scenario_for(scenario_name, seed, episode)
                    run_id = f"p2-v2-{seed}-{scenario_name}-{episode}"
                    seed_balanced_bank(storage, run_id=run_id, scenario=base,
                                       seed=seed, episode=episode)
                    obs = base.observe()
                    obs_dict = base.to_observation_dict(obs)
                    query = {"alarm": obs.alarm,
                             "proposition": next(iter(obs.propositions), "NO_PROPOSITION")}
                    retrieval_started = time.perf_counter_ns()
                    raw_hits = retrieval.retrieve(
                        run_id=run_id, query=query, limit=prereg["n_raw"],
                        candidate_pool_size=prereg["n_raw"], scenario_name=scenario_name,
                        scenario_version=getattr(base.config, "version", None),
                        scenario_filter_mode="strict_same_scenario",
                    )
                    retrieval_ms = (time.perf_counter_ns() - retrieval_started) / 1e6
                    raw = FrozenRetrieval.freeze(raw_hits)
                    external_input = ((seed % 17) - 8) / 1000.0
                    snapshot = PreActionSnapshot.build({
                        "scenario": scenario_name, "seed": seed, "episode_index": episode,
                        "observation": obs_dict,
                        "allowed_interventions": list(base.config.interventions),
                        "external_input": external_input,
                    })
                    _, context, ref_candidate = context_for(reference, base, seed, episode, raw)
                    direction = base.causal_signature.optimization_direction
                    ref_directive = derive_n3_shadow_directive(
                        ref_candidate, candidate_hash=canonical_sha256(ref_candidate),
                        optimization_direction=direction,
                        alarm_threshold=base.config.alarm_threshold,
                    )
                    payload = N3Adapter().model_payload(context, ref_candidate)
                    request = NeuralInferenceRequest(
                        inference_id=f"p2-v2-trained-{seed}-{scenario_name}-{episode}",
                        run_id=run_id, organ="N3", capability="temporal_reference_state",
                        payload=payload, seed=seed, scope=InferenceScope.LAB,
                    )
                    trained_candidate = dict(trained.infer(request).candidate_output)
                    trained_directive = derive_n3_shadow_directive(
                        trained_candidate, candidate_hash=canonical_sha256(trained_candidate),
                        optimization_direction=direction,
                        alarm_threshold=base.config.alarm_threshold,
                    )
                    for arm, directive in (("canonical", None),
                                           ("n3-reference", ref_directive),
                                           ("n3-trained", trained_directive)):
                        fresh = scenario_for(scenario_name, seed, episode)
                        row = evaluate_arm(
                            campaign_id=prereg["campaign_id"], scenario=fresh,
                            scenario_name=scenario_name, seed=seed, episode_index=episode,
                            arm_id=arm, snapshot=snapshot, raw_pool=raw,
                            k_exposed=prereg["k_exposed"], external_input=external_input,
                            directive=directive,
                        )
                        row["latency"]["canonical_retrieval_latency_ms"] = retrieval_ms
                        receipts.append(row)
    finally:
        trained.unload()
        storage.close()

    receipt_path = out / "decision-receipts.jsonl"
    receipt_path.write_text("".join(json.dumps(x, sort_keys=True, separators=(",", ":"),
                                               allow_nan=False) + "\n" for x in receipts))
    from scripts.audit_p2_v2_n3_causal import recompute_matrix
    matrix = recompute_matrix(receipts, prereg)
    dump(out / "matrix.json", matrix)
    dump(out / "manifest.json", {
        "schema_version": "p2-v2-manifest-v1", "campaign_id": prereg["campaign_id"],
        "repository": "Necktral/RNE_v16", "branch": git("branch", "--show-current"),
        "base_commit": prereg["p1_closure_commit"], "execution_commit": execution_commit,
        "python_version": sys.version, "platform": platform.platform(),
        "n_raw": prereg["n_raw"], "k_exposed": prereg["k_exposed"],
        "arms": list(ARMS), "receipt_count": len(receipts),
        "real_memory_retrieval_used": True, "single_retrieval_per_unit": True,
        "external_reasoner_enabled": False, "training_enabled": False,
        "gpu_required": False, "authority_ceiling": "controlled_paired_sandbox",
        "trained_artifact": prereg["trained_artifact"], "duration_seconds": time.time() - started,
        "receipt_sha256": file_sha(receipt_path),
        "ind_memory_sensitivity_control": controls,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
