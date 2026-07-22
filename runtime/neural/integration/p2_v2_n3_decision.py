"""Fail-closed P2-v2 harness for causal N3 retrieval reranking.

This module is deliberately disconnected from the live organism.  One real MFM
retrieval is frozen, N3 may only permute it, IND sees only the exposed top-k,
and the scenario oracle cannot open before the decision has been sealed.
"""

from __future__ import annotations

import hashlib
import itertools
import math
import random
import statistics
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping, Sequence

from runtime.neural.integration.contracts import canonical_sha256
from runtime.neural.integration.p1_n3 import N3ShadowDirective
from runtime.reasoning.families.core_inference import induce
from runtime.world.intervention_override import outcome_effectiveness

ARMS = ("canonical", "n3-reference", "n3-trained")
SCALES = ("micro", "meso", "macro")


def _finite(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("p2_v2_nonfinite_number")
    return result


@dataclass(frozen=True, slots=True)
class FrozenRetrieval:
    hits: tuple[Mapping[str, Any], ...]
    sha256: str

    @classmethod
    def freeze(cls, hits: Sequence[Mapping[str, Any]]) -> "FrozenRetrieval":
        frozen = tuple(dict(hit) for hit in hits)
        ids = tuple(str(hit.get("memory_id") or "") for hit in frozen)
        if not frozen or any(not item for item in ids) or len(ids) != len(set(ids)):
            raise ValueError("p2_v2_raw_pool_empty_or_ids_invalid")
        for hit in frozen:
            _finite(hit.get("score"))
        return cls(frozen, canonical_sha256(list(frozen)))

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(str(hit["memory_id"]) for hit in self.hits)


def rerank(
    pool: FrozenRetrieval,
    *,
    arm_id: str,
    directive: N3ShadowDirective | None = None,
) -> tuple[Mapping[str, Any], ...]:
    """Return a stable permutation; canonical is byte/order faithful to MFM."""
    if arm_id not in ARMS:
        raise ValueError("p2_v2_arm_invalid")
    if arm_id == "canonical":
        return pool.hits
    if directive is None or not directive.eligible:
        raise ValueError("p2_v2_n3_directive_not_eligible")
    signals = directive.scale_signals  # official P1 semantic bridge
    indexed = list(enumerate(pool.hits))
    indexed.sort(
        key=lambda pair: (
            -(0.75 + 0.25 * signals[str(pair[1].get("scale"))])
            * _finite(pair[1]["score"]),
            pair[0],
        )
    )
    ordered = tuple(hit for _, hit in indexed)
    if len(ordered) != len(pool.hits) or {str(x["memory_id"]) for x in ordered} != set(pool.ids):
        raise ValueError("P2_V2_INVALID_CANDIDATE_POOL_MUTATION")
    if any(dict(a) != dict(b) for a, b in zip(sorted(ordered, key=lambda x: str(x["memory_id"])),
                                               sorted(pool.hits, key=lambda x: str(x["memory_id"])))):
        raise ValueError("P2_V2_INVALID_CANDIDATE_CONTENT_MUTATION")
    return ordered


@dataclass(frozen=True, slots=True)
class PreActionSnapshot:
    payload: Mapping[str, Any]
    sha256: str

    @classmethod
    def build(cls, payload: Mapping[str, Any]) -> "PreActionSnapshot":
        frozen = dict(payload)
        return cls(frozen, canonical_sha256(frozen))


@dataclass(frozen=True, slots=True)
class DecisionSeal:
    payload: Mapping[str, Any]
    sha256: str

    @classmethod
    def create(cls, payload: Mapping[str, Any]) -> "DecisionSeal":
        required = {"arm_id", "actual_pre_action_state_sha256", "raw_candidate_pool_sha256",
                    "exposed_memory_sha256", "chosen_intervention"}
        if not required.issubset(payload):
            raise ValueError("p2_v2_decision_seal_incomplete")
        frozen = dict(payload)
        return cls(frozen, canonical_sha256(frozen))


def open_oracle(
    *, scenario: Any, external_input: float, allowed_interventions: Sequence[str],
    seal: DecisionSeal | None,
) -> tuple[dict[str, float], str]:
    if not isinstance(seal, DecisionSeal):
        raise ValueError("P2_ORACLE_BEFORE_DECISION_SEAL")
    utilities: dict[str, float] = {}
    for intervention in allowed_interventions:
        transition = scenario.simulate_counterfactual(
            intervention=intervention, external_input=float(external_input)
        )
        value = _finite(transition.state[scenario.config.main_variable])
        utilities[str(intervention)] = _finite(outcome_effectiveness(
            value=value,
            alarm_threshold=scenario.config.alarm_threshold,
            alarm_semantics=scenario.causal_signature.alarm_semantics,
        ))
    optimal = max(allowed_interventions, key=lambda x: (utilities[str(x)], -list(allowed_interventions).index(x)))
    return utilities, str(optimal)


def evaluate_arm(
    *, campaign_id: str, scenario: Any, scenario_name: str, seed: int,
    episode_index: int, arm_id: str, snapshot: PreActionSnapshot,
    raw_pool: FrozenRetrieval, k_exposed: int, external_input: float,
    directive: N3ShadowDirective | None = None,
    decision_fn: Callable[[Mapping[str, Any]], Mapping[str, Any]] = induce,
) -> dict[str, Any]:
    started = time.perf_counter_ns()
    rerank_started = time.perf_counter_ns()
    ordered = rerank(raw_pool, arm_id=arm_id, directive=directive)
    rerank_ms = (time.perf_counter_ns() - rerank_started) / 1e6
    if not 0 < k_exposed < len(ordered):
        raise ValueError("p2_v2_invalid_exposure_cut")
    exposed = ordered[:k_exposed]
    observation = scenario.observe()
    actual_payload = {
        "scenario": scenario_name,
        "seed": seed,
        "episode_index": episode_index,
        "observation": scenario.to_observation_dict(observation),
        "allowed_interventions": list(scenario.config.interventions),
        "external_input": float(external_input),
    }
    actual_sha = canonical_sha256(actual_payload)
    if actual_sha != snapshot.sha256:
        raise ValueError("P2_V2_ACTUAL_STATE_SNAPSHOT_MISMATCH")
    state = {
        "observation": {**dict(observation.state), "alarm": observation.alarm,
                        "propositions": list(observation.propositions)},
        "retrieved_memory": [dict(hit) for hit in exposed],
        "scenario_metadata": {
            "scenario_name": scenario.config.name,
            "main_variable": scenario.config.main_variable,
            "alarm_threshold": scenario.config.alarm_threshold,
            "interventions": list(scenario.config.interventions),
            "optimization_direction": scenario.causal_signature.optimization_direction,
            "causal_signature": scenario.causal_signature,
        },
    }
    decision_started = time.perf_counter_ns()
    result = decision_fn(state)
    recommendation = result.get("state_delta", {}).get("ind_best_intervention")
    chosen = str(recommendation or scenario.select_intervention(observation))
    if chosen not in scenario.config.interventions:
        raise ValueError("p2_v2_unauthorized_intervention")
    decision_ms = (time.perf_counter_ns() - decision_started) / 1e6
    exposed_ids = tuple(str(x["memory_id"]) for x in exposed)
    seal = DecisionSeal.create({
        "arm_id": arm_id,
        "actual_pre_action_state_sha256": actual_sha,
        "raw_candidate_pool_sha256": raw_pool.sha256,
        "exposed_memory_sha256": canonical_sha256(list(exposed_ids)),
        "chosen_intervention": chosen,
    })
    oracle_started = time.perf_counter_ns()
    utilities, optimal = open_oracle(
        scenario=scenario, external_input=external_input,
        allowed_interventions=scenario.config.interventions, seal=seal,
    )
    oracle_ms = (time.perf_counter_ns() - oracle_started) / 1e6
    canonical_ids = raw_pool.ids
    ordered_ids = tuple(str(x["memory_id"]) for x in ordered)
    canonical_exposed = set(canonical_ids[:k_exposed])
    exposed_set = set(exposed_ids)
    scores = [_finite(x["score"]) for x in raw_pool.hits]
    return {
        "campaign_id": campaign_id, "scenario": scenario_name, "seed": seed,
        "episode_index": episode_index, "arm_id": arm_id,
        "snapshot_sha256": snapshot.sha256,
        "actual_pre_action_state_sha256": actual_sha,
        "snapshot_matches_actual": True,
        "raw_candidate_pool_sha256": raw_pool.sha256,
        "raw_candidate_ids": list(canonical_ids),
        "raw_candidate_scores": scores,
        "raw_candidate_scales": [str(x.get("scale")) for x in raw_pool.hits],
        "raw_candidate_structures": [x.get("structure") for x in raw_pool.hits],
        "canonical_order_ids": list(canonical_ids), "arm_order_ids": list(ordered_ids),
        "exposed_memory_ids": list(exposed_ids),
        "exposed_memory_sha256": canonical_sha256(list(exposed_ids)),
        "raw_pool_set_parity": set(ordered_ids) == set(canonical_ids),
        "exposed_set_overlap": len(canonical_exposed & exposed_set),
        "exposed_set_changed": exposed_set != canonical_exposed,
        "top1_changed": ordered_ids[0] != canonical_ids[0],
        "full_order_changed": ordered_ids != canonical_ids,
        "unique_score_count": len(set(scores)), "score_range": max(scores) - min(scores),
        "tie_count": len(scores) - len(set(scores)),
        "decision_sealed": True, "decision_sha256": seal.sha256,
        "chosen_intervention": chosen, "chosen_intervention_is_allowed": True,
        "optimal_intervention": optimal, "chosen_utility": utilities[chosen],
        "optimal_utility": utilities[optimal], "regret": utilities[optimal] - utilities[chosen],
        "oracle_opened_after_seal": True, "external_reasoner_used": False,
        "training_executed": False, "live_authority": False,
        "closure_passed": None, "certified": None, "full_safety_evaluation": None,
        "measurement_status": "NOT_MEASURED_IN_P2_V2",
        "latency": {"canonical_retrieval_latency_ms": None,
                    "n3_rerank_latency_ms": rerank_ms, "decision_latency_ms": decision_ms,
                    "oracle_latency_ms": oracle_ms,
                    "total_arm_latency_ms": (time.perf_counter_ns() - started) / 1e6},
    }


def seed_values(prefix: str = "rnfe-p2-v2-n3-causal", count: int = 12) -> list[int]:
    return [int.from_bytes(hashlib.sha256(f"{prefix}:{i}".encode()).digest()[:4], "big") & 0x7FFFFFFF
            for i in range(count)]


def contrast_statistics(values: Sequence[float], *, name: str,
                        bootstrap_samples: int = 10_000) -> dict[str, Any]:
    vals = [_finite(v) for v in values]
    if len(vals) != 12:
        raise ValueError("p2_v2_contrast_requires_12_seed_values")
    rng = random.Random(int.from_bytes(hashlib.sha256(f"p2-v2:{name}".encode()).digest()[:8], "big"))
    boot = sorted(statistics.mean(rng.choice(vals) for _ in vals) for _ in range(bootstrap_samples))
    lo, hi = boot[int(.025 * bootstrap_samples)], boot[min(bootstrap_samples - 1, int(.975 * bootstrap_samples))]
    observed = abs(statistics.mean(vals))
    extreme = sum(abs(statistics.mean(v * s for v, s in zip(vals, signs))) >= observed - 1e-15
                  for signs in itertools.product((-1.0, 1.0), repeat=12))
    result = {"mean": statistics.mean(vals), "median": statistics.median(vals),
              "standard_deviation_ddof1": statistics.stdev(vals), "minimum": min(vals),
              "maximum": max(vals), "positive_count": sum(v > 0 for v in vals),
              "zero_count": sum(v == 0 for v in vals), "negative_count": sum(v < 0 for v in vals),
              "bootstrap_ci95": [lo, hi], "bootstrap_samples": bootstrap_samples,
              "exact_sign_flip_p_value": extreme / 4096, "assignments_enumerated": 4096,
              "seed_values": vals}
    result["gate_passed"] = bool(result["mean"] > 0 and lo > 0 and result["exact_sign_flip_p_value"] < .05
                                  and result["positive_count"] >= 9 and result["negative_count"] <= 3)
    return result
