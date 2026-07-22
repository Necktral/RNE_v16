"""Pure P2 harness for causal N3 memory-reranking experiments.

The module has no live hooks.  It freezes one candidate set, permits each N3
arm to permute that set, seals an IND decision, and only then opens the
scenario-owned counterfactual oracle.
"""

from __future__ import annotations

import hashlib
import itertools
import math
import random
import statistics
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from runtime.neural.integration.contracts import canonical_sha256
from runtime.reasoning.families.core_inference import induce
from runtime.world.intervention_override import outcome_effectiveness


ARMS = ("canonical", "n3-reference", "n3-trained")


@dataclass(frozen=True, slots=True)
class P2CandidatePool:
    candidates: tuple[Mapping[str, Any], ...]
    sha256: str

    @classmethod
    def freeze(cls, candidates: Sequence[Mapping[str, Any]]) -> "P2CandidatePool":
        frozen = tuple(dict(item) for item in candidates)
        ids = [str(item.get("memory_id") or "") for item in frozen]
        if not frozen or any(not item for item in ids) or len(ids) != len(set(ids)):
            raise ValueError("p2_candidate_pool_empty_or_ids_invalid")
        return cls(frozen, canonical_sha256(list(frozen)))

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(str(item["memory_id"]) for item in self.candidates)


@dataclass(frozen=True, slots=True)
class P2PreActionSnapshot:
    scenario: str
    seed: int
    episode_index: int
    observation: Mapping[str, Any]
    allowed_interventions: tuple[str, ...]
    external_input: float
    sha256: str

    @classmethod
    def build(cls, *, scenario: str, seed: int, episode_index: int,
              observation: Mapping[str, Any], allowed_interventions: Sequence[str],
              external_input: float) -> "P2PreActionSnapshot":
        payload = {
            "scenario": scenario, "seed": seed, "episode_index": episode_index,
            "observation": dict(observation),
            "allowed_interventions": list(allowed_interventions),
            "external_input": float(external_input),
        }
        return cls(scenario, seed, episode_index, dict(observation),
                   tuple(allowed_interventions), float(external_input),
                   canonical_sha256(payload))


@dataclass(frozen=True, slots=True)
class P2DecisionReceipt:
    campaign_id: str
    scenario: str
    seed: int
    episode_index: int
    arm_id: str
    pre_action_state_sha256: str
    candidate_pool_sha256: str
    ordered_memory_sha256: str
    decision_sha256: str
    candidate_memory_ids: tuple[str, ...]
    ordered_memory_ids: tuple[str, ...]
    chosen_intervention: str
    optimal_intervention: str
    chosen_utility: float
    optimal_utility: float
    regret: float
    closure_passed: bool = True
    certified: bool = True
    safety_violations: int = 0
    causal_attestation: Mapping[str, Any] | None = None
    risk_advisory: Mapping[str, Any] | None = None
    latency_ms: float = 0.0
    external_reasoner_used: bool = False
    training_executed: bool = False
    live_authority: bool = False
    shared_state_writes: int = 0
    oracle_leakage: int = 0
    first_intended_divergence_stage: str = "memory_reranking"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def rank_pool(pool: P2CandidatePool, *, arm_id: str,
              scale_signals: Mapping[str, float] | None = None) -> tuple[Mapping[str, Any], ...]:
    if arm_id not in ARMS:
        raise ValueError("p2_arm_invalid")
    if arm_id == "canonical":
        return pool.candidates
    signals = dict(scale_signals or {})
    if set(signals) - {"micro", "meso", "macro"}:
        raise ValueError("p2_n3_scale_signal_invalid")
    indexed = list(enumerate(pool.candidates))
    indexed.sort(key=lambda pair: (
        -(0.75 + 0.25 * float(signals.get(str(pair[1].get("scale")), 0.0)))
        * float(pair[1].get("score", 0.0)),
        pair[0],
    ))
    ranked = tuple(item for _, item in indexed)
    if set(pool.ids) != {str(item["memory_id"]) for item in ranked} or len(ranked) != len(pool.ids):
        raise ValueError("P2_INVALID_CANDIDATE_POOL_MUTATION")
    return ranked


class P2N3DecisionEvaluator:
    def __init__(self, *, campaign_id: str):
        self.campaign_id = campaign_id

    def decide(self, *, scenario: Any, snapshot: P2PreActionSnapshot,
               pool: P2CandidatePool, arm_id: str,
               scale_signals: Mapping[str, float] | None = None,
               risk_advisory: Mapping[str, Any] | None = None) -> P2DecisionReceipt:
        ordered = rank_pool(pool, arm_id=arm_id, scale_signals=scale_signals)
        observation = scenario.observe()
        state = {
            "observation": {**dict(observation.state), "alarm": observation.alarm,
                            "propositions": list(observation.propositions)},
            "retrieved_memory": [dict(item) for item in ordered],
            "scenario_metadata": {
                "scenario_name": scenario.config.name,
                "main_variable": scenario.config.main_variable,
                "alarm_threshold": scenario.config.alarm_threshold,
                "interventions": list(scenario.config.interventions),
                "optimization_direction": scenario.causal_signature.optimization_direction,
                "causal_signature": scenario.causal_signature,
            },
        }
        recommendation = induce(state)["state_delta"].get("ind_best_intervention")
        chosen = str(recommendation or scenario.select_intervention(observation))
        if chosen not in snapshot.allowed_interventions:
            raise ValueError("p2_invalid_intervention")
        ordered_ids = tuple(str(item["memory_id"]) for item in ordered)
        decision_payload = {
            "arm_id": arm_id, "pre_action_state_sha256": snapshot.sha256,
            "ordered_memory_ids": list(ordered_ids), "chosen_intervention": chosen,
        }
        decision_sha = canonical_sha256(decision_payload)  # oracle opens below this line

        sig = scenario.causal_signature
        utilities: dict[str, float] = {}
        transitions: dict[str, Any] = {}
        for intervention in snapshot.allowed_interventions:
            transition = scenario.simulate_counterfactual(
                intervention=intervention, external_input=snapshot.external_input)
            value = float(transition.state[scenario.config.main_variable])
            utility = outcome_effectiveness(
                value=value, alarm_threshold=scenario.config.alarm_threshold,
                alarm_semantics=sig.alarm_semantics)
            if not math.isfinite(utility):
                raise ValueError("p2_nonfinite_utility")
            utilities[intervention] = utility
            transitions[intervention] = transition
        optimal = max(snapshot.allowed_interventions, key=lambda item: (utilities[item], -snapshot.allowed_interventions.index(item)))
        regret = utilities[optimal] - utilities[chosen]
        return P2DecisionReceipt(
            campaign_id=self.campaign_id, scenario=snapshot.scenario, seed=snapshot.seed,
            episode_index=snapshot.episode_index, arm_id=arm_id,
            pre_action_state_sha256=snapshot.sha256, candidate_pool_sha256=pool.sha256,
            ordered_memory_sha256=canonical_sha256(list(ordered_ids)), decision_sha256=decision_sha,
            candidate_memory_ids=pool.ids, ordered_memory_ids=ordered_ids,
            chosen_intervention=chosen, optimal_intervention=optimal,
            chosen_utility=utilities[chosen], optimal_utility=utilities[optimal], regret=regret,
            causal_attestation={"utility_source": "outcome_effectiveness", "oracle_opened_after_decision_sha256": True},
            risk_advisory=dict(risk_advisory or {}),
        )


def seed_values(prefix: str = "rnfe-p2-n3-causal-v1", count: int = 12) -> list[int]:
    return [int.from_bytes(hashlib.sha256(f"{prefix}:{i}".encode()).digest()[:4], "big") & 0x7FFFFFFF for i in range(count)]


def contrast_statistics(values: Sequence[float], *, name: str, bootstrap_samples: int = 10000) -> dict[str, Any]:
    vals = [float(item) for item in values]
    if len(vals) != 12 or any(not math.isfinite(item) for item in vals):
        raise ValueError("p2_contrast_requires_12_finite_seed_values")
    rng_seed = int.from_bytes(hashlib.sha256(f"rnfe-p2-bootstrap:{name}".encode()).digest()[:8], "big")
    rng = random.Random(rng_seed)
    boot = sorted(sum(rng.choice(vals) for _ in vals) / len(vals) for _ in range(bootstrap_samples))
    lo = boot[int(0.025 * bootstrap_samples)]
    hi = boot[min(bootstrap_samples - 1, int(0.975 * bootstrap_samples))]
    observed = abs(statistics.mean(vals))
    assignments = 1 << len(vals)
    extreme = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(vals)):
        if abs(statistics.mean(v * s for v, s in zip(vals, signs))) >= observed - 1e-15:
            extreme += 1
    p = extreme / assignments
    result = {
        "mean": statistics.mean(vals), "median": statistics.median(vals),
        "standard_deviation_ddof1": statistics.stdev(vals), "minimum": min(vals), "maximum": max(vals),
        "positive_count": sum(v > 0 for v in vals), "zero_count": sum(v == 0 for v in vals),
        "negative_count": sum(v < 0 for v in vals), "bootstrap_ci95": [lo, hi],
        "bootstrap_samples": bootstrap_samples, "exact_sign_flip_p_value": p,
        "assignments_enumerated": assignments, "seed_values": vals,
    }
    result["gate_passed"] = bool(result["mean"] > 0 and lo > 0 and p < 0.05
                                  and result["positive_count"] >= 9 and result["negative_count"] <= 3)
    return result
