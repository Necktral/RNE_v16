"""W0-R3 seven-arm causal treatment kernel.

The kernel executes one already-snapshotted experimental unit.  It never
derives N3 again, never changes canonical retrieval, and opens the scenario
oracle only after a decision seal has been produced.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from runtime.neural.integration.contracts import canonical_json_bytes, canonical_sha256
from runtime.neural.integration.n3_scoring import n3_adjusted_score
from runtime.neural.integration.p2_arm_isolation import (
    ArmExecutionContext,
    UnitStateSnapshot,
    dispose_arm_context,
    instantiate_arm_context,
    verify_arm_prestate,
)
from runtime.reasoning.families.core_inference import induce
from runtime.world.intervention_override import outcome_effectiveness


FACTORIAL_ARMS = ("C", "R-S", "R-M", "R-SM", "T-S", "T-M", "T-SM")
FACTORIAL_SCHEMA_VERSION = "p2-factorial-unit-v1"

_ARM_FACTORS: Mapping[str, tuple[str, str, str]] = {
    "C": ("canonical", "canonical", "none"),
    "R-S": ("canonical", "n3", "reference"),
    "R-M": ("n3", "canonical", "reference"),
    "R-SM": ("n3", "n3", "reference"),
    "T-S": ("canonical", "n3", "trained"),
    "T-M": ("n3", "canonical", "trained"),
    "T-SM": ("n3", "n3", "trained"),
}


@dataclass(frozen=True, slots=True)
class ArmPlan:
    arm_id: str
    membership_factor: str
    sequence_factor: str
    backend_factor: str
    mechanism_probe: str | None
    execution_position: int


@dataclass(frozen=True, slots=True)
class MembershipTreatmentReceipt:
    eligible: bool
    delivered: bool
    ineligibility_reason: str | None
    swap_in_id: str | None
    swap_out_id: str | None
    swap_margin: float | None
    membership_before_hash: str
    membership_after_hash: str
    membership_before_ids: tuple[str, ...]
    membership_after_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SequenceTreatmentReceipt:
    delivered: bool
    sequence_before_hash: str
    sequence_after_hash: str
    sequence_before_ids: tuple[str, ...]
    sequence_after_ids: tuple[str, ...]
    kendall_tau: float
    spearman_distance: int
    inversion_count: int
    top1_changed: bool


@dataclass(frozen=True, slots=True)
class DecisionSeal:
    chosen_action: str
    decision_order: int
    decision_hash: str
    oracle_unopened: bool
    allowed_action_validated: bool
    ordered_memory_hash: str


@dataclass(frozen=True, slots=True)
class OracleReceipt:
    decision_hash: str
    oracle_order: int
    decision_already_sealed: bool
    opened_after_seal: bool
    chosen_utility: float
    optimal_utility: float
    optimal_action: str
    regret: float
    outcome_set_hash: str


@dataclass(frozen=True, slots=True)
class FactorialArmReceipt:
    schema_version: str
    campaign_id: str
    unit_id: str
    arm_plan: ArmPlan
    unit_snapshot_hash: str
    raw_pool_hash: str
    backend_output_hash: str | None
    adjusted_scores: tuple[tuple[str, float], ...]
    membership: MembershipTreatmentReceipt
    sequence: SequenceTreatmentReceipt
    decision_seal: DecisionSeal
    oracle: OracleReceipt
    authority_effect: str = "none"
    training_executed: bool = False
    shared_state_writes: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FactorialArmReceipt":
        plan = dict(value["arm_plan"])
        membership = dict(value["membership"])
        sequence = dict(value["sequence"])
        seal = dict(value["decision_seal"])
        oracle = dict(value["oracle"])
        receipt = cls(
            schema_version=str(value["schema_version"]),
            campaign_id=str(value["campaign_id"]),
            unit_id=str(value["unit_id"]),
            arm_plan=ArmPlan(**plan),
            unit_snapshot_hash=str(value["unit_snapshot_hash"]),
            raw_pool_hash=str(value["raw_pool_hash"]),
            backend_output_hash=(
                str(value["backend_output_hash"])
                if value.get("backend_output_hash") is not None
                else None
            ),
            adjusted_scores=tuple(
                (str(item[0]), float(item[1])) for item in value["adjusted_scores"]
            ),
            membership=MembershipTreatmentReceipt(
                **{
                    **membership,
                    "membership_before_ids": tuple(
                        membership["membership_before_ids"]
                    ),
                    "membership_after_ids": tuple(
                        membership["membership_after_ids"]
                    ),
                }
            ),
            sequence=SequenceTreatmentReceipt(
                **{
                    **sequence,
                    "sequence_before_ids": tuple(sequence["sequence_before_ids"]),
                    "sequence_after_ids": tuple(sequence["sequence_after_ids"]),
                }
            ),
            decision_seal=DecisionSeal(**seal),
            oracle=OracleReceipt(**oracle),
            authority_effect=str(value.get("authority_effect") or "none"),
            training_executed=bool(value.get("training_executed", False)),
            shared_state_writes=int(value.get("shared_state_writes", 0)),
        )
        return receipt

    @property
    def receipt_hash(self) -> str:
        return canonical_sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class GoldenUnitArtifact:
    schema_version: str
    campaign_id: str
    unit_id: str
    snapshot_hash: str
    top_k: int
    canonical_arm_order: tuple[str, ...]
    executed_arm_order: tuple[str, ...]
    receipts: tuple[FactorialArmReceipt, ...]
    artifact_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "campaign_id": self.campaign_id,
            "unit_id": self.unit_id,
            "snapshot_hash": self.snapshot_hash,
            "top_k": self.top_k,
            "canonical_arm_order": list(self.canonical_arm_order),
            "executed_arm_order": list(self.executed_arm_order),
            "receipts": [receipt.to_dict() for receipt in self.receipts],
            "receipt_hashes": [receipt.receipt_hash for receipt in self.receipts],
            "artifact_hash": self.artifact_hash,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GoldenUnitArtifact":
        artifact = cls(
            schema_version=str(value["schema_version"]),
            campaign_id=str(value["campaign_id"]),
            unit_id=str(value["unit_id"]),
            snapshot_hash=str(value["snapshot_hash"]),
            top_k=int(value["top_k"]),
            canonical_arm_order=tuple(value["canonical_arm_order"]),
            executed_arm_order=tuple(value["executed_arm_order"]),
            receipts=tuple(
                FactorialArmReceipt.from_dict(item) for item in value["receipts"]
            ),
            artifact_hash=str(value["artifact_hash"]),
        )
        verify_golden_unit_artifact(artifact)
        return artifact


def _candidate_id(candidate: Mapping[str, Any]) -> str:
    value = str(candidate.get("memory_id") or "")
    if not value:
        raise ValueError("p2_factorial_candidate_id_required")
    return value


def _canonical_score(candidate: Mapping[str, Any]) -> float:
    raw = candidate.get("canonical_score", candidate.get("score"))
    score = float(raw)
    if not math.isfinite(score):
        raise ValueError("p2_factorial_canonical_score_nonfinite")
    return score


def _canonical_rank(candidate: Mapping[str, Any], fallback: int) -> int:
    rank = int(candidate.get("canonical_rank", fallback))
    if rank < 0:
        raise ValueError("p2_factorial_canonical_rank_invalid")
    return rank


def _normalize_candidate(
    candidate: Mapping[str, Any], *, fallback_rank: int
) -> dict[str, Any]:
    structure = candidate.get("structure")
    if not isinstance(structure, Mapping):
        structure_json = candidate.get("structure_json")
        if not isinstance(structure_json, str):
            raise ValueError("p2_factorial_candidate_structure_required")
        structure = json.loads(structure_json)
    normalized = {
        "memory_id": _candidate_id(candidate),
        "scale": str(candidate.get("scale") or ""),
        "score": _canonical_score(candidate),
        "canonical_score": _canonical_score(candidate),
        "canonical_rank": _canonical_rank(candidate, fallback_rank),
        "structure": dict(structure),
    }
    if normalized["scale"] not in {"micro", "meso", "macro"}:
        raise ValueError("p2_factorial_candidate_scale_invalid")
    return normalized


def _raw_pool(snapshot: UnitStateSnapshot) -> tuple[dict[str, Any], ...]:
    raw = snapshot.payload["canonical_scored_pool"]
    candidates = raw.get("candidates") if isinstance(raw, Mapping) else None
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("p2_factorial_raw_pool_required")
    normalized = tuple(
        _normalize_candidate(item, fallback_rank=index)
        for index, item in enumerate(candidates)
        if isinstance(item, Mapping)
    )
    ids = [_candidate_id(item) for item in normalized]
    ranks = [_canonical_rank(item, index) for index, item in enumerate(normalized)]
    if (
        len(normalized) != len(candidates)
        or len(ids) != len(set(ids))
        or len(ranks) != len(set(ranks))
        or ranks != sorted(ranks)
    ):
        raise ValueError("p2_factorial_raw_pool_invalid")
    return normalized


def _signals(snapshot: UnitStateSnapshot, backend: str) -> tuple[dict[str, float], str]:
    outputs = snapshot.payload["backend_outputs"]
    if backend == "reference":
        raw = outputs["reference"]
    elif backend == "trained":
        raw = outputs["trained"]
    else:
        raise ValueError("p2_factorial_backend_invalid")
    source = raw.get("scale_signals") if isinstance(raw, Mapping) else None
    if source is None and isinstance(raw, Mapping):
        source = {
            key: raw[key]
            for key in ("micro", "meso", "macro")
            if key in raw
        }
    if not isinstance(source, Mapping) or set(source) != {"micro", "meso", "macro"}:
        raise ValueError(f"p2_factorial_scale_signals_missing:{backend}")
    signals = {key: float(source[key]) for key in ("micro", "meso", "macro")}
    if any(not math.isfinite(value) for value in signals.values()):
        raise ValueError("p2_factorial_scale_signal_nonfinite")
    return signals, canonical_sha256(raw)


def _adjusted_score(candidate: Mapping[str, Any], signals: Mapping[str, float]) -> float:
    return n3_adjusted_score(candidate, signals)


def _membership_treatment(
    *,
    raw_pool: Sequence[Mapping[str, Any]],
    top_k: int,
    signals: Mapping[str, float] | None,
    apply_membership: bool,
) -> tuple[tuple[dict[str, Any], ...], MembershipTreatmentReceipt]:
    canonical = tuple(dict(item) for item in raw_pool[:top_k])
    before_ids = tuple(_candidate_id(item) for item in canonical)
    before_hash = canonical_sha256(sorted(before_ids))
    if not apply_membership:
        return canonical, MembershipTreatmentReceipt(
            eligible=False,
            delivered=False,
            ineligibility_reason="factor_disabled",
            swap_in_id=None,
            swap_out_id=None,
            swap_margin=None,
            membership_before_hash=before_hash,
            membership_after_hash=before_hash,
            membership_before_ids=before_ids,
            membership_after_ids=before_ids,
        )
    if signals is None:
        raise ValueError("p2_factorial_membership_signals_required")
    outside = tuple(dict(item) for item in raw_pool[top_k:])
    if not outside:
        return canonical, MembershipTreatmentReceipt(
            eligible=False,
            delivered=False,
            ineligibility_reason="no_outside_candidate",
            swap_in_id=None,
            swap_out_id=None,
            swap_margin=None,
            membership_before_hash=before_hash,
            membership_after_hash=before_hash,
            membership_before_ids=before_ids,
            membership_after_ids=before_ids,
        )
    swap_in = min(
        outside,
        key=lambda item: (
            -_adjusted_score(item, signals),
            _canonical_rank(item, 0),
        ),
    )
    swap_out = min(
        canonical,
        key=lambda item: (
            _adjusted_score(item, signals),
            -_canonical_rank(item, 0),
        ),
    )
    margin = _adjusted_score(swap_in, signals) - _adjusted_score(swap_out, signals)
    eligible = margin > 0.0
    if not eligible:
        return canonical, MembershipTreatmentReceipt(
            eligible=False,
            delivered=False,
            ineligibility_reason="nonpositive_swap_margin",
            swap_in_id=_candidate_id(swap_in),
            swap_out_id=_candidate_id(swap_out),
            swap_margin=margin,
            membership_before_hash=before_hash,
            membership_after_hash=before_hash,
            membership_before_ids=before_ids,
            membership_after_ids=before_ids,
        )
    treated = tuple(
        dict(swap_in) if _candidate_id(item) == _candidate_id(swap_out) else dict(item)
        for item in canonical
    )
    after_ids = tuple(_candidate_id(item) for item in treated)
    after_hash = canonical_sha256(sorted(after_ids))
    if (
        len(set(before_ids).difference(after_ids)) != 1
        or len(set(after_ids).difference(before_ids)) != 1
    ):
        raise RuntimeError("p2_factorial_membership_not_exactly_one_swap")
    return treated, MembershipTreatmentReceipt(
        eligible=True,
        delivered=True,
        ineligibility_reason=None,
        swap_in_id=_candidate_id(swap_in),
        swap_out_id=_candidate_id(swap_out),
        swap_margin=margin,
        membership_before_hash=before_hash,
        membership_after_hash=after_hash,
        membership_before_ids=before_ids,
        membership_after_ids=after_ids,
    )


def _sequence_metrics(
    before: Sequence[str], after: Sequence[str]
) -> tuple[float, int, int]:
    if set(before) != set(after) or len(before) != len(after):
        raise ValueError("p2_factorial_sequence_membership_changed")
    positions = {memory_id: index for index, memory_id in enumerate(before)}
    permutation = [positions[memory_id] for memory_id in after]
    inversions = sum(
        permutation[left] > permutation[right]
        for left in range(len(permutation))
        for right in range(left + 1, len(permutation))
    )
    pairs = len(permutation) * (len(permutation) - 1) // 2
    tau = 1.0 if pairs == 0 else 1.0 - (2.0 * inversions / pairs)
    after_positions = {memory_id: index for index, memory_id in enumerate(after)}
    distance = sum(
        abs(index - after_positions[memory_id])
        for index, memory_id in enumerate(before)
    )
    return tau, distance, inversions


def _sequence_treatment(
    *,
    membership: Sequence[Mapping[str, Any]],
    signals: Mapping[str, float] | None,
    apply_sequence: bool,
) -> tuple[tuple[dict[str, Any], ...], SequenceTreatmentReceipt]:
    canonical = tuple(
        sorted(
            (dict(item) for item in membership),
            key=lambda item: _canonical_rank(item, 0),
        )
    )
    before_ids = tuple(_candidate_id(item) for item in canonical)
    if apply_sequence:
        if signals is None:
            raise ValueError("p2_factorial_sequence_signals_required")
        ordered = tuple(
            sorted(
                (dict(item) for item in canonical),
                key=lambda item: (
                    -_adjusted_score(item, signals),
                    _canonical_rank(item, 0),
                ),
            )
        )
    else:
        ordered = canonical
    after_ids = tuple(_candidate_id(item) for item in ordered)
    tau, distance, inversions = _sequence_metrics(before_ids, after_ids)
    delivered = before_ids != after_ids
    return ordered, SequenceTreatmentReceipt(
        delivered=delivered,
        sequence_before_hash=canonical_sha256(list(before_ids)),
        sequence_after_hash=canonical_sha256(list(after_ids)),
        sequence_before_ids=before_ids,
        sequence_after_ids=after_ids,
        kendall_tau=tau,
        spearman_distance=distance,
        inversion_count=inversions,
        top1_changed=bool(before_ids and before_ids[0] != after_ids[0]),
    )


def _decision_seal(
    *,
    context: ArmExecutionContext,
    arm_id: str,
    ordered: Sequence[Mapping[str, Any]],
    allowed_actions: Sequence[str],
) -> DecisionSeal:
    observation = context.scenario.observe()
    state = {
        "observation": {
            **dict(observation.state),
            "alarm": observation.alarm,
            "propositions": list(observation.propositions),
        },
        "retrieved_memory": [dict(item) for item in ordered],
        "scenario_metadata": {
            "scenario_name": context.scenario.config.name,
            "main_variable": context.scenario.config.main_variable,
            "alarm_threshold": context.scenario.config.alarm_threshold,
            "interventions": list(context.scenario.config.interventions),
            "optimization_direction": (
                context.scenario.causal_signature.optimization_direction
            ),
            "causal_signature": context.scenario.causal_signature,
        },
    }
    recommendation = induce(state)["state_delta"].get("ind_best_intervention")
    chosen = str(recommendation or context.scenario.select_intervention(observation))
    allowed = chosen in allowed_actions
    if not allowed:
        raise ValueError("p2_factorial_action_not_allowed")
    ordered_ids = tuple(_candidate_id(item) for item in ordered)
    ordered_hash = canonical_sha256(list(ordered_ids))
    payload = {
        "arm_id": arm_id,
        "unit_snapshot_hash": context.snapshot_hash,
        "ordered_memory_hash": ordered_hash,
        "chosen_action": chosen,
        "oracle_unopened": True,
        "allowed_action_validated": True,
        "decision_order": 1,
    }
    decision_hash = canonical_sha256(payload)
    context.scenario.seal_preaction_decision(decision_hash)
    return DecisionSeal(
        chosen_action=chosen,
        decision_order=1,
        decision_hash=decision_hash,
        oracle_unopened=True,
        allowed_action_validated=True,
        ordered_memory_hash=ordered_hash,
    )


def _oracle_receipt(
    *,
    context: ArmExecutionContext,
    seal: DecisionSeal,
    allowed_actions: Sequence[str],
    external_input: float,
) -> OracleReceipt:
    signature = context.scenario.causal_signature
    utilities: dict[str, float] = {}
    outcomes: list[dict[str, Any]] = []
    for intervention in allowed_actions:
        transition = context.scenario.simulate_counterfactual(
            intervention=intervention,
            external_input=external_input,
        )
        value = float(transition.state[context.scenario.config.main_variable])
        utility = outcome_effectiveness(
            value=value,
            alarm_threshold=context.scenario.config.alarm_threshold,
            alarm_semantics=signature.alarm_semantics,
        )
        if not math.isfinite(value) or not math.isfinite(utility):
            raise ValueError("p2_factorial_oracle_nonfinite")
        utilities[intervention] = utility
        outcomes.append(
            {"intervention": intervention, "value": value, "utility": utility}
        )
    optimal = max(
        allowed_actions,
        key=lambda item: (utilities[item], -allowed_actions.index(item)),
    )
    regret = utilities[optimal] - utilities[seal.chosen_action]
    if not math.isfinite(regret) or regret < -1e-12:
        raise ValueError("p2_factorial_regret_invalid")
    return OracleReceipt(
        decision_hash=seal.decision_hash,
        oracle_order=2,
        decision_already_sealed=True,
        opened_after_seal=True,
        chosen_utility=utilities[seal.chosen_action],
        optimal_utility=utilities[optimal],
        optimal_action=optimal,
        regret=max(0.0, regret),
        outcome_set_hash=canonical_sha256(outcomes),
    )


def _execute_arm(
    *,
    snapshot: UnitStateSnapshot,
    campaign_id: str,
    arm_id: str,
    execution_position: int,
    top_k: int,
    raw_pool: Sequence[Mapping[str, Any]],
    raw_pool_hash: str,
) -> FactorialArmReceipt:
    membership_factor, sequence_factor, backend_factor = _ARM_FACTORS[arm_id]
    context = instantiate_arm_context(snapshot, arm_id)
    try:
        signals: dict[str, float] | None = None
        backend_hash: str | None = None
        if backend_factor != "none":
            signals, backend_hash = _signals(snapshot, backend_factor)
        membership, membership_receipt = _membership_treatment(
            raw_pool=raw_pool,
            top_k=top_k,
            signals=signals,
            apply_membership=membership_factor == "n3",
        )
        ordered, sequence_receipt = _sequence_treatment(
            membership=membership,
            signals=signals,
            apply_sequence=sequence_factor == "n3",
        )
        seal = _decision_seal(
            context=context,
            arm_id=arm_id,
            ordered=ordered,
            allowed_actions=snapshot.payload["allowed_actions"],
        )
        oracle = _oracle_receipt(
            context=context,
            seal=seal,
            allowed_actions=snapshot.payload["allowed_actions"],
            external_input=float(snapshot.payload["external_input"]),
        )
        verify_arm_prestate(snapshot, context)
        scores = tuple(
            (
                _candidate_id(item),
                (
                    _canonical_score(item)
                    if signals is None
                    else _adjusted_score(item, signals)
                ),
            )
            for item in raw_pool
        )
        receipt = FactorialArmReceipt(
            schema_version=FACTORIAL_SCHEMA_VERSION,
            campaign_id=str(campaign_id),
            unit_id=snapshot.unit_id,
            arm_plan=ArmPlan(
                arm_id=arm_id,
                membership_factor=membership_factor,
                sequence_factor=sequence_factor,
                backend_factor=backend_factor,
                mechanism_probe=None,
                execution_position=execution_position,
            ),
            unit_snapshot_hash=snapshot.snapshot_hash,
            raw_pool_hash=raw_pool_hash,
            backend_output_hash=backend_hash,
            adjusted_scores=scores,
            membership=membership_receipt,
            sequence=sequence_receipt,
            decision_seal=seal,
            oracle=oracle,
        )
        verify_factorial_receipt(receipt, raw_pool_ids=tuple(_candidate_id(item) for item in raw_pool))
        return receipt
    finally:
        dispose_arm_context(context)


def verify_factorial_receipt(
    receipt: FactorialArmReceipt, *, raw_pool_ids: Sequence[str]
) -> bool:
    if receipt.schema_version != FACTORIAL_SCHEMA_VERSION:
        raise ValueError("p2_factorial_receipt_schema_invalid")
    if receipt.arm_plan.arm_id not in FACTORIAL_ARMS:
        raise ValueError("p2_factorial_receipt_arm_invalid")
    expected_factors = _ARM_FACTORS[receipt.arm_plan.arm_id]
    observed_factors = (
        receipt.arm_plan.membership_factor,
        receipt.arm_plan.sequence_factor,
        receipt.arm_plan.backend_factor,
    )
    if observed_factors != expected_factors:
        raise ValueError("p2_factorial_receipt_factors_invalid")
    before = set(receipt.membership.membership_before_ids)
    after = set(receipt.membership.membership_after_ids)
    if not before.issubset(raw_pool_ids) or not after.issubset(raw_pool_ids):
        raise ValueError("p2_factorial_receipt_candidate_invented")
    if receipt.membership.delivered:
        if len(before - after) != 1 or len(after - before) != 1:
            raise ValueError("p2_factorial_receipt_membership_delivery_invalid")
    elif before != after:
        raise ValueError("p2_factorial_receipt_membership_leakage")
    if set(receipt.sequence.sequence_before_ids) != set(
        receipt.sequence.sequence_after_ids
    ):
        raise ValueError("p2_factorial_receipt_sequence_membership_leakage")
    if receipt.arm_plan.sequence_factor == "canonical" and receipt.sequence.delivered:
        raise ValueError("p2_factorial_receipt_sequence_factor_leakage")
    if receipt.oracle.decision_hash != receipt.decision_seal.decision_hash:
        raise ValueError("p2_factorial_receipt_decision_oracle_hash_mismatch")
    if (
        not receipt.decision_seal.oracle_unopened
        or receipt.decision_seal.decision_order >= receipt.oracle.oracle_order
        or not receipt.oracle.decision_already_sealed
        or not receipt.oracle.opened_after_seal
    ):
        raise ValueError("p2_factorial_receipt_oracle_order_invalid")
    canonical_json_bytes(receipt.to_dict())
    return True


def execute_factorial_unit(
    *,
    snapshot: UnitStateSnapshot,
    campaign_id: str,
    top_k: int,
    arm_order: Sequence[str] = FACTORIAL_ARMS,
) -> GoldenUnitArtifact:
    if not str(campaign_id).strip():
        raise ValueError("p2_factorial_campaign_id_required")
    if tuple(sorted(arm_order)) != tuple(sorted(FACTORIAL_ARMS)):
        raise ValueError("p2_factorial_arm_order_must_be_complete")
    raw_pool = _raw_pool(snapshot)
    if top_k <= 0 or top_k > len(raw_pool):
        raise ValueError("p2_factorial_top_k_invalid")
    raw_pool_hash = canonical_sha256(list(raw_pool))
    by_arm: dict[str, FactorialArmReceipt] = {}
    for position, arm_id in enumerate(arm_order):
        by_arm[arm_id] = _execute_arm(
            snapshot=snapshot,
            campaign_id=campaign_id,
            arm_id=arm_id,
            execution_position=position,
            top_k=top_k,
            raw_pool=raw_pool,
            raw_pool_hash=raw_pool_hash,
        )
    receipts = tuple(by_arm[arm_id] for arm_id in FACTORIAL_ARMS)
    payload = {
        "schema_version": FACTORIAL_SCHEMA_VERSION,
        "campaign_id": str(campaign_id),
        "unit_id": snapshot.unit_id,
        "snapshot_hash": snapshot.snapshot_hash,
        "top_k": int(top_k),
        "canonical_arm_order": list(FACTORIAL_ARMS),
        "executed_arm_order": list(arm_order),
        "receipts": [receipt.to_dict() for receipt in receipts],
    }
    artifact = GoldenUnitArtifact(
        schema_version=FACTORIAL_SCHEMA_VERSION,
        campaign_id=str(campaign_id),
        unit_id=snapshot.unit_id,
        snapshot_hash=snapshot.snapshot_hash,
        top_k=int(top_k),
        canonical_arm_order=FACTORIAL_ARMS,
        executed_arm_order=tuple(arm_order),
        receipts=receipts,
        artifact_hash=canonical_sha256(payload),
    )
    verify_golden_unit_artifact(artifact)
    return artifact


def verify_golden_unit_artifact(artifact: GoldenUnitArtifact) -> bool:
    if tuple(receipt.arm_plan.arm_id for receipt in artifact.receipts) != FACTORIAL_ARMS:
        raise ValueError("p2_factorial_golden_arms_invalid")
    for receipt in artifact.receipts:
        verify_factorial_receipt(
            receipt,
            raw_pool_ids=tuple(memory_id for memory_id, _ in receipt.adjusted_scores),
        )
    payload = {
        "schema_version": artifact.schema_version,
        "campaign_id": artifact.campaign_id,
        "unit_id": artifact.unit_id,
        "snapshot_hash": artifact.snapshot_hash,
        "top_k": artifact.top_k,
        "canonical_arm_order": list(artifact.canonical_arm_order),
        "executed_arm_order": list(artifact.executed_arm_order),
        "receipts": [receipt.to_dict() for receipt in artifact.receipts],
    }
    if canonical_sha256(payload) != artifact.artifact_hash:
        raise ValueError("p2_factorial_golden_hash_invalid")
    return True


def write_golden_unit_artifact(path: str | Path, artifact: GoldenUnitArtifact) -> str:
    """Persist one deterministic unit artifact; this is not a campaign runner."""

    verify_golden_unit_artifact(artifact)
    target = Path(path)
    target.write_bytes(canonical_json_bytes(artifact.to_dict()) + b"\n")
    return artifact.artifact_hash
