"""Passive cognitive contracts for RNFE-CG-01 / ASCG v1.1 Phase 1."""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum
import math
import re
from typing import Any

from .canonical import canonical_sha256

CONTRACT_SCHEMA_VERSION = "rnfe-cg-contracts-v1"
LEDGER_SCHEMA_VERSION = "rnfe-cg-ledger-event-v1"
GENESIS_PREVIOUS_EVENT_HASH = "urn:rnfe:cg:genesis:v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ProducerType(str, Enum):
    DETERMINISTIC_ORACLE = "deterministic_oracle"
    LOCAL_7B = "local_7b"
    FRONTIER_TEACHER = "frontier_teacher"
    HUMAN_DEFINED = "human_defined"


class VerificationVerdict(str, Enum):
    VALID = "valid"
    INVALID = "invalid"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    CONTRADICTED = "contradicted"
    OUT_OF_SCOPE = "out_of_scope"
    QUARANTINED = "quarantined"


class LessonStatus(str, Enum):
    ACTIVE = "active"
    SUSPENDED = "suspended"
    SUPERSEDED = "superseded"
    REVOKED = "revoked"
    EXPIRED = "expired"
    QUARANTINED = "quarantined"


class InfluenceType(str, Enum):
    SCAR_BIAS = "scar_bias"
    VALIDATED_LESSON = "validated_lesson"


class CognitiveAdmissionVerdict(str, Enum):
    ADMITTED = "admitted"
    REJECTED = "rejected"
    QUARANTINED = "quarantined"


class LedgerEventType(str, Enum):
    EXPERIENCE_OBSERVED = "experience_observed"
    LESSON_CANDIDATE_GENERATED = "lesson_candidate_generated"
    LESSON_VERIFICATION_COMPLETED = "lesson_verification_completed"
    LESSON_ACTIVATED = "lesson_activated"
    LESSON_RETRIEVED = "lesson_retrieved"
    INFLUENCE_PROPOSED = "influence_proposed"
    INFLUENCE_ADMITTED = "influence_admitted"
    INFLUENCE_REJECTED = "influence_rejected"
    INTERVENTION_COMMITTED = "intervention_committed"
    OUTCOME_OBSERVED = "outcome_observed"
    EFFECT_EVALUATED = "effect_evaluated"
    LESSON_SUSPENDED = "lesson_suspended"
    LESSON_REVOKED = "lesson_revoked"


class LedgerWriteStatus(str, Enum):
    APPENDED = "appended"
    IDEMPOTENT_REPLAY = "idempotent_replay"
    DIVERGENCE_QUARANTINED = "divergence_quarantined"
    CHAIN_ALREADY_QUARANTINED = "chain_already_quarantined"
    REJECTED_INVALID_SEQUENCE = "rejected_invalid_sequence"
    REJECTED_INVALID_PREVIOUS_HASH = "rejected_invalid_previous_hash"
    REJECTED_INVALID_EVENT = "rejected_invalid_event"


class QuarantineReason(str, Enum):
    DIVERGENT_EVENT_ID = "divergent_event_id"
    INVALID_HASH_CHAIN = "invalid_hash_chain"
    INVALID_SEQUENCE = "invalid_sequence"
    INVALID_EVENT = "invalid_event"


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name}_required")
    return value.strip()


def _optional_text(value: Any, name: str) -> str | None:
    return None if value is None else _text(value, name)


def _hash(value: Any, name: str) -> str:
    value = _text(value, name)
    if not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{name}_must_be_sha256")
    return value


def _number(value: Any, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name}_must_be_finite")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name}_must_be_finite")
    value = 0.0 if value == 0.0 else value
    if minimum is not None and value < minimum:
        raise ValueError(f"{name}_must_be_gte_{minimum}")
    return value


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name}_must_be_integer_gte_{minimum}")
    return value


def _enum(value: Any, kind: type[Enum], name: str) -> None:
    if not isinstance(value, kind):
        raise ValueError(f"{name}_must_be_{kind.__name__}")


def _texts(value: Any, name: str, *, allow_empty: bool = False, unique: bool = False) -> tuple[str, ...]:
    if not isinstance(value, (tuple, list)):
        raise ValueError(f"{name}_must_be_sequence")
    value = tuple(_text(item, name) for item in value)
    if not allow_empty and not value:
        raise ValueError(f"{name}_required")
    if unique and len(set(value)) != len(value):
        raise ValueError(f"{name}_must_be_unique")
    return value


def _hashes(value: Any, name: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    if not isinstance(value, (tuple, list)):
        raise ValueError(f"{name}_must_be_sequence")
    value = tuple(_hash(item, name) for item in value)
    if not allow_empty and not value:
        raise ValueError(f"{name}_required")
    return value


def _json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return value.to_dict()
    return value


def contract_to_payload(value: Any, *, exclude: tuple[str, ...] = ()) -> dict[str, Any]:
    excluded = set(exclude)
    return {item.name: _json_value(getattr(value, item.name)) for item in fields(value) if item.name not in excluded}


def _derive(instance: Any, field_name: str) -> None:
    object.__setattr__(instance, field_name, canonical_sha256(contract_to_payload(instance, exclude=(field_name,))))


def _set_texts(instance: Any, names: tuple[str, ...]) -> None:
    for name in names:
        object.__setattr__(instance, name, _text(getattr(instance, name), name))


def _set_hashes(instance: Any, names: tuple[str, ...]) -> None:
    for name in names:
        object.__setattr__(instance, name, _hash(getattr(instance, name), name))


@dataclass(frozen=True, slots=True)
class CausalEpisodeEvidence:
    evidence_id: str
    experiment_id: str
    arm_id: str
    chain_id: str
    scenario_id: str
    scenario_version: str
    causal_signature_hash: str
    episode_id: str
    paired_run_id: str
    seed: int
    state_before_hash: str
    available_interventions: tuple[str, ...]
    baseline_intervention: str
    committed_intervention: str
    state_after_hash: str
    reward: float
    severity: float
    regret: float
    outcome_hash: str
    evidence_hash: str = ""

    def __post_init__(self) -> None:
        _set_texts(self, ("evidence_id", "experiment_id", "arm_id", "chain_id", "scenario_id", "scenario_version", "episode_id", "paired_run_id"))
        _set_hashes(self, ("causal_signature_hash", "state_before_hash", "state_after_hash", "outcome_hash"))
        object.__setattr__(self, "seed", _integer(self.seed, "seed"))
        actions = _texts(self.available_interventions, "available_interventions", unique=True)
        object.__setattr__(self, "available_interventions", actions)
        for name in ("baseline_intervention", "committed_intervention"):
            action = _text(getattr(self, name), name)
            if action not in actions:
                raise ValueError(f"{name}_not_available")
            object.__setattr__(self, name, action)
        object.__setattr__(self, "reward", _number(self.reward, "reward"))
        severity = _number(self.severity, "severity")
        if not 0.0 <= severity <= 1.0:
            raise ValueError("severity_out_of_range")
        object.__setattr__(self, "severity", severity)
        object.__setattr__(self, "regret", _number(self.regret, "regret", minimum=0.0))
        _derive(self, "evidence_hash")


@dataclass(frozen=True, slots=True)
class LessonCandidate:
    candidate_id: str
    producer_type: ProducerType
    producer_version: str
    source_evidence_ids: tuple[str, ...]
    applicability_conditions_hash: str
    preferred_intervention: str
    avoided_intervention: str | None
    proposition_hash: str
    supporting_evidence_hash: str
    expected_effect: float
    refutation_conditions_hash: str
    candidate_hash: str = ""

    def __post_init__(self) -> None:
        _set_texts(self, ("candidate_id", "producer_version", "preferred_intervention"))
        _enum(self.producer_type, ProducerType, "producer_type")
        object.__setattr__(self, "source_evidence_ids", _texts(self.source_evidence_ids, "source_evidence_ids", unique=True))
        object.__setattr__(self, "avoided_intervention", _optional_text(self.avoided_intervention, "avoided_intervention"))
        _set_hashes(self, ("applicability_conditions_hash", "proposition_hash", "supporting_evidence_hash", "refutation_conditions_hash"))
        object.__setattr__(self, "expected_effect", _number(self.expected_effect, "expected_effect"))
        _derive(self, "candidate_hash")


@dataclass(frozen=True, slots=True)
class VerificationRecord:
    verification_id: str
    candidate_id: str
    verifier_id: str
    verifier_version: str
    verification_method: str
    evidence_checked_ids: tuple[str, ...]
    counterfactuals_checked_hash: str
    verdict: VerificationVerdict
    scope_adjustments_hash: str
    rejection_reasons: tuple[str, ...]
    verification_hash: str = ""

    def __post_init__(self) -> None:
        _set_texts(self, ("verification_id", "candidate_id", "verifier_id", "verifier_version", "verification_method"))
        object.__setattr__(self, "evidence_checked_ids", _texts(self.evidence_checked_ids, "evidence_checked_ids", unique=True))
        _set_hashes(self, ("counterfactuals_checked_hash", "scope_adjustments_hash"))
        _enum(self.verdict, VerificationVerdict, "verdict")
        object.__setattr__(self, "rejection_reasons", _texts(self.rejection_reasons, "rejection_reasons", allow_empty=True))
        _derive(self, "verification_hash")


@dataclass(frozen=True, slots=True)
class ValidatedLesson:
    lesson_id: str
    candidate_id: str
    verification_id: str
    lesson_version: str
    status: LessonStatus
    applicability_scope_hash: str
    causal_signature_hash: str
    preferred_intervention: str
    avoided_intervention: str | None
    expected_effect: float
    confidence_bound: float
    authorized_consumers: tuple[str, ...]
    maximum_influence_budget: float
    expires_at_logical_time: int
    revocation_conditions_hash: str
    lesson_hash: str = ""

    def __post_init__(self) -> None:
        _set_texts(self, ("lesson_id", "candidate_id", "verification_id", "lesson_version", "preferred_intervention"))
        _enum(self.status, LessonStatus, "status")
        _set_hashes(self, ("applicability_scope_hash", "causal_signature_hash", "revocation_conditions_hash"))
        object.__setattr__(self, "avoided_intervention", _optional_text(self.avoided_intervention, "avoided_intervention"))
        object.__setattr__(self, "expected_effect", _number(self.expected_effect, "expected_effect"))
        confidence = _number(self.confidence_bound, "confidence_bound")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence_bound_out_of_range")
        object.__setattr__(self, "confidence_bound", confidence)
        object.__setattr__(self, "authorized_consumers", _texts(self.authorized_consumers, "authorized_consumers", unique=True))
        object.__setattr__(self, "maximum_influence_budget", _number(self.maximum_influence_budget, "maximum_influence_budget", minimum=0.0))
        object.__setattr__(self, "expires_at_logical_time", _integer(self.expires_at_logical_time, "expires_at_logical_time"))
        _derive(self, "lesson_hash")


@dataclass(frozen=True, slots=True)
class RetrievalDecision:
    retrieval_id: str
    episode_id: str
    query_context_hash: str
    eligible_lesson_ids: tuple[str, ...]
    retrieved_lesson_ids: tuple[str, ...]
    rejected_lesson_ids: tuple[str, ...]
    scope_checks_hash: str
    version_checks_hash: str
    retrieval_budget: float
    retrieval_hash: str = ""

    def __post_init__(self) -> None:
        _set_texts(self, ("retrieval_id", "episode_id"))
        _set_hashes(self, ("query_context_hash", "scope_checks_hash", "version_checks_hash"))
        for name in ("eligible_lesson_ids", "retrieved_lesson_ids", "rejected_lesson_ids"):
            object.__setattr__(self, name, _texts(getattr(self, name), name, allow_empty=True, unique=True))
        object.__setattr__(self, "retrieval_budget", _number(self.retrieval_budget, "retrieval_budget", minimum=0.0))
        _derive(self, "retrieval_hash")


@dataclass(frozen=True, slots=True)
class CognitiveInfluenceProposal:
    proposal_id: str
    influence_type: InfluenceType
    source_artifact_id: str
    original_intervention: str
    proposed_intervention: str
    evidence_hashes: tuple[str, ...]
    expected_effect: float
    requested_budget: float
    proposal_hash: str = ""

    def __post_init__(self) -> None:
        _set_texts(self, ("proposal_id", "source_artifact_id", "original_intervention", "proposed_intervention"))
        _enum(self.influence_type, InfluenceType, "influence_type")
        object.__setattr__(self, "evidence_hashes", _hashes(self.evidence_hashes, "evidence_hashes"))
        object.__setattr__(self, "expected_effect", _number(self.expected_effect, "expected_effect"))
        object.__setattr__(self, "requested_budget", _number(self.requested_budget, "requested_budget", minimum=0.0))
        _derive(self, "proposal_hash")


@dataclass(frozen=True, slots=True)
class CognitiveAdmissionDecision:
    decision_id: str
    proposal_id: str
    verdict: CognitiveAdmissionVerdict
    checks_executed: tuple[str, ...]
    reasons: tuple[str, ...]
    authorized_budget: float
    decision_hash: str = ""

    def __post_init__(self) -> None:
        _set_texts(self, ("decision_id", "proposal_id"))
        _enum(self.verdict, CognitiveAdmissionVerdict, "verdict")
        object.__setattr__(self, "checks_executed", _texts(self.checks_executed, "checks_executed", allow_empty=True))
        object.__setattr__(self, "reasons", _texts(self.reasons, "reasons", allow_empty=True))
        budget = _number(self.authorized_budget, "authorized_budget", minimum=0.0)
        if self.verdict is not CognitiveAdmissionVerdict.ADMITTED and budget != 0.0:
            raise ValueError("non_admitted_budget_must_be_zero")
        object.__setattr__(self, "authorized_budget", budget)
        _derive(self, "decision_hash")


@dataclass(frozen=True, slots=True)
class CausalInfluenceReceipt:
    receipt_id: str
    experiment_id: str
    arm_id: str
    chain_id: str
    episode_id: str
    source_artifact_id: str
    retrieval_id: str
    proposal_id: str
    cognitive_admission_decision_id: str
    baseline_intervention: str
    proposed_intervention: str
    committed_intervention: str
    action_changed: bool
    budget_consumed: float
    reversible: bool
    observed_effect_id: str | None
    receipt_hash: str = ""

    def __post_init__(self) -> None:
        _set_texts(self, ("receipt_id", "experiment_id", "arm_id", "chain_id", "episode_id", "source_artifact_id", "retrieval_id", "proposal_id", "cognitive_admission_decision_id", "baseline_intervention", "proposed_intervention", "committed_intervention"))
        if not isinstance(self.action_changed, bool) or not isinstance(self.reversible, bool):
            raise ValueError("receipt_boolean_invalid")
        if self.action_changed != (self.baseline_intervention != self.committed_intervention):
            raise ValueError("action_changed_inconsistent")
        object.__setattr__(self, "budget_consumed", _number(self.budget_consumed, "budget_consumed", minimum=0.0))
        object.__setattr__(self, "observed_effect_id", _optional_text(self.observed_effect_id, "observed_effect_id"))
        _derive(self, "receipt_hash")


@dataclass(frozen=True, slots=True)
class ObservedEffect:
    effect_id: str
    receipt_id: str
    episode_id: str
    paired_run_id: str
    state_after_hash: str
    reward: float
    severity: float
    regret: float
    constraint_violations: tuple[str, ...]
    resource_cost_hash: str
    latency: float
    effect_hash: str = ""

    def __post_init__(self) -> None:
        _set_texts(self, ("effect_id", "receipt_id", "episode_id", "paired_run_id"))
        _set_hashes(self, ("state_after_hash", "resource_cost_hash"))
        object.__setattr__(self, "reward", _number(self.reward, "reward"))
        severity = _number(self.severity, "severity")
        if not 0.0 <= severity <= 1.0:
            raise ValueError("severity_out_of_range")
        object.__setattr__(self, "severity", severity)
        object.__setattr__(self, "regret", _number(self.regret, "regret", minimum=0.0))
        object.__setattr__(self, "constraint_violations", _texts(self.constraint_violations, "constraint_violations", allow_empty=True))
        object.__setattr__(self, "latency", _number(self.latency, "latency", minimum=0.0))
        _derive(self, "effect_hash")


@dataclass(frozen=True, slots=True)
class CausalLedgerEvent:
    event_id: str
    event_type: LedgerEventType
    experiment_id: str
    arm_id: str
    chain_id: str
    event_sequence: int
    previous_event_hash: str
    logical_time: int
    wall_clock_time: str
    payload_contract_type: str
    payload_hash: str
    event_hash: str = ""
    schema_version: str = LEDGER_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != LEDGER_SCHEMA_VERSION:
            raise ValueError("ledger_schema_version_mismatch")
        _set_texts(self, ("event_id", "experiment_id", "arm_id", "chain_id", "wall_clock_time", "payload_contract_type"))
        _enum(self.event_type, LedgerEventType, "event_type")
        object.__setattr__(self, "event_sequence", _integer(self.event_sequence, "event_sequence"))
        object.__setattr__(self, "logical_time", _integer(self.logical_time, "logical_time"))
        object.__setattr__(self, "payload_hash", _hash(self.payload_hash, "payload_hash"))
        if self.event_sequence == 0:
            if self.logical_time != 0:
                raise ValueError("genesis_logical_time_must_be_zero")
            if self.previous_event_hash != GENESIS_PREVIOUS_EVENT_HASH:
                raise ValueError("genesis_previous_hash_invalid")
        else:
            object.__setattr__(self, "previous_event_hash", _hash(self.previous_event_hash, "previous_event_hash"))
            if self.logical_time == 0:
                raise ValueError("non_genesis_logical_time_must_be_positive")
        object.__setattr__(
            self,
            "event_hash",
            canonical_sha256(
                contract_to_payload(
                    self,
                    exclude=("event_hash", "wall_clock_time"),
                )
            ),
        )

    @classmethod
    def create(cls, *, payload: Any, **values: Any) -> "CausalLedgerEvent":
        return cls(payload_hash=canonical_sha256(payload), **values)


@dataclass(frozen=True, slots=True)
class ExperimentalQuarantineRecord:
    quarantine_id: str
    experiment_id: str
    arm_id: str
    chain_id: str
    reason: QuarantineReason
    conflicting_event_id: str
    stored_payload_hash: str
    incoming_payload_hash: str
    observed_sequence: int
    evidence_hash: str
    logical_time: int
    quarantine_hash: str = ""

    def __post_init__(self) -> None:
        _set_texts(self, ("quarantine_id", "experiment_id", "arm_id", "chain_id", "conflicting_event_id"))
        _enum(self.reason, QuarantineReason, "reason")
        _set_hashes(self, ("stored_payload_hash", "incoming_payload_hash", "evidence_hash"))
        object.__setattr__(self, "observed_sequence", _integer(self.observed_sequence, "observed_sequence"))
        object.__setattr__(self, "logical_time", _integer(self.logical_time, "logical_time"))
        _derive(self, "quarantine_hash")


@dataclass(frozen=True, slots=True)
class LedgerChainState:
    experiment_id: str
    arm_id: str
    chain_id: str
    head_sequence: int
    head_event_hash: str
    head_logical_time: int
    quarantined: bool = False
    quarantine_id: str | None = None

    def __post_init__(self) -> None:
        _set_texts(self, ("experiment_id", "arm_id", "chain_id"))
        if isinstance(self.head_sequence, bool) or not isinstance(self.head_sequence, int) or self.head_sequence < -1:
            raise ValueError("head_sequence_invalid")
        if isinstance(self.head_logical_time, bool) or not isinstance(self.head_logical_time, int) or self.head_logical_time < -1:
            raise ValueError("head_logical_time_invalid")
        if self.head_sequence == -1:
            if self.head_event_hash != GENESIS_PREVIOUS_EVENT_HASH or self.head_logical_time != -1:
                raise ValueError("empty_chain_state_invalid")
        else:
            object.__setattr__(self, "head_event_hash", _hash(self.head_event_hash, "head_event_hash"))
        if not isinstance(self.quarantined, bool):
            raise ValueError("quarantined_must_be_bool")
        object.__setattr__(self, "quarantine_id", _optional_text(self.quarantine_id, "quarantine_id"))
        if self.quarantined != (self.quarantine_id is not None):
            raise ValueError("quarantine_state_inconsistent")


@dataclass(frozen=True, slots=True)
class LedgerWriteResult:
    status: LedgerWriteStatus
    stored_event: CausalLedgerEvent | None
    chain_state: LedgerChainState | None
    quarantine_record: ExperimentalQuarantineRecord | None
    detail: str = ""

    def __post_init__(self) -> None:
        _enum(self.status, LedgerWriteStatus, "status")
        if self.stored_event is not None and not isinstance(self.stored_event, CausalLedgerEvent):
            raise ValueError("stored_event_invalid")
        if self.chain_state is not None and not isinstance(self.chain_state, LedgerChainState):
            raise ValueError("chain_state_invalid")
        if self.quarantine_record is not None and not isinstance(self.quarantine_record, ExperimentalQuarantineRecord):
            raise ValueError("quarantine_record_invalid")
        if not isinstance(self.detail, str):
            raise ValueError("detail_must_be_string")


__all__ = [
    "CONTRACT_SCHEMA_VERSION", "LEDGER_SCHEMA_VERSION", "GENESIS_PREVIOUS_EVENT_HASH",
    "ProducerType", "VerificationVerdict", "LessonStatus", "InfluenceType",
    "CognitiveAdmissionVerdict", "LedgerEventType", "LedgerWriteStatus",
    "QuarantineReason", "CausalEpisodeEvidence", "LessonCandidate",
    "VerificationRecord", "ValidatedLesson", "RetrievalDecision",
    "CognitiveInfluenceProposal", "CognitiveAdmissionDecision",
    "CausalInfluenceReceipt", "ObservedEffect", "CausalLedgerEvent",
    "ExperimentalQuarantineRecord", "LedgerChainState", "LedgerWriteResult",
    "contract_to_payload",
]
