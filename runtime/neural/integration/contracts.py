"""Contratos internos versionados de la integración simbiótica neuronal."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping


SYMBIOSIS_TRACE_SCHEMA_VERSION_V1 = "neural-symbiosis-trace-v1"
SYMBIOSIS_TRACE_SCHEMA_VERSION = "neural-symbiosis-trace-v2"
CONSUMER_RECEIPT_SCHEMA_VERSION = "neural-consumer-receipt-v1"


class AuthorityEffect(str, Enum):
    NONE = "none"
    EVIDENCE_ONLY = "evidence_only"
    BOUNDED_PROPOSAL = "bounded_proposal"
    AUTHORITATIVE = "authoritative"


_AUTHORITY_RANK = {
    AuthorityEffect.NONE: 0,
    AuthorityEffect.EVIDENCE_ONLY: 1,
    AuthorityEffect.BOUNDED_PROPOSAL: 2,
    AuthorityEffect.AUTHORITATIVE: 3,
}


CONSUMER_AUTHORITY_CEILINGS: Mapping[tuple[str, str], AuthorityEffect] = {
    ("N1", "scheduler_comparison"): AuthorityEffect.EVIDENCE_ONLY,
    ("N1", "delayed_outcome_observer"): AuthorityEffect.EVIDENCE_ONLY,
    ("N2", "ded_verifier"): AuthorityEffect.EVIDENCE_ONLY,
    ("N2", "lotf_verifier"): AuthorityEffect.EVIDENCE_ONLY,
    ("N2", "nesy_verifier"): AuthorityEffect.EVIDENCE_ONLY,
    ("N3", "next_episode_state"): AuthorityEffect.EVIDENCE_ONLY,
    ("N3", "checkpoint_continuity"): AuthorityEffect.EVIDENCE_ONLY,
    ("N4", "canonical_causal_comparator"): AuthorityEffect.EVIDENCE_ONLY,
    ("N4", "certification_metadata"): AuthorityEffect.EVIDENCE_ONLY,
    ("N5", "smg_write_result"): AuthorityEffect.EVIDENCE_ONLY,
    ("N5", "mfm_candidate_gate"): AuthorityEffect.EVIDENCE_ONLY,
    ("N6", "sandbox"): AuthorityEffect.EVIDENCE_ONLY,
    ("N6", "certification"): AuthorityEffect.EVIDENCE_ONLY,
    ("N6", "autoevolution_evidence_observer"): AuthorityEffect.EVIDENCE_ONLY,
}
REQUIRED_CONSUMERS: Mapping[str, frozenset[str]] = {
    organ: frozenset(
        consumer for receipt_organ, consumer in CONSUMER_AUTHORITY_CEILINGS if receipt_organ == organ
    )
    for organ in (f"N{index}" for index in range(1, 7))
}
_MODE_AUTHORITY_CEILINGS = {
    "off": AuthorityEffect.NONE,
    "experimental": AuthorityEffect.EVIDENCE_ONLY,
    "shadow": AuthorityEffect.EVIDENCE_ONLY,
    "provisional": AuthorityEffect.BOUNDED_PROPOSAL,
}


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True, slots=True)
class SymbiosisIdentity:
    trace_group_id: str
    organism_id: str
    lineage_id: str
    run_id: str
    episode_id: str
    scenario_id: str
    decision_id: str | None = None

    def __post_init__(self) -> None:
        required = (
            "trace_group_id",
            "organism_id",
            "lineage_id",
            "run_id",
            "episode_id",
            "scenario_id",
        )
        missing = [name for name in required if not str(getattr(self, name) or "").strip()]
        if missing:
            raise ValueError(f"symbiosis_identity_missing:{','.join(missing)}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class OrganTrace:
    identity: SymbiosisIdentity
    organ: str
    capability: str
    requested_mode: str
    effective_mode: str
    authority_ceiling: str
    input_hash: str
    candidate_hash: str | None
    consumer: str
    consumer_verdict: str
    latency_ms: float
    ram_mb: float | None = None
    vram_mb: float | None = None
    fallback_reason: str | None = None
    manifest_sha256: str | None = None
    artifact_sha256: str | None = None
    candidate: Any = None
    abstained: bool = False
    cost: Mapping[str, Any] = field(default_factory=dict)
    generated_at: str = field(default_factory=utc_now_iso)

    def to_dict(self, *, include_candidate: bool = True) -> dict[str, Any]:
        data = {
            **self.identity.to_dict(),
            "organ": self.organ,
            "capability": self.capability,
            "requested_mode": self.requested_mode,
            "effective_mode": self.effective_mode,
            "authority_ceiling": self.authority_ceiling,
            "input_hash": self.input_hash,
            "candidate_hash": self.candidate_hash,
            "consumer": self.consumer,
            "consumer_verdict": self.consumer_verdict,
            "latency": round(float(self.latency_ms), 6),
            "RAM": self.ram_mb,
            "VRAM": self.vram_mb,
            "fallback_reason": self.fallback_reason,
            "manifest_sha256": self.manifest_sha256,
            "artifact_sha256": self.artifact_sha256,
            "abstained": self.abstained,
            "cost": dict(self.cost),
            "generated_at": self.generated_at,
            "schema_version": SYMBIOSIS_TRACE_SCHEMA_VERSION,
        }
        if include_candidate:
            data["candidate"] = self.candidate
        return data


@dataclass(frozen=True, slots=True)
class ConsumerReceipt:
    receipt_id: str
    identity: SymbiosisIdentity
    organ: str
    candidate_hash: str
    consumer_id: str
    consumer_contract_version: str
    consumer_input_hash: str
    consumer_output_hash: str
    verdict: str
    evidence_refs: tuple[str, ...]
    authority_effect: AuthorityEffect
    persisted: bool
    generated_at: str = field(default_factory=utc_now_iso)
    schema_version: str = CONSUMER_RECEIPT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CONSUMER_RECEIPT_SCHEMA_VERSION:
            raise ValueError("consumer_receipt_schema_mismatch")
        required = {
            "receipt_id": self.receipt_id,
            "organ": self.organ,
            "candidate_hash": self.candidate_hash,
            "consumer_id": self.consumer_id,
            "consumer_contract_version": self.consumer_contract_version,
            "consumer_input_hash": self.consumer_input_hash,
            "consumer_output_hash": self.consumer_output_hash,
            "verdict": self.verdict,
            "generated_at": self.generated_at,
        }
        missing = [name for name, value in required.items() if not str(value or "").strip()]
        if missing:
            raise ValueError(f"consumer_receipt_missing:{','.join(missing)}")
        object.__setattr__(self, "organ", self.organ.upper())
        object.__setattr__(self, "evidence_refs", tuple(str(item) for item in self.evidence_refs))
        try:
            datetime.fromisoformat(self.generated_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("consumer_receipt_generated_at_invalid") from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "receipt_id": self.receipt_id,
            **self.identity.to_dict(),
            "organ": self.organ,
            "candidate_hash": self.candidate_hash,
            "consumer_id": self.consumer_id,
            "consumer_contract_version": self.consumer_contract_version,
            "consumer_input_hash": self.consumer_input_hash,
            "consumer_output_hash": self.consumer_output_hash,
            "verdict": self.verdict,
            "evidence_refs": list(self.evidence_refs),
            "authority_effect": self.authority_effect.value,
            "persisted": self.persisted,
            "generated_at": self.generated_at,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ConsumerReceipt":
        return cls(
            receipt_id=str(raw.get("receipt_id") or ""),
            identity=SymbiosisIdentity(**{name: raw.get(name) for name in _IDENTITY_FIELDS}),
            organ=str(raw.get("organ") or ""),
            candidate_hash=str(raw.get("candidate_hash") or ""),
            consumer_id=str(raw.get("consumer_id") or ""),
            consumer_contract_version=str(raw.get("consumer_contract_version") or ""),
            consumer_input_hash=str(raw.get("consumer_input_hash") or ""),
            consumer_output_hash=str(raw.get("consumer_output_hash") or ""),
            verdict=str(raw.get("verdict") or ""),
            evidence_refs=tuple(str(item) for item in raw.get("evidence_refs") or ()),
            authority_effect=AuthorityEffect(str(raw.get("authority_effect") or "none")),
            persisted=bool(raw.get("persisted")),
            generated_at=str(raw.get("generated_at") or ""),
            schema_version=str(raw.get("schema_version") or ""),
        )


def validate_consumer_receipt(
    receipt: ConsumerReceipt,
    *,
    trace_identity: SymbiosisIdentity,
    organ_trace: OrganTrace,
) -> None:
    if receipt.identity != trace_identity or organ_trace.identity != trace_identity:
        raise ValueError("consumer_receipt_identity_mismatch")
    if receipt.organ != organ_trace.organ:
        raise ValueError("consumer_receipt_organ_mismatch")
    if not organ_trace.candidate_hash or receipt.candidate_hash != organ_trace.candidate_hash:
        raise ValueError("consumer_receipt_candidate_hash_mismatch")
    ceiling = CONSUMER_AUTHORITY_CEILINGS.get((receipt.organ, receipt.consumer_id))
    if ceiling is None:
        raise ValueError("consumer_receipt_consumer_unknown")
    if _AUTHORITY_RANK[receipt.authority_effect] > _AUTHORITY_RANK[ceiling]:
        raise ValueError("consumer_receipt_authority_exceeds_ceiling")
    organ_ceiling = _MODE_AUTHORITY_CEILINGS.get(organ_trace.authority_ceiling)
    if organ_ceiling is None or _AUTHORITY_RANK[receipt.authority_effect] > _AUTHORITY_RANK[organ_ceiling]:
        raise ValueError("consumer_receipt_authority_exceeds_organ_ceiling")
    receipt_time = datetime.fromisoformat(receipt.generated_at.replace("Z", "+00:00"))
    candidate_time = datetime.fromisoformat(organ_trace.generated_at.replace("Z", "+00:00"))
    if receipt_time < candidate_time:
        raise ValueError("consumer_receipt_predates_candidate")


@dataclass(slots=True)
class SymbiosisTrace:
    identity: SymbiosisIdentity
    organs: list[OrganTrace] = field(default_factory=list)
    episode_result: Mapping[str, Any] | None = None
    certificate: Mapping[str, Any] | None = None
    trace_health: Mapping[str, Any] = field(default_factory=dict)
    consumer_receipts: list[ConsumerReceipt] = field(default_factory=list)
    state_before_hash: str | None = None
    state_after_hash: str | None = None
    previous_transition_hash: str | None = None
    life_transition_id: str | None = None
    active_regime: str | None = None
    policy_versions: Mapping[str, str] = field(default_factory=dict)
    organ_contract_versions: Mapping[str, str] = field(default_factory=dict)
    backend_identities: Mapping[str, Any] = field(default_factory=dict)
    memory_read_references: tuple[str, ...] = ()
    memory_write_references: tuple[str, ...] = ()
    resource_state: Mapping[str, Any] = field(default_factory=dict)
    measurement_status: Mapping[str, str] = field(default_factory=dict)
    unmeasured_fields: tuple[str, ...] = ()
    not_applicable_fields: tuple[str, ...] = ()
    certificate_reference: str | None = None

    def to_dict(self, *, include_candidates: bool = True) -> dict[str, Any]:
        return {
            "schema_version": SYMBIOSIS_TRACE_SCHEMA_VERSION,
            **self.identity.to_dict(),
            "organs": [
                entry.to_dict(include_candidate=include_candidates) for entry in self.organs
            ],
            "consumer_receipts": [receipt.to_dict() for receipt in self.consumer_receipts],
            "state_before_hash": self.state_before_hash,
            "state_after_hash": self.state_after_hash,
            "previous_transition_hash": self.previous_transition_hash,
            "life_transition_id": self.life_transition_id,
            "active_regime": self.active_regime,
            "policy_versions": dict(self.policy_versions),
            "organ_contract_versions": dict(self.organ_contract_versions),
            "backend_identities": dict(self.backend_identities),
            "memory_read_references": list(self.memory_read_references),
            "memory_write_references": list(self.memory_write_references),
            "resource_state": dict(self.resource_state),
            "measurement_status": dict(self.measurement_status),
            "unmeasured_fields": list(self.unmeasured_fields),
            "not_applicable_fields": list(self.not_applicable_fields),
            "certificate_reference": self.certificate_reference,
            "episode_result": dict(self.episode_result or {}),
            "certificate": dict(self.certificate or {}),
            "trace_health": dict(self.trace_health),
            "trace_complete": self.is_complete,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "SymbiosisTrace":
        """Lee trazas v1 y v2; v1 permanece incompleta por carecer de receipts."""

        schema = str(raw.get("schema_version") or "")
        if schema not in {SYMBIOSIS_TRACE_SCHEMA_VERSION_V1, SYMBIOSIS_TRACE_SCHEMA_VERSION}:
            raise ValueError("symbiosis_trace_schema_unsupported")
        identity = SymbiosisIdentity(**{name: raw.get(name) for name in _IDENTITY_FIELDS})
        organs = []
        for item in raw.get("organs") or ():
            item_identity = SymbiosisIdentity(
                **{name: item.get(name, getattr(identity, name)) for name in _IDENTITY_FIELDS}
            )
            organs.append(
                OrganTrace(
                    identity=item_identity,
                    organ=str(item.get("organ") or ""),
                    capability=str(item.get("capability") or ""),
                    requested_mode=str(item.get("requested_mode") or "off"),
                    effective_mode=str(item.get("effective_mode") or "off"),
                    authority_ceiling=str(item.get("authority_ceiling") or "off"),
                    input_hash=str(item.get("input_hash") or ""),
                    candidate_hash=item.get("candidate_hash"),
                    consumer=str(item.get("consumer") or ""),
                    consumer_verdict=str(item.get("consumer_verdict") or ""),
                    latency_ms=float(item.get("latency", item.get("latency_ms", 0.0)) or 0.0),
                    ram_mb=item.get("RAM", item.get("ram_mb")),
                    vram_mb=item.get("VRAM", item.get("vram_mb")),
                    fallback_reason=item.get("fallback_reason"),
                    manifest_sha256=item.get("manifest_sha256"),
                    artifact_sha256=item.get("artifact_sha256"),
                    candidate=item.get("candidate"),
                    abstained=bool(item.get("abstained")),
                    cost=dict(item.get("cost") or {}),
                    generated_at=str(item.get("generated_at") or utc_now_iso()),
                )
            )
        return cls(
            identity=identity,
            organs=organs,
            episode_result=dict(raw.get("episode_result") or {}),
            certificate=dict(raw.get("certificate") or {}),
            trace_health=dict(raw.get("trace_health") or {}),
            consumer_receipts=[
                ConsumerReceipt.from_dict(item) for item in raw.get("consumer_receipts") or ()
            ],
            state_before_hash=raw.get("state_before_hash"),
            state_after_hash=raw.get("state_after_hash"),
            previous_transition_hash=raw.get("previous_transition_hash"),
            life_transition_id=raw.get("life_transition_id"),
            active_regime=raw.get("active_regime"),
            policy_versions=dict(raw.get("policy_versions") or {}),
            organ_contract_versions=dict(raw.get("organ_contract_versions") or {}),
            backend_identities=dict(raw.get("backend_identities") or {}),
            memory_read_references=tuple(raw.get("memory_read_references") or ()),
            memory_write_references=tuple(raw.get("memory_write_references") or ()),
            resource_state=dict(raw.get("resource_state") or {}),
            measurement_status=dict(raw.get("measurement_status") or {}),
            unmeasured_fields=tuple(raw.get("unmeasured_fields") or ()),
            not_applicable_fields=tuple(raw.get("not_applicable_fields") or ()),
            certificate_reference=raw.get("certificate_reference"),
        )

    @property
    def is_complete(self) -> bool:
        expected = {"N1", "N2", "N3", "N4", "N5", "N6"}
        present = {entry.organ for entry in self.organs}
        if not expected.issubset(present):
            return False
        if not self.life_transition_id or not self.previous_transition_hash:
            return False
        receipts_by_organ: dict[str, list[ConsumerReceipt]] = {}
        for receipt in self.consumer_receipts:
            receipts_by_organ.setdefault(receipt.organ, []).append(receipt)
        for entry in self.organs:
            if entry.identity != self.identity:
                return False
            if entry.candidate is None:
                continue
            receipts = receipts_by_organ.get(entry.organ, [])
            if not receipts:
                return False
            if not REQUIRED_CONSUMERS[entry.organ].issubset(
                {receipt.consumer_id for receipt in receipts}
            ):
                return False
            try:
                for receipt in receipts:
                    validate_consumer_receipt(
                        receipt, trace_identity=self.identity, organ_trace=entry
                    )
            except ValueError:
                return False
        return True


_IDENTITY_FIELDS = (
    "trace_group_id",
    "organism_id",
    "lineage_id",
    "run_id",
    "episode_id",
    "scenario_id",
    "decision_id",
)
