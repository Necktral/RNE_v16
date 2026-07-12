"""Contratos internos versionados de la integración simbiótica neuronal."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


SYMBIOSIS_TRACE_SCHEMA_VERSION = "neural-symbiosis-trace-v1"


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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
            "schema_version": SYMBIOSIS_TRACE_SCHEMA_VERSION,
        }
        if include_candidate:
            data["candidate"] = self.candidate
        return data


@dataclass(slots=True)
class SymbiosisTrace:
    identity: SymbiosisIdentity
    organs: list[OrganTrace] = field(default_factory=list)
    episode_result: Mapping[str, Any] | None = None
    certificate: Mapping[str, Any] | None = None
    trace_health: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self, *, include_candidates: bool = True) -> dict[str, Any]:
        return {
            "schema_version": SYMBIOSIS_TRACE_SCHEMA_VERSION,
            **self.identity.to_dict(),
            "organs": [
                entry.to_dict(include_candidate=include_candidates) for entry in self.organs
            ],
            "episode_result": dict(self.episode_result or {}),
            "certificate": dict(self.certificate or {}),
            "trace_health": dict(self.trace_health),
            "trace_complete": self.is_complete,
        }

    @property
    def is_complete(self) -> bool:
        expected = {"N1", "N2", "N3", "N4", "N5", "N6"}
        present = {entry.organ for entry in self.organs}
        if not expected.issubset(present):
            return False
        return all(
            entry.identity.trace_group_id == self.identity.trace_group_id
            and bool(entry.consumer)
            and bool(entry.consumer_verdict)
            for entry in self.organs
        )
