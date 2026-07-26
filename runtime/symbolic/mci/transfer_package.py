"""Contratos inmutables para transferir conocimiento entre Transition IR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from runtime.symbolic.schemas import _Schema, _freeze, sealed_sha256


@dataclass(frozen=True)
class TransferPackage(_Schema):
    package_id: str
    source_spec_sha256: str
    source_overlay_id: str
    source_overlay_sha256: str
    morphism_id: str
    normalized_claims: Mapping[str, Any]
    evidence_refs: tuple[str, ...]
    source_metrics: Mapping[str, float]
    confidence: float

    def __post_init__(self) -> None:
        confidence = float(self.confidence)
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("transfer_confidence_out_of_range")
        object.__setattr__(self, "confidence", round(confidence, 6))
        object.__setattr__(self, "normalized_claims", _freeze(self.normalized_claims))
        object.__setattr__(self, "source_metrics", _freeze(self.source_metrics))
        object.__setattr__(self, "evidence_refs", tuple(sorted(set(self.evidence_refs))))
        if not self.package_id:
            payload = {
                "source_spec_sha256": self.source_spec_sha256,
                "source_overlay_sha256": self.source_overlay_sha256,
                "morphism_id": self.morphism_id,
                "normalized_claims": self.normalized_claims,
                "evidence_refs": self.evidence_refs,
            }
            object.__setattr__(
                self, "package_id", f"transfer-package-{sealed_sha256(payload)[:24]}"
            )

    @property
    def sha256(self) -> str:
        return sealed_sha256(self.to_dict())


@dataclass(frozen=True)
class TransferredHypothesis(_Schema):
    hypothesis_id: str
    package_id: str
    morphism_id: str
    kind: str
    target_id: str
    expression: str | None
    proposed_value: float | None
    transfer_confidence: float
    transfer_penalty: float
    source_evidence_refs: tuple[str, ...]
    logical_time: int

    def __post_init__(self) -> None:
        if self.kind not in {"precondition", "parameter"}:
            raise ValueError("transferred_hypothesis_kind_not_actionable")
        confidence = float(self.transfer_confidence)
        penalty = float(self.transfer_penalty)
        if not 0.0 <= confidence <= 1.0 or not 0.0 <= penalty <= 1.0:
            raise ValueError("transferred_hypothesis_probability_out_of_range")
        object.__setattr__(self, "transfer_confidence", round(confidence, 6))
        object.__setattr__(self, "transfer_penalty", round(penalty, 6))
        object.__setattr__(
            self, "source_evidence_refs", tuple(sorted(set(self.source_evidence_refs)))
        )
        if not self.hypothesis_id:
            payload = {
                "package_id": self.package_id,
                "morphism_id": self.morphism_id,
                "kind": self.kind,
                "target_id": self.target_id,
                "expression": self.expression,
                "proposed_value": self.proposed_value,
                "logical_time": int(self.logical_time),
            }
            object.__setattr__(
                self, "hypothesis_id", f"transfer-{sealed_sha256(payload)[:24]}"
            )
