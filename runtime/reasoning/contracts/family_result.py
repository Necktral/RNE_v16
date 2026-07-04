"""Contrato canónico de salida para familias de razonamiento."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping


_OPTIONAL_FIELDS = {"recommended_next_family", "artifacts", "failure_mode"}
_RESERVED_FIELDS = {"family", "status", "state_delta", "confidence", "cost"} | _OPTIONAL_FIELDS


@dataclass(frozen=True)
class FamilyResult:
    family: str
    status: str = "ok"
    state_delta: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    cost: float = 0.0
    recommended_next_family: str | None = None
    artifacts: Dict[str, Any] | None = None
    failure_mode: str | None = None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "family": self.family,
            "status": self.status,
            "state_delta": dict(self.state_delta),
            "confidence": float(self.confidence),
            "cost": float(self.cost),
        }
        if self.recommended_next_family:
            payload["recommended_next_family"] = self.recommended_next_family
        if self.artifacts is not None:
            payload["artifacts"] = dict(self.artifacts)
        if self.failure_mode:
            payload["failure_mode"] = self.failure_mode
        return payload


def _as_nonnegative_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, (int, float)):
        return max(0.0, float(value))
    return default


def normalize_family_result(
    raw_result: Mapping[str, Any] | None,
    *,
    family_hint: str | None = None,
) -> Dict[str, Any]:
    raw = dict(raw_result or {})
    family = str(raw.get("family") or family_hint or "").strip().upper()
    status = str(raw.get("status") or "ok").strip() or "ok"
    state_delta = raw.get("state_delta")
    if not isinstance(state_delta, dict):
        state_delta = {}

    recommended_next = raw.get("recommended_next_family")
    recommended_next_family = None
    if isinstance(recommended_next, str) and recommended_next.strip():
        recommended_next_family = recommended_next.strip().upper()

    artifacts = raw.get("artifacts")
    if not isinstance(artifacts, dict):
        artifacts = None

    failure_mode = raw.get("failure_mode")
    if not isinstance(failure_mode, str) or not failure_mode.strip():
        failure_mode = None

    normalized = FamilyResult(
        family=family,
        status=status,
        state_delta=state_delta,
        confidence=_as_nonnegative_float(raw.get("confidence"), default=0.0),
        cost=_as_nonnegative_float(raw.get("cost"), default=0.0),
        recommended_next_family=recommended_next_family,
        artifacts=artifacts,
        failure_mode=failure_mode,
    ).to_dict()

    for key, value in raw.items():
        if key not in _RESERVED_FIELDS and value is not None:
            normalized[key] = value
    return normalized
