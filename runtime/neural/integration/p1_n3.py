"""P1 N3: contratos puros para evaluar influencia temporal en sombra.

Este modulo no conecta N3 con ninguna autoridad viva.  Convierte una salida N3
ya admitida en una directiva acotada y compara dos listas de retrieval sin
escribir MFM, ejecutar el scheduler ni mutar el candidato original.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from .contracts import canonical_sha256


N3_SHADOW_DIRECTIVE_SCHEMA_VERSION = "n3-shadow-directive-v1"
N3_SHADOW_COUNTERFACTUAL_SCHEMA_VERSION = "n3-shadow-counterfactual-v1"
_SCALES = ("micro", "meso", "macro")


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    resolved = float(value)
    return resolved if math.isfinite(resolved) else None


def _unit(value: Any) -> float | None:
    resolved = _finite(value)
    if resolved is None or not 0.0 <= resolved <= 1.0:
        return None
    return resolved


def _clamp(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


@dataclass(frozen=True, slots=True)
class N3ShadowDirective:
    """Directiva no autoritativa derivada de un candidato N3 consumible."""

    status: str
    reason: str
    candidate_hash: str | None
    optimization_direction: str
    trend: float | None = None
    uncertainty: float | None = None
    retrieval_priority: float | None = None
    importance: float | None = None
    risk: float | None = None
    continuity: float | None = None
    retrieval_limit_delta: int = 0
    schema_version: str = N3_SHADOW_DIRECTIVE_SCHEMA_VERSION
    authority_effect: str = "none"

    @property
    def eligible(self) -> bool:
        return self.status == "eligible"

    @property
    def scale_signals(self) -> dict[str, float]:
        if not self.eligible:
            return {}
        return {
            "micro": float(self.risk or 0.0),
            "meso": float(self.importance or 0.0),
            "macro": float(self.continuity or 0.0),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "eligible": self.eligible, "scale_signals": self.scale_signals}


@dataclass(frozen=True, slots=True)
class N3ShadowCounterfactualReport:
    """Comparacion auditable; deliberadamente incapaz de representar actuacion."""

    status: str
    reason: str
    directive: Mapping[str, Any]
    canonical_memory_ids: tuple[str, ...]
    shadow_memory_ids: tuple[str, ...]
    overlap_count: int
    canonical_retrieval_hash: str
    shadow_retrieval_hash: str
    canonical_scale_counts: Mapping[str, int]
    shadow_scale_counts: Mapping[str, int]
    canonical_scheduler_sequence: tuple[str, ...] = ()
    shadow_scheduler_sequence: tuple[str, ...] = ()
    snapshot_match: bool | None = None
    schema_version: str = N3_SHADOW_COUNTERFACTUAL_SCHEMA_VERSION
    authority_effect: str = "none"
    writes_performed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def derive_n3_shadow_directive(
    candidate: Mapping[str, Any] | None,
    *,
    candidate_hash: str | None,
    optimization_direction: str,
    alarm_threshold: float | None,
) -> N3ShadowDirective:
    """Derive bounded retrieval signals without treating missing data as zero.

    A trained candidate may provide the four bounded temporal heads directly.
    The deterministic reference requires a measured trend and derives the same
    four signals from its direction and magnitude.  ``target_band`` is rejected
    until the scenario provides both bounds; a single alarm threshold cannot
    identify which direction is adverse.
    """

    direction = str(optimization_direction or "").strip().lower()
    unavailable = lambda status, reason: N3ShadowDirective(
        status=status,
        reason=reason,
        candidate_hash=candidate_hash,
        optimization_direction=direction or "unknown",
    )
    if not isinstance(candidate, Mapping):
        return unavailable("unavailable", "candidate_missing_or_not_mapping")
    if not candidate_hash:
        return unavailable("unavailable", "admitted_candidate_hash_required")
    if direction == "target_band":
        return unavailable("unavailable", "target_band_requires_explicit_lower_and_upper_bounds")
    if direction not in {"minimize", "maximize"}:
        return unavailable("unavailable", "optimization_direction_unsupported")

    uncertainty = _unit(candidate.get("uncertainty"))
    if uncertainty is None:
        return unavailable("unavailable", "uncertainty_missing_nonfinite_or_out_of_range")

    trained = {
        name: _unit(candidate.get(name))
        for name in ("retrieval_priority", "importance", "risk", "continuity")
    }
    present_trained = [value is not None for value in trained.values()]
    if any(present_trained) and not all(present_trained):
        return unavailable("unavailable", "trained_temporal_heads_incomplete_or_invalid")
    if all(present_trained):
        priority = float(trained["retrieval_priority"])
        return N3ShadowDirective(
            status="eligible",
            reason="bounded_trained_temporal_heads",
            candidate_hash=candidate_hash,
            optimization_direction=direction,
            trend=_finite(candidate.get("trend")),
            uncertainty=uncertainty,
            retrieval_priority=priority,
            importance=float(trained["importance"]),
            risk=float(trained["risk"]),
            continuity=float(trained["continuity"]),
            retrieval_limit_delta=min(2, max(0, int(math.floor((2.0 * priority) + 0.5)))),
        )

    trend = _finite(candidate.get("trend"))
    if trend is None:
        return unavailable("warmup", "reference_trend_not_measured")
    threshold = _finite(alarm_threshold)
    if threshold is None or abs(threshold) < 1e-9:
        return unavailable("unavailable", "finite_nonzero_alarm_threshold_required")

    magnitude = _clamp(abs(trend) / abs(threshold))
    adverse = max(trend, 0.0) if direction == "minimize" else max(-trend, 0.0)
    risk = _clamp(adverse / abs(threshold))
    priority = max(uncertainty, risk)
    return N3ShadowDirective(
        status="eligible",
        reason="bounded_reference_trend",
        candidate_hash=candidate_hash,
        optimization_direction=direction,
        trend=trend,
        uncertainty=uncertainty,
        retrieval_priority=priority,
        importance=magnitude,
        risk=risk,
        continuity=1.0 - magnitude,
        retrieval_limit_delta=min(2, max(0, int(math.floor((2.0 * priority) + 0.5)))),
    )


def retrieval_scale_weights(directive: N3ShadowDirective) -> dict[str, float]:
    """Map [0,1] temporal signals to conservative [0.75,1] multipliers."""

    if not directive.eligible:
        return {}
    return {
        scale: 0.75 + (0.25 * signal)
        for scale, signal in directive.scale_signals.items()
    }


def shadow_retrieval_limit(
    directive: N3ShadowDirective,
    *,
    canonical_limit: int,
    maximum: int = 8,
) -> int:
    if canonical_limit < 1 or maximum < 1:
        raise ValueError("retrieval_limits_must_be_positive")
    if not directive.eligible:
        return min(canonical_limit, maximum)
    return min(maximum, canonical_limit + directive.retrieval_limit_delta)


def _memory_ids(hits: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    return tuple(str(hit.get("memory_id") or "") for hit in hits)


def _scale_counts(hits: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = {scale: 0 for scale in _SCALES}
    for hit in hits:
        scale = str(hit.get("scale") or "")
        if scale in counts:
            counts[scale] += 1
    return counts


def compare_n3_shadow_retrieval(
    *,
    directive: N3ShadowDirective,
    canonical_hits: Sequence[Mapping[str, Any]],
    shadow_hits: Sequence[Mapping[str, Any]],
    canonical_scheduler_sequence: Sequence[str] = (),
    shadow_scheduler_sequence: Sequence[str] = (),
    snapshot_match: bool | None = None,
) -> N3ShadowCounterfactualReport:
    """Build a pure report; caller remains responsible for snapshot attestation."""

    canonical_ids = _memory_ids(canonical_hits)
    shadow_ids = _memory_ids(shadow_hits)
    status = "compared" if directive.eligible else directive.status
    return N3ShadowCounterfactualReport(
        status=status,
        reason="retrieval_compared_without_authority" if directive.eligible else directive.reason,
        directive=directive.to_dict(),
        canonical_memory_ids=canonical_ids,
        shadow_memory_ids=shadow_ids,
        overlap_count=len(set(canonical_ids).intersection(shadow_ids)),
        canonical_retrieval_hash=canonical_sha256(list(canonical_hits)),
        shadow_retrieval_hash=canonical_sha256(list(shadow_hits)),
        canonical_scale_counts=_scale_counts(canonical_hits),
        shadow_scale_counts=_scale_counts(shadow_hits),
        canonical_scheduler_sequence=tuple(str(item).upper() for item in canonical_scheduler_sequence),
        shadow_scheduler_sequence=tuple(str(item).upper() for item in shadow_scheduler_sequence),
        snapshot_match=snapshot_match,
    )

