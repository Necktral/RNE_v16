"""Detección pura y determinista de fronteras causales en evidencia observada."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, Sequence

if TYPE_CHECKING:
    from .causal_learning import TransitionEvidence


@dataclass(frozen=True, slots=True)
class BoundaryCandidate:
    variable: str
    operator: str
    threshold: float
    lower_support: int
    upper_support: int
    local_purity: float
    separation: float
    action: str
    effect_key: str
    source: str = "change_boundary"

    @property
    def expression(self) -> str:
        return f"{self.variable} {self.operator} {self.threshold:.9g}"


def detect_transition_boundaries(
    evidence: Sequence[TransitionEvidence],
    *,
    variable: str,
    action: str | None = None,
    effect_key: str | None = None,
    min_support: int = 2,
) -> tuple[BoundaryCandidate, ...]:
    """Detecta cambios estables de concordancia predicción/observación.

    Una transición es exitosa cuando los campos observados que el modelo predijo
    coinciden (con tolerancia numérica). La función usa únicamente evidencia ya
    observada, nunca metadatos oracle del escenario.
    """

    if min_support < 1:
        raise ValueError("boundary_min_support_must_be_positive")
    grouped: dict[tuple[str, str], list[tuple[float, bool]]] = {}
    for row in evidence:
        if action is not None and row.action != action:
            continue
        value = row.state.get(variable)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        numeric = float(value)
        if not math.isfinite(numeric):
            continue
        signature = _effect_signature(row)
        if effect_key is not None and effect_key != signature:
            continue
        grouped.setdefault((row.action, signature), []).append(
            (numeric, _prediction_matches(row.predicted, row.observed))
        )

    candidates: dict[tuple[str, float, str, str], BoundaryCandidate] = {}
    for (group_action, signature), rows in sorted(grouped.items()):
        by_value: dict[float, list[bool]] = {}
        for value, success in rows:
            by_value.setdefault(value, []).append(success)
        points = sorted(
            (value, sum(labels) / len(labels), len(labels))
            for value, labels in by_value.items()
        )
        for split in range(1, len(points)):
            lower = points[:split]
            upper = points[split:]
            lower_support = sum(item[2] for item in lower)
            upper_support = sum(item[2] for item in upper)
            if lower_support < min_support or upper_support < min_support:
                continue
            lower_rate = sum(rate * count for _, rate, count in lower) / lower_support
            upper_rate = sum(rate * count for _, rate, count in upper) / upper_support
            separation = abs(upper_rate - lower_rate)
            purity = (
                max(lower_rate, 1.0 - lower_rate) * lower_support
                + max(upper_rate, 1.0 - upper_rate) * upper_support
            ) / (lower_support + upper_support)
            if separation < 0.5 or purity < 0.75:
                continue
            operator = ">" if upper_rate > lower_rate else "<"
            threshold = round((lower[-1][0] + upper[0][0]) / 2.0, 9)
            candidate = BoundaryCandidate(
                variable=variable,
                operator=operator,
                threshold=threshold,
                lower_support=lower_support,
                upper_support=upper_support,
                local_purity=round(purity, 9),
                separation=round(separation, 9),
                action=group_action,
                effect_key=signature,
            )
            key = (operator, threshold, group_action, signature)
            previous = candidates.get(key)
            if previous is None or _quality(candidate) > _quality(previous):
                candidates[key] = candidate
    stable: dict[tuple[str, str, str], BoundaryCandidate] = {}
    for candidate in candidates.values():
        key = (candidate.operator, candidate.action, candidate.effect_key)
        previous = stable.get(key)
        if previous is None or _quality(candidate) > _quality(previous):
            stable[key] = candidate
    return tuple(
        sorted(
            stable.values(),
            key=lambda item: (
                -item.separation,
                -item.local_purity,
                item.threshold,
                item.operator,
                item.action,
                item.effect_key,
            ),
        )
    )


def _prediction_matches(
    predicted: Mapping[str, Any], observed: Mapping[str, Any]
) -> bool:
    keys = sorted(set(predicted) & set(observed))
    if not keys:
        return False
    for key in keys:
        left, right = predicted[key], observed[key]
        if isinstance(left, bool) or isinstance(right, bool):
            if left != right:
                return False
        elif isinstance(left, (int, float)) and isinstance(right, (int, float)):
            if not math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=1e-9):
                return False
        elif left != right:
            return False
    return True


def _effect_signature(row: TransitionEvidence) -> str:
    changed = sorted(
        key
        for key, value in row.predicted.items()
        if key not in row.state or row.state.get(key) != value
    )
    return ",".join(changed) or "no_predicted_change"


def _quality(candidate: BoundaryCandidate) -> tuple[float, float, int]:
    return (
        candidate.separation,
        candidate.local_purity,
        min(candidate.lower_support, candidate.upper_support),
    )
