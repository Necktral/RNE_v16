"""Promoción de candidatos EML a modo advisory (sin control online)."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List


@dataclass(frozen=True, slots=True)
class AdvisoryDecision:
    promoted: bool
    reason: str
    support: int
    mean_score: float
    std_score: float
    signature: str | None = None


def _signature(candidate: Dict[str, Any]) -> str:
    return str(candidate.get("expr", {}))


def evaluate_advisory_promotion(
    *,
    candidates: Iterable[Dict[str, Any]],
    continuity_alert: bool,
    min_support: int = 3,
    min_mean_score: float = 0.72,
    max_std_score: float = 0.08,
) -> AdvisoryDecision:
    if continuity_alert:
        return AdvisoryDecision(
            promoted=False,
            reason="continuity_alert",
            support=0,
            mean_score=0.0,
            std_score=0.0,
        )
    grouped: Dict[str, List[float]] = {}
    for item in candidates:
        sig = _signature(item)
        score = item.get("composite_score", 0.0)
        if not isinstance(score, (int, float)):
            continue
        grouped.setdefault(sig, []).append(float(score))
    if not grouped:
        return AdvisoryDecision(
            promoted=False,
            reason="no_candidates",
            support=0,
            mean_score=0.0,
            std_score=0.0,
        )

    best_sig = None
    best_values: List[float] = []
    for sig, values in grouped.items():
        if len(values) > len(best_values):
            best_sig = sig
            best_values = values
    support = len(best_values)
    mean_score = mean(best_values) if best_values else 0.0
    std_score = pstdev(best_values) if len(best_values) > 1 else 0.0
    promoted = bool(
        support >= min_support and mean_score >= min_mean_score and std_score <= max_std_score
    )
    return AdvisoryDecision(
        promoted=promoted,
        reason="eligible" if promoted else "threshold_not_met",
        support=support,
        mean_score=mean_score,
        std_score=std_score,
        signature=best_sig,
    )

