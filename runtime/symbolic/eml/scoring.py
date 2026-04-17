"""Scoring de candidatos EML: fit, estabilidad y validez de dominio."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Any, Iterable, List

from .safe_eval import DomainError, safe_eval
from .tree import ExprNode


@dataclass(frozen=True, slots=True)
class CandidateScore:
    fit_score: float
    stability_score: float
    domain_valid_ratio: float
    composite_score: float
    predictions: list[float]
    errors: list[float]


def score_candidate(
    expr: ExprNode,
    rows: Iterable[dict[str, Any]],
    *,
    target_key: str = "y",
    clip_abs: float = 1_000_000.0,
) -> CandidateScore:
    predictions: List[float] = []
    errors: List[float] = []
    valid = 0
    total = 0
    for row in rows:
        total += 1
        if target_key not in row or not isinstance(row[target_key], (int, float)):
            continue
        target = float(row[target_key])
        try:
            pred = safe_eval(expr, row, clip_abs=clip_abs)
            error = abs(pred - target)
            valid += 1
        except DomainError:
            pred = 0.0
            error = 1_000_000.0
        predictions.append(pred)
        errors.append(error)

    domain_valid_ratio = (valid / total) if total else 0.0
    mae = mean(errors) if errors else 1_000_000.0
    std_err = pstdev(errors) if len(errors) > 1 else 0.0
    fit_score = 1.0 / (1.0 + mae)
    stability_score = 1.0 / (1.0 + std_err)
    composite = (0.6 * fit_score) + (0.25 * stability_score) + (0.15 * domain_valid_ratio)
    return CandidateScore(
        fit_score=fit_score,
        stability_score=stability_score,
        domain_valid_ratio=domain_valid_ratio,
        composite_score=composite,
        predictions=predictions,
        errors=errors,
    )

