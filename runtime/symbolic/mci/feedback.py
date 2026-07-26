"""Relay fail-open de evaluaciones MCI hacia el proveedor que las originó."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .contracts import NeuralHypothesisEvaluation
from .provider_protocol import HypothesisProvider


def evaluations_from_payload(
    payload: Sequence[Mapping[str, Any]],
) -> tuple[NeuralHypothesisEvaluation, ...]:
    return tuple(
        NeuralHypothesisEvaluation(
            hypothesis_id=str(item["hypothesis_id"]),
            status=str(item["status"]),
            reason=str(item["reason"]),
            train_mae_before=item.get("train_mae_before"),
            train_mae_after=item.get("train_mae_after"),
            holdout_mae_before=item.get("holdout_mae_before"),
            holdout_mae_after=item.get("holdout_mae_after"),
            promoted_overlay_id=item.get("promoted_overlay_id"),
        )
        for item in payload
    )


def relay_feedback(
    provider: HypothesisProvider,
    payload: Sequence[Mapping[str, Any]],
) -> tuple[NeuralHypothesisEvaluation, ...]:
    evaluations = evaluations_from_payload(payload)
    if evaluations:
        provider.receive_feedback(evaluations)
    return evaluations
