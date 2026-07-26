"""Proveedor proposal-only configurable para campañas controladas."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from runtime.symbolic.mci import (
    ExternalHypothesis,
    NeuralHypothesisEvaluation,
)


class ScheduledHypothesisProvider:
    def __init__(
        self,
        schedule: Mapping[int, Sequence[ExternalHypothesis]],
    ) -> None:
        self._schedule = {
            int(episode): tuple(proposals)
            for episode, proposals in schedule.items()
        }
        self._episode = 0
        self.feedback: list[NeuralHypothesisEvaluation] = []

    def infer_hypotheses(
        self, context: Mapping[str, Any]
    ) -> Sequence[ExternalHypothesis]:
        del context
        self._episode += 1
        return self._schedule.get(self._episode, ())

    def receive_feedback(
        self, evaluations: Sequence[NeuralHypothesisEvaluation]
    ) -> None:
        self.feedback.extend(evaluations)
