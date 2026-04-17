"""Detector de colapso para continuidad operativa."""

from __future__ import annotations

from typing import Sequence

from runtime.storage import RealityAssessmentRecord


class CollapseDetector:
    def __init__(self, *, continuity_threshold: float = 0.35, streak: int = 3):
        self.continuity_threshold = continuity_threshold
        self.streak = streak

    def detect(
        self,
        *,
        closure_passed: bool,
        trace_integrity: bool,
        continuity_score: float,
        recent_assessments: Sequence[RealityAssessmentRecord] | None,
    ) -> bool:
        if not closure_passed or not trace_integrity:
            return True
        if continuity_score < self.continuity_threshold:
            history = [
                item.continuity_score
                for item in (recent_assessments or [])
                if item.continuity_score < self.continuity_threshold
            ]
            if len(history) >= self.streak - 1:
                return True
        return False
