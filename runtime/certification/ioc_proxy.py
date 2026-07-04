"""IoC* proxy operativo para evaluación episódica."""

from __future__ import annotations


class IoCProxy:
    def compute(
        self,
        *,
        continuity_score: float,
        closure_passed: bool,
        trace_integrity: bool,
        collapse_detected: bool,
        uncertainty: float,
    ) -> float:
        continuity = max(0.0, min(1.0, float(continuity_score)))
        closure = 1.0 if closure_passed else 0.0
        trace = 1.0 if trace_integrity else 0.0
        collapse_penalty = 1.0 if collapse_detected else 0.0
        uncertainty_penalty = max(0.0, min(1.0, float(uncertainty)))
        ioc = (
            0.45 * continuity
            + 0.25 * closure
            + 0.2 * trace
            - 0.06 * uncertainty_penalty
            - 0.14 * collapse_penalty
        )
        return max(0.0, min(1.0, ioc))
