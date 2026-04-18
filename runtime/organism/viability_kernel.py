"""Kernel de viabilidad trayectorial (RNFE-T4)."""

from __future__ import annotations

from dataclasses import dataclass

from .constitution_flow import ConstitutionalFlowResult
from .trajectory import OrganismTrajectory


@dataclass(frozen=True)
class TrajectoryViabilityAssessment:
    viability_score: float
    drift_pressure: float
    hysteresis_pressure: float
    recovery_capacity: float
    rollback_required: bool


class TrajectoryViabilityKernel:
    def assess(
        self,
        *,
        trajectory: OrganismTrajectory,
        flow_result: ConstitutionalFlowResult,
        window_size: int = 8,
    ) -> TrajectoryViabilityAssessment:
        digest = trajectory.digest(window_size=window_size)
        viability_score = max(
            0.0,
            min(
                1.0,
                (
                    0.40 * (1.0 - digest.drift_score)
                    + 0.25 * (1.0 - digest.hysteresis_score)
                    + 0.20 * digest.recovery_score
                    + 0.15 * (1.0 - flow_result.erosion)
                ),
            ),
        )
        rollback_required = flow_result.rollback_obligation or viability_score < 0.25
        return TrajectoryViabilityAssessment(
            viability_score=round(viability_score, 4),
            drift_pressure=round(digest.drift_score, 4),
            hysteresis_pressure=round(digest.hysteresis_score, 4),
            recovery_capacity=round(digest.recovery_score, 4),
            rollback_required=rollback_required,
        )
