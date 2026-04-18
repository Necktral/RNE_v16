"""Constitucion de flujo para RNFE-T5."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Tuple

from .trajectory import OrganismTrajectory, TrajectoryInvariantReport


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    q = min(1.0, max(0.0, q))
    xs = sorted(values)
    idx = (len(xs) - 1) * q
    lo = int(idx)
    hi = min(len(xs) - 1, lo + 1)
    w = idx - lo
    return xs[lo] * (1.0 - w) + xs[hi] * w


@dataclass(frozen=True)
class ConstitutionalFlowResult:
    flow_validity: bool
    erosion: float
    phase_drift: float
    rollback_obligation: bool
    invariants: Dict[str, float]
    thresholds: Dict[str, float]
    point_violations: Tuple[str, ...]
    flow_violations: Tuple[str, ...]
    violations: Tuple[str, ...]


@dataclass
class ConstitutionalFlowEngine:
    """Motor de flujo constitucional con calibracion trayectorial."""

    identity_curvature_bound: float = 0.35
    policy_phase_jump_bound: float = 0.30
    memory_purity_integral_min: float = 0.55
    trace_integrity_persistence_min: float = 0.60
    recovery_debt_integral_max: float = 0.65
    constitutional_erosion_rate_max: float = 0.40
    calibration_min_samples: int = 5
    calibration_window: int = 128
    upper_quantile: float = 0.85
    lower_quantile: float = 0.15
    _history: Dict[str, list[Dict[str, float]]] = field(default_factory=dict, repr=False)

    def _default_thresholds(self) -> Dict[str, float]:
        return {
            "identity_curvature_bound": self.identity_curvature_bound,
            "policy_phase_jump_bound": self.policy_phase_jump_bound,
            "memory_purity_integral": self.memory_purity_integral_min,
            "trace_integrity_persistence": self.trace_integrity_persistence_min,
            "recovery_debt_integral": self.recovery_debt_integral_max,
            "constitutional_erosion_rate": self.constitutional_erosion_rate_max,
        }

    def _calibrated_thresholds(
        self,
        *,
        trajectory_id: str,
        invariants: Dict[str, float],
    ) -> Dict[str, float]:
        history = self._history.setdefault(trajectory_id, [])
        thresholds = self._default_thresholds()

        if len(history) >= self.calibration_min_samples:
            for key in ("identity_curvature_bound", "policy_phase_jump_bound", "recovery_debt_integral", "constitutional_erosion_rate"):
                values = [row.get(key, thresholds[key]) for row in history]
                qv = _quantile(values, self.upper_quantile)
                thresholds[key] = round(min(1.0, max(thresholds[key], qv * 1.10)), 4)

            for key in ("memory_purity_integral", "trace_integrity_persistence"):
                values = [row.get(key, thresholds[key]) for row in history]
                qv = _quantile(values, self.lower_quantile)
                thresholds[key] = round(max(0.0, min(thresholds[key], qv * 0.90)), 4)

        history.append(dict(invariants))
        if len(history) > self.calibration_window:
            self._history[trajectory_id] = history[-self.calibration_window :]

        return thresholds

    def evaluate(
        self,
        trajectory: OrganismTrajectory,
        *,
        window_size: int = 8,
    ) -> ConstitutionalFlowResult:
        window = trajectory.window(window_size)
        snaps = window.snapshots
        if not snaps:
            invariants = {
                "identity_curvature_bound": 0.0,
                "policy_phase_jump_bound": 0.0,
                "memory_purity_integral": 1.0,
                "trace_integrity_persistence": 1.0,
                "recovery_debt_integral": 0.0,
                "constitutional_erosion_rate": 0.0,
            }
            thresholds = self._default_thresholds()
            return ConstitutionalFlowResult(
                flow_validity=True,
                erosion=0.0,
                phase_drift=0.0,
                rollback_obligation=False,
                invariants=invariants,
                thresholds=thresholds,
                point_violations=(),
                flow_violations=(),
                violations=(),
            )

        digest = trajectory.digest(window_size=window_size)
        memory_integral = sum(s.belief.memory_purity_estimate for s in snaps) / len(snaps)
        trace_integral = sum(s.belief.trace_integrity_confidence for s in snaps) / len(snaps)
        recovery_debt_integral = sum(s.viability.recovery_debt for s in snaps) / len(snaps)

        erosion = min(
            1.0,
            (
                0.25 * digest.identity_curvature
                + 0.20 * digest.policy_phase_drift
                + 0.20 * (1.0 - memory_integral)
                + 0.20 * (1.0 - trace_integral)
                + 0.15 * recovery_debt_integral
            ),
        )

        invariants = {
            "identity_curvature_bound": round(digest.identity_curvature, 4),
            "policy_phase_jump_bound": round(digest.policy_phase_drift, 4),
            "memory_purity_integral": round(memory_integral, 4),
            "trace_integrity_persistence": round(trace_integral, 4),
            "recovery_debt_integral": round(recovery_debt_integral, 4),
            "constitutional_erosion_rate": round(erosion, 4),
        }
        thresholds = self._calibrated_thresholds(
            trajectory_id=trajectory.trajectory_id,
            invariants=invariants,
        )

        latest = snaps[-1]
        point_violations: list[str] = []
        if latest.belief.memory_purity_estimate < thresholds["memory_purity_integral"]:
            point_violations.append("point_memory_purity")
        if latest.belief.trace_integrity_confidence < thresholds["trace_integrity_persistence"]:
            point_violations.append("point_trace_integrity")
        if latest.policy.accumulated_drift > thresholds["policy_phase_jump_bound"]:
            point_violations.append("point_policy_phase_jump")

        flow_violations: list[str] = []
        if invariants["identity_curvature_bound"] > thresholds["identity_curvature_bound"]:
            flow_violations.append("identity_curvature_bound")
        if invariants["policy_phase_jump_bound"] > thresholds["policy_phase_jump_bound"]:
            flow_violations.append("policy_phase_jump_bound")
        if invariants["memory_purity_integral"] < thresholds["memory_purity_integral"]:
            flow_violations.append("memory_purity_integral")
        if invariants["trace_integrity_persistence"] < thresholds["trace_integrity_persistence"]:
            flow_violations.append("trace_integrity_persistence")
        if invariants["recovery_debt_integral"] > thresholds["recovery_debt_integral"]:
            flow_violations.append("recovery_debt_integral")
        if invariants["constitutional_erosion_rate"] > thresholds["constitutional_erosion_rate"]:
            flow_violations.append("constitutional_erosion_rate")

        rollback = (
            "identity_curvature_bound" in flow_violations
            or "trace_integrity_persistence" in flow_violations
            or "constitutional_erosion_rate" in flow_violations
        )
        all_violations = tuple(dict.fromkeys(point_violations + flow_violations))
        flow_validity = len(flow_violations) == 0

        return ConstitutionalFlowResult(
            flow_validity=flow_validity,
            erosion=round(erosion, 4),
            phase_drift=round(digest.policy_phase_drift, 4),
            rollback_obligation=rollback,
            invariants=invariants,
            thresholds=thresholds,
            point_violations=tuple(point_violations),
            flow_violations=tuple(flow_violations),
            violations=all_violations,
        )

    def to_report(
        self,
        *,
        trajectory: OrganismTrajectory,
        result: ConstitutionalFlowResult,
        window_size: int = 8,
    ) -> TrajectoryInvariantReport:
        window = trajectory.window(window_size)
        return TrajectoryInvariantReport(
            trajectory_id=trajectory.trajectory_id,
            window_start_episode=window.start_episode,
            window_end_episode=window.end_episode,
            invariants={
                **result.invariants,
                "point_violations_count": float(len(result.point_violations)),
                "flow_violations_count": float(len(result.flow_violations)),
            },
            violations=result.violations,
            rollback_obligation=result.rollback_obligation,
        )

