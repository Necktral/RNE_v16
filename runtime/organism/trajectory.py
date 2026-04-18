"""Modelo trayectorial nativo del organismo (RNFE-T4)."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Sequence

from .snapshot import OrganismSnapshot


@dataclass(frozen=True)
class BeliefHistory:
    alarm_probability: tuple[float, ...]
    intervention_efficacy: tuple[float, ...]
    causal_support_confidence: tuple[float, ...]
    memory_purity_estimate: tuple[float, ...]
    trace_integrity_confidence: tuple[float, ...]


@dataclass(frozen=True)
class PolicyHistory:
    accumulated_drift: tuple[float, ...]
    sensitivity: tuple[float, ...]
    perturbation_tolerance: tuple[float, ...]


@dataclass(frozen=True)
class ViabilityHistory:
    viability_margin: tuple[float, ...]
    accumulated_degradation: tuple[float, ...]
    recovery_debt: tuple[float, ...]


@dataclass(frozen=True)
class TrajectoryDigest:
    trajectory_id: str
    window_start_episode: int
    window_end_episode: int
    drift_score: float
    hysteresis_score: float
    recovery_score: float
    volatility_score: float
    identity_curvature: float
    policy_phase_drift: float


@dataclass(frozen=True)
class TrajectoryInvariantReport:
    trajectory_id: str
    window_start_episode: int
    window_end_episode: int
    invariants: Dict[str, float]
    violations: tuple[str, ...]
    rollback_obligation: bool


@dataclass(frozen=True)
class TrajectoryPoint:
    snapshot: OrganismSnapshot
    regime: str
    observation: Dict[str, Any]
    intervention: Dict[str, Any]
    counterfactual: Dict[str, Any]
    memory_context: Dict[str, Any]


@dataclass(frozen=True)
class TrajectoryWindow:
    trajectory_id: str
    start_episode: int
    end_episode: int
    points: tuple[TrajectoryPoint, ...]

    @property
    def snapshots(self) -> tuple[OrganismSnapshot, ...]:
        return tuple(point.snapshot for point in self.points)

    def belief_history(self) -> BeliefHistory:
        snaps = self.snapshots
        return BeliefHistory(
            alarm_probability=tuple(s.belief.alarm_probability for s in snaps),
            intervention_efficacy=tuple(s.belief.intervention_efficacy for s in snaps),
            causal_support_confidence=tuple(s.belief.causal_support_confidence for s in snaps),
            memory_purity_estimate=tuple(s.belief.memory_purity_estimate for s in snaps),
            trace_integrity_confidence=tuple(s.belief.trace_integrity_confidence for s in snaps),
        )

    def policy_history(self) -> PolicyHistory:
        snaps = self.snapshots
        return PolicyHistory(
            accumulated_drift=tuple(s.policy.accumulated_drift for s in snaps),
            sensitivity=tuple(s.policy.sensitivity for s in snaps),
            perturbation_tolerance=tuple(s.policy.perturbation_tolerance for s in snaps),
        )

    def viability_history(self) -> ViabilityHistory:
        snaps = self.snapshots
        return ViabilityHistory(
            viability_margin=tuple(s.viability.viability_margin for s in snaps),
            accumulated_degradation=tuple(s.viability.accumulated_degradation for s in snaps),
            recovery_debt=tuple(s.viability.recovery_debt for s in snaps),
        )


@dataclass
class OrganismTrajectory:
    trajectory_id: str
    points: List[TrajectoryPoint] = field(default_factory=list)

    def append(
        self,
        *,
        snapshot: OrganismSnapshot,
        regime: str,
        observation: Mapping[str, Any] | None = None,
        intervention: Mapping[str, Any] | None = None,
        counterfactual: Mapping[str, Any] | None = None,
        memory_context: Mapping[str, Any] | None = None,
    ) -> None:
        self.points.append(
            TrajectoryPoint(
                snapshot=snapshot,
                regime=regime,
                observation=dict(observation or {}),
                intervention=dict(intervention or {}),
                counterfactual=dict(counterfactual or {}),
                memory_context=dict(memory_context or {}),
            )
        )

    def window(self, size: int = 8) -> TrajectoryWindow:
        if not self.points:
            return TrajectoryWindow(
                trajectory_id=self.trajectory_id,
                start_episode=0,
                end_episode=0,
                points=(),
            )
        points = tuple(self.points[-max(1, size) :])
        return TrajectoryWindow(
            trajectory_id=self.trajectory_id,
            start_episode=points[0].snapshot.episode_count,
            end_episode=points[-1].snapshot.episode_count,
            points=points,
        )

    def digest(self, *, window_size: int = 8) -> TrajectoryDigest:
        window = self.window(window_size)
        snaps = window.snapshots
        if len(snaps) <= 1:
            return TrajectoryDigest(
                trajectory_id=self.trajectory_id,
                window_start_episode=window.start_episode,
                window_end_episode=window.end_episode,
                drift_score=0.0,
                hysteresis_score=0.0,
                recovery_score=0.0,
                volatility_score=0.0,
                identity_curvature=0.0,
                policy_phase_drift=0.0,
            )

        deltas: list[float] = []
        signs: list[int] = []
        margin_values: list[float] = []
        identity_dist: list[float] = []
        policy_drift_values: list[float] = []
        for idx in range(1, len(snaps)):
            prev = snaps[idx - 1]
            cur = snaps[idx]
            delta = cur.belief.distance_to(prev.belief)
            deltas.append(delta)
            signs.append(1 if delta >= 0 else -1)
            margin_values.append(cur.viability.viability_margin - prev.viability.viability_margin)
            identity_dist.append(cur.identity.identity_distance(prev.identity))
            policy_drift_values.append(abs(cur.policy.accumulated_drift - prev.policy.accumulated_drift))

        drift_score = sum(abs(v) for v in deltas) / len(deltas)
        sign_flips = 0
        for idx in range(1, len(signs)):
            if signs[idx] != signs[idx - 1]:
                sign_flips += 1
        hysteresis = sign_flips / max(1, len(signs) - 1)

        recovery_gain = 0.0
        for idx in range(1, len(margin_values)):
            if margin_values[idx - 1] < 0 and margin_values[idx] > 0:
                recovery_gain += margin_values[idx]
        recovery_score = max(0.0, min(1.0, recovery_gain + (sum(1 for v in margin_values if v > 0) / len(margin_values)) * 0.25))

        mean_delta = sum(deltas) / len(deltas)
        volatility = math.sqrt(sum((v - mean_delta) ** 2 for v in deltas) / len(deltas))

        identity_curvature = sum(identity_dist) / len(identity_dist)
        policy_phase_drift = sum(policy_drift_values) / len(policy_drift_values)

        return TrajectoryDigest(
            trajectory_id=self.trajectory_id,
            window_start_episode=window.start_episode,
            window_end_episode=window.end_episode,
            drift_score=round(drift_score, 4),
            hysteresis_score=round(max(0.0, min(1.0, hysteresis)), 4),
            recovery_score=round(max(0.0, min(1.0, recovery_score)), 4),
            volatility_score=round(max(0.0, min(1.0, volatility)), 4),
            identity_curvature=round(max(0.0, min(1.0, identity_curvature)), 4),
            policy_phase_drift=round(max(0.0, min(1.0, policy_phase_drift)), 4),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trajectory_id": self.trajectory_id,
            "points": [
                {
                    "snapshot": point.snapshot.to_dict(),
                    "regime": point.regime,
                    "observation": point.observation,
                    "intervention": point.intervention,
                    "counterfactual": point.counterfactual,
                    "memory_context": point.memory_context,
                }
                for point in self.points
            ],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "OrganismTrajectory":
        traj = cls(trajectory_id=str(payload.get("trajectory_id", "trajectory")))
        for raw in payload.get("points", []):
            snapshot = OrganismSnapshot.from_dict(dict(raw.get("snapshot", {})))
            traj.append(
                snapshot=snapshot,
                regime=str(raw.get("regime", snapshot.active_regime)),
                observation=dict(raw.get("observation", {})),
                intervention=dict(raw.get("intervention", {})),
                counterfactual=dict(raw.get("counterfactual", {})),
                memory_context=dict(raw.get("memory_context", {})),
            )
        return traj
