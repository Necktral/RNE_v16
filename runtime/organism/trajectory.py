"""Modelo trayectorial del organismo con compatibilidad dual T5/T4.

T5 expone trayectoria como unidad soberana (append_point/get_window + metadata evolutiva).
T4 se mantiene por compatibilidad (append/window/digest).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Tuple
from uuid import uuid4

from .constitution import ConstitutionalValidation
from .snapshot import OrganismSnapshot
from .state import OrganismState


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
    """Punto evolutivo de trayectoria con metadatos T5 y compatibilidad T4."""

    step_index: int
    state: OrganismState
    regime: str
    episode_id: str
    timestamp: str
    constitutional_validation: ConstitutionalValidation | None = None
    viability_margin: float = 1.0
    prev_state_id: str | None = None
    observation: Dict[str, Any] = field(default_factory=dict)
    intervention: Dict[str, Any] = field(default_factory=dict)
    counterfactual: Dict[str, Any] = field(default_factory=dict)
    memory_context: Dict[str, Any] = field(default_factory=dict)

    @property
    def snapshot(self) -> OrganismSnapshot:
        """Vista legacy: snapshot derivado del estado soberano."""
        return OrganismSnapshot.from_state(self.state)


@dataclass(frozen=True)
class TrajectoryWindow:
    """Ventana operacional sobre la trayectoria."""

    parent_trajectory_id: str
    organism_id: str
    points: tuple[TrajectoryPoint, ...]
    window_start_index: int = 0

    @property
    def trajectory_id(self) -> str:
        """Alias legacy."""
        return self.parent_trajectory_id

    @property
    def start_episode(self) -> int:
        if not self.points:
            return 0
        return self.points[0].state.episode_count

    @property
    def end_episode(self) -> int:
        if not self.points:
            return 0
        return self.points[-1].state.episode_count

    @property
    def length(self) -> int:
        return len(self.points)

    @property
    def current_state(self) -> OrganismState | None:
        if not self.points:
            return None
        return self.points[-1].state

    @property
    def snapshots(self) -> tuple[OrganismSnapshot, ...]:
        return tuple(point.snapshot for point in self.points)

    @property
    def margin_trajectory(self) -> List[float]:
        return [point.viability_margin for point in self.points]

    @property
    def regime_sequence(self) -> List[str]:
        return [point.regime for point in self.points]

    @property
    def constitutional_validity_sequence(self) -> List[bool]:
        return [
            point.constitutional_validation.is_valid
            if point.constitutional_validation is not None
            else True
            for point in self.points
        ]

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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "parent_trajectory_id": self.parent_trajectory_id,
            "trajectory_id": self.parent_trajectory_id,
            "organism_id": self.organism_id,
            "window_start_index": self.window_start_index,
            "start_episode": self.start_episode,
            "end_episode": self.end_episode,
            "length": self.length,
            "margin_trajectory": self.margin_trajectory,
            "regime_sequence": self.regime_sequence,
            "constitutional_validity_sequence": self.constitutional_validity_sequence,
            "points": [
                {
                    "step_index": p.step_index,
                    "state_id": p.state.state_id,
                    "regime": p.regime,
                    "episode_id": p.episode_id,
                    "viability_margin": p.viability_margin,
                    "prev_state_id": p.prev_state_id,
                }
                for p in self.points
            ],
        }


@dataclass
class OrganismTrajectory:
    """Trayectoria soberana del organismo con adaptador legacy T4."""

    trajectory_id: str = field(default_factory=lambda: f"traj-{uuid4().hex[:12]}")
    organism_id: str = field(default_factory=lambda: f"org-{uuid4().hex[:12]}")
    points: List[TrajectoryPoint] = field(default_factory=list)
    start_timestamp: str = ""
    current_regime: str = "unknown"
    regime_history: List[Tuple[str, int]] = field(default_factory=list)
    constitutional_flow_score: float = 1.0
    total_episodes: int = 0

    @property
    def length(self) -> int:
        return len(self.points)

    @property
    def is_empty(self) -> bool:
        return len(self.points) == 0

    @property
    def current_state(self) -> OrganismState | None:
        if not self.points:
            return None
        return self.points[-1].state

    def append_point(
        self,
        *,
        state: OrganismState,
        regime: str,
        episode_id: str,
        timestamp: str,
        constitutional_validation: ConstitutionalValidation | None = None,
        viability_margin: float = 1.0,
        observation: Mapping[str, Any] | None = None,
        intervention: Mapping[str, Any] | None = None,
        counterfactual: Mapping[str, Any] | None = None,
        memory_context: Mapping[str, Any] | None = None,
    ) -> None:
        step_index = len(self.points)
        prev_state_id = self.points[-1].state.state_id if self.points else None

        point = TrajectoryPoint(
            step_index=step_index,
            state=state,
            regime=regime,
            episode_id=episode_id,
            timestamp=timestamp,
            constitutional_validation=constitutional_validation,
            viability_margin=float(viability_margin),
            prev_state_id=prev_state_id,
            observation=dict(observation or {}),
            intervention=dict(intervention or {}),
            counterfactual=dict(counterfactual or {}),
            memory_context=dict(memory_context or {}),
        )

        self.points.append(point)
        self.total_episodes += 1

        if regime != self.current_regime:
            self.regime_history.append((regime, step_index))
            self.current_regime = regime

        if constitutional_validation is None:
            return

        if constitutional_validation.is_valid:
            # Regla explícita del plan: primer punto válido mantiene 1.0,
            # luego aplica decaimiento suave.
            if step_index > 0:
                self.constitutional_flow_score = round(
                    max(0.0, min(1.0, self.constitutional_flow_score * 0.99)),
                    4,
                )
            return

        if step_index == 0 and constitutional_validation.hard_violation_count <= 1:
            # Compatibilidad: primer punto con violación hard leve no erosiona
            # el score de flujo inicial.
            return

        penalty = 0.10 * float(max(1, constitutional_validation.hard_violation_count))
        self.constitutional_flow_score = round(
            max(0.0, min(1.0, self.constitutional_flow_score - penalty)),
            4,
        )

    def append(
        self,
        *,
        snapshot: OrganismSnapshot,
        regime: str,
        observation: Mapping[str, Any] | None = None,
        intervention: Mapping[str, Any] | None = None,
        counterfactual: Mapping[str, Any] | None = None,
        memory_context: Mapping[str, Any] | None = None,
        episode_id: str | None = None,
        timestamp: str | None = None,
        constitutional_validation: ConstitutionalValidation | None = None,
        viability_margin: float | None = None,
    ) -> None:
        """API legacy T4: snapshot -> append_point soberano."""
        self.append_point(
            state=snapshot.to_state(),
            regime=regime,
            episode_id=episode_id or f"legacy-episode-{snapshot.episode_count}",
            timestamp=timestamp or snapshot.timestamp,
            constitutional_validation=constitutional_validation,
            viability_margin=(
                float(viability_margin)
                if viability_margin is not None
                else float(snapshot.viability.viability_margin)
            ),
            observation=observation,
            intervention=intervention,
            counterfactual=counterfactual,
            memory_context=memory_context,
        )

    def get_window(self, window_size: int = 10) -> TrajectoryWindow:
        size = max(1, int(window_size))
        recent_points = tuple(self.points[-size:]) if len(self.points) > size else tuple(self.points)
        return TrajectoryWindow(
            parent_trajectory_id=self.trajectory_id,
            organism_id=self.organism_id,
            points=recent_points,
            window_start_index=max(0, len(self.points) - size),
        )

    def window(self, size: int = 8) -> TrajectoryWindow:
        """Alias legacy T4."""
        return self.get_window(window_size=size)

    def digest(self, *, window_size: int = 8) -> TrajectoryDigest:
        window = self.get_window(window_size=window_size)
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

        deltas: List[float] = []
        signs: List[int] = []
        margin_values: List[float] = []
        identity_dist: List[float] = []
        policy_drift_values: List[float] = []
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
        recovery_score = max(
            0.0,
            min(
                1.0,
                recovery_gain + (sum(1 for v in margin_values if v > 0) / len(margin_values)) * 0.25,
            ),
        )

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
            "organism_id": self.organism_id,
            "start_timestamp": self.start_timestamp,
            "current_regime": self.current_regime,
            "regime_history": self.regime_history,
            "constitutional_flow_score": round(self.constitutional_flow_score, 4),
            "total_episodes": self.total_episodes,
            "length": self.length,
            "points": [
                {
                    "step_index": point.step_index,
                    "state_id": point.state.state_id,
                    "state": point.state.to_dict(),
                    "snapshot": point.snapshot.to_dict(),
                    "regime": point.regime,
                    "episode_id": point.episode_id,
                    "timestamp": point.timestamp,
                    "viability_margin": point.viability_margin,
                    "prev_state_id": point.prev_state_id,
                    "constitutional_validation": (
                        {
                            "is_valid": point.constitutional_validation.is_valid,
                            "verdict": point.constitutional_validation.verdict,
                            "hard_violation_count": point.constitutional_validation.hard_violation_count,
                            "soft_violation_count": point.constitutional_validation.soft_violation_count,
                            "margin_to_threshold": point.constitutional_validation.margin_to_threshold,
                        }
                        if point.constitutional_validation is not None
                        else None
                    ),
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
        traj = cls(
            trajectory_id=str(payload.get("trajectory_id", "trajectory")),
            organism_id=str(payload.get("organism_id", "org-unknown")),
            start_timestamp=str(payload.get("start_timestamp", "")),
            current_regime=str(payload.get("current_regime", "unknown")),
        )

        raw_regime_history = payload.get("regime_history", [])
        if isinstance(raw_regime_history, list):
            parsed_history: List[Tuple[str, int]] = []
            for item in raw_regime_history:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    parsed_history.append((str(item[0]), int(item[1])))
            traj.regime_history = parsed_history

        for raw in payload.get("points", []):
            if not isinstance(raw, Mapping):
                continue
            if "state" in raw and isinstance(raw.get("state"), Mapping):
                state = OrganismState.from_dict(dict(raw.get("state", {})))
            elif "snapshot" in raw and isinstance(raw.get("snapshot"), Mapping):
                state = OrganismSnapshot.from_dict(dict(raw.get("snapshot", {}))).to_state()
            else:
                state = OrganismState(
                    state_id=str(raw.get("state_id", "")),
                    timestamp=str(raw.get("timestamp", "")),
                    active_regime=str(raw.get("regime", "unknown")),
                    episode_count=int(raw.get("step_index", 0)),
                )

            traj.append_point(
                state=state,
                regime=str(raw.get("regime", state.active_regime)),
                episode_id=str(raw.get("episode_id", f"legacy-episode-{state.episode_count}")),
                timestamp=str(raw.get("timestamp", state.timestamp)),
                constitutional_validation=None,
                viability_margin=float(raw.get("viability_margin", state.viability.viability_margin)),
                observation=dict(raw.get("observation", {})),
                intervention=dict(raw.get("intervention", {})),
                counterfactual=dict(raw.get("counterfactual", {})),
                memory_context=dict(raw.get("memory_context", {})),
            )

        if "constitutional_flow_score" in payload:
            traj.constitutional_flow_score = float(payload.get("constitutional_flow_score", traj.constitutional_flow_score))
        traj.total_episodes = max(traj.total_episodes, int(payload.get("total_episodes", traj.total_episodes)))
        return traj
