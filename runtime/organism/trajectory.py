"""Trayectoria de organismo RNFE-T5 como unidad soberana.

La trayectoria es la entidad primaria del runtime T5, no el snapshot.
Una trayectoria es una secuencia evolutiva del organismo con:
  - estados (snapshots) como puntos de muestreo
  - flujo constitucional acumulado
  - régimen(es) transitados
  - riesgo secuencial propagado
  - viabilidad marginal dinámica

La trayectoria NO es simplemente List[OrganismState].
Es una entidad con semántica evolutiva propia.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any
from uuid import uuid4

from .state import OrganismState
from .constitution import ConstitutionalValidation


@dataclass(frozen=True)
class TrajectoryPoint:
    """Punto de muestreo en la trayectoria del organismo.

    Combina snapshot con metadata de transición y régimen.
    """

    step_index: int
    state: OrganismState
    regime: str
    episode_id: str
    timestamp: str
    constitutional_validation: ConstitutionalValidation | None = None
    viability_margin: float = 1.0
    prev_state_id: str | None = None


@dataclass
class OrganismTrajectory:
    """Trayectoria soberana del organismo RNFE-T5.

    La trayectoria es la unidad primaria de:
      - evolución del organismo
      - certificación
      - viabilidad
      - riesgo

    No es un mero contenedor de snapshots, sino una entidad
    con dinámica evolutiva propia.

    Attributes:
        trajectory_id: ID único de la trayectoria.
        organism_id: ID del organismo que evoluciona.
        points: Secuencia de puntos de muestreo.
        start_timestamp: Timestamp de inicio de trayectoria.
        current_regime: Régimen actual.
        regime_history: Historia de regímenes transitados.
        constitutional_flow_score: Score acumulado de flujo constitucional.
        total_episodes: Número total de episodios en la trayectoria.
    """

    trajectory_id: str = field(default_factory=lambda: f"traj-{uuid4()}")
    organism_id: str = field(default_factory=lambda: f"org-{uuid4()}")
    points: List[TrajectoryPoint] = field(default_factory=list)
    start_timestamp: str = ""
    current_regime: str = "unknown"
    regime_history: List[Tuple[str, int]] = field(default_factory=list)  # (regime, step_index)
    constitutional_flow_score: float = 1.0
    total_episodes: int = 0

    @property
    def current_state(self) -> OrganismState | None:
        """Estado actual del organismo (último punto)."""
        if not self.points:
            return None
        return self.points[-1].state

    @property
    def length(self) -> int:
        """Longitud de la trayectoria (número de puntos)."""
        return len(self.points)

    @property
    def is_empty(self) -> bool:
        """True si la trayectoria no tiene puntos."""
        return len(self.points) == 0

    def append_point(
        self,
        *,
        state: OrganismState,
        regime: str,
        episode_id: str,
        timestamp: str,
        constitutional_validation: ConstitutionalValidation | None = None,
        viability_margin: float = 1.0,
    ) -> None:
        """Añade un punto a la trayectoria.

        Este método mantiene la coherencia evolutiva de la trayectoria.
        """
        step_index = len(self.points)
        prev_state_id = self.points[-1].state.state_id if self.points else None

        point = TrajectoryPoint(
            step_index=step_index,
            state=state,
            regime=regime,
            episode_id=episode_id,
            timestamp=timestamp,
            constitutional_validation=constitutional_validation,
            viability_margin=viability_margin,
            prev_state_id=prev_state_id,
        )

        self.points.append(point)
        self.total_episodes += 1

        # Update regime tracking
        if regime != self.current_regime:
            self.regime_history.append((regime, step_index))
            self.current_regime = regime

        # Update constitutional flow score
        if constitutional_validation is not None:
            if constitutional_validation.is_valid:
                self.constitutional_flow_score *= 0.99  # slight decay
            else:
                # Penalize by severity
                penalty = 0.10 * constitutional_validation.hard_violation_count
                self.constitutional_flow_score = max(0.0, self.constitutional_flow_score - penalty)

    def get_window(self, window_size: int = 10) -> TrajectoryWindow:
        """Obtiene ventana reciente de la trayectoria.

        Args:
            window_size: Tamaño de la ventana (número de puntos recientes).

        Returns:
            TrajectoryWindow con los últimos N puntos.
        """
        recent_points = self.points[-window_size:] if len(self.points) > window_size else self.points
        return TrajectoryWindow(
            parent_trajectory_id=self.trajectory_id,
            organism_id=self.organism_id,
            points=recent_points,
            window_start_index=max(0, len(self.points) - window_size),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serializa trayectoria a dict."""
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
                    "step_index": p.step_index,
                    "state_id": p.state.state_id,
                    "regime": p.regime,
                    "episode_id": p.episode_id,
                    "timestamp": p.timestamp,
                    "viability_margin": p.viability_margin,
                    "prev_state_id": p.prev_state_id,
                }
                for p in self.points
            ],
        }


@dataclass(frozen=True)
class TrajectoryWindow:
    """Ventana reciente de trayectoria para operaciones de runtime.

    El runtime opera sobre ventanas, no sobre trayectorias completas,
    para mantener eficiencia computacional.
    """

    parent_trajectory_id: str
    organism_id: str
    points: List[TrajectoryPoint]
    window_start_index: int = 0

    @property
    def length(self) -> int:
        """Longitud de la ventana."""
        return len(self.points)

    @property
    def current_state(self) -> OrganismState | None:
        """Estado actual (último punto de la ventana)."""
        if not self.points:
            return None
        return self.points[-1].state

    @property
    def margin_trajectory(self) -> List[float]:
        """Trayectoria de márgenes de viabilidad."""
        return [p.viability_margin for p in self.points]

    @property
    def regime_sequence(self) -> List[str]:
        """Secuencia de regímenes en la ventana."""
        return [p.regime for p in self.points]

    @property
    def constitutional_validity_sequence(self) -> List[bool]:
        """Secuencia de validez constitucional."""
        return [
            p.constitutional_validation.is_valid
            if p.constitutional_validation is not None
            else True
            for p in self.points
        ]

    def to_dict(self) -> Dict[str, Any]:
        """Serializa ventana a dict."""
        return {
            "parent_trajectory_id": self.parent_trajectory_id,
            "organism_id": self.organism_id,
            "window_start_index": self.window_start_index,
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
                }
                for p in self.points
            ],
        }
