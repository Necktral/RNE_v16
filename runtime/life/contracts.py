"""Contratos publicos del Life Kernel RNFE.

Estos tipos son deliberadamente pequenos y JSON-friendly: el runtime vivo los
usa para dejar evidencia auditable sin requerir migraciones de schema.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Literal
from uuid import uuid4

from runtime.storage.records import utc_now_iso


GoalKind = Literal[
    "survival",
    "continuity",
    "risk_reduction",
    "cognitive_gain",
    "exploration",
    "memory_maintenance",
]
GoalStatus = Literal["active", "satisfied", "paused", "failed"]
AutonomyAction = Literal[
    "act",
    "observe",
    "explore",
    "consult_external",
    "self_modify",
    "rollback",
    "quarantine",
    "sleep",
    "shutdown",
]
VitalMode = Literal[
    "normal",
    "conservative",
    "recovery",
    "quarantine",
    "rollback",
    "shutdown_safe",
]
EvolutionProposalStatus = Literal[
    "proposed",
    "sandboxed",
    "shadowing",
    "accepted",
    "committed",
    "reverted",
    "rejected",
]


@dataclass(frozen=True, slots=True)
class GoalState:
    """Objetivo interno persistente del organismo."""

    goal_id: str
    kind: GoalKind
    priority: float
    horizon_episodes: int
    success_metric: str
    risk_budget: float
    status: GoalStatus = "active"
    progress: float = 0.0
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        kind: GoalKind,
        priority: float,
        horizon_episodes: int,
        success_metric: str,
        risk_budget: float,
        metadata: Dict[str, Any] | None = None,
    ) -> "GoalState":
        return cls(
            goal_id=f"goal-{kind}-{uuid4().hex[:10]}",
            kind=kind,
            priority=float(priority),
            horizon_episodes=int(horizon_episodes),
            success_metric=success_metric,
            risk_budget=float(risk_budget),
            metadata=dict(metadata or {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "GoalState":
        return cls(
            goal_id=str(payload.get("goal_id") or f"goal-restored-{uuid4().hex[:10]}"),
            kind=payload.get("kind", "survival"),
            priority=float(payload.get("priority", 1.0)),
            horizon_episodes=int(payload.get("horizon_episodes", 1)),
            success_metric=str(payload.get("success_metric", "")),
            risk_budget=float(payload.get("risk_budget", 0.5)),
            status=payload.get("status", "active"),
            progress=float(payload.get("progress", 0.0)),
            created_at=str(payload.get("created_at") or utc_now_iso()),
            updated_at=str(payload.get("updated_at") or utc_now_iso()),
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass(frozen=True, slots=True)
class AutonomyDecision:
    """Decision soberana de alto nivel para un ciclo vital."""

    action: AutonomyAction
    mode: VitalMode
    reason: str
    priority: float = 0.5
    scenario: str | None = None
    external_input: float | None = None
    directives: Dict[str, Any] = field(default_factory=dict)
    decision_id: str = field(default_factory=lambda: f"dec-{uuid4().hex[:12]}")
    created_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EvolutionProposalV2:
    """Mutacion versionada, auditable y reversible."""

    proposal_id: str
    target: str
    semantic_diff: Dict[str, Any]
    sandbox_result: Dict[str, Any]
    shadow_evidence: Dict[str, Any]
    rollback_plan: Dict[str, Any]
    certificate: Dict[str, Any]
    status: EvolutionProposalStatus = "proposed"
    created_at: str = field(default_factory=utc_now_iso)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        target: str,
        semantic_diff: Dict[str, Any],
        rollback_plan: Dict[str, Any],
        metadata: Dict[str, Any] | None = None,
    ) -> "EvolutionProposalV2":
        return cls(
            proposal_id=f"evo2-{uuid4().hex[:12]}",
            target=target,
            semantic_diff=dict(semantic_diff),
            sandbox_result={},
            shadow_evidence={},
            rollback_plan=dict(rollback_plan),
            certificate={},
            metadata=dict(metadata or {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class VitalSignsSnapshot:
    """Snapshot compacto de salud viva y autonomia operacional."""

    run_id: str
    episode_count: int
    mode: VitalMode
    viability_margin: float
    continuity_score: float
    ioc_proxy: float
    risk_score: float
    memory_purity: float
    cognitive_quality: float
    resource_pressure: float
    recovery_debt: float
    accumulated_drift: float
    reversible: bool
    identity_continuity: float
    certified: bool
    snapshot_id: str = field(default_factory=lambda: f"vital-{uuid4().hex[:12]}")
    created_at: str = field(default_factory=utc_now_iso)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_stable(self) -> bool:
        return (
            self.certified
            and self.viability_margin >= 0.45
            and self.continuity_score >= 0.60
            and self.risk_score < 0.60
            and self.memory_purity >= 0.75
        )

    @property
    def is_restorable(self) -> bool:
        """Estado sano y REVERSIBLE al que es seguro rodar atrás (refugio E5).

        A diferencia de ``is_stable``, NO exige la certificación formal del episodio
        (``certified``): en vida real las certificaciones suelen quedar ``rejected``
        por closure/trace, de modo que ``is_stable`` es inalcanzable y el organismo
        nunca acumula un refugio — el callejón sin salida que dejó a aeon-01 atascado
        en cuarentena sin a dónde volver. Para el refugio lo que importa es la salud
        genuina: viabilidad, continuidad, riesgo acotado, memoria pura y REVERSIBILIDAD.
        Umbrales de salud MÁS estrictos que ``is_stable`` para compensar la ausencia
        del sello de certificación: solo un estado realmente bueno sirve de refugio.
        """
        return (
            self.reversible
            and self.viability_margin >= 0.55
            and self.continuity_score >= 0.75
            and self.risk_score < 0.50
            and self.memory_purity >= 0.85
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "VitalSignsSnapshot":
        return cls(
            run_id=str(payload.get("run_id") or "unknown"),
            episode_count=int(payload.get("episode_count", 0)),
            mode=payload.get("mode", "normal"),
            viability_margin=float(payload.get("viability_margin", 1.0)),
            continuity_score=float(payload.get("continuity_score", 1.0)),
            ioc_proxy=float(payload.get("ioc_proxy", 0.0)),
            risk_score=float(payload.get("risk_score", 0.0)),
            memory_purity=float(payload.get("memory_purity", 1.0)),
            cognitive_quality=float(payload.get("cognitive_quality", 0.5)),
            resource_pressure=float(payload.get("resource_pressure", 0.0)),
            recovery_debt=float(payload.get("recovery_debt", 0.0)),
            accumulated_drift=float(payload.get("accumulated_drift", 0.0)),
            reversible=bool(payload.get("reversible", True)),
            identity_continuity=float(payload.get("identity_continuity", 1.0)),
            certified=bool(payload.get("certified", False)),
            snapshot_id=str(payload.get("snapshot_id") or f"vital-{uuid4().hex[:12]}"),
            created_at=str(payload.get("created_at") or utc_now_iso()),
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass(frozen=True, slots=True)
class RestoredIdentity:
    """Identidad viva restaurada desde el ultimo checkpoint.

    B41: separa los tres ejes. ``run_id`` es el de la corrida bajo la que se guardó
    el checkpoint (queda como ``run_id`` ANTERIOR en la genealogía de corridas);
    ``organism_id`` es el genoma persistente que el kernel adopta como clave (con
    fallback legado ``organism_id := run_id``); ``lineage_id`` es el linaje μ_t.
    """

    run_id: str
    organism_state: Any
    lineage: Any
    goals: list[GoalState]
    vital_signs: VitalSignsSnapshot | None
    total_steps: int
    scenario_index: int
    checkpoint_payload: Dict[str, Any]
    checkpoint_artifact_id: str | None = None
    organism_id: str = ""
    lineage_id: str = ""


@dataclass(frozen=True, slots=True)
class LifeStepResult:
    """Resultado de un ciclo vital del Life Kernel."""

    run_id: str
    step_index: int
    decision: AutonomyDecision
    vital_signs: VitalSignsSnapshot
    goals: list[GoalState]
    episode_result: Dict[str, Any] | None
    checkpoint_artifact_id: str | None
    msrc: Dict[str, Any] = field(default_factory=dict)
    operational: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "step_index": self.step_index,
            "decision": self.decision.to_dict(),
            "vital_signs": self.vital_signs.to_dict(),
            "goals": [goal.to_dict() for goal in self.goals],
            "episode_result": self.episode_result,
            "checkpoint_artifact_id": self.checkpoint_artifact_id,
            "msrc": dict(self.msrc),
            "operational": dict(self.operational),
        }
