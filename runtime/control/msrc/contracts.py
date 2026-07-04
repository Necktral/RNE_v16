"""Contratos del controlador multiescala (MSRC)."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Literal, Optional, Tuple

ScaleActionType = Literal[
    "keep_scale",
    "upgrade_scale",
    "downgrade_scale",
    "fork_probe",
    "commit_probe_result",
    "discard_probe_result",
    "lock_scale_for_n_steps",
]


@dataclass(frozen=True)
class ScaleSpec:
    """Especificación formal de una escala representacional."""

    scale_id: str
    grid_shape: Tuple[int, int]
    resolution_rank: int
    scenario_name: str
    is_executable: bool
    supports_local_structure: bool
    supports_local_intervention: bool
    supports_spatial_memory: bool
    expected_time_cost: float
    expected_artifact_cost: float
    expected_information_gain_prior: float
    memory_compatibility_profile: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ScaleEstimate:
    """Salida estructurada del estimador de demanda cognitiva."""

    required_resolution_score: float
    heterogeneity_score: float
    epistemic_insufficiency_score: float
    risk_score: float
    operational_pressure_score: float
    vram_headroom: float
    vram_pressure: float
    vram_fragmentation_risk: float
    vram_opportunity_score: float
    recommended_scale_candidates: List[str]
    signals: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProbeResult:
    """Resultado de una sonda de escala."""

    target_scale_id: str
    cognitive_gain_delta: float
    viability_preserved: bool
    evidence_score: float
    outcome: Literal["positive", "negative", "inconclusive", "error"]
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ScaleAction:
    """Acción elegida por la política de selección de escala."""

    action_type: ScaleActionType
    target_scale_id: Optional[str] = None
    reason: str = ""
    expected_gain: float = 0.0
    expected_cost_penalty: float = 0.0
    lock_steps: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ScalePolicyState:
    """Estado acumulado para hysteresis, probe y antioscilación."""

    current_scale_id: str
    step_index: int = 0
    upgrade_evidence: int = 0
    downgrade_evidence: int = 0
    cooldown_remaining: int = 0
    lock_remaining: int = 0
    probe_inflight_target: Optional[str] = None
    last_actions: List[str] = field(default_factory=list)
    oscillation_events: int = 0
    upgrade_regret: int = 0
    downgrade_regret: int = 0
    missed_upgrade_regret: int = 0
    regime_history: List[str] = field(default_factory=list)

    def register_action(self, action_type: str) -> None:
        if self.last_actions and self.last_actions[-1] != action_type:
            if {self.last_actions[-1], action_type} <= {"upgrade_scale", "downgrade_scale"}:
                self.oscillation_events += 1
        self.last_actions.append(action_type)
        if len(self.last_actions) > 12:
            self.last_actions = self.last_actions[-12:]

    def register_regime(self, regime_label: str) -> None:
        self.regime_history.append(regime_label)
        if len(self.regime_history) > 24:
            self.regime_history = self.regime_history[-24:]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_scale_id": self.current_scale_id,
            "step_index": self.step_index,
            "upgrade_evidence": self.upgrade_evidence,
            "downgrade_evidence": self.downgrade_evidence,
            "cooldown_remaining": self.cooldown_remaining,
            "lock_remaining": self.lock_remaining,
            "probe_inflight_target": self.probe_inflight_target,
            "last_actions": list(self.last_actions),
            "oscillation_events": self.oscillation_events,
            "upgrade_regret": self.upgrade_regret,
            "downgrade_regret": self.downgrade_regret,
            "missed_upgrade_regret": self.missed_upgrade_regret,
            "regime_history": list(self.regime_history),
        }


@dataclass(frozen=True)
class ScaleDecisionRecord:
    """Registro auditable de una decisión de escala."""

    run_id: str
    episode_id: str
    step_index: int
    current_scale_id: str
    action: ScaleAction
    estimate: ScaleEstimate
    selected_scale_id: str
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "episode_id": self.episode_id,
            "step_index": self.step_index,
            "current_scale_id": self.current_scale_id,
            "action": self.action.to_dict(),
            "estimate": self.estimate.to_dict(),
            "selected_scale_id": self.selected_scale_id,
            "timestamp": self.timestamp,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ScaleTransitionRecord:
    """Registro de transición de escala."""

    run_id: str
    episode_id: str
    action_type: ScaleActionType
    source_scale_id: str
    target_scale_id: str
    reason: str
    estimated_time_cost: float
    estimated_artifact_cost: float
    real_time_cost: float
    real_artifact_cost: float
    ioc_delta: float
    viability_delta: float
    rollback_applied: bool
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CrossScaleMemoryReport:
    """Resultado de validación de memoria entre escalas."""

    contamination_detected: bool
    blocked_fields_count: int
    allowed_fields_count: int
    cross_scale_memory_contamination_rate: float
    sanitized_payload: Dict[str, Any]
    blocked_fields: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
