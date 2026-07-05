"""Contracts for operational conjunction and compensation.

The contracts are intentionally JSON-friendly so they can be persisted in the
existing event ledger without schema migrations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Literal, Sequence
from uuid import uuid4

from runtime.storage.records import utc_now_iso


ComputeTier = Literal[
    "tier_0_deterministic",
    "tier_1_local_light",
    "tier_2_specialized",
    "tier_3_external",
]
TaskType = Literal[
    "life_cycle",
    "causal_decision",
    "external_consultation",
    "self_modification",
    "agent_action",
    "maintenance",
]
ValidationStatus = Literal["pass", "warn", "fail"]
CompensationStatus = Literal["none", "applied", "required", "blocked"]
ConfidenceState = Literal["high", "medium", "low", "conflicted"]
FinalDecision = Literal[
    "allow",
    "allow_with_compensation",
    "degrade",
    "block",
    "observe",
]


TIER_ORDER: tuple[ComputeTier, ...] = (
    "tier_0_deterministic",
    "tier_1_local_light",
    "tier_2_specialized",
    "tier_3_external",
)


def tier_rank(tier: ComputeTier) -> int:
    return TIER_ORDER.index(tier)


def clamp01(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


@dataclass(frozen=True, slots=True)
class EvidenceItem:
    evidence_id: str
    kind: str
    source: str
    confidence: float
    payload: Dict[str, Any] = field(default_factory=dict)
    canonical: bool = False

    @classmethod
    def create(
        cls,
        *,
        kind: str,
        source: str,
        confidence: float,
        payload: Dict[str, Any] | None = None,
        canonical: bool = False,
    ) -> "EvidenceItem":
        return cls(
            evidence_id=f"ev-{uuid4().hex[:12]}",
            kind=kind,
            source=source,
            confidence=round(clamp01(confidence), 4),
            payload=dict(payload or {}),
            canonical=bool(canonical),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CausalAssumption:
    name: str
    statement: str
    supported: bool
    strength: float = 0.0
    support_evidence_ids: tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class OperationalConstraints:
    max_compute_tier: ComputeTier = "tier_2_specialized"
    allow_external: bool = False
    resource_pressure_limit: float = 0.85
    require_evidence_for_critical: bool = True
    critical_actions: tuple[str, ...] = (
        "self_modify",
        "consult_external",
        "rollback",
        "shutdown",
    )
    permitted_actions: tuple[str, ...] = (
        "act",
        "observe",
        "explore",
        "sleep",
        "rollback",
        "quarantine",
        "shutdown",
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AgentPolicy:
    role: str
    allowed_actions: tuple[str, ...]
    prohibited_actions: tuple[str, ...] = ()
    budget_units: float = 1.0
    requires_plan_for_critical: bool = True
    rollback_required_for_critical: bool = True
    human_approval_required_actions: tuple[str, ...] = ()

    def allows(self, action: str) -> bool:
        return action in self.allowed_actions and action not in self.prohibited_actions

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class OperationContext:
    operation_id: str
    run_id: str | None
    user_intent: str
    task_type: TaskType
    requested_action: str
    scenario: str | None = None
    required_evidence: tuple[str, ...] = ()
    available_evidence: tuple[EvidenceItem, ...] = ()
    causal_assumptions: tuple[CausalAssumption, ...] = ()
    constraints: OperationalConstraints = field(default_factory=OperationalConstraints)
    agent_policy: AgentPolicy | None = None
    risk_score: float = 0.0
    complexity_score: float = 0.0
    resource_pressure: float = 0.0
    uncertainty_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now_iso)

    @classmethod
    def create(
        cls,
        *,
        run_id: str | None,
        user_intent: str,
        task_type: TaskType,
        requested_action: str,
        scenario: str | None = None,
        required_evidence: Sequence[str] | None = None,
        available_evidence: Sequence[EvidenceItem] | None = None,
        causal_assumptions: Sequence[CausalAssumption] | None = None,
        constraints: OperationalConstraints | None = None,
        agent_policy: AgentPolicy | None = None,
        risk_score: float = 0.0,
        complexity_score: float = 0.0,
        resource_pressure: float = 0.0,
        uncertainty_score: float = 0.0,
        metadata: Dict[str, Any] | None = None,
    ) -> "OperationContext":
        return cls(
            operation_id=f"op-{uuid4().hex[:12]}",
            run_id=run_id,
            user_intent=user_intent,
            task_type=task_type,
            requested_action=requested_action,
            scenario=scenario,
            required_evidence=tuple(required_evidence or ()),
            available_evidence=tuple(available_evidence or ()),
            causal_assumptions=tuple(causal_assumptions or ()),
            constraints=constraints or OperationalConstraints(),
            agent_policy=agent_policy,
            risk_score=round(clamp01(risk_score), 4),
            complexity_score=round(clamp01(complexity_score), 4),
            resource_pressure=round(clamp01(resource_pressure), 4),
            uncertainty_score=round(clamp01(uncertainty_score), 4),
            metadata=dict(metadata or {}),
        )

    def evidence_kinds(self) -> set[str]:
        return {item.kind for item in self.available_evidence}

    def has_evidence(self, kind: str) -> bool:
        return kind in self.evidence_kinds()

    def is_critical_action(self) -> bool:
        return self.requested_action in self.constraints.critical_actions

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["available_evidence"] = [item.to_dict() for item in self.available_evidence]
        data["causal_assumptions"] = [item.to_dict() for item in self.causal_assumptions]
        data["constraints"] = self.constraints.to_dict()
        data["agent_policy"] = self.agent_policy.to_dict() if self.agent_policy else None
        return data


@dataclass(frozen=True, slots=True)
class ComputeRoute:
    selected_compute_tier: ComputeTier
    selected_reasoning_path: str
    reason: str
    estimated_cost: float = 0.0
    expected_quality: float = 0.0
    trace: tuple[Dict[str, Any], ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ValidationFinding:
    code: str
    status: ValidationStatus
    message: str
    validator: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CompensationAction:
    code: str
    status: CompensationStatus
    reason: str
    directive: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class OperationalConjunctionResult:
    operation_id: str
    selected_compute_tier: ComputeTier
    selected_reasoning_path: str
    validation_status: ValidationStatus
    compensation_status: CompensationStatus
    confidence_state: ConfidenceState
    execution_permissions: Dict[str, Any]
    final_decision: FinalDecision
    validation_findings: tuple[ValidationFinding, ...]
    compensations: tuple[CompensationAction, ...]
    trace: tuple[Dict[str, Any], ...]
    context_summary: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now_iso)

    @property
    def blocked(self) -> bool:
        return self.final_decision == "block"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "operation_id": self.operation_id,
            "selected_compute_tier": self.selected_compute_tier,
            "selected_reasoning_path": self.selected_reasoning_path,
            "validation_status": self.validation_status,
            "compensation_status": self.compensation_status,
            "confidence_state": self.confidence_state,
            "execution_permissions": dict(self.execution_permissions),
            "final_decision": self.final_decision,
            "validation_findings": [item.to_dict() for item in self.validation_findings],
            "compensations": [item.to_dict() for item in self.compensations],
            "trace": [dict(item) for item in self.trace],
            "context_summary": dict(self.context_summary),
            "created_at": self.created_at,
        }
