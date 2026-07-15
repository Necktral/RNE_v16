"""Contratos deterministas para la capa de agentes neurales.

Los agentes sólo observan, contrastan y proponen modulación acotada. Ningún
contrato de este módulo concede autoridad de decisión ni mutación del conectoma.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Mapping, Sequence

from runtime.neural.integration.contracts import (
    AuthorityEffect,
    SymbiosisIdentity,
    canonical_sha256,
    canonicalize,
)


NEURAL_AGENT_REPORT_SCHEMA_VERSION = "rnfe-neural-agent-report-v1"
NEURAL_AGENT_CYCLE_SCHEMA_VERSION = "rnfe-neural-agent-cycle-v1"
NEURAL_SPECIALIZED_AGENT_BUNDLE_SCHEMA_VERSION = "rnfe-neural-agent-extensions-v1"


class AgentRole(str, Enum):
    ORCHESTRATION = "orchestration"
    CONNECTOMICS = "connectomics"
    LATENT_COMMUNICATION = "latent_communication"
    ADVERSARIAL = "adversarial"
    SYMBIOSIS_SYNERGY = "symbiosis_synergy"
    METACOGNITIVE_EPISTEMIC = "metacognitive_epistemic"
    MEMORY_CONSOLIDATION = "memory_consolidation"
    PEDAGOGICAL_TEACHER = "pedagogical_teacher"
    MODEL_DATA_IMMUNE = "model_data_immune"
    CURRICULUM_LEARNING = "curriculum_learning"
    SENSORIMOTOR_WORLD_MODEL = "sensorimotor_world_model"
    INTEROCEPTIVE_HOMEOSTATIC = "interoceptive_homeostatic"
    METABOLIC_BUDGET = "metabolic_budget"
    DEVELOPMENT_LINEAGE = "development_lineage"
    HORIZONTAL_CREATIVITY = "horizontal_creativity"
    SOCIAL_EXOCORTEX = "social_exocortex"


CORE_AGENT_ROLES = frozenset(
    {
        AgentRole.ORCHESTRATION,
        AgentRole.CONNECTOMICS,
        AgentRole.LATENT_COMMUNICATION,
        AgentRole.ADVERSARIAL,
        AgentRole.SYMBIOSIS_SYNERGY,
    }
)


class AgentState(str, Enum):
    OBSERVED = "observed"
    DEGRADED = "degraded"
    ABSTAINED = "abstained"
    BLOCKED = "blocked"


class FindingSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass(frozen=True, slots=True)
class AgentFinding:
    code: str
    severity: FindingSeverity
    detail: str
    subject: str | None = None
    evidence_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.code.strip() or not self.detail.strip():
            raise ValueError("neural_agent_finding_identity_required")
        if any(not isinstance(item, str) or not item for item in self.evidence_refs):
            raise ValueError("neural_agent_finding_evidence_invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity.value,
            "detail": self.detail,
            "subject": self.subject,
            "evidence_refs": list(self.evidence_refs),
        }


@dataclass(frozen=True, slots=True)
class GainModulation:
    modulation_id: str
    source: str
    target: str
    latent_vector: tuple[float, float, float, float]
    proposed_gain: float
    reason: str
    evidence_refs: tuple[str, ...]
    apply_authorized: bool = False
    authority_effect: AuthorityEffect = AuthorityEffect.NONE

    def __post_init__(self) -> None:
        required = (self.modulation_id, self.source, self.target, self.reason)
        if not all(str(value).strip() for value in required):
            raise ValueError("latent_modulation_identity_required")
        if len(self.latent_vector) != 4 or any(
            not math.isfinite(float(value)) for value in self.latent_vector
        ):
            raise ValueError("latent_modulation_vector_invalid")
        if not math.isfinite(self.proposed_gain) or not 0.75 <= self.proposed_gain <= 1.25:
            raise ValueError("latent_modulation_gain_out_of_bounds")
        if self.apply_authorized or self.authority_effect is not AuthorityEffect.NONE:
            raise ValueError("latent_modulation_authority_forbidden")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["latent_vector"] = list(self.latent_vector)
        data["evidence_refs"] = list(self.evidence_refs)
        data["authority_effect"] = self.authority_effect.value
        return data


@dataclass(frozen=True, slots=True)
class AgentReport:
    agent_id: str
    role: AgentRole
    identity: SymbiosisIdentity
    state: AgentState
    authority_effect: AuthorityEffect
    metrics: Mapping[str, Any]
    findings: tuple[AgentFinding, ...]
    outputs: Mapping[str, Any]
    report_hash: str
    experimental: bool = True
    schema_version: str = NEURAL_AGENT_REPORT_SCHEMA_VERSION

    @classmethod
    def create(
        cls,
        *,
        agent_id: str,
        role: AgentRole,
        identity: SymbiosisIdentity,
        state: AgentState,
        authority_effect: AuthorityEffect,
        metrics: Mapping[str, Any] | None = None,
        findings: Sequence[AgentFinding] = (),
        outputs: Mapping[str, Any] | None = None,
    ) -> "AgentReport":
        if not agent_id.strip():
            raise ValueError("neural_agent_id_required")
        if authority_effect not in {AuthorityEffect.NONE, AuthorityEffect.EVIDENCE_ONLY}:
            raise ValueError("neural_agent_authority_forbidden")
        ordered_findings = tuple(
            sorted(findings, key=lambda item: (item.severity.value, item.code, item.subject or ""))
        )
        normalized_metrics = canonicalize(dict(metrics or {}))
        normalized_outputs = canonicalize(dict(outputs or {}))
        payload = {
            "schema_version": NEURAL_AGENT_REPORT_SCHEMA_VERSION,
            "agent_id": agent_id,
            "role": role.value,
            "experimental": True,
            **identity.to_dict(),
            "state": state.value,
            "authority_effect": authority_effect.value,
            "metrics": normalized_metrics,
            "findings": [item.to_dict() for item in ordered_findings],
            "outputs": normalized_outputs,
        }
        return cls(
            agent_id=agent_id,
            role=role,
            identity=identity,
            state=state,
            authority_effect=authority_effect,
            metrics=normalized_metrics,
            findings=ordered_findings,
            outputs=normalized_outputs,
            report_hash=canonical_sha256(payload),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "agent_id": self.agent_id,
            "role": self.role.value,
            "experimental": self.experimental,
            **self.identity.to_dict(),
            "state": self.state.value,
            "authority_effect": self.authority_effect.value,
            "metrics": canonicalize(dict(self.metrics)),
            "findings": [item.to_dict() for item in self.findings],
            "outputs": canonicalize(dict(self.outputs)),
            "report_hash": self.report_hash,
        }


@dataclass(frozen=True, slots=True)
class AgentCycleReport:
    identity: SymbiosisIdentity
    reports: tuple[AgentReport, ...]
    blocked: bool
    cycle_hash: str
    experimental: bool = True
    schema_version: str = NEURAL_AGENT_CYCLE_SCHEMA_VERSION

    @classmethod
    def create(
        cls,
        *,
        identity: SymbiosisIdentity,
        reports: Sequence[AgentReport],
    ) -> "AgentCycleReport":
        ordered = tuple(reports)
        expected = set(CORE_AGENT_ROLES)
        roles = [report.role for report in ordered]
        if len(roles) != len(expected) or set(roles) != expected:
            raise ValueError("neural_agent_cycle_requires_five_unique_roles")
        if any(report.identity != identity for report in ordered):
            raise ValueError("neural_agent_cycle_identity_mismatch")
        blocked = any(report.state is AgentState.BLOCKED for report in ordered)
        payload = {
            "schema_version": NEURAL_AGENT_CYCLE_SCHEMA_VERSION,
            **identity.to_dict(),
            "experimental": True,
            "blocked": blocked,
            "reports": [report.to_dict() for report in ordered],
        }
        return cls(
            identity=identity,
            reports=ordered,
            blocked=blocked,
            cycle_hash=canonical_sha256(payload),
        )

    def by_role(self, role: AgentRole) -> AgentReport:
        return next(report for report in self.reports if report.role is role)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            **self.identity.to_dict(),
            "experimental": self.experimental,
            "blocked": self.blocked,
            "reports": [report.to_dict() for report in self.reports],
            "cycle_hash": self.cycle_hash,
        }


@dataclass(frozen=True, slots=True)
class SpecializedAgentBundle:
    """Extensiones especializadas sin alterar el ciclo estable de cinco agentes."""

    identity: SymbiosisIdentity
    reports: tuple[AgentReport, ...]
    bundle_hash: str
    experimental: bool = True
    schema_version: str = NEURAL_SPECIALIZED_AGENT_BUNDLE_SCHEMA_VERSION

    @classmethod
    def create(
        cls,
        *,
        identity: SymbiosisIdentity,
        reports: Sequence[AgentReport],
    ) -> "SpecializedAgentBundle":
        ordered = tuple(sorted(reports, key=lambda report: report.role.value))
        if not ordered:
            raise ValueError("specialized_agent_bundle_requires_report")
        roles = [report.role for report in ordered]
        if len(set(roles)) != len(roles):
            raise ValueError("specialized_agent_bundle_duplicate_role")
        if any(role in CORE_AGENT_ROLES for role in roles):
            raise ValueError("specialized_agent_bundle_core_role_forbidden")
        if any(report.identity != identity for report in ordered):
            raise ValueError("specialized_agent_bundle_identity_mismatch")
        payload = {
            "schema_version": NEURAL_SPECIALIZED_AGENT_BUNDLE_SCHEMA_VERSION,
            **identity.to_dict(),
            "experimental": True,
            "reports": [report.to_dict() for report in ordered],
        }
        return cls(
            identity=identity,
            reports=ordered,
            bundle_hash=canonical_sha256(payload),
        )

    def by_role(self, role: AgentRole) -> AgentReport:
        return next(report for report in self.reports if report.role is role)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            **self.identity.to_dict(),
            "experimental": self.experimental,
            "reports": [report.to_dict() for report in self.reports],
            "bundle_hash": self.bundle_hash,
        }
