"""Agente de desarrollo: linaje, evolución propuesta y reversibilidad."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from runtime.neural.integration.contracts import AuthorityEffect, OrganTrace, SymbiosisIdentity

from .contracts import AgentFinding, AgentReport, AgentRole, AgentState, FindingSeverity


class DevelopmentLineageAgent:
    agent_id = "agent-development-lineage-v1"

    def assess(self, *, identity: SymbiosisIdentity, viability: Mapping[str, Any], organs: Sequence[OrganTrace]) -> AgentReport:
        n6 = next((item for item in organs if item.organ == "N6"), None)
        proposal = dict(n6.candidate or {}) if n6 else {}
        token = str(proposal.get("rollback_token") or "")
        rollback_required = bool(viability.get("rollback_required"))
        lineage_bound = bool(identity.lineage_id and str(proposal.get("lineage_id") or identity.lineage_id) == identity.lineage_id)
        findings = []
        if proposal and not token:
            findings.append(AgentFinding("development_rollback_token_missing", FindingSeverity.CRITICAL, "La propuesta N6 carece de token de rollback.", subject="N6"))
        if rollback_required:
            findings.append(AgentFinding("development_rollback_required", FindingSeverity.WARNING, "Viabilidad exige refugio o rollback."))
        if proposal and not token:
            state, cls, next_step = AgentState.BLOCKED, "irreversible_proposal", "quarantine_n6_proposal"
        elif rollback_required:
            state, cls, next_step = AgentState.DEGRADED, "rollback_required", "require_healthy_checkpoint"
        else:
            state, cls, next_step = AgentState.OBSERVED, "lineage_stable_observation", "retain_shadow_evolution"
        return AgentReport.create(agent_id=self.agent_id, role=AgentRole.DEVELOPMENT_LINEAGE, identity=identity, state=state, authority_effect=AuthorityEffect.NONE, metrics={"n6_proposal_present": bool(proposal), "rollback_token_present": bool(token), "rollback_required": rollback_required, "lineage_bound": lineage_bound}, findings=findings, outputs={"evidence_pipeline": ["measure", "classify", "analyze", "deliberate"], "stages": {"measure": {"lineage_id": identity.lineage_id, "mutation_type": proposal.get("mutation_type"), "rollback_token": token or None}, "classify": {"development_class": cls}, "analyze": {"reversibility_observed": bool(token) if proposal else None}, "deliberate": {"proposal": next_step, "mutation_authorized": False, "rollback_authorized": False}}, "lineage_authority_preserved": True, "decision_influence": "none"})
