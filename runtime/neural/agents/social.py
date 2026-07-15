"""Agente social/exocórtex para evidencia externa con procedencia."""

from __future__ import annotations

from typing import Any, Mapping

from runtime.neural.integration.contracts import AuthorityEffect, SymbiosisIdentity

from .contracts import AgentFinding, AgentReport, AgentRole, AgentState, FindingSeverity


class SocialExocortexAgent:
    agent_id = "agent-social-exocortex-v1"

    def assess(self, *, identity: SymbiosisIdentity, scenario_metadata: Mapping[str, Any]) -> AgentReport:
        evidence = scenario_metadata.get("external_evidence")
        rows = list(evidence) if isinstance(evidence, (list, tuple)) else []
        traceable = [row for row in rows if isinstance(row, Mapping) and row.get("source_id") and row.get("content_hash")]
        untraceable = len(rows) - len(traceable)
        findings = []
        if untraceable:
            findings.append(AgentFinding("social_external_provenance_missing", FindingSeverity.WARNING, "Evidencia externa sin fuente y hash permanece en cuarentena."))
        if not rows:
            state, cls, proposal = AgentState.ABSTAINED, "exocortex_disconnected", "define_attested_external_channel"
        elif untraceable:
            state, cls, proposal = AgentState.DEGRADED, "external_evidence_untraceable", "quarantine_untraceable_external_evidence"
        else:
            state, cls, proposal = AgentState.OBSERVED, "external_evidence_traceable", "compare_without_granting_authority"
        return AgentReport.create(agent_id=self.agent_id, role=AgentRole.SOCIAL_EXOCORTEX, identity=identity, state=state, authority_effect=AuthorityEffect.NONE, metrics={"external_evidence_count": len(rows), "traceable_evidence_count": len(traceable), "untraceable_evidence_count": untraceable}, findings=findings, outputs={"evidence_pipeline": ["measure", "classify", "analyze", "deliberate"], "stages": {"measure": {"source_ids": sorted(str(row.get("source_id")) for row in traceable)}, "classify": {"social_class": cls}, "analyze": {"trust_score": None, "trust_status": "not_inferred_from_identity_alone"}, "deliberate": {"proposal": proposal, "external_write_authorized": False, "decision_authority": "none"}}, "exocortex_boundary_preserved": True, "decision_influence": "none"})
