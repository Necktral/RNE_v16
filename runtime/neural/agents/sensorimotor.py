"""Cierre sensoriomotor entre observación, modelo causal y acción comprometida."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from runtime.neural.integration.contracts import AuthorityEffect, OrganTrace, SymbiosisIdentity

from .contracts import AgentFinding, AgentReport, AgentRole, AgentState, FindingSeverity


class SensorimotorWorldModelAgent:
    agent_id = "agent-sensorimotor-world-model-v1"

    def assess(self, *, identity: SymbiosisIdentity, observation: Mapping[str, Any], causal_attestation: Mapping[str, Any], organs: Sequence[OrganTrace]) -> AgentReport:
        n4 = next((item for item in organs if item.organ == "N4"), None)
        comparison = dict((n4.candidate or {}).get("canonical_comparison") or {}) if n4 else {}
        committed = comparison.get("temporal_binding") == "committed_action"
        attested = bool(causal_attestation.get("main_variable"))
        findings = []
        if n4 and not committed:
            findings.append(AgentFinding("sensorimotor_action_not_committed", FindingSeverity.WARNING, "N4 no está ligado a la acción final."))
        if not observation:
            state, cls, proposal = AgentState.ABSTAINED, "observation_missing", "measure_environment"
        elif not n4 or not committed or not attested:
            state, cls, proposal = AgentState.DEGRADED, "open_loop", "close_observation_action_causal_loop"
        else:
            state, cls, proposal = AgentState.OBSERVED, "closed_loop_observed", "compare_predicted_and_realized_transition_post_outcome"
        return AgentReport.create(agent_id=self.agent_id, role=AgentRole.SENSORIMOTOR_WORLD_MODEL, identity=identity, state=state, authority_effect=AuthorityEffect.NONE, metrics={"observation_field_count": len(observation), "causal_attestation_present": attested, "committed_n4_binding": committed}, findings=findings, outputs={"evidence_pipeline": ["measure", "classify", "analyze", "deliberate"], "stages": {"measure": {"main_variable": causal_attestation.get("main_variable"), "committed_intervention": comparison.get("committed_intervention")}, "classify": {"sensorimotor_class": cls}, "analyze": {"prediction_error": None, "prediction_error_status": "requires_post_outcome_transition"}, "deliberate": {"proposal": proposal, "actuation_authorized": False}}, "world_model_authority_preserved": True, "decision_influence": "none"})
