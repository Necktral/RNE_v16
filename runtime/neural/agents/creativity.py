"""Creatividad horizontal medible sobre familias y alternativas existentes."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from runtime.neural.integration.contracts import AuthorityEffect, SymbiosisIdentity

from .contracts import AgentFinding, AgentReport, AgentRole, AgentState, FindingSeverity


class HorizontalCreativityAgent:
    agent_id = "agent-horizontal-creativity-v1"

    def assess(self, *, identity: SymbiosisIdentity, reasoning: Mapping[str, Any], memory_hits: Sequence[Mapping[str, Any]]) -> AgentReport:
        sequence = [str(item).upper() for item in reasoning.get("sequence") or ()]
        state_map = dict(reasoning.get("state") or {})
        viewpoints = sum(key in state_map for key in ("abd_hypothesis", "ana_pattern", "cau_link", "ctf_checked", "ded_valid", "prob_point"))
        cross_memory = len({str(item.get("scenario_name") or item.get("scenario") or "") for item in memory_hits if item.get("scenario_name") or item.get("scenario")})
        if not sequence:
            state, cls, proposal = AgentState.ABSTAINED, "reasoning_absent", "collect_reasoning_trace"
        elif viewpoints < 2:
            state, cls, proposal = AgentState.DEGRADED, "narrow_search", "propose_independent_reasoning_family"
        else:
            state, cls, proposal = AgentState.OBSERVED, "horizontal_breadth_observed", "test_novel_alternative_in_shadow"
        return AgentReport.create(agent_id=self.agent_id, role=AgentRole.HORIZONTAL_CREATIVITY, identity=identity, state=state, authority_effect=AuthorityEffect.NONE, metrics={"family_count": len(set(sequence)), "viewpoint_count": viewpoints, "memory_domain_count": cross_memory}, outputs={"evidence_pipeline": ["measure", "classify", "analyze", "deliberate"], "stages": {"measure": {"families": sequence, "observed_viewpoints": viewpoints}, "classify": {"creativity_class": cls}, "analyze": {"novelty": None, "novelty_status": "unmeasured_without_outcome_comparison", "mamba2_transport_required": False}, "deliberate": {"proposal": proposal, "alternative_selection_authorized": False}}, "horizontal_not_latent_transport": True, "decision_influence": "none"})
