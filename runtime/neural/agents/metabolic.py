"""Agente metabólico: hace explícitos presupuesto MSRC y presión física."""

from __future__ import annotations

from typing import Any, Mapping

from runtime.neural.integration.contracts import AuthorityEffect, SymbiosisIdentity

from .contracts import AgentFinding, AgentReport, AgentRole, AgentState, FindingSeverity


class MetabolicBudgetAgent:
    agent_id = "agent-metabolic-budget-v1"

    def assess(self, *, identity: SymbiosisIdentity, resources: Mapping[str, Any]) -> AgentReport:
        budget = bool(resources.get("msrc_budget_available", True))
        pressures = {name: float(resources.get(name, 0.0) or 0.0) for name in ("cpu_pressure", "memory_pressure", "thermal_pressure", "vram_pressure")}
        peak = max(pressures.values(), default=0.0)
        physical_ids = bool(resources.get("msrc_scale_id"))
        findings = []
        if not budget:
            findings.append(AgentFinding("metabolic_msrc_budget_unavailable", FindingSeverity.CRITICAL, "El presupuesto MSRC está cerrado."))
        elif peak >= 0.9:
            findings.append(AgentFinding("metabolic_pressure_high", FindingSeverity.WARNING, "La presión física medida supera el sobre de operación conservador."))
        if not budget:
            state, cls, proposal = AgentState.BLOCKED, "budget_exhausted", "defer_optional_neural_work"
        elif peak >= 0.9:
            state, cls, proposal = AgentState.DEGRADED, "physical_pressure_high", "reduce_optional_compute"
        else:
            state, cls, proposal = AgentState.OBSERVED, "within_declared_budget", "retain_current_budget"
        return AgentReport.create(agent_id=self.agent_id, role=AgentRole.METABOLIC_BUDGET, identity=identity, state=state, authority_effect=AuthorityEffect.NONE, metrics={**pressures, "peak_pressure": peak, "msrc_budget_available": budget}, findings=findings, outputs={"evidence_pipeline": ["measure", "classify", "analyze", "deliberate"], "stages": {"measure": {"resources": dict(resources), "physical_scale_attested": physical_ids}, "classify": {"metabolic_class": cls}, "analyze": {"energy_joules": None, "energy_status": "unmeasured", "budget_signal_observed": True}, "deliberate": {"proposal": proposal, "budget_mutation_authorized": False}}, "n0_resource_authority_preserved": True, "decision_influence": "none"})
