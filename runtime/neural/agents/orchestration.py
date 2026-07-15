"""Agente de orquestación para un ciclo verificable de cinco agentes."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from runtime.neural.connectome import ConnectomeRuntime
from runtime.neural.integration.contracts import (
    AuthorityEffect,
    ConsumerReceipt,
    OrganTrace,
    SymbiosisIdentity,
)

from .adversarial import AdversarialAgent
from .connectomics import ConnectomicsAgent
from .contracts import AgentCycleReport, AgentReport, AgentRole, AgentState
from .latent import LatentCommunicationAgent
from .symbiosis import SymbiosisSynergyAgent


class NeuralOrchestrationAgent:
    """Ordena análisis sin reemplazar al coordinador ni al scheduler."""

    agent_id = "agent-orchestration-v1"
    execution_order = (
        AgentRole.CONNECTOMICS,
        AgentRole.ADVERSARIAL,
        AgentRole.LATENT_COMMUNICATION,
        AgentRole.SYMBIOSIS_SYNERGY,
    )

    def __init__(self, *, connectome: ConnectomeRuntime) -> None:
        self.connectomics = ConnectomicsAgent(connectome=connectome)
        self.adversarial = AdversarialAgent()
        self.latent = LatentCommunicationAgent()
        self.symbiosis = SymbiosisSynergyAgent()

    def run_cycle(
        self,
        *,
        identity: SymbiosisIdentity,
        organs: Sequence[OrganTrace],
        receipts: Sequence[ConsumerReceipt],
        connectome_activity: Mapping[str, Any] | None,
    ) -> AgentCycleReport:
        connectomics = self.connectomics.analyze(
            identity=identity,
            activity=connectome_activity,
            organs=organs,
            receipts=receipts,
        )
        adversarial = self.adversarial.inspect(
            identity=identity,
            organs=organs,
            receipts=receipts,
            activity=connectome_activity,
        )
        latent = self.latent.modulate(
            identity=identity,
            organs=organs,
            receipts=receipts,
            quarantined_organs=tuple(
                adversarial.outputs.get("quarantined_organs") or ()
            ),
        )
        symbiosis = self.symbiosis.assess(
            identity=identity,
            organs=organs,
            receipts=receipts,
            connectomics=connectomics,
            latent=latent,
            adversarial=adversarial,
        )
        blocked_roles = [
            report.role.value
            for report in (connectomics, adversarial, latent, symbiosis)
            if report.state is AgentState.BLOCKED
        ]
        degraded_roles = [
            report.role.value
            for report in (connectomics, adversarial, latent, symbiosis)
            if report.state in {AgentState.DEGRADED, AgentState.ABSTAINED}
        ]
        orchestration = AgentReport.create(
            agent_id=self.agent_id,
            role=AgentRole.ORCHESTRATION,
            identity=identity,
            state=(
                AgentState.BLOCKED
                if blocked_roles
                else AgentState.DEGRADED
                if degraded_roles
                else AgentState.OBSERVED
            ),
            authority_effect=AuthorityEffect.NONE,
            metrics={
                "agent_count": 5,
                "blocked_agent_count": len(blocked_roles),
                "degraded_agent_count": len(degraded_roles),
            },
            outputs={
                "execution_order": [role.value for role in self.execution_order],
                "blocked_roles": blocked_roles,
                "degraded_roles": degraded_roles,
                "scheduler_authority_preserved": True,
                "decision_influence": "none",
            },
        )
        return AgentCycleReport.create(
            identity=identity,
            reports=(
                orchestration,
                connectomics,
                latent,
                adversarial,
                symbiosis,
            ),
        )
