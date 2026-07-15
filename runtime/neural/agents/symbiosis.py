"""Agente de simbiosis y sinergia basado en consumo comprobable."""

from __future__ import annotations

from typing import Sequence

from runtime.neural.integration.contracts import (
    AuthorityEffect,
    ConsumerReceipt,
    OrganTrace,
    SymbiosisIdentity,
)

from .contracts import (
    AgentFinding,
    AgentReport,
    AgentRole,
    AgentState,
    FindingSeverity,
)


class SymbiosisSynergyAgent:
    """Mide integración exacta; no confunde presencia con cooperación."""

    agent_id = "agent-symbiosis-synergy-v1"

    def assess(
        self,
        *,
        identity: SymbiosisIdentity,
        organs: Sequence[OrganTrace],
        receipts: Sequence[ConsumerReceipt],
        connectomics: AgentReport,
        latent: AgentReport,
        adversarial: AgentReport,
    ) -> AgentReport:
        active = {
            organ.organ
            for organ in organs
            if organ.candidate_hash and organ.effective_mode != "off"
        }
        receipt_count = {
            organ: sum(receipt.organ == organ for receipt in receipts) for organ in active
        }
        connected = {organ for organ, count in receipt_count.items() if count > 0}
        isolated = sorted(active - connected)
        coverage = len(connected) / len(active) if active else None
        findings: list[AgentFinding] = []
        for organ in isolated:
            findings.append(
                AgentFinding(
                    "symbiosis_isolated_active_organ",
                    FindingSeverity.WARNING,
                    "El órgano está activo pero no hay evidencia de consumo por otro subsistema.",
                    subject=organ,
                )
            )

        if adversarial.state is AgentState.BLOCKED:
            synergy_state = "blocked"
            state = AgentState.BLOCKED
            findings.append(
                AgentFinding(
                    "symbiosis_adversarial_block",
                    FindingSeverity.CRITICAL,
                    "La sinergia no puede afirmarse mientras exista evidencia en cuarentena.",
                    evidence_refs=(adversarial.report_hash,),
                )
            )
        elif not active:
            synergy_state = "inactive"
            state = AgentState.ABSTAINED
        elif isolated:
            synergy_state = "fragmented"
            state = AgentState.DEGRADED
        elif int(latent.metrics.get("modulation_count") or 0) == 0:
            synergy_state = "connected_unmodulated"
            state = AgentState.DEGRADED
            findings.append(
                AgentFinding(
                    "symbiosis_latent_channel_unmeasured",
                    FindingSeverity.WARNING,
                    "Hay consumo conectado pero no evidencia informativa para modular ganancia.",
                    evidence_refs=(latent.report_hash,),
                )
            )
        else:
            synergy_state = "integrated"
            state = AgentState.OBSERVED

        matrix = {
            organ: {
                "active": True,
                "consumer_receipt_count": receipt_count[organ],
                "connected": organ in connected,
            }
            for organ in sorted(active)
        }
        return AgentReport.create(
            agent_id=self.agent_id,
            role=AgentRole.SYMBIOSIS_SYNERGY,
            identity=identity,
            state=state,
            authority_effect=AuthorityEffect.EVIDENCE_ONLY,
            metrics={
                "active_organ_count": len(active),
                "connected_organ_count": len(connected),
                "connectivity_coverage": coverage,
                "latent_modulation_count": int(
                    latent.metrics.get("modulation_count") or 0
                ),
            },
            findings=findings,
            outputs={
                "synergy_state": synergy_state,
                "integration_matrix": matrix,
                "isolated_organs": isolated,
                "connectomics_report_hash": connectomics.report_hash,
                "latent_report_hash": latent.report_hash,
                "adversarial_report_hash": adversarial.report_hash,
                "decision_influence": "none",
            },
        )
