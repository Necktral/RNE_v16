"""Agente inmunológico para artefactos, receipts y persistencia neural."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from runtime.neural.integration.contracts import (
    AuthorityEffect,
    ConsumerReceipt,
    ConsumerVerdictClass,
    OrganTrace,
    SymbiosisIdentity,
)

from .contracts import AgentFinding, AgentReport, AgentRole, AgentState, FindingSeverity


class ModelDataImmuneAgent:
    """Propone cuarentena; N0, manifest admission y certificación siguen decidiendo."""

    agent_id = "agent-model-data-immune-v1"

    def assess(
        self,
        *,
        identity: SymbiosisIdentity,
        organs: Sequence[OrganTrace],
        receipts: Sequence[ConsumerReceipt],
        trace_health: Mapping[str, Any],
    ) -> AgentReport:
        findings: list[AgentFinding] = []
        quarantine: set[str] = set()
        artifact_backed = 0
        for organ in organs:
            has_manifest = bool(organ.manifest_sha256)
            has_artifact = bool(organ.artifact_sha256)
            artifact_backed += int(has_manifest and has_artifact)
            if has_manifest != has_artifact:
                quarantine.add(organ.organ)
                findings.append(
                    AgentFinding(
                        "immune_incomplete_artifact_binding",
                        FindingSeverity.CRITICAL,
                        "Manifiesto y artefacto deben estar enlazados juntos.",
                        subject=organ.organ,
                    )
                )
            if organ.fallback_reason and has_artifact:
                findings.append(
                    AgentFinding(
                        "immune_artifact_fell_back",
                        FindingSeverity.WARNING,
                        "Un artefacto configurado degradó al backend de referencia.",
                        subject=organ.organ,
                        evidence_refs=(organ.artifact_sha256 or "artifact",),
                    )
                )

        negative_receipts = []
        for receipt in receipts:
            if receipt.verdict_class in {
                ConsumerVerdictClass.INVALID,
                ConsumerVerdictClass.PERSISTENCE_DEGRADED,
            }:
                quarantine.add(receipt.organ)
                negative_receipts.append(receipt.receipt_id)
                findings.append(
                    AgentFinding(
                        "immune_receipt_integrity_failure",
                        FindingSeverity.CRITICAL,
                        "El recibo es inválido o perdió persistencia durable.",
                        subject=receipt.organ,
                        evidence_refs=(receipt.receipt_id,),
                    )
                )
            elif receipt.verdict_class is ConsumerVerdictClass.REJECTED:
                negative_receipts.append(receipt.receipt_id)

        pending = int(trace_health.get("pending_events", 0) or 0)
        dropped = int(trace_health.get("dropped_events", 0) or 0)
        if pending or dropped or bool(trace_health.get("degraded")):
            findings.append(
                AgentFinding(
                    "immune_trace_persistence_degraded",
                    FindingSeverity.WARNING,
                    "La evidencia neural no está completamente durable.",
                )
            )

        critical = any(item.severity is FindingSeverity.CRITICAL for item in findings)
        warning = any(item.severity is FindingSeverity.WARNING for item in findings)
        state = AgentState.BLOCKED if critical else AgentState.DEGRADED if warning else AgentState.OBSERVED
        proposal = (
            "quarantine_sources_and_reject_training"
            if critical
            else "hold_training_until_persistence_recovers"
            if warning
            else "admit_as_observation_only"
        )
        return AgentReport.create(
            agent_id=self.agent_id,
            role=AgentRole.MODEL_DATA_IMMUNE,
            identity=identity,
            state=state,
            authority_effect=AuthorityEffect.NONE,
            metrics={
                "organ_count": len(organs),
                "artifact_backed_organ_count": artifact_backed,
                "negative_receipt_count": len(negative_receipts),
                "pending_event_count": pending,
                "dropped_event_count": dropped,
            },
            findings=findings,
            outputs={
                "evidence_pipeline": ["measure", "classify", "analyze", "deliberate"],
                "stages": {
                    "measure": {
                        "trace_health": dict(trace_health),
                        "negative_receipt_ids": negative_receipts,
                    },
                    "classify": {
                        "immune_class": "quarantine" if critical else "degraded" if warning else "clean_observation",
                        "quarantined_organs": sorted(quarantine),
                    },
                    "analyze": {
                        "training_evidence_eligible": not critical and not warning,
                        "promotion_evidence_eligible": False,
                    },
                    "deliberate": {
                        "proposal": proposal,
                        "quarantine_authorized": False,
                        "unload_authorized": False,
                        "training_authorized": False,
                    },
                },
                "n0_admission_authority_preserved": True,
                "decision_influence": "none",
            },
        )
