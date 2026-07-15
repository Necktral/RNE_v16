"""Agente adversarial: verifica identidad, integridad y techo de autoridad."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence

from runtime.neural.integration.contracts import (
    AuthorityEffect,
    ConsumerReceipt,
    ConsumerVerdictClass,
    OrganTrace,
    SymbiosisIdentity,
    canonical_sha256,
)

from .contracts import (
    AgentFinding,
    AgentReport,
    AgentRole,
    AgentState,
    FindingSeverity,
)


_NEGATIVE_VERDICTS = {
    ConsumerVerdictClass.REJECTED,
    ConsumerVerdictClass.INVALID,
    ConsumerVerdictClass.PERSISTENCE_DEGRADED,
}


class AdversarialAgent:
    """Bloquea inconsistencias estructurales; no sustituye certificación."""

    agent_id = "agent-adversarial-v1"

    def inspect(
        self,
        *,
        identity: SymbiosisIdentity,
        organs: Sequence[OrganTrace],
        receipts: Sequence[ConsumerReceipt],
        activity: Mapping[str, Any] | None,
    ) -> AgentReport:
        findings: list[AgentFinding] = []
        quarantined: set[str] = set()
        organ_names = Counter(organ.organ for organ in organs)
        receipt_ids = Counter(receipt.receipt_id for receipt in receipts)
        organ_map = {organ.organ: organ for organ in organs}

        for organ in organs:
            if organ.identity != identity:
                quarantined.add(organ.organ)
                findings.append(
                    AgentFinding(
                        "adversarial_organ_identity_mismatch",
                        FindingSeverity.CRITICAL,
                        "La traza del órgano cruza una identidad causal distinta.",
                        subject=organ.organ,
                    )
                )
            if organ_names[organ.organ] > 1:
                quarantined.add(organ.organ)
                findings.append(
                    AgentFinding(
                        "adversarial_duplicate_organ_trace",
                        FindingSeverity.CRITICAL,
                        "Existen múltiples trazas para el mismo órgano.",
                        subject=organ.organ,
                    )
                )
            if organ.authority_ceiling == AuthorityEffect.AUTHORITATIVE.value:
                quarantined.add(organ.organ)
                findings.append(
                    AgentFinding(
                        "adversarial_authority_escalation",
                        FindingSeverity.CRITICAL,
                        "Un órgano neural declaró autoridad soberana prohibida.",
                        subject=organ.organ,
                    )
                )
            if organ.candidate is not None and organ.candidate_hash != canonical_sha256(
                organ.candidate
            ):
                quarantined.add(organ.organ)
                findings.append(
                    AgentFinding(
                        "adversarial_candidate_hash_mismatch",
                        FindingSeverity.CRITICAL,
                        "El candidato no coincide con su hash canónico.",
                        subject=organ.organ,
                    )
                )

        for receipt in receipts:
            if receipt_ids[receipt.receipt_id] > 1:
                quarantined.add(receipt.organ)
                findings.append(
                    AgentFinding(
                        "adversarial_receipt_replay",
                        FindingSeverity.CRITICAL,
                        "El identificador de recibo fue observado más de una vez.",
                        subject=receipt.organ,
                        evidence_refs=(receipt.receipt_id,),
                    )
                )
            organ = organ_map.get(receipt.organ)
            if receipt.identity != identity or organ is None:
                quarantined.add(receipt.organ)
                findings.append(
                    AgentFinding(
                        "adversarial_receipt_identity_or_organ_mismatch",
                        FindingSeverity.CRITICAL,
                        "El recibo no pertenece al episodio o no tiene órgano productor.",
                        subject=receipt.organ,
                        evidence_refs=(receipt.receipt_id,),
                    )
                )
                continue
            if receipt.candidate_hash != organ.candidate_hash:
                quarantined.add(receipt.organ)
                findings.append(
                    AgentFinding(
                        "adversarial_receipt_candidate_mismatch",
                        FindingSeverity.CRITICAL,
                        "El recibo referencia un candidato distinto al observado.",
                        subject=receipt.organ,
                        evidence_refs=(receipt.receipt_id,),
                    )
                )
            if receipt.authority_effect not in {
                AuthorityEffect.NONE,
                AuthorityEffect.EVIDENCE_ONLY,
            }:
                quarantined.add(receipt.organ)
                findings.append(
                    AgentFinding(
                        "adversarial_receipt_authority_escalation",
                        FindingSeverity.CRITICAL,
                        "El consumidor intenta elevar evidencia a autoridad de decisión.",
                        subject=receipt.organ,
                        evidence_refs=(receipt.receipt_id,),
                    )
                )
            if receipt.verdict_class in _NEGATIVE_VERDICTS:
                findings.append(
                    AgentFinding(
                        "adversarial_negative_consumer_evidence",
                        FindingSeverity.WARNING,
                        "Un consumidor reportó evidencia negativa o degradada.",
                        subject=receipt.organ,
                        evidence_refs=(receipt.receipt_id,),
                    )
                )

        activity = dict(activity or {})
        if activity.get("authority_effect") not in {None, AuthorityEffect.NONE.value}:
            findings.append(
                AgentFinding(
                    "adversarial_connectome_authority_escalation",
                    FindingSeverity.CRITICAL,
                    "La observación conectómica declaró efecto de autoridad.",
                )
            )
        if activity.get("graph_mutated") is True:
            findings.append(
                AgentFinding(
                    "adversarial_unauthorized_graph_mutation",
                    FindingSeverity.CRITICAL,
                    "El ciclo reportó una mutación conectómica no autorizada.",
                )
            )

        critical_count = sum(
            item.severity is FindingSeverity.CRITICAL for item in findings
        )
        warning_count = sum(
            item.severity is FindingSeverity.WARNING for item in findings
        )
        state = (
            AgentState.BLOCKED
            if critical_count
            else AgentState.DEGRADED
            if warning_count
            else AgentState.OBSERVED
        )
        return AgentReport.create(
            agent_id=self.agent_id,
            role=AgentRole.ADVERSARIAL,
            identity=identity,
            state=state,
            authority_effect=AuthorityEffect.NONE,
            metrics={
                "organ_trace_count": len(organs),
                "receipt_count": len(receipts),
                "critical_count": critical_count,
                "warning_count": warning_count,
            },
            findings=findings,
            outputs={
                "quarantined_organs": sorted(quarantined),
                "decision_influence": "none",
                "certification_required": True,
            },
        )
