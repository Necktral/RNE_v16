"""Agente de memoria: audita procedencia y propone consolidación gobernada."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence

from runtime.neural.integration.contracts import (
    AuthorityEffect,
    ConsumerReceipt,
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


class MemoryConsolidationAgent:
    """Conecta N3/N5 con MFM sin escribir ni promocionar memoria directamente."""

    agent_id = "agent-memory-consolidation-v1"

    def assess(
        self,
        *,
        identity: SymbiosisIdentity,
        memory_hits: Sequence[Mapping[str, Any]],
        organs: Sequence[OrganTrace],
        receipts: Sequence[ConsumerReceipt],
    ) -> AgentReport:
        hits = [dict(item) for item in memory_hits]
        refs = [_memory_ref(item) for item in hits]
        traceable_refs = [ref for ref in refs if ref is not None]
        ref_counts = Counter(traceable_refs)
        duplicate_refs = sorted(ref for ref, count in ref_counts.items() if count > 1)
        untraceable_count = len(hits) - len(traceable_refs)
        current_scenario = identity.scenario_id.split("@", 1)[0]
        cross_scenario_refs = sorted(
            ref or canonical_sha256(item)
            for item, ref in zip(hits, refs)
            if (scenario := _scenario_name(item)) is not None
            and scenario.split("@", 1)[0] != current_scenario
        )
        organ_map = {organ.organ: organ for organ in organs}
        n3 = organ_map.get("N3")
        n5 = organ_map.get("N5")
        n3_candidate = _candidate(n3)
        n5_candidate = _candidate(n5)
        n5_candidates = n5_candidate.get("memory_candidates")
        n5_candidate_count = len(n5_candidates) if isinstance(n5_candidates, list) else 0
        n3_receipt_count = sum(receipt.organ == "N3" for receipt in receipts)
        n5_receipt_count = sum(receipt.organ == "N5" for receipt in receipts)

        findings: list[AgentFinding] = []
        if untraceable_count:
            findings.append(
                AgentFinding(
                    "memory_provenance_missing",
                    FindingSeverity.WARNING,
                    "Hay memorias recuperadas sin id de episodio o registro.",
                )
            )
        if duplicate_refs:
            findings.append(
                AgentFinding(
                    "memory_duplicate_retrieval",
                    FindingSeverity.WARNING,
                    "La recuperación repite referencias que deben deduplicarse.",
                    evidence_refs=tuple(duplicate_refs),
                )
            )
        if cross_scenario_refs:
            findings.append(
                AgentFinding(
                    "memory_cross_scenario_requires_attestation",
                    FindingSeverity.WARNING,
                    "La memoria cruza escenario y necesita compatibilidad certificada.",
                    evidence_refs=tuple(cross_scenario_refs),
                )
            )

        if not hits and n5_candidate_count == 0:
            memory_class = "empty"
            report_state = AgentState.ABSTAINED
            proposal = "no_consolidation_candidate"
        elif untraceable_count or cross_scenario_refs:
            memory_class = "provenance_degraded"
            report_state = AgentState.DEGRADED
            proposal = "quarantine_untraceable_or_unattested_memory"
        elif duplicate_refs:
            memory_class = "deduplication_required"
            report_state = AgentState.DEGRADED
            proposal = "deduplicate_before_mfm_gate"
        else:
            memory_class = "traceable_candidate"
            report_state = AgentState.OBSERVED
            proposal = "defer_to_existing_mfm_certification_gate"

        unique_traceable_refs = sorted(set(traceable_refs))
        return AgentReport.create(
            agent_id=self.agent_id,
            role=AgentRole.MEMORY_CONSOLIDATION,
            identity=identity,
            state=report_state,
            authority_effect=AuthorityEffect.NONE,
            metrics={
                "retrieved_count": len(hits),
                "traceable_count": len(traceable_refs),
                "untraceable_count": untraceable_count,
                "duplicate_reference_count": len(duplicate_refs),
                "cross_scenario_count": len(cross_scenario_refs),
                "n3_receipt_count": n3_receipt_count,
                "n5_receipt_count": n5_receipt_count,
                "n5_memory_candidate_count": n5_candidate_count,
            },
            findings=findings,
            outputs={
                "evidence_pipeline": ["measure", "classify", "analyze", "deliberate"],
                "stages": {
                    "measure": {
                        "retrieved_refs": unique_traceable_refs,
                        "n3_measurement_status": n3_candidate.get("measurement_status"),
                        "n5_candidate_count": n5_candidate_count,
                    },
                    "classify": {
                        "memory_class": memory_class,
                        "duplicate_refs": duplicate_refs,
                        "cross_scenario_refs": cross_scenario_refs,
                    },
                    "analyze": {
                        "consolidation_candidate_refs": unique_traceable_refs,
                        "provenance_complete": untraceable_count == 0,
                        "certificate_observed": False,
                    },
                    "deliberate": {
                        "proposal": proposal,
                        "writes_memory": False,
                        "promotion_authorized": False,
                        "certification_required": True,
                    },
                },
                "mfm_authority_preserved": True,
                "smg_authority_preserved": True,
                "decision_influence": "none",
            },
        )


def _memory_ref(item: Mapping[str, Any]) -> str | None:
    value = item.get("episode_id") or item.get("id") or item.get("memory_id")
    return str(value) if value is not None and value != "" else None


def _scenario_name(item: Mapping[str, Any]) -> str | None:
    metadata = item.get("metadata") if isinstance(item.get("metadata"), Mapping) else {}
    value = item.get("scenario_name") or metadata.get("scenario_name")
    return str(value) if value is not None and value != "" else None


def _candidate(organ: OrganTrace | None) -> dict[str, Any]:
    if organ is None or not isinstance(organ.candidate, Mapping):
        return {}
    return dict(organ.candidate)
