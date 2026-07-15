"""Agente metacognitivo: mide certeza comprometida sin gobernar META."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from runtime.neural.integration.contracts import (
    AuthorityEffect,
    ConsumerReceipt,
    ConsumerVerdictClass,
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


class MetacognitiveEpistemicAgent:
    """Integra señales de META y PROB; nunca inventa calibración ausente."""

    agent_id = "agent-metacognitive-epistemic-v1"

    def assess(
        self,
        *,
        identity: SymbiosisIdentity,
        reasoning: Mapping[str, Any],
        organs: Sequence[OrganTrace],
        receipts: Sequence[ConsumerReceipt],
    ) -> AgentReport:
        state = _mapping(reasoning.get("state"))
        validation = _mapping(reasoning.get("sequence_validation"))
        sequence = tuple(str(item).upper() for item in reasoning.get("sequence") or ())
        prob_point = _finite_unit(state.get("prob_point"))
        prob_lcb = _finite_unit(state.get("prob_lcb"))
        cau = _mapping(state.get("cau_link"))
        ctf = _mapping(state.get("ctf_checked"))
        cau_support = _optional_bool(cau.get("helps_goal"))
        ctf_support = _optional_bool(ctf.get("supports_choice"))
        causal_conflict = (
            cau_support is not None
            and ctf_support is not None
            and cau_support != ctf_support
        )
        interval_width = (
            round(max(0.0, prob_point - prob_lcb), 6)
            if prob_point is not None and prob_lcb is not None
            else None
        )
        sequence_validated = (
            bool(validation.get("validated_passed"))
            if "validated_passed" in validation
            else None
        )
        measured = {
            "prob_point": prob_point is not None,
            "prob_lcb": prob_lcb is not None,
            "cau_support": cau_support is not None,
            "ctf_support": ctf_support is not None,
            "sequence_validation": sequence_validated is not None,
        }
        coverage = sum(measured.values()) / len(measured)

        findings: list[AgentFinding] = []
        if prob_point is None or prob_lcb is None:
            epistemic_class = "unmeasured"
            report_state = AgentState.ABSTAINED
            deliberation = "require_probabilistic_measurement"
            escalation_targets = ["PROB"]
            findings.append(
                AgentFinding(
                    "epistemic_calibration_missing",
                    FindingSeverity.WARNING,
                    "Falta punto posterior o cota inferior; no se declara certeza.",
                    evidence_refs=tuple(sequence),
                )
            )
        elif causal_conflict:
            epistemic_class = "measured_conflicted"
            report_state = AgentState.DEGRADED
            deliberation = "propose_deeper_critique_or_teacher_consultation"
            escalation_targets = ["DIA_ADV", "FAL_GUARD", "tier_3_external"]
            findings.append(
                AgentFinding(
                    "epistemic_causal_counterfactual_conflict",
                    FindingSeverity.WARNING,
                    "CAU y CTF sostienen conclusiones incompatibles.",
                    evidence_refs=("CAU", "CTF"),
                )
            )
        else:
            epistemic_class = "measured_consistent"
            report_state = AgentState.OBSERVED
            deliberation = "defer_gain_judgment_until_outcome"
            escalation_targets = []

        if sequence_validated is False:
            report_state = AgentState.DEGRADED
            findings.append(
                AgentFinding(
                    "epistemic_sequence_not_validated",
                    FindingSeverity.WARNING,
                    "La secuencia ejecutada no tiene validación positiva.",
                    evidence_refs=tuple(sequence),
                )
            )

        verdict_counts = _verdict_counts(receipts)
        active_organs = sum(
            bool(organ.candidate_hash and organ.effective_mode != "off") for organ in organs
        )
        return AgentReport.create(
            agent_id=self.agent_id,
            role=AgentRole.METACOGNITIVE_EPISTEMIC,
            identity=identity,
            state=report_state,
            authority_effect=AuthorityEffect.NONE,
            metrics={
                "measurement_coverage": round(coverage, 6),
                "committed_certainty": prob_lcb,
                "posterior_point": prob_point,
                "posterior_interval_width": interval_width,
                "causal_conflict": causal_conflict,
                "active_organ_count": active_organs,
                **verdict_counts,
            },
            findings=findings,
            outputs={
                "evidence_pipeline": ["measure", "classify", "analyze", "deliberate"],
                "stages": {
                    "measure": {
                        "signals": measured,
                        "coverage": round(coverage, 6),
                        "prob_point": prob_point,
                        "prob_lcb": prob_lcb,
                    },
                    "classify": {
                        "epistemic_class": epistemic_class,
                        "causal_conflict": causal_conflict,
                    },
                    "analyze": {
                        "committed_certainty": prob_lcb,
                        "interval_width": interval_width,
                        "epistemic_gain": None,
                        "gain_status": "unmeasured_pre_outcome",
                    },
                    "deliberate": {
                        "proposal": deliberation,
                        "escalation_targets": escalation_targets,
                        "scheduler_authority_preserved": True,
                        "teacher_invocation_authorized": False,
                    },
                },
                "objective_currency": "epistemic_gain_per_committed_certainty",
                "decision_influence": "none",
                "training_authorized": False,
            },
        )


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _finite_unit(value: Any) -> float | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        return None
    return number


def _optional_bool(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _verdict_counts(receipts: Sequence[ConsumerReceipt]) -> dict[str, int]:
    positive = sum(
        receipt.verdict_class is ConsumerVerdictClass.ACCEPTED for receipt in receipts
    )
    negative = sum(
        receipt.verdict_class
        in {
            ConsumerVerdictClass.REJECTED,
            ConsumerVerdictClass.INVALID,
            ConsumerVerdictClass.PERSISTENCE_DEGRADED,
        }
        for receipt in receipts
    )
    return {
        "informative_positive_receipt_count": positive,
        "informative_negative_receipt_count": negative,
        "neutral_receipt_count": len(receipts) - positive - negative,
    }
