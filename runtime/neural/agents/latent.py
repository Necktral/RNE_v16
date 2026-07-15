"""Agente de comunicación latente mediante modulación de ganancia acotada."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Sequence

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
    GainModulation,
)


_POSITIVE = {ConsumerVerdictClass.ACCEPTED}
_NEGATIVE = {
    ConsumerVerdictClass.REJECTED,
    ConsumerVerdictClass.INVALID,
    ConsumerVerdictClass.PERSISTENCE_DEGRADED,
}


class LatentCommunicationAgent:
    """Propone ganancia, no mensajes ni decisiones.

    Este canal no usa Mamba2: Mamba2 permanece como alternativa experimental de
    memoria/creatividad horizontal en SHADOW, no como transporte canónico. El
    vector de control es objetivo-agnóstico: confianza, incertidumbre,
    evidencia positiva y evidencia negativa. Las clases neutrales no se
    convierten en apoyo y la propuesta nunca se aplica en esta versión.
    """

    agent_id = "agent-latent-communication-v1"

    def modulate(
        self,
        *,
        identity: SymbiosisIdentity,
        organs: Sequence[OrganTrace],
        receipts: Sequence[ConsumerReceipt],
        quarantined_organs: Sequence[str] = (),
    ) -> AgentReport:
        quarantined = set(quarantined_organs)
        receipts_by_organ: dict[str, list[ConsumerReceipt]] = defaultdict(list)
        for receipt in receipts:
            receipts_by_organ[receipt.organ].append(receipt)

        findings: list[AgentFinding] = []
        modulations: list[GainModulation] = []
        measurements: list[dict[str, object]] = []
        classifications: list[dict[str, object]] = []
        analyses: list[dict[str, object]] = []
        deliberations: list[dict[str, object]] = []
        for organ in sorted(organs, key=lambda item: item.organ):
            if not organ.candidate_hash or organ.effective_mode == "off":
                continue
            organ_receipts = sorted(
                receipts_by_organ.get(organ.organ, ()), key=lambda item: item.receipt_id
            )
            confidence = _finite_unit(organ.confidence)
            uncertainty = _finite_unit(organ.uncertainty)
            measurements.append(
                {
                    "organ": organ.organ,
                    "confidence": confidence,
                    "uncertainty": uncertainty,
                    "consumer_receipt_count": len(organ_receipts),
                    "status": (
                        "measured"
                        if confidence is not None and uncertainty is not None
                        else "unmeasured"
                    ),
                }
            )
            if organ.organ in quarantined:
                findings.append(
                    AgentFinding(
                        "latent_source_quarantined",
                        FindingSeverity.WARNING,
                        "La modulación se abstiene porque la fuente quedó en cuarentena.",
                        subject=organ.organ,
                    )
                )
                deliberations.append(
                    _deliberation(organ.organ, "abstain", "source_quarantined")
                )
                continue
            if confidence is None or uncertainty is None:
                findings.append(
                    AgentFinding(
                        "latent_measurement_missing",
                        FindingSeverity.WARNING,
                        "Falta confianza o incertidumbre medida; no se fabrica una ganancia.",
                        subject=organ.organ,
                    )
                )
                deliberations.append(
                    _deliberation(organ.organ, "abstain", "measurement_missing")
                )
                continue
            if not organ_receipts:
                findings.append(
                    AgentFinding(
                        "latent_consumer_evidence_missing",
                        FindingSeverity.WARNING,
                        "No hay recibos de consumo; el canal latente se abstiene.",
                        subject=organ.organ,
                    )
                )
                classifications.append(
                    {
                        "organ": organ.organ,
                        "positive_count": 0,
                        "negative_count": 0,
                        "neutral_count": 0,
                        "status": "unavailable",
                    }
                )
                deliberations.append(
                    _deliberation(organ.organ, "abstain", "consumer_evidence_missing")
                )
                continue

            informative = [
                receipt
                for receipt in organ_receipts
                if receipt.verdict_class in _POSITIVE | _NEGATIVE
            ]
            positive = sum(receipt.verdict_class in _POSITIVE for receipt in informative)
            negative = sum(receipt.verdict_class in _NEGATIVE for receipt in informative)
            neutral = len(organ_receipts) - len(informative)
            classifications.append(
                {
                    "organ": organ.organ,
                    "positive_count": positive,
                    "negative_count": negative,
                    "neutral_count": neutral,
                    "status": "informative" if informative else "non_informative",
                }
            )
            if not informative:
                findings.append(
                    AgentFinding(
                        "latent_consumer_evidence_non_informative",
                        FindingSeverity.INFO,
                        "Los recibos son observacionales; no justifican aumentar ni reducir ganancia.",
                        subject=organ.organ,
                        evidence_refs=tuple(receipt.receipt_id for receipt in organ_receipts),
                    )
                )
                deliberations.append(
                    _deliberation(organ.organ, "abstain", "evidence_non_informative")
                )
                continue
            positive_fraction = positive / len(informative)
            negative_fraction = negative / len(informative)
            evidence_balance = positive_fraction - negative_fraction
            certainty_balance = confidence - uncertainty
            proposed_gain = round(
                max(0.75, min(1.25, 1.0 + 0.25 * certainty_balance * evidence_balance)),
                6,
            )
            analyses.append(
                {
                    "organ": organ.organ,
                    "certainty_balance": round(certainty_balance, 6),
                    "evidence_balance": round(evidence_balance, 6),
                    "candidate_gain": proposed_gain,
                    "hypothesis": "epistemic_gain_modulates_consumer_weight",
                    "benefit_measured": False,
                }
            )
            deliberations.append(
                _deliberation(
                    organ.organ,
                    "propose",
                    "bounded_reference_hypothesis",
                    proposed_gain=proposed_gain,
                )
            )
            for receipt in informative:
                modulations.append(
                    GainModulation(
                        modulation_id=f"latent:{organ.organ}:{receipt.consumer_id}:{receipt.receipt_id}",
                        source=organ.organ,
                        target=receipt.consumer_id,
                        latent_vector=(
                            confidence,
                            uncertainty,
                            positive_fraction,
                            negative_fraction,
                        ),
                        proposed_gain=proposed_gain,
                        reason="bounded_epistemic_gain_reference_v1",
                        evidence_refs=(organ.candidate_hash, receipt.receipt_id),
                    )
                )

        warning_count = sum(
            finding.severity is FindingSeverity.WARNING for finding in findings
        )
        if modulations and warning_count:
            state = AgentState.DEGRADED
        elif modulations:
            state = AgentState.OBSERVED
        else:
            state = AgentState.ABSTAINED
        return AgentReport.create(
            agent_id=self.agent_id,
            role=AgentRole.LATENT_COMMUNICATION,
            identity=identity,
            state=state,
            authority_effect=AuthorityEffect.NONE,
            metrics={
                "modulation_count": len(modulations),
                "abstention_count": warning_count,
                "quarantined_source_count": len(quarantined),
            },
            findings=findings,
            outputs={
                "experimental": True,
                "policy": "bounded_epistemic_gain_reference_v1",
                "evidence_pipeline": [
                    "measure",
                    "classify",
                    "analyze",
                    "deliberate",
                ],
                "stages": {
                    "measure": measurements,
                    "classify": classifications,
                    "analyze": analyses,
                    "deliberate": deliberations,
                },
                "modulations": [item.to_dict() for item in modulations],
                "gain_bounds": [0.75, 1.25],
                "gain_bounds_kind": "safety_envelope_not_setpoint",
                "setpoint_status": "unlearned",
                "hypothesis_status": "reference_untrained",
                "apply_authorized": False,
                "training_authorized": False,
            },
        )


def _finite_unit(value: float | None) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        return None
    return number


def _deliberation(
    organ: str,
    verdict: str,
    reason: str,
    *,
    proposed_gain: float | None = None,
) -> dict[str, object]:
    return {
        "organ": organ,
        "verdict": verdict,
        "reason": reason,
        "proposed_gain": proposed_gain,
        "apply_authorized": False,
        "training_authorized": False,
    }
