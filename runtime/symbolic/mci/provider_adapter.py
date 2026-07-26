"""Adaptación determinista de propuestas externas al contrato del MCI."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from runtime.symbolic.schemas import sealed_sha256

from .causal_learning import TransitionEvidence
from .contracts import NeuralHypothesis, TransitionSpec
from .evidence_mapper import map_external_evidence_refs
from .hypothesis_mapping import HypothesisMappingRegistry
from .provider_protocol import ExternalHypothesis


def adapt_hypotheses(
    proposals: Sequence[ExternalHypothesis],
    *,
    registry: HypothesisMappingRegistry,
    spec: TransitionSpec,
    recent_evidence: Sequence[TransitionEvidence],
    logical_time: int,
) -> list[NeuralHypothesis]:
    """Convierte únicamente propuestas que poseen una semántica local verificable."""

    registry.validate_for(spec)
    parameters = set(spec.parameters)
    effects = {
        effect.effect_id
        for equation in spec.equations
        for effect in equation.effects
    }
    adapted: list[NeuralHypothesis] = []
    evidence_aliases = {
        alias
        for item in recent_evidence
        for alias in (
            item.evidence_id,
            item.replay_unit_id,
            item.decision_trace_sha256 or "",
        )
        if alias
    }
    for proposal in proposals:
        kind = proposal.kind
        target_id = proposal.target_id
        expression = proposal.expression
        proposed_value = proposal.proposed_value
        if kind == "edge":
            mapping = registry.resolve(proposal.source, proposal.target)
            if mapping is None:
                continue
            kind = mapping.kind
            target_id = mapping.target_id
            if kind == "precondition":
                expression = mapping.precondition_template
            elif proposed_value is None:
                continue
        if kind == "parameter":
            if target_id not in parameters or proposed_value is None:
                continue
        elif kind == "precondition":
            if target_id not in effects or not expression:
                continue
        else:
            continue
        evidence_refs = map_external_evidence_refs(
            proposal.evidence_refs, recent_evidence
        )
        payload: dict[str, Any] = {
            "source_hypothesis_id": proposal.hypothesis_id,
            "kind": kind,
            "target_id": target_id,
            "expression": expression,
            "proposed_value": proposed_value,
            "confidence": float(proposal.confidence),
            "evidence_refs": evidence_refs,
            "provider": proposal.provider,
            "model_ref": proposal.model_ref,
            "logical_time": int(logical_time),
        }
        adapted.append(
            NeuralHypothesis(
                hypothesis_id=(
                    proposal.hypothesis_id
                    or f"external-{sealed_sha256(payload)[:24]}"
                ),
                kind=kind,
                target_id=target_id,
                expression=expression,
                proposed_value=proposed_value,
                confidence=proposal.confidence,
                evidence_refs=evidence_refs,
                provider=proposal.provider,
                model_ref=proposal.model_ref,
                logical_time=logical_time,
                submitted_evidence_refs=tuple(proposal.evidence_refs),
                unresolved_evidence_refs=tuple(
                    sorted(
                        str(item)
                        for item in proposal.evidence_refs
                        if str(item) not in evidence_aliases
                    )
                ),
            )
        )
    return sorted(adapted, key=lambda item: item.hypothesis_id)
