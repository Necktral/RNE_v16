"""Etiquetado temporal de candidatos N4 sin reutilizar evidencia de features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from runtime.neural.contracts import canonical_sha256
from runtime.neural.hypothesis_generator import StructuralCandidate
from runtime.symbolic.mci.causal_learning import TransitionEvidence
from runtime.symbolic.mci.compiler import TransitionCompiler
from runtime.symbolic.mci.contracts import TransitionSpec
from runtime.symbolic.mci.specs import with_overlay


@dataclass(frozen=True, slots=True)
class CandidateLabel:
    candidate_set_id: str
    hypothesis_id: str
    cutoff_logical_time: int
    valid: bool
    mae_gain: float
    invariant_risk: float
    evaluation_count: int


def label_candidate_counterfactually(
    *,
    spec: TransitionSpec,
    candidate: StructuralCandidate,
    feature_evidence: Sequence[TransitionEvidence],
    label_evidence: Sequence[TransitionEvidence],
) -> CandidateLabel:
    """Evalúa un overlay sobre contextos posteriores o rollouts independientes."""

    features = tuple(feature_evidence)
    labels = tuple(label_evidence)
    if not features or not labels:
        raise ValueError("n4_labeler_requires_feature_and_label_evidence")
    cutoff = max(item.logical_time for item in features)
    if any(item.logical_time <= cutoff for item in labels):
        raise ValueError("n4_label_leakage_detected")
    candidate_spec = _candidate_spec(spec, candidate)
    baseline = TransitionCompiler(spec)
    proposed = TransitionCompiler(candidate_spec)
    before = _mae(baseline, labels, spec.main_variable)
    after = _mae(proposed, labels, spec.main_variable)
    gain = (before - after) / max(before, after, 1e-9)
    risk = _invariant_risk(proposed, labels, candidate_spec)
    candidate_set_id = "n4-set-" + canonical_sha256(
        {
            "cutoff": cutoff,
            "evidence_ids": sorted(item.evidence_id for item in features),
        }
    )[:24]
    return CandidateLabel(
        candidate_set_id=candidate_set_id,
        hypothesis_id=candidate.hypothesis_id,
        cutoff_logical_time=cutoff,
        valid=bool(gain > 0.0 and risk == 0.0),
        mae_gain=round(gain, 9),
        invariant_risk=round(risk, 9),
        evaluation_count=len(labels),
    )


def _candidate_spec(
    spec: TransitionSpec, candidate: StructuralCandidate
) -> TransitionSpec:
    if candidate.kind == "parameter":
        if candidate.proposed_value is None:
            raise ValueError("n4_parameter_candidate_requires_value")
        return with_overlay(spec, {candidate.target_id: candidate.proposed_value})
    if candidate.kind == "precondition":
        if not candidate.expression:
            raise ValueError("n4_precondition_candidate_requires_expression")
        return with_overlay(spec, {}, {candidate.target_id: candidate.expression})
    raise ValueError(f"n4_candidate_kind_not_labelable:{candidate.kind}")


def _mae(
    compiler: TransitionCompiler,
    rows: Sequence[TransitionEvidence],
    main_variable: str,
) -> float:
    return sum(
        abs(
            float(
                compiler.execute(
                    row.state,
                    action=row.action,
                    external_input=row.external_input,
                )[main_variable]
            )
            - float(row.observed[main_variable])
        )
        for row in rows
    ) / len(rows)


def _invariant_risk(
    compiler: TransitionCompiler,
    rows: Sequence[TransitionEvidence],
    spec: TransitionSpec,
) -> float:
    bounded = {
        item.variable for item in spec.invariants if item.operator == "bounds"
    }
    if not bounded:
        return 0.0
    severity = 0.0
    for row in rows:
        prediction = compiler.execute(
            row.state,
            action=row.action,
            external_input=row.external_input,
        )
        for variable in bounded:
            value = float(prediction[variable])
            severity += max(0.0, -value, value - 1.0)
    return min(1.0, severity / (len(rows) * len(bounded)))
