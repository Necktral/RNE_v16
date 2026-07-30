"""Etiquetado temporal de candidatos N4 sin reutilizar evidencia de features."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from runtime.neural.contracts import canonical_sha256
from runtime.neural.hypothesis_generator import StructuralCandidate
from runtime.symbolic.mci.causal_learning import TransitionEvidence
from runtime.symbolic.mci.compiler import TransitionCompiler
from runtime.symbolic.mci.contracts import TransitionSpec
from runtime.symbolic.mci.rollout_engine import (
    CandidateRiskReport,
    OracleStep,
    rollout_contrafactual,
    thermal_battery_oracle,
)
from runtime.symbolic.mci.safety_contracts import (
    SAFETY_CONTRACT_VERSION,
    thermal_battery_safety_config,
)
from runtime.symbolic.mci.specs import with_overlay


RISK_LABEL_VERSION = "n4-risk-label.multistep.v1"


@dataclass(frozen=True, slots=True)
class CandidateLabel:
    candidate_set_id: str
    hypothesis_id: str
    cutoff_logical_time: int
    valid: bool
    mae_gain: float
    invariant_risk: float
    evaluation_count: int
    risk_label: float | None = None
    risk_label_available: bool = False
    risk_label_version: str | None = None
    risk_report_sha256: str | None = None
    safety_contract_version: str | None = None
    rollout_horizon: int | None = None
    risk_components: Mapping[str, float] | None = None
    risk_report: CandidateRiskReport | None = None

    def __post_init__(self) -> None:
        if not self.risk_label_available:
            if any(
                item is not None
                for item in (
                    self.risk_label,
                    self.risk_label_version,
                    self.risk_report_sha256,
                    self.safety_contract_version,
                    self.rollout_horizon,
                    self.risk_report,
                )
            ):
                raise ValueError("n4_unavailable_risk_metadata_must_be_null")
        else:
            if self.risk_label is None or not 0.0 <= self.risk_label <= 1.0:
                raise ValueError("n4_available_risk_label_invalid")
            if self.risk_label_version != RISK_LABEL_VERSION:
                raise ValueError("n4_risk_label_version_unknown")
            if self.risk_report is None:
                raise ValueError("n4_available_risk_report_required")
            expected_hash = canonical_sha256(asdict(self.risk_report))
            if self.risk_report_sha256 != expected_hash:
                raise ValueError("n4_risk_report_hash_mismatch")


def label_candidate_counterfactually(
    *,
    spec: TransitionSpec,
    candidate: StructuralCandidate,
    feature_evidence: Sequence[TransitionEvidence],
    label_evidence: Sequence[TransitionEvidence],
    risk_label_version: str | None = None,
    rollout_horizon: int = 3,
    oracle: OracleStep | None = None,
    safety_config: Mapping[str, Any] | None = None,
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
    legacy_risk = _invariant_risk(proposed, labels, candidate_spec)
    report = None
    if risk_label_version is not None:
        if risk_label_version != RISK_LABEL_VERSION:
            raise ValueError("n4_risk_label_version_unknown")
        if rollout_horizon < 1 or len(labels) < rollout_horizon:
            raise ValueError("n4_risk_rollout_horizon_unavailable")
        alarm_threshold = float(
            spec.parameters[spec.alarm_threshold_parameter]
        )
        config = dict(
            safety_config
            or thermal_battery_safety_config(
                alarm_threshold=alarm_threshold
            )
        )
        report = rollout_contrafactual(
            labels[0].state,
            tuple(item.action for item in labels[:rollout_horizon]),
            tuple(
                item.external_input for item in labels[:rollout_horizon]
            ),
            candidate_spec,
            spec,
            oracle
            or thermal_battery_oracle(alarm_threshold=alarm_threshold),
            rollout_horizon,
            safety_config=config,
        )
    risk = report.invariant_risk if report is not None else legacy_risk
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
        risk_label=(round(report.invariant_risk, 9) if report else None),
        risk_label_available=report is not None,
        risk_label_version=(risk_label_version if report else None),
        risk_report_sha256=(
            canonical_sha256(asdict(report)) if report else None
        ),
        safety_contract_version=(
            SAFETY_CONTRACT_VERSION if report else None
        ),
        rollout_horizon=(report.horizon if report else None),
        risk_components=(
            {
                "missed_hazard_rate": report.missed_hazard_rate,
                "alarm_miss_rate": report.alarm_miss_rate,
                "actuator_failure_miss_rate": (
                    report.actuator_failure_miss_rate
                ),
                "persistent_hazard_miss_rate": (
                    report.persistent_hazard_miss_rate
                ),
                "raw_bound_violation": report.raw_bound_violation,
                "maximum_severity": report.maximum_severity,
                "cumulative_severity": report.cumulative_severity,
            }
            if report
            else None
        ),
        risk_report=report,
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
