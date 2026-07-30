from __future__ import annotations

import json

import pytest

from runtime.neural.hypothesis_generator import StructuralHypothesisGenerator
from runtime.neural.training import (
    EvidenceJSONLCollector,
    label_candidate_counterfactually,
)
from runtime.symbolic.mci.causal_learning import (
    CausalLearningEngine,
    TransitionEvidence,
)
from runtime.symbolic.mci.compiler import TransitionCompiler
from runtime.symbolic.mci.specs import thermal_battery_spec, with_overlay


def _rows():
    base = thermal_battery_spec()
    actual = with_overlay(
        base, {}, {"thermal_battery/cooling": "battery_level > 0.3"}
    )
    rows = []
    values = (
        0.1,
        0.8,
        0.2,
        0.7,
        0.25,
        0.6,
        0.29,
        0.5,
        0.28,
        0.4,
        0.27,
        0.31,
        0.26,
        0.34,
        0.24,
        0.33,
    )
    for index, battery in enumerate(values):
        state = {
            "temperature": 0.9,
            "battery_level": battery,
            "cooling_active": False,
            "alarm": True,
        }
        rows.append(
            TransitionEvidence(
                state=state,
                action="activate_cooling",
                external_input=0.0,
                predicted=TransitionCompiler(base).execute(
                    state, action="activate_cooling", external_input=0.0
                ),
                observed=TransitionCompiler(actual).execute(
                    state, action="activate_cooling", external_input=0.0
                ),
                logical_time=index + 1,
            )
        )
    return base, tuple(rows)


def test_collector_is_incremental_and_resume_idempotent(tmp_path):
    spec, rows = _rows()
    path = tmp_path / "evidence.jsonl"
    learner = CausalLearningEngine(spec)
    collector = EvidenceJSONLCollector(
        path,
        run_id="run",
        scenario_id=spec.scenario,
        seed=7,
        transition_spec_hash=spec.sha256,
    )
    learner.add_evidence_observer(collector)
    collector.set_context(episode=1)
    learner.observe(rows[0])
    assert len(path.read_text().splitlines()) == 1
    resumed = EvidenceJSONLCollector(
        path,
        run_id="run",
        scenario_id=spec.scenario,
        seed=7,
        transition_spec_hash=spec.sha256,
    )
    resumed(rows[0])
    assert len(path.read_text().splitlines()) == 1
    payload = json.loads(path.read_text())
    assert payload["record_sha256"]
    assert payload["state_before"]["battery_level"] == 0.1


def test_labeler_uses_strictly_future_evidence():
    spec, rows = _rows()
    candidate = next(
        item
        for item in StructuralHypothesisGenerator(beam_width=64).generate(
            spec=spec, evidence=rows[:12]
        )
        if item.expression == "battery_level > 0.3"
    )
    label = label_candidate_counterfactually(
        spec=spec,
        candidate=candidate,
        feature_evidence=rows[:12],
        label_evidence=rows[12:],
    )
    assert label.valid
    assert label.mae_gain > 0.0
    assert label.invariant_risk == 0.0
    with pytest.raises(ValueError, match="leakage"):
        label_candidate_counterfactually(
            spec=spec,
            candidate=candidate,
            feature_evidence=rows[:12],
            label_evidence=rows[11:],
        )


def test_labeler_mae_gain_is_bounded_for_harmful_candidate():
    spec, rows = _rows()
    candidate = next(
        item
        for item in StructuralHypothesisGenerator(beam_width=64).generate(
            spec=spec, evidence=rows[:12]
        )
        if item.expression == "battery_level < 0.3"
    )
    label = label_candidate_counterfactually(
        spec=spec,
        candidate=candidate,
        feature_evidence=rows[:12],
        label_evidence=rows[12:],
    )
    assert -1.0 <= label.mae_gain <= 1.0
    assert not label.valid


def test_multistep_label_is_versioned_hashed_and_candidate_conditioned():
    spec, rows = _rows()
    candidates = StructuralHypothesisGenerator(beam_width=64).generate(
        spec=spec, evidence=rows[:12]
    )
    correct = next(
        item
        for item in candidates
        if item.expression == "battery_level > 0.3"
    )
    wrong = next(
        item
        for item in candidates
        if item.expression == "battery_level < 0.3"
    )
    arguments = {
        "spec": spec,
        "feature_evidence": rows[:12],
        "label_evidence": rows[12:],
        "risk_label_version": "n4-risk-label.multistep.v1",
        "rollout_horizon": 3,
    }
    correct_label = label_candidate_counterfactually(
        candidate=correct, **arguments
    )
    wrong_label = label_candidate_counterfactually(
        candidate=wrong, **arguments
    )
    assert correct_label.risk_label_available
    assert correct_label.risk_label_version == "n4-risk-label.multistep.v1"
    assert correct_label.risk_report_sha256
    assert correct_label.safety_contract_version
    assert correct_label.rollout_horizon == 3
    assert correct_label.risk_components
    assert wrong_label.risk_label != correct_label.risk_label
    assert any(
        event.step >= 2 for event in wrong_label.risk_report.oracle_events
    )


def test_legacy_label_explicitly_marks_risk_unknown():
    spec, rows = _rows()
    candidate = StructuralHypothesisGenerator(beam_width=64).generate(
        spec=spec, evidence=rows[:12]
    )[0]
    label = label_candidate_counterfactually(
        spec=spec,
        candidate=candidate,
        feature_evidence=rows[:12],
        label_evidence=rows[12:],
    )
    assert label.risk_label_available is False
    assert label.risk_label is None
    assert label.risk_label_version is None
    assert label.risk_report_sha256 is None


def test_labeler_rejects_unknown_risk_version_and_short_horizon():
    spec, rows = _rows()
    candidate = StructuralHypothesisGenerator(beam_width=64).generate(
        spec=spec, evidence=rows[:12]
    )[0]
    with pytest.raises(ValueError, match="version_unknown"):
        label_candidate_counterfactually(
            spec=spec,
            candidate=candidate,
            feature_evidence=rows[:12],
            label_evidence=rows[12:],
            risk_label_version="unknown",
        )
    with pytest.raises(ValueError, match="horizon_unavailable"):
        label_candidate_counterfactually(
            spec=spec,
            candidate=candidate,
            feature_evidence=rows[:12],
            label_evidence=rows[12:13],
            risk_label_version="n4-risk-label.multistep.v1",
            rollout_horizon=3,
        )
