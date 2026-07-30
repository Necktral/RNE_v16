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
