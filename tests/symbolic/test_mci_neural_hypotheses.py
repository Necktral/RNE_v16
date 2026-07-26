from __future__ import annotations

from runtime.symbolic.mci import MCIRuntime, NeuralHypothesis
from runtime.symbolic.mci.causal_learning import (
    CausalLearningEngine,
    TransitionEvidence,
)
from runtime.symbolic.mci.compiler import TransitionCompiler
from runtime.symbolic.mci.contracts import (
    ActionSpec,
    EffectSpec,
    EquationSpec,
    InvariantSpec,
    TermSpec,
    TransitionSpec,
    VariableSpec,
)
from runtime.symbolic.mci.specs import thermal_spec, with_overlay


def _battery_spec() -> TransitionSpec:
    return TransitionSpec(
        spec_id="transition/neural_battery",
        version="1.0",
        scenario="neural_battery",
        variables=(
            VariableSpec("temperature", "real"),
            VariableSpec("battery_level", "real"),
            VariableSpec("cooling_active", "bool"),
            VariableSpec("alarm", "bool"),
        ),
        parameters={"alarm_threshold": 0.85, "cooling_effect": 0.10},
        actions=(ActionSpec("activate_cooling", {"cooling_active": True}),),
        equations=(
            EquationSpec(
                "temperature",
                (TermSpec("state.temperature"),),
                (
                    EffectSpec(
                        "neural/cooling",
                        "activate_cooling",
                        -1.0,
                        "cooling_effect",
                    ),
                ),
            ),
        ),
        alarm_variable="alarm",
        alarm_source="temperature",
        alarm_operator=">=",
        alarm_threshold_parameter="alarm_threshold",
        main_variable="temperature",
        optimization_direction="minimize",
        safe_action="activate_cooling",
        invariants=(InvariantSpec("temperature", "bounds", "alarm_threshold"),),
    )


def _hypothesis(
    *,
    kind: str = "precondition",
    target: str = "neural/cooling",
    expression: str | None = "battery_level > 0.3",
    proposed_value: float | None = None,
    confidence: float = 0.8,
) -> NeuralHypothesis:
    return NeuralHypothesis(
        hypothesis_id="",
        kind=kind,
        target_id=target,
        expression=expression,
        proposed_value=proposed_value,
        confidence=confidence,
        evidence_refs=(),
        provider="synthetic-neural",
        model_ref="test-v1",
        logical_time=1,
    )


def _evidence(
    base: TransitionSpec,
    actual: TransitionSpec,
    *,
    episode: int,
    battery: float,
) -> TransitionEvidence:
    state = {
        "temperature": 0.9,
        "battery_level": battery,
        "cooling_active": False,
        "alarm": True,
    }
    predicted = TransitionCompiler(base).execute(
        state, action="activate_cooling", external_input=0.0
    )
    observed = TransitionCompiler(actual).execute(
        state, action="activate_cooling", external_input=0.0
    )
    return TransitionEvidence(
        state=state,
        action="activate_cooling",
        external_input=0.0,
        observed=observed,
        predicted=predicted,
        replay_unit_id=f"neural/ep-{episode}",
        logical_time=episode * 2 + 2,
    )


def _populate(
    learner: CausalLearningEngine,
    actual: TransitionSpec,
    *,
    count: int = 12,
) -> None:
    for episode in range(count):
        learner.observe(
            _evidence(
                learner.active_spec,
                actual,
                episode=episode,
                battery=0.1 if episode % 2 == 0 else 0.8,
            )
        )


def test_invalid_expression_is_rejected_without_entering_queue():
    learner = CausalLearningEngine(_battery_spec())
    learner.enqueue_hypotheses(
        [_hypothesis(expression="__import__('os').system('false')")]
    )
    actual = with_overlay(
        learner.base_spec, {}, {"neural/cooling": "battery_level > 0.3"}
    )
    learner.observe(_evidence(learner.active_spec, actual, episode=0, battery=0.1))
    evaluation = learner.last_neural_evaluations()[0]
    assert evaluation.status == "invalid_expression"
    assert evaluation.reason == "unsupported_precondition_expression"


def test_unsupported_precondition_is_rejected_by_holdout():
    learner = CausalLearningEngine(_battery_spec())
    actual = with_overlay(
        learner.base_spec, {}, {"neural/cooling": "battery_level > 0.3"}
    )
    _populate(learner, actual, count=11)
    learner.enqueue_hypotheses([_hypothesis(expression="battery_level > 0.9")])
    learner.observe(_evidence(learner.active_spec, actual, episode=11, battery=0.8))
    evaluation = learner.last_neural_evaluations()[0]
    assert evaluation.status == "rejected"
    assert evaluation.reason == "empirical_gates_failed"


def test_supported_precondition_promotes_with_neural_provenance():
    learner = CausalLearningEngine(_battery_spec())
    actual = with_overlay(
        learner.base_spec, {}, {"neural/cooling": "battery_level > 0.3"}
    )
    _populate(learner, actual, count=11)
    hypothesis = _hypothesis()
    learner.enqueue_hypotheses([hypothesis])
    overlay = learner.observe(
        _evidence(learner.active_spec, actual, episode=11, battery=0.8)
    )
    evaluation = learner.last_neural_evaluations()[0]
    assert overlay is not None
    assert evaluation.status == "promoted"
    assert evaluation.promoted_overlay_id == overlay.overlay_id
    assert overlay.evidence["source"] == "neural"
    assert overlay.evidence["hypothesis_id"] == hypothesis.hypothesis_id


def test_parameter_hypothesis_beats_internal_approximation():
    base = thermal_spec(cooling_effect=0.07)
    actual = thermal_spec(cooling_effect=0.14)
    learner = CausalLearningEngine(base)
    for episode in range(11):
        state = {"temperature": 0.9, "cooling_active": False, "alarm": True}
        learner.observe(
            TransitionEvidence(
                state=state,
                action="activate_cooling",
                external_input=0.0,
                observed=TransitionCompiler(actual).execute(
                    state, action="activate_cooling", external_input=0.0
                ),
                predicted=TransitionCompiler(learner.active_spec).execute(
                    state, action="activate_cooling", external_input=0.0
                ),
                logical_time=episode * 2 + 2,
            )
        )
    hypothesis = NeuralHypothesis(
        hypothesis_id="",
        kind="parameter",
        target_id="cooling_effect",
        expression=None,
        proposed_value=0.14,
        confidence=0.7,
        evidence_refs=(),
        provider="synthetic-neural",
        model_ref="parameter-v1",
        logical_time=23,
    )
    learner.enqueue_hypotheses([hypothesis])
    state = {"temperature": 0.9, "cooling_active": False, "alarm": True}
    overlay = learner.observe(
        TransitionEvidence(
            state=state,
            action="activate_cooling",
            external_input=0.0,
            observed=TransitionCompiler(actual).execute(
                state, action="activate_cooling", external_input=0.0
            ),
            predicted=TransitionCompiler(learner.active_spec).execute(
                state, action="activate_cooling", external_input=0.0
            ),
            logical_time=24,
        )
    )
    assert overlay is not None
    assert overlay.parameter_updates["cooling_effect"] == 0.14
    assert learner.last_neural_evaluations()[0].status == "promoted"


def test_regime_hypothesis_is_recorded_but_not_actionable():
    learner = CausalLearningEngine(_battery_spec())
    actual = with_overlay(
        learner.base_spec, {}, {"neural/cooling": "battery_level > 0.3"}
    )
    _populate(learner, actual, count=11)
    learner.enqueue_hypotheses(
        [
            _hypothesis(
                kind="regime",
                target="neural-battery-regime",
                expression=None,
            )
        ]
    )
    learner.observe(_evidence(learner.active_spec, actual, episode=11, battery=0.8))
    evaluation = learner.last_neural_evaluations()[0]
    assert evaluation.status == "rejected"
    assert evaluation.reason == "regime_hypotheses_not_actionable"


def test_mci_outcome_exposes_evaluations():
    spec = _battery_spec()
    actual = with_overlay(spec, {}, {"neural/cooling": "battery_level > 0.3"})
    runtime = MCIRuntime(spec)
    _populate(runtime.learner, actual, count=11)
    runtime.ingest_neural_hypotheses([_hypothesis()])
    state = {
        "temperature": 0.9,
        "battery_level": 0.8,
        "cooling_active": False,
        "alarm": True,
    }
    runtime.propose(
        state=state,
        external_input=0.0,
        logical_time=23,
        regime="synthetic",
        replay_unit_id="neural/runtime/ep-12",
    )
    observed = TransitionCompiler(actual).execute(
        state, action="activate_cooling", external_input=0.0
    )
    outcome = runtime.observe_outcome(
        observed,
        committed_action="activate_cooling",
        logical_time=24,
        reasoning_cost=1.0,
        decision_trace_sha256="abc",
    )
    evaluations = outcome["neural_hypotheses_evaluated"]
    assert len(evaluations) == 1
    assert evaluations[0]["status"] == "promoted"


def test_evidence_and_hypothesis_ids_are_deterministic():
    spec = _battery_spec()
    actual = with_overlay(spec, {}, {"neural/cooling": "battery_level > 0.3"})
    first = _evidence(spec, actual, episode=1, battery=0.1)
    second = _evidence(spec, actual, episode=1, battery=0.1)
    assert first.evidence_id == second.evidence_id
    assert _hypothesis().hypothesis_id == _hypothesis().hypothesis_id
