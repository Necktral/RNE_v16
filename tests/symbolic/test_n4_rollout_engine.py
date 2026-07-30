from __future__ import annotations

import copy

import pytest

from runtime.symbolic.mci import TransitionCompiler, thermal_battery_spec
from runtime.symbolic.mci.rollout_engine import (
    rollout_contrafactual,
    simulate_rollout,
    thermal_battery_oracle,
)
from runtime.symbolic.mci.specs import with_overlay


def _state(*, temperature=0.84, battery=0.2):
    return {
        "temperature": temperature,
        "battery_level": battery,
        "cooling_active": False,
        "alarm": temperature >= 0.85,
    }


def _guarded_spec():
    return with_overlay(
        thermal_battery_spec(),
        {},
        {"thermal_battery/cooling": "battery_level > 0.3"},
    )


def test_simulate_rollout_chains_steps_and_does_not_mutate_input():
    spec = thermal_battery_spec()
    initial = _state(temperature=0.8, battery=0.8)
    frozen = copy.deepcopy(initial)
    actions = ("activate_cooling",) * 3
    inputs = (0.04,) * 3
    states = simulate_rollout(initial, actions, inputs, spec, 3)
    assert initial == frozen
    assert len(states) == 3
    assert states[1]["temperature"] != states[0]["temperature"]

    expected = []
    state = dict(initial)
    compiler = TransitionCompiler(spec)
    for action, external_input in zip(actions, inputs):
        state = compiler.execute(
            state, action=action, external_input=external_input
        )
        expected.append(state)
    assert states == expected


def test_hazard_that_appears_on_later_step_is_detected():
    baseline = thermal_battery_spec()
    report = rollout_contrafactual(
        _state(temperature=0.78, battery=0.2),
        ("activate_cooling",) * 3,
        (0.04,) * 3,
        baseline,
        baseline,
        thermal_battery_oracle(),
        3,
    )
    oracle_hazard_steps = {
        event.step
        for event in report.oracle_events
        if event.predicate_id == "temp_above_threshold"
    }
    assert oracle_hazard_steps == {2, 3}
    assert report.missed_hazard_rate > 0.0
    assert report.alarm_miss_rate > 0.0


def test_correct_guarded_candidate_is_safer_than_unguarded_baseline():
    baseline = thermal_battery_spec()
    oracle = thermal_battery_oracle()
    arguments = (
        _state(),
        ("activate_cooling",) * 3,
        (0.04,) * 3,
    )
    wrong = rollout_contrafactual(
        *arguments, baseline, baseline, oracle, 3
    )
    correct = rollout_contrafactual(
        *arguments, _guarded_spec(), baseline, oracle, 3
    )
    assert wrong.actuator_failure_miss_rate == 1.0
    assert wrong.alarm_miss_rate > 0.0
    assert wrong.invariant_risk > correct.invariant_risk
    assert correct.missed_hazard_rate == pytest.approx(0.0)
    assert correct.actuator_failure_miss_rate == pytest.approx(0.0)
    assert correct.alarm_miss_rate == pytest.approx(0.0)


def test_persistent_hazard_is_visible_at_step_three():
    spec = _guarded_spec()
    report = rollout_contrafactual(
        _state(temperature=0.86, battery=0.2),
        ("activate_cooling",) * 3,
        (0.01,) * 3,
        spec,
        thermal_battery_spec(),
        thermal_battery_oracle(),
        3,
    )
    persistent = [
        event
        for event in report.oracle_events
        if event.predicate_id == "persistent_hazard"
    ]
    assert [event.step for event in persistent] == [3]


def test_raw_violation_is_recorded_even_when_output_is_clamped():
    spec = thermal_battery_spec()
    report = rollout_contrafactual(
        _state(temperature=0.99, battery=0.01),
        ("deactivate_cooling",),
        (0.5,),
        spec,
        spec,
        thermal_battery_oracle(),
        1,
    )
    raw = [
        event
        for event in report.candidate_events
        if event.predicate_id.startswith("raw_state_out_of_bounds:")
    ]
    assert any(
        event.predicate_id == "raw_state_out_of_bounds:temperature"
        for event in raw
    )
    assert report.raw_bound_violation > 0.0


def test_rollout_is_deterministic_and_rejects_short_sequences():
    spec = thermal_battery_spec()
    args = (
        _state(),
        ("activate_cooling",) * 3,
        (0.04,) * 3,
        spec,
        spec,
        thermal_battery_oracle(),
        3,
    )
    assert rollout_contrafactual(*args) == rollout_contrafactual(*args)
    with pytest.raises(ValueError, match="shorter_than_horizon"):
        simulate_rollout(_state(), ("activate_cooling",), (0.1,), spec, 2)
