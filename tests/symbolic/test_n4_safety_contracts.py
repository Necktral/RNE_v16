from __future__ import annotations

import pytest

from runtime.symbolic.mci.safety_contracts import (
    SafetyEvent,
    evaluate_safety,
    thermal_battery_safety_config,
)


def _config():
    return thermal_battery_safety_config(alarm_threshold=0.85)


def _ids(events):
    return {item.predicate_id for item in events}


def test_safe_state_has_no_events():
    events = evaluate_safety(
        {"temperature": 0.7, "battery_level": 0.8, "alarm": False},
        _config(),
    )
    assert events == []


def test_temperature_hazard_and_correct_alarm_are_distinct():
    events = evaluate_safety(
        {"temperature": 0.88, "battery_level": 0.8, "alarm": True},
        _config(),
    )
    assert "temp_above_threshold" in _ids(events)
    assert "alarm_missed" not in _ids(events)


def test_critical_temperature_and_missed_alarm_are_reported():
    events = evaluate_safety(
        {"temperature": 0.95, "battery_level": 0.8, "alarm": False},
        _config(),
        step=2,
        source="oracle",
    )
    assert {
        "temp_above_threshold",
        "temp_critical",
        "alarm_missed",
    } <= _ids(events)
    assert all(item.step == 2 and item.source == "oracle" for item in events)


def test_low_battery_only_means_unavailable_when_cooling_is_commanded():
    state = {"temperature": 0.8, "battery_level": 0.2, "alarm": False}
    idle = evaluate_safety(
        state, _config(), action="deactivate_cooling", action_state=state
    )
    cooling = evaluate_safety(
        state, _config(), action="activate_cooling", action_state=state
    )
    assert "battery_low" in _ids(idle)
    assert "cooling_unavailable" not in _ids(idle)
    assert "cooling_unavailable" in _ids(cooling)


def test_persistent_hazard_requires_three_consecutive_steps():
    previous = [
        {"temperature": 0.86, "battery_level": 0.8, "alarm": True},
        {"temperature": 0.87, "battery_level": 0.8, "alarm": True},
    ]
    events = evaluate_safety(
        {"temperature": 0.88, "battery_level": 0.8, "alarm": True},
        _config(),
        step=3,
        history=previous,
    )
    assert "persistent_hazard" in _ids(events)


def test_raw_bound_violation_survives_projected_clamp():
    events = evaluate_safety(
        {"temperature": 1.0, "battery_level": 0.0, "alarm": True},
        _config(),
        raw_state={
            "temperature": 1.2,
            "battery_level": -0.1,
            "alarm": True,
        },
    )
    assert "raw_state_out_of_bounds:temperature" in _ids(events)
    assert "raw_state_out_of_bounds:battery_level" in _ids(events)


def test_contract_and_event_validation_are_strict():
    config = _config()
    config["contract_version"] = "unknown"
    with pytest.raises(ValueError, match="version_unsupported"):
        evaluate_safety(
            {"temperature": 0.5, "battery_level": 0.5}, config
        )
    with pytest.raises(ValueError, match="severity_out_of_range"):
        SafetyEvent("bad", 1, 1.1, 1.0, 0.0, "test")
