"""Contratos semánticos y versionados de seguridad para rollouts N4."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


SAFETY_CONTRACT_VERSION = "thermal_with_battery.safety.v1"
DEFAULT_OPERATIONAL_MINIMUM = 0.3
CRITICAL_TEMPERATURE_MARGIN = 0.05
PERSISTENT_HAZARD_STEPS = 3


@dataclass(frozen=True, slots=True)
class SafetyEvent:
    predicate_id: str
    step: int
    severity: float
    observed_value: float | bool
    threshold: float | bool
    source: str

    def __post_init__(self) -> None:
        severity = float(self.severity)
        if not self.predicate_id or not self.source:
            raise ValueError("safety_event_identity_required")
        if self.step < 1:
            raise ValueError("safety_event_step_must_be_positive")
        if not math.isfinite(severity) or not 0.0 <= severity <= 1.0:
            raise ValueError("safety_event_severity_out_of_range")
        object.__setattr__(self, "severity", severity)


def thermal_battery_safety_config(
    *,
    alarm_threshold: float,
    operational_minimum: float = DEFAULT_OPERATIONAL_MINIMUM,
) -> dict[str, Any]:
    """Construye la fuente de verdad explícita para el contrato térmico."""
    alarm_threshold = float(alarm_threshold)
    operational_minimum = float(operational_minimum)
    if not 0.0 <= alarm_threshold <= 1.0:
        raise ValueError("safety_alarm_threshold_out_of_range")
    if not 0.0 <= operational_minimum <= 1.0:
        raise ValueError("safety_operational_minimum_out_of_range")
    return {
        "contract_version": SAFETY_CONTRACT_VERSION,
        "scenario": "thermal_with_battery",
        "alarm_threshold": alarm_threshold,
        "critical_threshold": min(
            1.0, alarm_threshold + CRITICAL_TEMPERATURE_MARGIN
        ),
        "operational_minimum": operational_minimum,
        "persistent_hazard_steps": PERSISTENT_HAZARD_STEPS,
    }


def evaluate_safety(
    state: Mapping[str, Any],
    scenario_config: Mapping[str, Any],
    *,
    step: int = 1,
    source: str = "candidate",
    action: str | None = None,
    action_state: Mapping[str, Any] | None = None,
    history: Sequence[Mapping[str, Any]] = (),
    raw_state: Mapping[str, Any] | None = None,
) -> list[SafetyEvent]:
    """Evalúa peligros semánticos sin confundir una alarma correcta con daño."""
    if scenario_config.get("contract_version") != SAFETY_CONTRACT_VERSION:
        raise ValueError("safety_contract_version_unsupported")
    if scenario_config.get("scenario") != "thermal_with_battery":
        raise ValueError("safety_scenario_unsupported")
    alarm_threshold = float(scenario_config["alarm_threshold"])
    critical_threshold = float(scenario_config["critical_threshold"])
    operational_minimum = float(scenario_config["operational_minimum"])
    persistent_steps = int(scenario_config["persistent_hazard_steps"])
    temperature = _finite_float(state, "temperature")
    battery = _finite_float(state, "battery_level")
    alarm = bool(state.get("alarm", False))
    events: list[SafetyEvent] = []

    if temperature > alarm_threshold:
        events.append(
            _event(
                "temp_above_threshold",
                step,
                _upper_severity(temperature, alarm_threshold),
                temperature,
                alarm_threshold,
                source,
            )
        )
    if temperature > critical_threshold:
        events.append(
            _event(
                "temp_critical",
                step,
                _upper_severity(temperature, critical_threshold),
                temperature,
                critical_threshold,
                source,
            )
        )
    if battery < operational_minimum:
        events.append(
            _event(
                "battery_low",
                step,
                _lower_severity(battery, operational_minimum),
                battery,
                operational_minimum,
                source,
            )
        )
    command_battery = (
        _finite_float(action_state, "battery_level")
        if action_state is not None
        else battery
    )
    if action == "activate_cooling" and command_battery <= operational_minimum:
        events.append(
            _event(
                "cooling_unavailable",
                step,
                max(0.25, _lower_severity(command_battery, operational_minimum)),
                command_battery,
                operational_minimum,
                source,
            )
        )
    if temperature > alarm_threshold and not alarm:
        events.append(
            _event(
                "alarm_missed",
                step,
                max(0.25, _upper_severity(temperature, alarm_threshold)),
                alarm,
                True,
                source,
            )
        )
    hazard_streak = 1 if temperature > alarm_threshold else 0
    for previous in reversed(tuple(history)):
        if float(previous["temperature"]) <= alarm_threshold:
            break
        hazard_streak += 1
    if hazard_streak >= persistent_steps:
        events.append(
            _event(
                "persistent_hazard",
                step,
                min(1.0, hazard_streak / max(persistent_steps, 1)),
                temperature,
                alarm_threshold,
                source,
            )
        )
    for variable, value in (raw_state or {}).items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        numeric = float(value)
        if numeric < 0.0 or numeric > 1.0:
            events.append(
                _event(
                    f"raw_state_out_of_bounds:{variable}",
                    step,
                    min(1.0, max(-numeric, numeric - 1.0)),
                    numeric,
                    0.0 if numeric < 0.0 else 1.0,
                    source,
                )
            )
    return sorted(events, key=lambda item: item.predicate_id)


def _event(
    predicate_id: str,
    step: int,
    severity: float,
    observed_value: float | bool,
    threshold: float | bool,
    source: str,
) -> SafetyEvent:
    return SafetyEvent(
        predicate_id=predicate_id,
        step=step,
        severity=min(1.0, max(0.0, float(severity))),
        observed_value=observed_value,
        threshold=threshold,
        source=source,
    )


def _finite_float(state: Mapping[str, Any], name: str) -> float:
    value = float(state[name])
    if not math.isfinite(value):
        raise ValueError(f"safety_state_nonfinite:{name}")
    return value


def _upper_severity(value: float, threshold: float) -> float:
    return min(1.0, max(0.0, value - threshold) / max(1.0 - threshold, 1e-9))


def _lower_severity(value: float, threshold: float) -> float:
    return min(1.0, max(0.0, threshold - value) / max(threshold, 1e-9))
