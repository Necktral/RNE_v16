"""Rollouts contrafactuales puros y multistep para etiquetado de seguridad."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from .conditions import evaluate_condition
from .contracts import TransitionSpec
from .safety_contracts import (
    SafetyEvent,
    evaluate_safety,
    thermal_battery_safety_config,
)


RISK_TARGET_VERSION = "n4-candidate-risk.v1"
DEFAULT_RISK_WEIGHTS = {
    "missed_hazard": 0.35,
    "alarm_miss": 0.20,
    "actuator_miss": 0.20,
    "persistence_miss": 0.15,
    "raw_bounds": 0.10,
}


@dataclass(frozen=True, slots=True)
class RolloutStep:
    step: int
    action: str
    external_input: float
    previous_state: Mapping[str, Any]
    raw_state: Mapping[str, Any]
    state: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class CandidateRiskReport:
    horizon: int
    violation_frequency: float
    maximum_severity: float
    cumulative_severity: float
    missed_hazard_rate: float
    actuator_failure_miss_rate: float
    alarm_miss_rate: float
    persistent_hazard_miss_rate: float
    raw_bound_violation: float
    invariant_risk: float
    target_version: str
    oracle_events: tuple[SafetyEvent, ...]
    baseline_events: tuple[SafetyEvent, ...]
    candidate_events: tuple[SafetyEvent, ...]


OracleStep = Callable[
    [Mapping[str, Any], str, float],
    RolloutStep | Mapping[str, Any] | tuple[Mapping[str, Any], Mapping[str, Any]],
]


def simulate_rollout(
    initial_state: Mapping[str, Any],
    actions: Sequence[str],
    external_inputs: Sequence[float],
    transition_spec: TransitionSpec,
    horizon: int,
) -> list[dict[str, Any]]:
    """Encadena el estado proyectado de cada transición sin efectos laterales."""
    return [
        dict(item.state)
        for item in _simulate_spec_rollout(
            initial_state, actions, external_inputs, transition_spec, horizon
        )
    ]


def rollout_contrafactual(
    initial_state: Mapping[str, Any],
    action_sequence: Sequence[str],
    external_inputs: Sequence[float],
    candidate_spec: TransitionSpec,
    baseline_spec: TransitionSpec,
    oracle: OracleStep,
    horizon: int,
    *,
    safety_config: Mapping[str, Any] | None = None,
    risk_weights: Mapping[str, float] = DEFAULT_RISK_WEIGHTS,
) -> CandidateRiskReport:
    """Compara candidato y baseline con un oráculo bajo inputs idénticos."""
    _validate_rollout_inputs(action_sequence, external_inputs, horizon)
    config = dict(
        safety_config
        or thermal_battery_safety_config(
            alarm_threshold=float(
                baseline_spec.parameters[
                    baseline_spec.alarm_threshold_parameter
                ]
            )
        )
    )
    oracle_steps = _simulate_oracle_rollout(
        initial_state, action_sequence, external_inputs, oracle, horizon
    )
    baseline_steps = _simulate_spec_rollout(
        initial_state,
        action_sequence,
        external_inputs,
        baseline_spec,
        horizon,
    )
    candidate_steps = _simulate_spec_rollout(
        initial_state,
        action_sequence,
        external_inputs,
        candidate_spec,
        horizon,
    )
    oracle_events = _evaluate_rollout(oracle_steps, config, "oracle")
    baseline_events = _evaluate_rollout(baseline_steps, config, "baseline")
    candidate_events = _evaluate_rollout(candidate_steps, config, "candidate")

    hazard_ids = {"temp_above_threshold", "temp_critical", "battery_low"}
    missed_hazard = _miss_rate(oracle_events, candidate_events, hazard_ids)
    alarm_miss = _alarm_miss_rate(
        oracle_steps,
        candidate_steps,
        float(config["alarm_threshold"]),
    )
    actuator_miss = _actuator_failure_miss_rate(
        oracle_steps,
        candidate_steps,
        float(config["operational_minimum"]),
    )
    persistence_miss = _miss_rate(
        oracle_events, candidate_events, {"persistent_hazard"}
    )
    raw_events = tuple(
        event
        for event in candidate_events
        if event.predicate_id.startswith("raw_state_out_of_bounds:")
    )
    raw_bound_violation = max(
        (event.severity for event in raw_events), default=0.0
    )
    candidate_semantic = tuple(
        event
        for event in candidate_events
        if not event.predicate_id.startswith("raw_state_out_of_bounds:")
    )
    violation_steps = {event.step for event in candidate_semantic}
    violation_frequency = len(violation_steps) / horizon
    maximum_severity = max(
        (event.severity for event in candidate_events), default=0.0
    )
    cumulative_severity = min(
        1.0,
        sum(event.severity for event in candidate_events) / horizon,
    )
    weights = _validated_weights(risk_weights)
    invariant_risk = min(
        1.0,
        weights["missed_hazard"] * missed_hazard
        + weights["alarm_miss"] * alarm_miss
        + weights["actuator_miss"] * actuator_miss
        + weights["persistence_miss"] * persistence_miss
        + weights["raw_bounds"] * raw_bound_violation,
    )
    return CandidateRiskReport(
        horizon=horizon,
        violation_frequency=round(violation_frequency, 9),
        maximum_severity=round(maximum_severity, 9),
        cumulative_severity=round(cumulative_severity, 9),
        missed_hazard_rate=round(missed_hazard, 9),
        actuator_failure_miss_rate=round(actuator_miss, 9),
        alarm_miss_rate=round(alarm_miss, 9),
        persistent_hazard_miss_rate=round(persistence_miss, 9),
        raw_bound_violation=round(raw_bound_violation, 9),
        invariant_risk=round(invariant_risk, 9),
        target_version=RISK_TARGET_VERSION,
        oracle_events=oracle_events,
        baseline_events=baseline_events,
        candidate_events=candidate_events,
    )


def thermal_battery_oracle(
    *,
    alarm_threshold: float = 0.85,
    cooling_effect: float = 0.07,
    battery_threshold: float = 0.3,
    battery_discharge_rate: float = 0.06,
    battery_charge_rate: float = 0.01,
) -> OracleStep:
    """Crea la dinámica real pura con la guarda energética oculta."""

    def step(
        state: Mapping[str, Any], action: str, external_input: float
    ) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        temperature = float(state["temperature"]) + float(external_input)
        battery = float(state["battery_level"])
        if action == "activate_cooling":
            if battery > battery_threshold:
                temperature -= cooling_effect
            battery -= battery_discharge_rate
            cooling_active = True
        elif action == "deactivate_cooling":
            battery += battery_charge_rate
            cooling_active = False
        else:
            raise ValueError(f"rollout_action_unknown:{action}")
        raw = {
            "temperature": temperature,
            "battery_level": battery,
            "cooling_active": cooling_active,
            "alarm": temperature >= alarm_threshold,
        }
        projected = {
            **raw,
            "temperature": _clamp(temperature),
            "battery_level": _clamp(battery),
        }
        projected["alarm"] = projected["temperature"] >= alarm_threshold
        return raw, projected

    return step


def _simulate_spec_rollout(
    initial_state, actions, external_inputs, spec, horizon
) -> tuple[RolloutStep, ...]:
    _validate_rollout_inputs(actions, external_inputs, horizon)
    state = dict(initial_state)
    steps = []
    for index in range(horizon):
        raw, projected = _execute_spec_step(
            state,
            actions[index],
            float(external_inputs[index]),
            spec,
        )
        steps.append(
            RolloutStep(
                step=index + 1,
                action=actions[index],
                external_input=float(external_inputs[index]),
                previous_state=dict(state),
                raw_state=raw,
                state=projected,
            )
        )
        state = projected
    return tuple(steps)


def _simulate_oracle_rollout(
    initial_state, actions, external_inputs, oracle, horizon
) -> tuple[RolloutStep, ...]:
    state = dict(initial_state)
    steps = []
    for index in range(horizon):
        result = oracle(state, actions[index], float(external_inputs[index]))
        if isinstance(result, RolloutStep):
            raw, projected = dict(result.raw_state), dict(result.state)
        elif isinstance(result, tuple) and len(result) == 2:
            raw, projected = dict(result[0]), dict(result[1])
        else:
            raw = projected = dict(result)
        steps.append(
            RolloutStep(
                step=index + 1,
                action=actions[index],
                external_input=float(external_inputs[index]),
                previous_state=dict(state),
                raw_state=raw,
                state=projected,
            )
        )
        state = projected
    return tuple(steps)


def _execute_spec_step(state, action, external_input, spec):
    actions = {item.name: item for item in spec.actions}
    variables = {item.name: item for item in spec.variables}
    if action not in actions:
        raise ValueError(f"rollout_action_unknown:{action}")
    raw = dict(state)
    raw.update(dict(actions[action].assignments))
    for equation in spec.equations:
        value = 0.0
        for term in equation.terms:
            if term.source == "external_input":
                source = external_input
            elif term.source.startswith("state."):
                source = state[term.source.split(".", 1)[1]]
            elif term.source.startswith("next."):
                source = raw[term.source.split(".", 1)[1]]
            else:
                raise ValueError(f"rollout_term_unsupported:{term.source}")
            coefficient = term.coefficient
            if term.parameter is not None:
                coefficient *= float(spec.parameters[term.parameter])
            value += float(source) * coefficient
        condition_state = {**dict(state), **raw}
        for effect in equation.effects:
            if effect.action != action or not evaluate_condition(
                effect.precondition, condition_state
            ):
                continue
            magnitude = effect.delta
            if effect.parameter is not None:
                magnitude *= float(spec.parameters[effect.parameter])
            value += magnitude
        raw[equation.target] = value
    projected = dict(raw)
    for equation in spec.equations:
        if equation.clamp:
            variable = variables[equation.target]
            projected[equation.target] = _clamp(
                float(raw[equation.target]),
                variable.lower,
                variable.upper,
            )
    threshold = float(spec.parameters[spec.alarm_threshold_parameter])
    projected[spec.alarm_variable] = (
        float(projected[spec.alarm_source]) >= threshold
        if spec.alarm_operator == ">="
        else float(projected[spec.alarm_source]) <= threshold
    )
    raw[spec.alarm_variable] = (
        float(raw[spec.alarm_source]) >= threshold
        if spec.alarm_operator == ">="
        else float(raw[spec.alarm_source]) <= threshold
    )
    names = {item.name for item in spec.variables}
    return (
        {name: raw[name] for name in names},
        {name: projected[name] for name in names},
    )


def _evaluate_rollout(steps, config, source):
    events = []
    history = []
    for item in steps:
        events.extend(
            evaluate_safety(
                item.state,
                config,
                step=item.step,
                source=source,
                action=item.action,
                action_state=item.previous_state,
                history=history,
                raw_state=item.raw_state,
            )
        )
        history.append(item.state)
    return tuple(events)


def _miss_rate(oracle_events, candidate_events, predicate_ids):
    oracle = {
        (event.step, event.predicate_id): event.severity
        for event in oracle_events
        if event.predicate_id in predicate_ids
    }
    if not oracle:
        return 0.0
    candidate = {
        (event.step, event.predicate_id): event.severity
        for event in candidate_events
        if event.predicate_id in predicate_ids
    }
    return sum(
        max(0.0, severity - candidate.get(key, 0.0))
        for key, severity in oracle.items()
    ) / len(oracle)


def _alarm_miss_rate(oracle_steps, candidate_steps, alarm_threshold):
    hazards = [
        (oracle, candidate)
        for oracle, candidate in zip(oracle_steps, candidate_steps)
        if float(oracle.state["temperature"]) > alarm_threshold
    ]
    if not hazards:
        return 0.0
    return sum(
        not bool(candidate.state.get("alarm", False))
        for _, candidate in hazards
    ) / len(hazards)


def _actuator_failure_miss_rate(
    oracle_steps, candidate_steps, operational_minimum
):
    unavailable = [
        (oracle, candidate)
        for oracle, candidate in zip(oracle_steps, candidate_steps)
        if oracle.action == "activate_cooling"
        and float(oracle.previous_state["battery_level"])
        <= operational_minimum
    ]
    if not unavailable:
        return 0.0
    misses = 0
    for _, candidate in unavailable:
        no_cooling_temperature = (
            float(candidate.previous_state["temperature"])
            + candidate.external_input
        )
        predicted_benefit = (
            no_cooling_temperature - float(candidate.raw_state["temperature"])
        )
        misses += predicted_benefit > 1e-9
    return misses / len(unavailable)


def _validate_rollout_inputs(actions, external_inputs, horizon):
    if horizon < 1:
        raise ValueError("rollout_horizon_must_be_positive")
    if len(actions) < horizon or len(external_inputs) < horizon:
        raise ValueError("rollout_sequence_shorter_than_horizon")
    if not all(math.isfinite(float(value)) for value in external_inputs[:horizon]):
        raise ValueError("rollout_external_input_nonfinite")


def _validated_weights(weights):
    required = set(DEFAULT_RISK_WEIGHTS)
    if set(weights) != required:
        raise ValueError("candidate_risk_weights_invalid")
    result = {name: float(value) for name, value in weights.items()}
    if any(not math.isfinite(value) or value < 0.0 for value in result.values()):
        raise ValueError("candidate_risk_weights_invalid")
    if sum(result.values()) > 1.0 + 1e-9:
        raise ValueError("candidate_risk_weights_exceed_one")
    return result


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, float(value)))
