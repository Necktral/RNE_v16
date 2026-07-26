"""Escenario térmico con una condición causal oculta dependiente de batería."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import cached_property

from .causal_signature import CausalEdge, InterventionEffect, ScenarioCausalSignature
from .compatibility import ScenarioStructuralProfile
from .scenario import (
    CognitiveScenario,
    ScenarioConfig,
    ScenarioObservation,
    ScenarioTransition,
)


@dataclass(frozen=True, slots=True)
class ThermalBatteryState:
    temperature: float
    battery_level: float
    cooling_active: bool
    alarm: bool


class ThermalBatteryScenario(CognitiveScenario):
    """Mundo real cuya guarda de enfriamiento falta en el IR inicial del MCI."""

    def __init__(
        self,
        *,
        initial_temperature: float = 0.85,
        initial_battery: float = 1.0,
        cooling_effect: float = 0.07,
        alarm_threshold: float = 0.85,
        battery_threshold: float = 0.3,
        battery_discharge_rate: float = 0.06,
        battery_charge_rate: float = 0.01,
    ) -> None:
        if cooling_effect < 0.0:
            raise ValueError("cooling_effect debe ser una magnitud positiva")
        if not 0.0 <= battery_threshold <= 1.0:
            raise ValueError("battery_threshold debe estar entre 0 y 1")
        if battery_discharge_rate < 0.0 or battery_charge_rate < 0.0:
            raise ValueError("Las tasas de batería deben ser no negativas")
        self._alarm_threshold = float(alarm_threshold)
        self._cooling_effect = float(cooling_effect)
        self._battery_threshold = float(battery_threshold)
        self._battery_discharge_rate = float(battery_discharge_rate)
        self._battery_charge_rate = float(battery_charge_rate)
        temperature = self._clamp(initial_temperature)
        battery = self._clamp(initial_battery)
        self._state = ThermalBatteryState(
            temperature=temperature,
            battery_level=battery,
            cooling_active=False,
            alarm=temperature >= self._alarm_threshold,
        )
        self._config = ScenarioConfig(
            name="thermal_with_battery",
            description="Control térmico con eficacia condicionada por batería",
            main_variable="temperature",
            alarm_threshold=self._alarm_threshold,
            interventions=["activate_cooling", "deactivate_cooling"],
            formula_template="TEMP_HIGH -> ACTIVATE_COOLING",
            type_context={"TEMP_HIGH": "bool", "ACTIVATE_COOLING": "bool"},
        )

    @staticmethod
    def _clamp(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    @property
    def config(self) -> ScenarioConfig:
        return self._config

    @property
    def alarm_threshold(self) -> float:
        return self._alarm_threshold

    @property
    def cooling_effect(self) -> float:
        return self._cooling_effect

    def set_regime(self, cooling_effect: float) -> None:
        """Cambia la dinámica real sin reescribir el IR certificado del runner."""
        if cooling_effect < 0.0:
            raise ValueError("cooling_effect debe ser una magnitud positiva")
        self._cooling_effect = float(cooling_effect)
        self.__dict__.pop("causal_signature", None)

    @cached_property
    def structural_profile(self) -> ScenarioStructuralProfile:
        blob = json.dumps(
            {
                "name": self._config.name,
                "alarm_threshold": self._alarm_threshold,
                "interventions": self._config.interventions,
                "battery_guard_hidden": True,
                "battery_threshold": self._battery_threshold,
                "battery_discharge_rate": self._battery_discharge_rate,
                "battery_charge_rate": self._battery_charge_rate,
            },
            sort_keys=True,
        )
        return ScenarioStructuralProfile(
            scenario_name=self._config.name,
            scenario_version="1.0",
            scenario_config_hash=hashlib.sha256(blob.encode()).hexdigest()[:12],
            control_topology="threshold_recovery_loop",
            optimization_direction="minimize",
            intervention_semantics=tuple(self._config.interventions),
            counterfactual_policy="opposite_intervention",
            relation_polarity="lower_is_better",
            main_variable="temperature",
        )

    @cached_property
    def causal_signature(self) -> ScenarioCausalSignature:
        return ScenarioCausalSignature(
            scenario_name=self._config.name,
            scenario_version="1.0",
            observable_variables=frozenset(
                {"temperature", "battery_level", "cooling_active"}
            ),
            control_variables=frozenset({"cooling_active"}),
            main_variable="temperature",
            optimization_direction="minimize",
            causal_polarity="lower_is_better",
            alarm_semantics="threshold_above",
            intervention_effects=(
                InterventionEffect(
                    intervention_name="activate_cooling",
                    target_variable="temperature",
                    expected_direction="-",
                    expected_magnitude=self._cooling_effect,
                    semantic_role="corrective",
                ),
                InterventionEffect(
                    intervention_name="deactivate_cooling",
                    target_variable="temperature",
                    expected_direction="+",
                    expected_magnitude=0.0,
                    semantic_role="neutral",
                ),
            ),
            counterfactual_policy="opposite_intervention",
            counterfactual_variable="temperature",
            causal_edges=(
                CausalEdge("external_input", "temperature", "+"),
                CausalEdge("battery_level", "cooling_active", "+"),
                CausalEdge("cooling_active", "temperature", "-"),
                CausalEdge("temperature", "alarm", "+"),
            ),
            proposition_vocabulary=frozenset(
                {
                    "TEMP_HIGH",
                    "TEMP_NORMAL",
                    "BATTERY_LOW",
                    "COOLING_ACTIVE",
                    "ACTIVATE_COOLING",
                    "KEEP_IDLE",
                }
            ),
            metadata={
                "hidden_precondition": (
                    f"battery_level > {self._battery_threshold}"
                )
            },
        )

    def _extract_propositions(self, state: ThermalBatteryState) -> list[str]:
        propositions = [
            "TEMP_HIGH"
            if state.temperature >= self._alarm_threshold
            else "TEMP_NORMAL"
        ]
        if state.battery_level <= self._battery_threshold:
            propositions.append("BATTERY_LOW")
        if state.cooling_active:
            propositions.append("COOLING_ACTIVE")
        return propositions

    def observe(self) -> ScenarioObservation:
        return ScenarioObservation(
            state={
                "temperature": self._state.temperature,
                "battery_level": self._state.battery_level,
                "cooling_active": self._state.cooling_active,
            },
            propositions=self._extract_propositions(self._state),
            alarm=self._state.alarm,
        )

    def _apply_activate_cooling(
        self, temperature: float, battery: float
    ) -> tuple[float, float, bool]:
        if battery > self._battery_threshold:
            temperature -= self._cooling_effect
        return temperature, battery - self._battery_discharge_rate, True

    def _apply_deactivate_cooling(
        self, temperature: float, battery: float
    ) -> tuple[float, float, bool]:
        return temperature, battery + self._battery_charge_rate, False

    def _compute_transition(
        self,
        state: ThermalBatteryState,
        *,
        intervention: str,
        external_input: float,
    ) -> ThermalBatteryState:
        temperature = state.temperature + float(external_input)
        battery = state.battery_level
        dispatch = {
            "activate_cooling": self._apply_activate_cooling,
            "deactivate_cooling": self._apply_deactivate_cooling,
        }
        action = dispatch.get(intervention)
        if action is None:
            raise ValueError(f"Intervención desconocida: {intervention}")
        temperature, battery, cooling_active = action(temperature, battery)
        temperature = self._clamp(temperature)
        return ThermalBatteryState(
            temperature=temperature,
            battery_level=self._clamp(battery),
            cooling_active=cooling_active,
            alarm=temperature >= self._alarm_threshold,
        )

    def _transition_result(self, state: ThermalBatteryState) -> ScenarioTransition:
        return ScenarioTransition(
            state={
                "temperature": state.temperature,
                "battery_level": state.battery_level,
                "cooling_active": state.cooling_active,
            },
            propositions=self._extract_propositions(state),
            alarm=state.alarm,
        )

    def factual_transition(
        self, *, intervention: str, external_input: float
    ) -> ScenarioTransition:
        self._state = self._compute_transition(
            self._state,
            intervention=intervention,
            external_input=external_input,
        )
        return self._transition_result(self._state)

    def simulate_counterfactual(
        self, *, intervention: str, external_input: float
    ) -> ScenarioTransition:
        return self._transition_result(
            self._compute_transition(
                self._state,
                intervention=intervention,
                external_input=external_input,
            )
        )

    def get_formula(self, observation: ScenarioObservation) -> str:
        return self._config.formula_template

    def select_intervention(self, observation: ScenarioObservation) -> str:
        return "activate_cooling" if observation.alarm else "deactivate_cooling"

    def get_main_proposition(self, observation: ScenarioObservation) -> str:
        return "TEMP_HIGH" if observation.alarm else "TEMP_NORMAL"

    def get_intervention_proposition(self, intervention: str) -> str:
        return (
            "ACTIVATE_COOLING"
            if intervention == "activate_cooling"
            else "KEEP_IDLE"
        )
