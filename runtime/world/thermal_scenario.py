"""Escenario térmico homeostático - implementación de referencia."""

from __future__ import annotations

from dataclasses import dataclass

from .scenario import (
    CognitiveScenario,
    ScenarioConfig,
    ScenarioObservation,
    ScenarioTransition,
)


@dataclass
class ThermalWorldState:
    """Estado interno del mundo térmico."""

    temperature: float
    cooling_active: bool
    alarm: bool


class ThermalScenario(CognitiveScenario):
    """Escenario homeostático de control de temperatura.

    Este escenario modela un sistema de control de temperatura con:
    - Variable principal: temperature (0.0 - 1.0)
    - Intervenciones: activate_cooling, deactivate_cooling
    - Umbral de alarma: configurable (default 0.85)
    - Dinámica: enfriamiento reduce temperatura, calor externo la aumenta
    """

    def __init__(
        self,
        *,
        initial_temperature: float = 0.82,
        alarm_threshold: float = 0.85,
        cooling_effect: float = 0.07,
    ):
        """Inicializa escenario térmico.

        Args:
            initial_temperature: Temperatura inicial (0.0-1.0).
            alarm_threshold: Umbral de alarma.
            cooling_effect: Efecto del enfriamiento por paso.
        """
        self._alarm_threshold = alarm_threshold
        self._cooling_effect = cooling_effect
        self._state = ThermalWorldState(
            temperature=initial_temperature,
            cooling_active=False,
            alarm=initial_temperature >= alarm_threshold,
        )
        self._config = ScenarioConfig(
            name="thermal_homeostasis",
            description="Control de temperatura homeostático con enfriamiento activo",
            main_variable="temperature",
            alarm_threshold=alarm_threshold,
            interventions=["activate_cooling", "deactivate_cooling"],
            formula_template="TEMP_HIGH -> ACTIVATE_COOLING",
            type_context={"TEMP_HIGH": "bool", "ACTIVATE_COOLING": "bool"},
        )

    @property
    def config(self) -> ScenarioConfig:
        return self._config

    @property
    def alarm_threshold(self) -> float:
        """Umbral de alarma para compatibilidad con código existente."""
        return self._alarm_threshold

    def observe(self) -> ScenarioObservation:
        temp_high = self._state.temperature >= self._alarm_threshold
        propositions = ["TEMP_HIGH"] if temp_high else ["TEMP_NORMAL"]
        if self._state.cooling_active:
            propositions.append("COOLING_ACTIVE")

        return ScenarioObservation(
            state={
                "temperature": self._state.temperature,
                "cooling_active": self._state.cooling_active,
            },
            propositions=propositions,
            alarm=self._state.alarm,
        )

    def _compute_transition(
        self,
        state: ThermalWorldState,
        *,
        intervention: str,
        external_input: float,
    ) -> ThermalWorldState:
        """Computa transición de estado."""
        cooling_active = state.cooling_active
        if intervention == "activate_cooling":
            cooling_active = True
        elif intervention == "deactivate_cooling":
            cooling_active = False

        cooling_delta = self._cooling_effect if cooling_active else 0.0
        next_temp = max(0.0, min(1.0, state.temperature + external_input - cooling_delta))

        return ThermalWorldState(
            temperature=next_temp,
            cooling_active=cooling_active,
            alarm=next_temp >= self._alarm_threshold,
        )

    def factual_transition(
        self,
        *,
        intervention: str,
        external_input: float,
    ) -> ScenarioTransition:
        self._state = self._compute_transition(
            self._state,
            intervention=intervention,
            external_input=external_input,
        )
        return ScenarioTransition(
            state={
                "temperature": self._state.temperature,
                "cooling_active": self._state.cooling_active,
            },
            propositions=self.observe().propositions,
            alarm=self._state.alarm,
        )

    def simulate_counterfactual(
        self,
        *,
        intervention: str,
        external_input: float,
    ) -> ScenarioTransition:
        simulated = self._compute_transition(
            self._state,
            intervention=intervention,
            external_input=external_input,
        )
        return ScenarioTransition(
            state={
                "temperature": simulated.temperature,
                "cooling_active": simulated.cooling_active,
            },
            propositions=[
                "TEMP_HIGH" if simulated.temperature >= self._alarm_threshold else "TEMP_NORMAL"
            ],
            alarm=simulated.alarm,
        )

    def get_formula(self, observation: ScenarioObservation) -> str:
        return self._config.formula_template

    def select_intervention(self, observation: ScenarioObservation) -> str:
        if observation.alarm:
            return "activate_cooling"
        return "deactivate_cooling"

    def get_main_proposition(self, observation: ScenarioObservation) -> str:
        if observation.alarm:
            return "TEMP_HIGH"
        return "TEMP_NORMAL"

    def get_intervention_proposition(self, intervention: str) -> str:
        if intervention == "activate_cooling":
            return "ACTIVATE_COOLING"
        return "KEEP_IDLE"


# Factory function para compatibilidad
def create_thermal_scenario(
    *,
    initial_temperature: float = 0.82,
    alarm_threshold: float = 0.85,
) -> ThermalScenario:
    """Crea escenario térmico con configuración por defecto."""
    return ThermalScenario(
        initial_temperature=initial_temperature,
        alarm_threshold=alarm_threshold,
    )
