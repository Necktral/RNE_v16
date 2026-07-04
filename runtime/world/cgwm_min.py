"""Causal world model mínimo orientado a homeostasis."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class WorldState:
    temperature: float
    cooling_active: bool
    alarm: bool


class CGWMMin:
    def __init__(self, *, initial_temperature: float = 0.82, alarm_threshold: float = 0.85):
        self.alarm_threshold = alarm_threshold
        self.state = WorldState(
            temperature=initial_temperature,
            cooling_active=False,
            alarm=initial_temperature >= alarm_threshold,
        )

    def observe(self) -> dict:
        return {
            "temperature": self.state.temperature,
            "cooling_active": self.state.cooling_active,
            "alarm": self.state.alarm,
        }

    def _transition(
        self,
        state: WorldState,
        *,
        intervention: str,
        external_heat: float,
    ) -> WorldState:
        cooling_active = state.cooling_active
        if intervention == "activate_cooling":
            cooling_active = True
        elif intervention == "deactivate_cooling":
            cooling_active = False

        cooling_delta = 0.07 if cooling_active else 0.0
        next_temp = max(0.0, min(1.0, state.temperature + external_heat - cooling_delta))
        return WorldState(
            temperature=next_temp,
            cooling_active=cooling_active,
            alarm=next_temp >= self.alarm_threshold,
        )

    def factual_transition(self, *, intervention: str, external_heat: float = 0.03) -> dict:
        self.state = self._transition(
            self.state, intervention=intervention, external_heat=external_heat
        )
        return self.observe()

    def simulate_counterfactual(
        self, *, intervention: str, external_heat: float = 0.03
    ) -> dict:
        simulated = self._transition(
            self.state, intervention=intervention, external_heat=external_heat
        )
        return {
            "temperature": simulated.temperature,
            "cooling_active": simulated.cooling_active,
            "alarm": simulated.alarm,
        }
