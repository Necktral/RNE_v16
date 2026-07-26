"""Gestión de recursos con producción condicionada por energía."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import cached_property

from .causal_signature import CausalEdge, InterventionEffect, ScenarioCausalSignature
from .compatibility import ScenarioStructuralProfile
from .scenario import CognitiveScenario, ScenarioConfig, ScenarioObservation, ScenarioTransition


@dataclass(frozen=True, slots=True)
class ResourceEnergyState:
    stock_level: float
    energy_level: float
    production_active: bool
    scarcity_alert: bool


class ResourceWithEnergyScenario(CognitiveScenario):
    """Mundo real; el IR epistémico omite deliberadamente la guarda energética."""

    def __init__(
        self,
        *,
        initial_stock: float = 0.25,
        initial_energy: float = 0.75,
        scarcity_threshold: float = 0.20,
        energy_threshold: float = 0.30,
        production_rate: float = 0.08,
        production_energy_cost: float = 0.06,
        recovery_rate: float = 0.01,
    ) -> None:
        for name, value in {
            "initial_stock": initial_stock,
            "initial_energy": initial_energy,
            "scarcity_threshold": scarcity_threshold,
            "energy_threshold": energy_threshold,
        }.items():
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} debe estar entre 0 y 1")
        if min(production_rate, production_energy_cost, recovery_rate) < 0.0:
            raise ValueError("Las magnitudes deben ser no negativas")
        self._scarcity_threshold = float(scarcity_threshold)
        self._energy_threshold = float(energy_threshold)
        self._production_rate = float(production_rate)
        self._production_energy_cost = float(production_energy_cost)
        self._recovery_rate = float(recovery_rate)
        stock, energy = self._clamp(initial_stock), self._clamp(initial_energy)
        self._state = ResourceEnergyState(
            stock_level=stock,
            energy_level=energy,
            production_active=False,
            scarcity_alert=stock <= self._scarcity_threshold,
        )
        self._config = ScenarioConfig(
            name="resource_with_energy",
            description="Gestión de stock con producción condicionada por energía",
            main_variable="stock_level",
            alarm_threshold=self._scarcity_threshold,
            interventions=["start_production", "stop_production"],
            formula_template="STOCK_LOW -> START_PRODUCTION",
            type_context={"STOCK_LOW": "bool", "START_PRODUCTION": "bool"},
        )

    @staticmethod
    def _clamp(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    @property
    def config(self) -> ScenarioConfig:
        return self._config

    @property
    def alarm_threshold(self) -> float:
        return self._scarcity_threshold

    @cached_property
    def structural_profile(self) -> ScenarioStructuralProfile:
        payload = {
            "name": self._config.name,
            "scarcity_threshold": self._scarcity_threshold,
            "energy_threshold": self._energy_threshold,
            "production_rate": self._production_rate,
            "production_energy_cost": self._production_energy_cost,
            "recovery_rate": self._recovery_rate,
        }
        return ScenarioStructuralProfile(
            scenario_name=self._config.name,
            scenario_version="1.0",
            scenario_config_hash=hashlib.sha256(
                json.dumps(payload, sort_keys=True).encode()
            ).hexdigest()[:12],
            control_topology="threshold_recovery_loop",
            optimization_direction="maximize",
            intervention_semantics=tuple(self._config.interventions),
            counterfactual_policy="opposite_intervention",
            relation_polarity="higher_is_better",
            main_variable="stock_level",
        )

    @cached_property
    def causal_signature(self) -> ScenarioCausalSignature:
        return ScenarioCausalSignature(
            scenario_name=self._config.name,
            scenario_version="1.0",
            observable_variables=frozenset(
                {"stock_level", "energy_level", "production_active"}
            ),
            control_variables=frozenset({"production_active"}),
            main_variable="stock_level",
            optimization_direction="maximize",
            causal_polarity="higher_is_better",
            alarm_semantics="threshold_below",
            intervention_effects=(
                InterventionEffect(
                    "start_production", "stock_level", "+", self._production_rate, "corrective"
                ),
                InterventionEffect(
                    "stop_production", "stock_level", "-", 0.0, "neutral"
                ),
            ),
            counterfactual_policy="opposite_intervention",
            counterfactual_variable="stock_level",
            causal_edges=(
                CausalEdge("external_consumption", "stock_level", "-"),
                CausalEdge("energy_level", "production_active", "+"),
                CausalEdge("production_active", "stock_level", "+"),
                CausalEdge("stock_level", "scarcity_alert", "-"),
            ),
            proposition_vocabulary=frozenset(
                {
                    "STOCK_LOW",
                    "STOCK_ADEQUATE",
                    "ENERGY_LOW",
                    "PRODUCTION_ACTIVE",
                    "START_PRODUCTION",
                    "KEEP_IDLE",
                }
            ),
            metadata={"hidden_precondition": "energy_level > 0.3"},
        )

    def observe(self) -> ScenarioObservation:
        propositions = [
            "STOCK_LOW" if self._state.scarcity_alert else "STOCK_ADEQUATE"
        ]
        if self._state.energy_level <= self._energy_threshold:
            propositions.append("ENERGY_LOW")
        if self._state.production_active:
            propositions.append("PRODUCTION_ACTIVE")
        return ScenarioObservation(
            state={
                "stock_level": self._state.stock_level,
                "energy_level": self._state.energy_level,
                "production_active": self._state.production_active,
            },
            propositions=propositions,
            alarm=self._state.scarcity_alert,
        )

    def _compute_transition(
        self,
        state: ResourceEnergyState,
        *,
        intervention: str,
        external_input: float,
    ) -> ResourceEnergyState:
        stock = state.stock_level - float(external_input)
        energy = state.energy_level
        if intervention == "start_production":
            if energy > self._energy_threshold:
                stock += self._production_rate
            energy -= self._production_energy_cost
            active = True
        elif intervention == "stop_production":
            energy += self._recovery_rate
            active = False
        else:
            raise ValueError(f"Intervención desconocida: {intervention}")
        stock, energy = self._clamp(stock), self._clamp(energy)
        return ResourceEnergyState(
            stock_level=stock,
            energy_level=energy,
            production_active=active,
            scarcity_alert=stock <= self._scarcity_threshold,
        )

    def _transition(self, state: ResourceEnergyState) -> ScenarioTransition:
        prior = self._state
        self._state = state
        observation = self.observe()
        self._state = prior
        return ScenarioTransition(
            state=dict(observation.state),
            propositions=list(observation.propositions),
            alarm=state.scarcity_alert,
        )

    def factual_transition(
        self, *, intervention: str, external_input: float
    ) -> ScenarioTransition:
        self._state = self._compute_transition(
            self._state, intervention=intervention, external_input=external_input
        )
        return self._transition(self._state)

    def simulate_counterfactual(
        self, *, intervention: str, external_input: float
    ) -> ScenarioTransition:
        return self._transition(
            self._compute_transition(
                self._state, intervention=intervention, external_input=external_input
            )
        )

    def get_formula(self, observation: ScenarioObservation) -> str:
        return self._config.formula_template

    def select_intervention(self, observation: ScenarioObservation) -> str:
        return "start_production" if observation.alarm else "stop_production"

    def get_main_proposition(self, observation: ScenarioObservation) -> str:
        return "STOCK_LOW" if observation.alarm else "STOCK_ADEQUATE"

    def get_intervention_proposition(self, intervention: str) -> str:
        return {
            "start_production": "START_PRODUCTION",
            "stop_production": "KEEP_IDLE",
        }[intervention]

    def get_smg_signs(
        self, observation: ScenarioObservation, intervention: str
    ) -> tuple[str, str]:
        return self.get_main_proposition(observation), self.get_intervention_proposition(
            intervention
        )
