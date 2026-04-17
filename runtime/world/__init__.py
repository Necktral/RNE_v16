"""Subsistema de mundo del runtime."""
"""Modelos de mundo del runtime RNFE."""

from .cgwm_min import CGWMMin, WorldState
from .min_cognitive_episode import MinimalCognitiveEpisodeRunner
from .scenario import (
    CognitiveScenario,
    ScenarioConfig,
    ScenarioObservation,
    ScenarioTransition,
)
from .thermal_scenario import ThermalScenario, create_thermal_scenario
from .resource_scenario import ResourceScenario, create_resource_scenario
from .registry import (
    SCENARIO_REGISTRY,
    DEFAULT_SCENARIO,
    get_scenario,
    list_scenarios,
    register_scenario,
)
from .scenario_runner import ScenarioEpisodeRunner

__all__ = [
    # Legacy
    "CGWMMin",
    "WorldState",
    "MinimalCognitiveEpisodeRunner",
    # Scenario interface
    "CognitiveScenario",
    "ScenarioConfig",
    "ScenarioObservation",
    "ScenarioTransition",
    # Scenarios
    "ThermalScenario",
    "ResourceScenario",
    "create_thermal_scenario",
    "create_resource_scenario",
    # Registry
    "SCENARIO_REGISTRY",
    "DEFAULT_SCENARIO",
    "get_scenario",
    "list_scenarios",
    "register_scenario",
    # Runner
    "ScenarioEpisodeRunner",
]
