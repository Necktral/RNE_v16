"""Registro de escenarios cognitivos disponibles."""

from __future__ import annotations

from typing import Dict, Type

from .scenario import CognitiveScenario, ScenarioConfig
from .thermal_scenario import ThermalScenario
from .resource_scenario import ResourceScenario


# Registro de escenarios disponibles
SCENARIO_REGISTRY: Dict[str, Type[CognitiveScenario]] = {
    "thermal_homeostasis": ThermalScenario,
    "resource_management": ResourceScenario,
}

# Escenario por defecto (baseline)
DEFAULT_SCENARIO = "thermal_homeostasis"


def get_scenario(name: str, **kwargs) -> CognitiveScenario:
    """Obtiene instancia de escenario por nombre.

    Args:
        name: Nombre del escenario registrado.
        **kwargs: Parámetros de configuración del escenario.

    Returns:
        Instancia del escenario.

    Raises:
        ValueError: Si el escenario no existe.
    """
    if name not in SCENARIO_REGISTRY:
        available = ", ".join(SCENARIO_REGISTRY.keys())
        raise ValueError(f"Escenario '{name}' no encontrado. Disponibles: {available}")

    scenario_class = SCENARIO_REGISTRY[name]
    return scenario_class(**kwargs)


def list_scenarios() -> Dict[str, ScenarioConfig]:
    """Lista configuraciones de todos los escenarios registrados.

    Returns:
        Dict con nombre -> configuración de cada escenario.
    """
    configs = {}
    for name, scenario_class in SCENARIO_REGISTRY.items():
        instance = scenario_class()
        configs[name] = instance.config
    return configs


def register_scenario(name: str, scenario_class: Type[CognitiveScenario]) -> None:
    """Registra un nuevo escenario.

    Args:
        name: Nombre único del escenario.
        scenario_class: Clase del escenario (debe heredar de CognitiveScenario).
    """
    SCENARIO_REGISTRY[name] = scenario_class
