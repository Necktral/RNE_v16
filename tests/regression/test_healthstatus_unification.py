"""Regresión: unificación de HealthStatus a un único tipo canónico.

Antes existían 3 definiciones incompatibles de `HealthStatus`
(contracts/types/aeon_types, core/homeo_controller, homeostasis/thermodynamic_governor).
El governor producía campos (energy/entropy/memory) que `shutdown_logic.evaluate_crisis`
NO leía (espera vram_usage/entropy_rate/power_consumption del canónico) → AttributeError
latente al cablear governor→shutdown.

Tras la unificación (A4) hay un único `HealthStatus` canónico
(`contracts/types/aeon_types`, expuesto también por el shim `src.aeon_types`).
"""

import dataclasses

from contracts.types.aeon_types import HealthStatus


def test_canonical_has_fields_shutdown_logic_reads():
    """El canónico expone los campos que lee shutdown_logic.evaluate_crisis."""
    fields = {f.name for f in dataclasses.fields(HealthStatus)}
    for name in ("vram_usage", "temperature", "entropy_rate", "power_consumption"):
        assert name in fields, f"falta campo canónico: {name}"
    # Campos que produce el governor (también deben existir en el canónico).
    for name in ("memory_load", "stability_index", "temp_forecast",
                 "thermal_gradient", "cognitive_load"):
        assert name in fields, f"falta campo del governor: {name}"


def test_governor_kwargs_construct_and_replace():
    """El canónico se construye con los kwargs de governor.assess_health y soporta replace."""
    health = HealthStatus(
        memory_load=0.3, power_consumption=0.5, entropy_rate=0.1,
        temperature=0.6, stability_index=1.0, cognitive_load=0.0,
        temp_forecast=0.62, thermal_gradient=0.01,
    )
    # homeo_controller.evaluate_state usa dataclasses.replace (canónico es frozen).
    updated = dataclasses.replace(health, vram_usage=0.4, thermal_gradient=0.02)
    assert updated.vram_usage == 0.4
    assert updated.entropy_rate == 0.1
    assert updated.power_consumption == 0.5


def test_core_homeo_controller_uses_canonical():
    """runtime/core/homeo_controller ya no define un HealthStatus propio."""
    from runtime.legacy.homeo_controller import HealthStatus as CoreHS, HomeoController

    assert CoreHS is HealthStatus
    controller = HomeoController(
        lambda: {"temperature": 0.5, "vram_usage_gb": 0.4, "entropy": 0.1}
    )
    status = controller.health_status()
    assert type(status) is HealthStatus
    assert status.temperature == 0.5
    assert status.vram_usage == 0.4
    assert status.entropy_rate == 0.1


def test_governor_to_shutdown_bridge_no_attribute_error():
    """governor.assess_health() → shutdown_logic.evaluate_crisis() sin AttributeError.

    Requiere psutil (cadena legacy del homeostasis); se salta con gracia si falta.
    """
    import pytest

    pytest.importorskip("psutil")

    from runtime.control.homeostasis.thermodynamic_governor import (
        ThermodynamicGovernor,
        HealthStatus as GovHS,
    )
    from runtime.control.homeostasis.shutdown_logic import PhasedShutdown

    # El governor usa ahora el HealthStatus canónico.
    assert GovHS is HealthStatus

    governor = ThermodynamicGovernor(config={})
    health = governor.assess_health()
    assert isinstance(health, HealthStatus)

    shutdown = PhasedShutdown(config={"shutdown_cooldown": 0})
    # No debe lanzar AttributeError leyendo vram_usage/entropy_rate/power_consumption.
    level = shutdown.evaluate_crisis(health)
    assert level is not None
