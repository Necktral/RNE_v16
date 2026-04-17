"""
protocols.py - Contratos de interacción para AEON
Autor: AEON-G
Fecha: Agosto 2025
"""
from typing import Protocol
from enum import Enum
from dataclasses import dataclass
from .state import AEONState

class SystemStatus(Enum):
    """Define el estado de salud general del sistema."""
    NOMINAL = "Nominal"     # Todo en orden.
    WARNING = "Warning"     # Parámetros acercándose a límites seguros.
    CRITICAL = "Critical"   # Límites de seguridad excedidos, riesgo inminente.

@dataclass
class HealthReport:
    """Estructura de datos para reportar el estado del hardware."""
    status: SystemStatus
    temp_c: float
    vram_gb: float
    vram_total_gb: float
    power_w: float
    message: str

class HomeoProto(Protocol):
    """Protocolo para cualquier controlador homeostático."""
    def health_status(self) -> HealthReport:
        """
        Evalúa y devuelve el estado de salud actual del sistema.
        """
        ...

class PlannerProto(Protocol):
    """
    Protocolo para cualquier sistema de planificación de AEON.
    Directiva Primaria de AEON:
    1. Supervivencia por encima de todo.
    2. Crecimiento si las condiciones son seguras.
    """
    def decide(self, state: AEONState) -> str:
        """
        Toma una decisión de acción basada en el estado completo de AEON.
        Devuelve un string que representa la acción a tomar.
        """
        ...