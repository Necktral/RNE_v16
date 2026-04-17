# aeon/systems/homeostasis/controller.py
import logging
from typing import Dict, Any

from ...state import AEONState
from ...protocols import HomeoProto
from ...vitals.sensors import SensorProto
from ...aeon_types import (
    HealthStatus, 
    SystemMode, 
    ThermodynamicState, 
    CognitiveState,
    HomeostasisThresholds
)

log = logging.getLogger("AEON.HomeoController")

class HomeoController(HomeoProto):
    """
    Interpreta los datos sensoriales para evaluar la salud general de AEON.
    Utiliza los tipos avanzados de aeon_types para generar un informe holístico.
    """
    def __init__(self, sensor: SensorProto, config: Dict[str, Any]):
        self.sensor = sensor
        # Usamos la dataclass de umbrales para una configuración limpia
        self.thresholds = HomeostasisThresholds(**config.get('thresholds', {}))
        log.info("Controlador Homeostático inicializado con umbrales.")

    def health_status(self, state: AEONState) -> HealthStatus:
        """
        Construye el informe de salud completo (HealthStatus) a partir de los
        datos sensoriales y el estado cognitivo actual.
        """
        # 1. Obtener datos físicos del sensor
        readout = self.sensor.read()
        
        # 2. Construir sub-estados
        thermo_state = ThermodynamicState(
            temperature_K=readout.temp_c + 273.15,
            power_W=readout.power_watts
            # Otros campos termodinámicos podrían ser calculados por modelos más complejos
        )
        
        cognitive_state = CognitiveState(
            learning_rate=state.optimizer.param_groups[0]['lr'] if state.optimizer else 0.0,
            uncertainty=state.metrics.loss, # Usamos la pérdida como un proxy de la incertidumbre
            memory_load=readout.vram_used_gb / readout.vram_total_gb if readout.vram_total_gb > 0 else 0
        )
        
        # 3. Instanciar el informe de salud principal
        health = HealthStatus(
            temperature=readout.temp_c,
            power_consumption=readout.power_watts,
            vram_usage=readout.vram_used_gb,
            thermal_state=thermo_state,
            cognitive_state=cognitive_state,
            stability_index=cognitive_state.stability_index
        )
        
        # 4. Calcular criticidad y determinar el modo del sistema
        criticality = health.criticality_score()
        system_mode = self._determine_system_mode(criticality)
        
        # 5. Devolver el informe final y completo, reemplazando el modo por defecto
        # Usamos replace ya que HealthStatus es un objeto inmutable (frozen=True)
        import dataclasses
        final_health = dataclasses.replace(health, system_mode=system_mode)
        
        return final_health

    def _determine_system_mode(self, criticality: float) -> SystemMode:
        """
        Traduce un puntaje de criticidad escalar a un modo operativo discreto.
        """
        if criticality > self.thresholds.power: # Umbral más alto y peligroso
            return SystemMode.EMERGENCY
        elif criticality > self.thresholds.temperature:
            return SystemMode.DEFENSIVE
        elif criticality > self.thresholds.memory:
            return SystemMode.CONSERVATIVE
        else:
            return SystemMode.NORMAL