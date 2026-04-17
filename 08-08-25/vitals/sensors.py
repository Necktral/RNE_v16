# aeon/vitals/sensors.py
import logging
from dataclasses import dataclass
from typing import Protocol

# pynvml es el binding oficial de Python para la NVIDIA Management Library (NVML)
# Es la forma más directa y eficiente de consultar el estado de la GPU.
# Necesita ser instalado: pip install nvidia-ml-py
try:
    import pynvml as nvml
    _PYNML_AVAILABLE = True
except ImportError:
    _PYNML_AVAILABLE = False
    
log = logging.getLogger("AEON.Sensor")

@dataclass
class SensorReadout:
    """Estructura de datos estandarizada para las lecturas de un sensor."""
    temp_c: float = 0.0
    vram_used_gb: float = 0.0
    vram_total_gb: float = 0.0
    power_watts: float = 0.0

class SensorProto(Protocol):
    """Define el contrato que todos los sensores deben seguir."""
    def read(self) -> SensorReadout:
        """Lee los datos del sensor y devuelve un readout estandarizado."""
        ...
        
    def shutdown(self) -> None:
        """Libera cualquier recurso utilizado por el sensor."""
        ...

class NVIDIASensor(SensorProto):
    """
    Sensor para monitorizar una GPU NVIDIA utilizando pynvml.
    """
    def __init__(self, device_index: int = 0):
        if not _PYNML_AVAILABLE:
            raise RuntimeError("La biblioteca pynvml no está instalada. No se puede monitorizar la GPU.")
        
        try:
            nvml.nvmlInit()
            self.handle = nvml.nvmlDeviceGetHandleByIndex(device_index)
            log.info(f"Sensor NVIDIA inicializado para el dispositivo {device_index}: {nvml.nvmlDeviceGetName(self.handle)}")
        except nvml.NVMLError as error:
            log.error(f"No se pudo inicializar NVML: {error}")
            raise

    def read(self) -> SensorReadout:
        """
        Obtiene las métricas actuales de la GPU.
        Maneja errores para evitar que el sistema se caiga si falla una lectura.
        """
        try:
            temp = nvml.nvmlDeviceGetTemperature(self.handle, nvml.NVML_TEMPERATURE_GPU)
            power = nvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0  # Convertir de mW a W
            memory = nvml.nvmlDeviceGetMemoryInfo(self.handle)
            
            # Convertir bytes a Gigabytes
            bytes_to_gb = 1 / (1024**3)
            
            return SensorReadout(
                temp_c=float(temp),
                vram_used_gb=memory.used * bytes_to_gb,
                vram_total_gb=memory.total * bytes_to_gb,
                power_watts=float(power)
            )
        except nvml.NVMLError as error:
            log.error(f"Error al leer datos de la GPU: {error}. Devolviendo lectura en cero.")
            return SensorReadout() # Devuelve un estado seguro y nulo en caso de error

    def shutdown(self) -> None:
        """Cierra la conexión con NVML."""
        log.info("Apagando el sensor NVIDIA.")
        nvml.nvmlShutdown()

class MockSensor(SensorProto):
    """
    Sensor simulado para pruebas en entornos sin GPU.
    """
    def read(self) -> SensorReadout:
        # Simula métricas nominales
        return SensorReadout(temp_c=45.0, vram_used_gb=2.5, vram_total_gb=8.0, power_watts=120.0)
    
    def shutdown(self) -> None:
        pass # No hace nada