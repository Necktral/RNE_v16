# [LEGACY cuarentena] HealthStatus unificado al canónico (contracts/types/aeon_types,
# vía el shim src.aeon_types). Antes definía un tipo mínimo propio (temp/vram/entropy).
from src.aeon_types import HealthStatus


class HomeoController:
    def __init__(self, metrics_provider):
        self.metrics_provider = metrics_provider

    def health_status(self) -> HealthStatus:
        metrics = self.metrics_provider()
        return HealthStatus(
            temperature=metrics.get('temperature', 0.0),
            vram_usage=metrics.get('vram_usage_gb', 0.0),
            entropy_rate=metrics.get('entropy', 0.0),
        )
