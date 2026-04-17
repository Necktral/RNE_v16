from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True, slots=True)
class HealthStatus:
    temp: float
    vram: float
    entropy: float
    extra: Any = None

class HomeoController:
    def __init__(self, metrics_provider):
        self.metrics_provider = metrics_provider

    def health_status(self) -> HealthStatus:
        metrics = self.metrics_provider()
        return HealthStatus(
            temp=metrics.get('temperature', 0.0),
            vram=metrics.get('vram_usage_gb', 0.0),
            entropy=metrics.get('entropy', 0.0),
            extra=metrics
        )
