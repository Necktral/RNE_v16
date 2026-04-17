from dataclasses import dataclass, field
from typing import Dict

# ────────────────────────────  CONSTANTES FÍSICAS  ────────────────────────────
MAX_VRAM_GB = 8                  # Límite de VRAM (RTX 2070 Super)
THERMAL_THRESHOLD = 85           # Umbral térmico (Max-Q)
MAX_ENTROPY = 1.0                # Entropía máxima computacional
CRITICAL_TEMP = 100              # Temperatura crítica de hardware

@dataclass
class SelfAwarenessMetrics:
    vram_usage_gb: float = 0.0
    dissipated_power: float = 0.0
    entropy: float = 0.0
    temperature: float = 0.0
    stability: float = 0.0
    loss: float = 0.0
    grad_norm: float = 0.0

    def as_dict(self) -> Dict[str, float]:
        d = {
            "Mem": self.vram_usage_gb / MAX_VRAM_GB,
            "Energy": self.dissipated_power / THERMAL_THRESHOLD,
            "Entropy": self.entropy / MAX_ENTROPY,
            "Temp": self.temperature / CRITICAL_TEMP,
            "Stability": self.stability,
            "loss": self.loss,
            "grad_norm": self.grad_norm,
            "entropy": self.entropy,
            "stability": self.stability
        }
        return d
