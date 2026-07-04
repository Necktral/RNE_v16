# aeon/state.py
from dataclasses import dataclass, field
from collections import deque
from .utils.metrics import SelfAwarenessMetrics

@dataclass
class AEONState:
    """Contenedor centralizado para todo el estado mutable de AEON."""
    metrics: SelfAwarenessMetrics = field(default_factory=SelfAwarenessMetrics)
    loss_history: deque[float] = field(default_factory=lambda: deque(maxlen=100))
    step: int = 0
    crisis: bool = False  # Señal global para parada de emergencia