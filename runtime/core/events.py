"""Tipo de evento compartido (extraído de infrastructure.py en la reorg 2026-07-01).

`Event` es el único símbolo de la antigua `infrastructure.py` usado por código
vivo (`runtime/control/crisis_router.py` y el runner moderno). Vive aquí, en
stdlib puro, para que el camino vivo no arrastre la cadena legacy
(pydantic/yaml) al importarlo. El resto de `infrastructure.py` (EventBus async,
WorkerPool, ConfigLoader) es legacy y está en `runtime/legacy/infrastructure.py`.
"""

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class Event:
    topic: str
    payload: Any = None
    severity: str = "INFO"
    timestamp: float = field(default_factory=time.time)
    source: str = "system"
