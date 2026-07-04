from dataclasses import dataclass, field
from typing import Any, Dict
import time


@dataclass
class ReasoningTraceStep:
    family: str
    status: str
    detail: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

