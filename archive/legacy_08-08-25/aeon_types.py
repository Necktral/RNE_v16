# src/aeon_types.py (renombrado para evitar conflicto con el módulo estándar types)
from dataclasses import dataclass, field, asdict, astuple
from typing import (
    Dict, List, Tuple, Any, Optional, Union, Callable, ClassVar, 
    TypeVar, Generic, NewType, Protocol, runtime_checkable
)
from enum import Enum, auto
try:
    from enum import StrEnum
except ImportError:
    class StrEnum(str, Enum):
        pass
import time
import numpy as np
from datetime import datetime
from collections import defaultdict
from math import sqrt, log

# Tipos personalizados para mayor claridad
ComponentID = NewType('ComponentID', str)

class SystemMode(Enum):
    HIGH_PERFORMANCE = 1
    NORMAL = 2
    CONSERVATIVE = 3
    DEFENSIVE = 4
    EMERGENCY = 5
    RECOVERY = 6
    CRITICAL_FAILURE = 7

class ComponentStatus(Enum):
    OPTIMAL = "optimal"
    STABLE = "stable"
    DEGRADED = "degraded"
    UNSTABLE = "unstable"
    FAILING = "failing"
    RECOVERING = "recovering"
    OFFLINE = "offline"

@dataclass(frozen=True, slots=True)
class Vector3D:
    x: float
    y: float
    z: float
    def magnitude(self) -> float:
        return sqrt(self.x**2 + self.y**2 + self.z**2)
    def normalized(self) -> 'Vector3D':
        mag = self.magnitude()
        return Vector3D(self.x/mag, self.y/mag, self.z/mag)
    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=np.float32)

@dataclass(frozen=True, slots=True)
class ThermodynamicState:
    timestamp: float = field(default_factory=time.time)
    temperature_K: float = 300.0
    power_W: float = 0.0
    entropy_rate: float = 0.0
    heat_dissipation: float = 0.0
    thermal_gradient: Vector3D = field(default_factory=lambda: Vector3D(0.0, 0.0, 0.0))
    cooling_efficiency: float = 1.0
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'temperature_K': self.temperature_K,
            'power_W': self.power_W,
            'entropy_rate': self.entropy_rate,
            'cooling_efficiency': self.cooling_efficiency
        }

@dataclass(frozen=True, slots=True)
class CognitiveState:
    attention_focus: float = 1.0
    memory_load: float = 0.0
    prediction_accuracy: float = 1.0
    uncertainty: float = 0.0
    learning_rate: float = 0.001
    complexity_level: int = 3
    @property
    def stability_index(self) -> float:
        return (
            self.prediction_accuracy * 0.4 + 
            (1 - self.uncertainty) * 0.3 +
            (1 - self.memory_load) * 0.3
        )

@dataclass(frozen=True, slots=True)
class HealthStatus:
    memory_load: float = 0.0
    cpu_utilization: float = 0.0
    gpu_utilization: float = 0.0
    power_consumption: float = 0.0
    temperature: float = 0.0
    vram_usage: float = 0.0
    thermal_state: ThermodynamicState = field(default_factory=ThermodynamicState)
    entropy_rate: float = 0.0
    free_energy: float = 0.0
    cognitive_state: CognitiveState = field(default_factory=CognitiveState)
    stability_index: float = 1.0
    operational_capacity: float = 1.0
    temp_forecast: float = 0.0
    failure_probability: float = 0.0
    time_to_recovery: float = 0.0
    system_mode: SystemMode = SystemMode.NORMAL
    critical_components: Dict[ComponentID, ComponentStatus] = field(default_factory=dict)
    def to_dict(self) -> Dict[str, Any]:
        return {
            **{k: v for k, v in asdict(self).items() if not k.startswith('_')},
            "thermal_state": self.thermal_state.to_dict(),
            "cognitive_state": asdict(self.cognitive_state),
            "system_mode": self.system_mode.name,
            "critical_components": {k: v.name for k, v in self.critical_components.items()}
        }
    def criticality_score(self) -> float:
        physical = max(
            self.memory_load,
            self.cpu_utilization,
            self.gpu_utilization,
            self.temperature
        ) * 0.4
        thermodynamic = max(
            self.thermal_state.temperature_K / 373.15,
            self.entropy_rate,
            1 - self.thermal_state.cooling_efficiency
        ) * 0.3
        cognitive = max(
            1 - self.cognitive_state.stability_index,
            self.cognitive_state.uncertainty,
            self.failure_probability
        ) * 0.3
        return min(1.0, physical + thermodynamic + cognitive)

@dataclass(frozen=True, slots=True)
class ActionLog:
    timestamp: float
    action_type: str
    parameters: Dict[str, Any]
    energy_cost: float
    cognitive_cost: float
    outcome: Optional[str] = None
    success: bool = True
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'action_type': self.action_type,
            'parameters': self.parameters,
            'energy_cost': self.energy_cost,
            'cognitive_cost': self.cognitive_cost,
            'outcome': self.outcome,
            'success': self.success
        }

@dataclass(frozen=True, slots=True)
class SystemSnapshot:
    timestamp: float = field(default_factory=time.time)
    health_status: HealthStatus = field(default_factory=HealthStatus)
    active_processes: List[str] = field(default_factory=list)
    resource_allocation: Dict[str, float] = field(default_factory=dict)
    action_log: List[ActionLog] = field(default_factory=list)
    anomaly_detections: List[Dict] = field(default_factory=list)
    def compress(self) -> Dict[str, Any]:
        return {
            "ts": self.timestamp,
            "health": {
                "temp": self.health_status.temperature,
                "power": self.health_status.power_consumption,
                "mem": self.health_status.memory_load,
                "vram": self.health_status.vram_usage,
                "mode": self.health_status.system_mode.name
            },
            "proc": self.active_processes,
            "res": self.resource_allocation,
            "anomalies": len(self.anomaly_detections)
        }

@dataclass(frozen=True, slots=True)
class OntologySpaces:
    observable_space: Vector3D
    latent_space: Vector3D
    action_space: Vector3D
    policy_space: Vector3D
    temporal_depth: int = 5
    def total_dimensions(self) -> int:
        return int(
            self.observable_space.magnitude() +
            self.latent_space.magnitude() +
            self.action_space.magnitude() +
            self.policy_space.magnitude()
        )

@dataclass(frozen=True, slots=True)
class HomeostasisThresholds:
    temperature: float = 0.85
    power: float = 0.90
    memory: float = 0.88
    entropy: float = 0.80
    uncertainty: float = 0.75
    learning_rate: float = 0.001
    def adaptive_adjust(self, stress_level: float) -> 'HomeostasisThresholds':
        adjustment = 0.01 * (stress_level - 0.5)
        return HomeostasisThresholds(
            temperature=min(0.95, max(0.7, self.temperature + adjustment)),
            power=min(0.98, max(0.8, self.power + adjustment)),
            memory=min(0.95, max(0.75, self.memory + adjustment * 1.2)),
            entropy=min(0.98, max(0.7, self.entropy + adjustment * 0.8)),
            uncertainty=min(0.95, max(0.6, self.uncertainty + adjustment * 0.5)),
            learning_rate=self.learning_rate
        )
    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

@dataclass(frozen=True, slots=True)
class NeuralSignature:
    layer_signatures: Dict[str, float]
    connectivity_matrix_hash: str
    activation_profile: Tuple[float, float, float]
    temporal_coherence: float
    def verify(self, reference: 'NeuralSignature') -> float:
        layer_match = sum(
            min(a/b, b/a) for a, b in zip(
                self.layer_signatures.values(),
                reference.layer_signatures.values()
            )
        ) / len(self.layer_signatures)
        matrix_match = 1.0 if self.connectivity_matrix_hash == reference.connectivity_matrix_hash else 0.3
        activation_match = 1 - abs(self.activation_profile[0] - reference.activation_profile[0])
        return (layer_match * 0.4 + matrix_match * 0.3 + activation_match * 0.3) * self.temporal_coherence
