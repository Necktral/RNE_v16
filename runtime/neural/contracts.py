"""Contratos canónicos del runtime neural N4."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Any, Mapping


def canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("neural_value_must_be_finite")
    return value


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


class NeuralMode(str, Enum):
    OFF = "off"
    EXPERIMENTAL = "experimental"
    SHADOW = "shadow"


@dataclass(frozen=True)
class ResourceSnapshot:
    cpu_pressure: float = 0.0
    memory_pressure: float = 0.0
    thermal_pressure: float = 0.0
    budget_available: bool = True

    def __post_init__(self) -> None:
        for name in ("cpu_pressure", "memory_pressure", "thermal_pressure"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name}_must_be_finite")
            object.__setattr__(self, name, min(1.0, max(0.0, value)))

    @property
    def pressure(self) -> float:
        return max(self.cpu_pressure, self.memory_pressure, self.thermal_pressure)


@dataclass(frozen=True)
class NeuralModelManifest:
    organ: str
    capability: str
    model_id: str
    version: str
    backend: str
    artifact_path: str
    artifact_sha256: str
    trained: bool
    training_provenance: Mapping[str, Any] = field(default_factory=dict)
    metrics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.organ.upper() != "N4":
            raise ValueError("manifest_requires_n4")
        path = PurePosixPath(self.artifact_path)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("artifact_path_must_be_relative")
        if len(self.artifact_sha256) != 64:
            raise ValueError("artifact_sha256_invalid")
        int(self.artifact_sha256, 16)
        object.__setattr__(self, "organ", "N4")
        object.__setattr__(self, "training_provenance", _freeze(self.training_provenance))
        object.__setattr__(self, "metrics", _freeze(self.metrics))
        if self.trained and not self.training_provenance:
            raise ValueError("trained_manifest_requires_provenance")

    def to_dict(self) -> dict[str, Any]:
        return {
            "organ": self.organ,
            "capability": self.capability,
            "model_id": self.model_id,
            "version": self.version,
            "backend": self.backend,
            "artifact_path": self.artifact_path,
            "artifact_sha256": self.artifact_sha256,
            "trained": self.trained,
            "training_provenance": _plain(self.training_provenance),
            "metrics": _plain(self.metrics),
        }

    @property
    def manifest_sha256(self) -> str:
        return canonical_sha256(self.to_dict())


@dataclass(frozen=True)
class NeuralInferenceRequest:
    inference_id: str
    run_id: str
    payload: Mapping[str, Any]
    logical_time: int
    resources: ResourceSnapshot = field(default_factory=ResourceSnapshot)

    def __post_init__(self) -> None:
        if not self.inference_id or not self.run_id:
            raise ValueError("inference_and_run_id_required")
        object.__setattr__(self, "payload", _freeze(self.payload))


@dataclass(frozen=True)
class BackendOutput:
    candidate_output: Mapping[str, Any]
    confidence: float
    uncertainty: float
    cost: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_output", _freeze(self.candidate_output))
        object.__setattr__(self, "cost", _freeze(self.cost))
        for name in ("confidence", "uncertainty"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name}_must_be_finite")
            object.__setattr__(self, name, min(1.0, max(0.0, value)))


@dataclass(frozen=True)
class NeuralInferenceResult:
    inference_id: str
    mode: NeuralMode
    candidate_output: Mapping[str, Any]
    confidence: float
    uncertainty: float
    model_manifest_sha256: str | None
    fallback_used: bool
    fallback_reason: str | None
    cost: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_output", _freeze(self.candidate_output))
        object.__setattr__(self, "cost", _freeze(self.cost))

    def to_dict(self) -> dict[str, Any]:
        return {
            "inference_id": self.inference_id,
            "mode": self.mode.value,
            "candidate_output": _plain(self.candidate_output),
            "confidence": self.confidence,
            "uncertainty": self.uncertainty,
            "model_manifest_sha256": self.model_manifest_sha256,
            "fallback_used": self.fallback_used,
            "fallback_reason": self.fallback_reason,
            "cost": _plain(self.cost),
        }
