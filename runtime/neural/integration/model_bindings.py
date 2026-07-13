"""Resolución lazy y fail-closed de manifiestos para órganos entrenados."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping

from runtime.neural.contracts import NeuralModelManifest, NeuralMode
from runtime.neural.registry import LazyBackendRegistry
from runtime.neural.technology_backends import (
    HNET_BACKEND_ID,
    MAMBA2_BACKEND_ID,
    HNetBoundaryTorchBackend,
    Mamba2TemporalTorchBackend,
)
from runtime.neural.organs.n1_router import CompactMLPRouterBackend
from runtime.neural.organs.n4_causal import CausalMessagePassingBackend


MODEL_MANIFEST_ENV: Mapping[str, str] = {
    "N1": "RNFE_NEURAL_N1_MANIFEST",
    "N3": "RNFE_NEURAL_N3_MANIFEST",
    "N4": "RNFE_NEURAL_N4_MANIFEST",
    "N5": "RNFE_NEURAL_N5_MANIFEST",
}


@dataclass(frozen=True, slots=True)
class ModelBinding:
    manifest: NeuralModelManifest
    manifest_path: str
    authority_ceiling: NeuralMode = NeuralMode.SHADOW


class ModelBindingResolver:
    """No abre manifiestos en OFF ni fuera del artifact root declarado."""

    _BACKENDS: Mapping[str, tuple[str, Callable[[], Any]]] = {
        "N1": ("rnfe-compact-mlp-router-v1", CompactMLPRouterBackend),
        "N3": (MAMBA2_BACKEND_ID, Mamba2TemporalTorchBackend),
        "N4": ("rnfe-trained-causal-graph-v1", CausalMessagePassingBackend),
        "N5": (HNET_BACKEND_ID, HNetBoundaryTorchBackend),
    }

    def __init__(self, *, registry: LazyBackendRegistry) -> None:
        self.registry = registry
        self._cache: dict[tuple[str, str], ModelBinding] = {}
        self._registered: set[str] = set()

    def resolve(
        self,
        *,
        organ: str,
        capability: str,
        mode: NeuralMode,
    ) -> ModelBinding | None:
        if mode is NeuralMode.OFF:
            return None
        env_name = MODEL_MANIFEST_ENV.get(organ)
        if env_name is None:
            return None
        raw_path = os.environ.get(env_name, "").strip()
        if not raw_path:
            return None
        relative = PurePosixPath(raw_path)
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValueError(f"model_manifest_path_unsafe:{organ}")
        key = (organ, raw_path)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        path = (self.registry.artifact_root / raw_path).resolve()
        try:
            path.relative_to(self.registry.artifact_root)
        except ValueError as exc:
            raise ValueError(f"model_manifest_path_escapes_root:{organ}") from exc
        if not path.is_file():
            raise FileNotFoundError(f"model_manifest_missing:{organ}:{raw_path}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"model_manifest_unreadable:{organ}") from exc
        manifest = NeuralModelManifest.from_dict(payload)
        expected_backend, factory = self._BACKENDS[organ]
        if (
            manifest.organ != organ
            or manifest.capability != capability
            or manifest.backend != expected_backend
        ):
            raise ValueError(f"model_manifest_contract_mismatch:{organ}")
        if expected_backend not in self._registered:
            self.registry.register(expected_backend, factory)
            self._registered.add(expected_backend)
        binding = ModelBinding(manifest=manifest, manifest_path=raw_path)
        self._cache[key] = binding
        return binding
