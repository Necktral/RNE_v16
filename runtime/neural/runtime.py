"""Runtime N4 acotado, determinista y completamente fail-open."""

from __future__ import annotations

from typing import Protocol

from .contracts import (
    BackendOutput,
    NeuralInferenceRequest,
    NeuralInferenceResult,
    NeuralMode,
    NeuralModelManifest,
)


class NeuralBackend(Protocol):
    def infer(self, request: NeuralInferenceRequest) -> BackendOutput: ...


class NeuralRuntime:
    def __init__(
        self,
        *,
        backend: NeuralBackend,
        manifest: NeuralModelManifest,
        mode: NeuralMode = NeuralMode.EXPERIMENTAL,
        pressure_limit: float = 0.90,
    ) -> None:
        self.backend = backend
        self.manifest = manifest
        self.mode = mode
        self.pressure_limit = float(pressure_limit)

    def infer(self, request: NeuralInferenceRequest) -> NeuralInferenceResult:
        if self.mode is NeuralMode.OFF:
            return self._fallback(request, "neural_mode_off")
        if not request.resources.budget_available:
            return self._fallback(request, "msrc_budget_unavailable")
        if request.resources.pressure >= self.pressure_limit:
            return self._fallback(request, "resource_pressure")
        try:
            output = self.backend.infer(request)
            return NeuralInferenceResult(
                inference_id=request.inference_id,
                mode=self.mode,
                candidate_output=output.candidate_output,
                confidence=output.confidence,
                uncertainty=output.uncertainty,
                model_manifest_sha256=self.manifest.manifest_sha256,
                fallback_used=False,
                fallback_reason=None,
                cost=output.cost,
            )
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            return self._fallback(
                request, f"backend_failure:{type(exc).__name__}"
            )

    def _fallback(
        self, request: NeuralInferenceRequest, reason: str
    ) -> NeuralInferenceResult:
        return NeuralInferenceResult(
            inference_id=request.inference_id,
            mode=self.mode,
            candidate_output={"schema": "n4-ranking.v1", "rankings": []},
            confidence=0.0,
            uncertainty=1.0,
            model_manifest_sha256=self.manifest.manifest_sha256,
            fallback_used=True,
            fallback_reason=reason,
            cost={},
        )
