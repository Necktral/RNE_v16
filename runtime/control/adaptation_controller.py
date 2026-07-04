"""Controlador de adaptación desacoplado del orquestador."""

from __future__ import annotations

from typing import Any, Dict, Optional

from runtime.core.metrics import CRITICAL_TEMP, MAX_VRAM_GB, SelfAwarenessMetrics


class AdaptationController:
    def build_context(
        self,
        *,
        metrics: SelfAwarenessMetrics,
        history: list[float],
        model: Any,
        optimizer: Any,
        device: Any,
        event_topic: Optional[str] = None,
        event_severity: Optional[str] = None,
    ) -> Dict[str, Any]:
        context: Dict[str, Any] = {
            "delta_epist": history[-1] - history[-2] if len(history) > 1 else 0.0,
            "mutual_info": metrics.stability,
            "thermal_risk": metrics.temperature / CRITICAL_TEMP,
            "vram_usage": metrics.vram_usage_gb / MAX_VRAM_GB,
            "pruning_intensity": 0.0,
            "neurogenesis_impact": {},
            "model": model,
            "optimizer": optimizer,
            "device": device,
            "vram_usage_gb": metrics.vram_usage_gb,
            "MAX_VRAM_GB": MAX_VRAM_GB,
            "history": list(history),
        }
        if event_topic:
            context["event_topic"] = event_topic
        if event_severity:
            context["event_severity"] = event_severity
        return context

    async def apply_adaptations(
        self,
        *,
        loop,
        executor,
        auto_mutator: Any,
        trainer: Any,
        optimizer: Any,
        context: Dict[str, Any],
        logger,
    ) -> Any:
        if auto_mutator is None:
            return optimizer

        payloads = await loop.run_in_executor(executor, lambda: auto_mutator.step(context))
        if not payloads:
            return optimizer

        logger.info(f"Aplicando {len(payloads)} adaptación(es)...")
        for payload in payloads:
            if hasattr(trainer, "apply_adaptation"):
                await loop.run_in_executor(executor, lambda: trainer.apply_adaptation(payload))

        if hasattr(trainer, "optimizer") and trainer.optimizer is not None:
            return trainer.optimizer
        return optimizer
