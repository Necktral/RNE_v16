"""Recolección de métricas operativas del runtime."""

from __future__ import annotations

import tracemalloc

from runtime.core.metrics import CRITICAL_TEMP, MAX_VRAM_GB, SelfAwarenessMetrics


class TelemetryCollector:
    """Recolector de métricas desacoplado del orquestador."""

    def update_metrics(
        self,
        *,
        metrics: SelfAwarenessMetrics,
        history: list[float],
        pynvml_module=None,
        gpu_handle=None,
    ) -> None:
        if pynvml_module:
            vmem = pynvml_module.nvmlDeviceGetMemoryInfo(gpu_handle)
            metrics.vram_usage_gb = vmem.used / (1024**3)
            metrics.temperature = pynvml_module.nvmlDeviceGetTemperature(
                gpu_handle, pynvml_module.NVML_TEMPERATURE_GPU
            )
            metrics.dissipated_power = (
                pynvml_module.nvmlDeviceGetPowerUsage(gpu_handle) / 1000
            )
        else:
            cur, _ = tracemalloc.get_traced_memory()
            metrics.vram_usage_gb = cur / (1024**3)
            metrics.temperature = 40.0

        try:
            import psutil

            cpu = psutil.cpu_percent() / 100
            mem = psutil.virtual_memory().percent / 100
            metrics.entropy = 0.7 * cpu + 0.3 * mem
        except ImportError:
            metrics.entropy = 0.5

        if len(history) > 1:
            metrics.stability = abs(history[-1] - history[-2])
        else:
            metrics.stability = 0.0

    def vitals_from_metrics(self, metrics: SelfAwarenessMetrics) -> dict[str, float]:
        """Payload normalizado para reglas de crisis."""
        payload = metrics.as_dict()
        payload["VRAM_GB"] = metrics.vram_usage_gb
        payload["TEMP_RAW"] = metrics.temperature
        payload["TEMP_LIMIT"] = CRITICAL_TEMP
        payload["VRAM_LIMIT_GB"] = MAX_VRAM_GB
        return payload
