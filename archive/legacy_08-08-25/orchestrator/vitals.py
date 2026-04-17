# aeon/orchestrator/vitals.py
# ----------------------------------------------------------------------
import asyncio, logging

log = logging.getLogger("AEON.Vitals")

try:
    import pynvml

    pynvml.nvmlInit()
    GPU_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
except Exception:
    pynvml = None
    GPU_HANDLE = None
    log.warning("pynvml no disponible: se simulará la telemetría GPU.")


async def monitor_vitals(metrics, shutdown_event):
    while not shutdown_event.is_set():
        if pynvml:
            info = pynvml.nvmlDeviceGetMemoryInfo(GPU_HANDLE)
            metrics.vram_usage_gb = info.used / (1024**3)
            metrics.temperature = pynvml.nvmlDeviceGetTemperature(
                GPU_HANDLE, pynvml.NVML_TEMPERATURE_GPU
            )
        await asyncio.sleep(2)
    if pynvml:
        pynvml.nvmlShutdown()
