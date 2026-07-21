"""Sampler de recursos de host (CPU/RAM/swap/térmica) para el camino vivo.

Complementa a ``NvidiaVRAMSampler`` (GPU) con sensado de host, de modo que el
organismo perciba presión de cómputo real. Diseñado para hardware modesto:
usa ``psutil`` si está disponible y degrada a stdlib puro (``/proc``,
``os.getloadavg``) en su ausencia — nunca introduce una dependencia dura en el
camino vivo.

El sensado real solo se activa cuando ``RNFE_HOST_SENSING`` está encendido
(ver :func:`host_sensing_enabled`); apagado, el kernel no construye snapshot y
la conducta nominal queda byte-idéntica.
"""

from __future__ import annotations

import os
import time
from collections import deque
from typing import Any, Deque, Dict, Optional


_TRUE = {"1", "true", "yes", "on"}


def host_sensing_enabled() -> bool:
    """True si ``RNFE_HOST_SENSING`` habilita el sensado real de host/GPU."""
    return os.environ.get("RNFE_HOST_SENSING", "").strip().lower() in _TRUE


def _clamp01(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return min(max(numeric, 0.0), 1.0)


def _try_import_psutil():
    try:
        import psutil  # type: ignore

        return psutil
    except Exception:
        return None


class HostResourceSampler:
    """Lee presión de CPU/RAM/swap/térmica del host.

    ``sample()`` devuelve un dict con la misma disciplina de campos que
    ``NvidiaVRAMSampler`` (``available``/``source``/``sample_ts``) más las
    señales de host. Los campos de GPU se dejan en cero/False; la composición
    con el snapshot de VRAM la hace :func:`build_resource_snapshot`.
    """

    def __init__(self, *, ttl_seconds: float = 0.5, history_size: int = 8):
        self.ttl_seconds = max(0.0, float(ttl_seconds))
        self._psutil = _try_import_psutil()
        self._cpu_history: Deque[float] = deque(maxlen=history_size)
        self._cached: Optional[Dict[str, Any]] = None
        self._cached_ts: float = 0.0

    def sample(self) -> Dict[str, Any]:
        now = time.time()
        if (
            self._cached is not None
            and self.ttl_seconds > 0.0
            and (now - self._cached_ts) < self.ttl_seconds
        ):
            return dict(self._cached)

        cpu = self._cpu_pressure()
        memory, swap = self._memory_pressure()
        thermal = self._thermal_pressure()

        snapshot: Dict[str, Any] = {
            "available": True,
            "source": "psutil" if self._psutil is not None else "proc",
            "sample_ts": now,
            "cpu_pressure": round(_clamp01(cpu), 6),
            "memory_pressure": round(_clamp01(memory), 6),
            "swap_pressure": round(_clamp01(swap), 6),
            "thermal_pressure": round(_clamp01(thermal), 6),
            # Campos de GPU: neutros aquí; los completa build_resource_snapshot.
            "vram_pressure": 0.0,
            "vram_headroom": 0.0,
            "gpu_available": False,
            "gpu_load": 0.0,
        }
        self._cached = snapshot
        self._cached_ts = now
        return dict(snapshot)

    # -- CPU ---------------------------------------------------------------
    def _cpu_pressure(self) -> float:
        if self._psutil is not None:
            try:
                pct = self._psutil.cpu_percent(interval=None)
                return float(pct) / 100.0
            except Exception:
                pass
        # Fallback stdlib: load average normalizado por número de CPUs.
        try:
            load1, _, _ = os.getloadavg()
            cpu_count = os.cpu_count() or 1
            return float(load1) / float(cpu_count)
        except (OSError, AttributeError):
            return 0.0

    # -- Memoria -----------------------------------------------------------
    def _memory_pressure(self) -> tuple[float, float]:
        if self._psutil is not None:
            try:
                vm = self._psutil.virtual_memory()
                mem = float(vm.percent) / 100.0
                swap_pct = 0.0
                try:
                    swap_pct = float(self._psutil.swap_memory().percent) / 100.0
                except Exception:
                    swap_pct = 0.0
                return mem, swap_pct
            except Exception:
                pass
        return self._memory_from_proc()

    def _memory_from_proc(self) -> tuple[float, float]:
        try:
            info: Dict[str, float] = {}
            with open("/proc/meminfo", "r", encoding="ascii") as handle:
                for line in handle:
                    parts = line.split(":", 1)
                    if len(parts) != 2:
                        continue
                    key = parts[0].strip()
                    value = parts[1].strip().split()[0]
                    try:
                        info[key] = float(value)
                    except ValueError:
                        continue
        except OSError:
            return 0.0, 0.0

        total = info.get("MemTotal", 0.0)
        available = info.get("MemAvailable", info.get("MemFree", 0.0))
        mem_pressure = 0.0
        if total > 0.0:
            mem_pressure = 1.0 - (available / total)

        swap_total = info.get("SwapTotal", 0.0)
        swap_free = info.get("SwapFree", swap_total)
        swap_pressure = 0.0
        if swap_total > 0.0:
            swap_pressure = 1.0 - (swap_free / swap_total)
        return mem_pressure, swap_pressure

    # -- Térmica -----------------------------------------------------------
    def _thermal_pressure(self) -> float:
        # Térmica es best-effort; su ausencia no es un fallo. Preferimos el sensor
        # real del paquete CPU/GPU (x86_pkg_temp/coretemp) y usamos una escala de
        # laptop realista: 80C -> 0, 100C -> 1.0 (la presión sube recién cerca del
        # throttle). Evita que un solo zone o el calor normal de trabajo pinnee a 1.0.
        base = "/sys/class/thermal"
        try:
            zones = sorted(z for z in os.listdir(base) if z.startswith("thermal_zone"))
        except OSError:
            return 0.0
        temps: Dict[str, float] = {}
        for zone in zones:
            try:
                ztype = ""
                type_path = os.path.join(base, zone, "type")
                if os.path.exists(type_path):
                    with open(type_path, "r", encoding="ascii") as th:
                        ztype = th.read().strip().lower()
                with open(os.path.join(base, zone, "temp"), "r", encoding="ascii") as handle:
                    celsius = float(handle.read().strip()) / 1000.0
            except (OSError, ValueError):
                continue
            if 0.0 < celsius < 130.0:  # descartar lecturas absurdas / trip-points fijos raros
                temps[ztype or zone] = celsius
        if not temps:
            return 0.0
        preferred: float | None = None
        for key in ("x86_pkg_temp", "coretemp", "gpu"):
            for ztype, celsius in temps.items():
                if key in ztype:
                    preferred = celsius if preferred is None else max(preferred, celsius)
        celsius = preferred if preferred is not None else max(temps.values())
        return (celsius - 80.0) / 20.0


def _null_snapshot() -> Dict[str, Any]:
    return {
        "available": False,
        "source": "none",
        "sample_ts": time.time(),
        "cpu_pressure": 0.0,
        "memory_pressure": 0.0,
        "swap_pressure": 0.0,
        "thermal_pressure": 0.0,
        "vram_pressure": 0.0,
        "vram_headroom": 0.0,
        "gpu_available": False,
        "gpu_load": 0.0,
    }


def build_resource_snapshot(
    *,
    host_sampler: HostResourceSampler | None = None,
    vram_sampler: Any | None = None,
) -> Dict[str, Any]:
    """Compone un snapshot unificado host+GPU para vitals y contexto de razonamiento.

    - ``host_sampler``: fuente de cpu/mem/swap/thermal (host).
    - ``vram_sampler``: cualquier objeto con ``.sample()`` estilo
      ``NvidiaVRAMSampler`` (opcional; si su ``available`` es False se ignora).

    El resultado incluye ``hardware_pressure`` (máx de las presiones) y
    ``gpu_acceleration`` (headroom de VRAM cuando hay GPU con holgura), listos
    para inyectarse en el contexto de razonamiento.
    """
    snapshot = _null_snapshot()
    if host_sampler is not None:
        snapshot.update(host_sampler.sample())

    if vram_sampler is not None:
        try:
            vram = vram_sampler.sample()
        except Exception:
            vram = None
        if isinstance(vram, dict) and vram.get("available"):
            snapshot["gpu_available"] = True
            snapshot["vram_pressure"] = _clamp01(vram.get("vram_pressure"))
            snapshot["vram_headroom"] = _clamp01(vram.get("vram_headroom"))
            snapshot["gpu_load"] = _clamp01(vram.get("vram_pressure"))
            snapshot["vram_fragmentation_risk"] = _clamp01(
                vram.get("vram_fragmentation_risk")
            )
            snapshot["vram_opportunity_score"] = _clamp01(
                vram.get("vram_opportunity_score")
            )
            # Mantener el alias histórico usado por vitals y los nombres
            # canónicos del sampler/MSRC. La telemetría absoluta también es
            # necesaria para que N0 pueda presupuestar CUDA sin inventar datos.
            snapshot["gpu_opportunity_score"] = snapshot["vram_opportunity_score"]
            for key in ("used_gb", "total_gb", "temperature_c"):
                snapshot[key] = vram.get(key)
            snapshot["vram_used_gb"] = snapshot["used_gb"]
            snapshot["vram_total_gb"] = snapshot["total_gb"]
            snapshot["gpu_temperature_c"] = snapshot["temperature_c"]
            snapshot["gpu_source"] = vram.get("source")
            snapshot["gpu_sample_ts"] = vram.get("sample_ts")

    hardware_pressure = max(
        snapshot["cpu_pressure"],
        snapshot["memory_pressure"],
        snapshot["vram_pressure"],
        snapshot["thermal_pressure"],
    )
    snapshot["hardware_pressure"] = round(_clamp01(hardware_pressure), 6)

    gpu_accel = 0.0
    if snapshot["gpu_available"]:
        opportunity = snapshot.get("gpu_opportunity_score")
        if isinstance(opportunity, (int, float)):
            gpu_accel = _clamp01(opportunity)
        else:
            gpu_accel = _clamp01(snapshot["vram_headroom"])
        if snapshot["vram_pressure"] >= 0.88 or snapshot["thermal_pressure"] >= 0.85:
            gpu_accel = min(gpu_accel, 0.25)
    snapshot["gpu_acceleration"] = round(gpu_accel, 6)
    return snapshot
