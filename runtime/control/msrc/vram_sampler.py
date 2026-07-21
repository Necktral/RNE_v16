"""Sampler de VRAM NVIDIA para MSRC."""

from __future__ import annotations

import subprocess
import time
from collections import deque
from typing import Any, Deque, Dict, Mapping, Optional


class NvidiaVRAMSampler:
    """Lee uso de VRAM/temperatura desde nvidia-smi.

    El riesgo de fragmentación se aproxima con presión + variabilidad temporal,
    ya que no hay métrica directa por proceso en este runtime.
    """

    def __init__(self, *, history_size: int = 12, command_timeout_sec: float = 0.8):
        self.history_size = history_size
        self.command_timeout_sec = command_timeout_sec
        self._ratio_history: Deque[float] = deque(maxlen=history_size)
        self._ts_history: Deque[float] = deque(maxlen=history_size)

    def sample(self) -> Dict[str, Any]:
        now = time.time()
        snapshot = self._read_nvidia_smi()
        if snapshot is None:
            return {
                "available": False,
                "source": "nvidia-smi",
                "used_gb": 0.0,
                "total_gb": 0.0,
                "temperature_c": 0.0,
                "vram_headroom": 0.0,
                "vram_pressure": 1.0,
                "vram_fragmentation_risk": 1.0,
                "vram_opportunity_score": 0.0,
                "sample_ts": now,
            }

        used_gb = snapshot["used_mb"] / 1024.0
        total_gb = max(snapshot["total_mb"] / 1024.0, 1e-6)
        pressure = min(max(used_gb / total_gb, 0.0), 1.0)
        headroom = max(1.0 - pressure, 0.0)

        self._ratio_history.append(pressure)
        self._ts_history.append(now)

        fragmentation_risk = self._estimate_fragmentation_risk()
        opportunity = self._estimate_opportunity_score(
            headroom=headroom,
            pressure=pressure,
            temperature_c=snapshot["temperature_c"],
            fragmentation_risk=fragmentation_risk,
        )

        return {
            "available": True,
            "source": "nvidia-smi",
            "used_gb": round(used_gb, 4),
            "total_gb": round(total_gb, 4),
            "temperature_c": float(snapshot["temperature_c"]),
            "vram_headroom": round(headroom, 6),
            "vram_pressure": round(pressure, 6),
            "vram_fragmentation_risk": round(fragmentation_risk, 6),
            "vram_opportunity_score": round(opportunity, 6),
            "sample_ts": now,
        }

    def _read_nvidia_smi(self) -> Optional[Dict[str, float]]:
        cmd = [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.total,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
        try:
            proc = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.command_timeout_sec,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None

        if proc.returncode != 0:
            return None
        line = (proc.stdout or "").strip().splitlines()
        if not line:
            return None

        parts = [chunk.strip() for chunk in line[0].split(",")]
        if len(parts) < 3:
            return None

        try:
            used_mb = float(parts[0])
            total_mb = float(parts[1])
            temperature_c = float(parts[2])
        except ValueError:
            return None

        return {
            "used_mb": max(used_mb, 0.0),
            "total_mb": max(total_mb, 1.0),
            "temperature_c": max(temperature_c, 0.0),
        }

    def _estimate_fragmentation_risk(self) -> float:
        if len(self._ratio_history) <= 1:
            return 0.1

        ratios = list(self._ratio_history)
        deltas = [abs(ratios[i] - ratios[i - 1]) for i in range(1, len(ratios))]
        mean_delta = sum(deltas) / len(deltas)
        pressure = ratios[-1]

        # Aproximación: alta presión + cambios bruscos -> más riesgo.
        risk = (0.55 * pressure) + (0.45 * min(mean_delta * 6.0, 1.0))
        return min(max(risk, 0.0), 1.0)

    def _estimate_opportunity_score(
        self,
        *,
        headroom: float,
        pressure: float,
        temperature_c: float,
        fragmentation_risk: float,
    ) -> float:
        thermal_risk = min(max((temperature_c - 70.0) / 25.0, 0.0), 1.0)
        # VRAM es activo estratégico: premiar headroom y castigar borde peligroso.
        score = (0.65 * headroom) + (0.25 * (1.0 - fragmentation_risk)) + (0.10 * (1.0 - thermal_risk))
        # Si ya estamos muy cerca del borde, recortar oportunidad.
        if pressure > 0.92:
            score *= 0.2
        elif pressure > 0.85:
            score *= 0.55
        return min(max(score, 0.0), 1.0)


class NullVRAMSampler:
    """Fallback cuando no se quiere o no se puede usar VRAM real."""

    def sample(self) -> Dict[str, Any]:
        return {
            "available": False,
            "source": "none",
            "used_gb": 0.0,
            "total_gb": 0.0,
            "temperature_c": 0.0,
            "vram_headroom": 0.0,
            "vram_pressure": 1.0,
            "vram_fragmentation_risk": 1.0,
            "vram_opportunity_score": 0.0,
            "sample_ts": time.time(),
        }


class FixedVRAMSampler:
    """Adapta un snapshot unificado sellado al contrato de VRAM de MSRC.

    Las campañas controladas inyectan el mismo
    ``LifeKernelConfig.resource_snapshot_override`` tanto al runtime neuronal
    como a MSRC. El sampler conserva una copia inmutable de la proyección GPU:
    ``sample()`` nunca vuelve a consultar el host ni cambia el timestamp.

    ``available`` del snapshot unificado representa disponibilidad del *host*;
    por eso la disponibilidad GPU se deriva exclusivamente de
    ``gpu_available``. Esto evita declarar una GPU inexistente cuando solo hay
    telemetría CPU/RAM.
    """

    def __init__(self, snapshot: Mapping[str, Any]):
        raw = dict(snapshot)
        self._snapshot: Dict[str, Any] = {
            "available": bool(raw.get("gpu_available", False)),
            "source": str(
                raw.get("gpu_source")
                or raw.get("vram_source")
                or raw.get("source")
                or "resource-snapshot-override"
            ),
            "used_gb": raw.get("used_gb", raw.get("vram_used_gb", 0.0)),
            "total_gb": raw.get("total_gb", raw.get("vram_total_gb", 0.0)),
            "temperature_c": raw.get(
                "temperature_c", raw.get("gpu_temperature_c", 0.0)
            ),
            "vram_headroom": raw.get("vram_headroom", 0.0),
            "vram_pressure": raw.get("vram_pressure", 1.0),
            "vram_fragmentation_risk": raw.get("vram_fragmentation_risk", 1.0),
            "vram_opportunity_score": raw.get(
                "vram_opportunity_score",
                raw.get("gpu_opportunity_score", 0.0),
            ),
            "sample_ts": raw.get("gpu_sample_ts", raw.get("sample_ts")),
        }

    def sample(self) -> Dict[str, Any]:
        return dict(self._snapshot)
