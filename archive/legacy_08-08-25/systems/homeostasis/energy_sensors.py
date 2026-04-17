# energy_sensors.py

import psutil
import numpy as np
import time
import platform
import json
from typing import Dict, Any
from pathlib import Path
from aeon.core.aeon_types import *

try:
    import pynvml
    pynvml.nvmlInit()
    GPU_AVAILABLE = True
    GPU_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
except Exception:
    GPU_AVAILABLE = False
    GPU_HANDLE = None

class EnergySensors:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.gpu_name = self._detect_gpu()
        self.cpu_name = platform.processor()
        self.system_limits = self._load_system_limits()
        self.history = {k: [] for k in ["cpu", "gpu", "mem", "temp", "entropy", "vram"]}
        self.log_path = Path(config.get("log_dir", "logs")) / "sensor_data.json"
        self._init_log_file()

    def _detect_gpu(self) -> str:
        if GPU_AVAILABLE:
            try:
                return pynvml.nvmlDeviceGetName(GPU_HANDLE).decode()
            except:
                return "NVIDIA GPU (detected)"
        return "GPU not available"

    def _load_system_limits(self) -> Dict[str, float]:
        return {
            "S_max": self.config.get("S_max", 1.0),
            "eta_silicio": self.config.get("eta_silicio", 100.0),
            "T_crit": self.config.get("T_crit", 85.0),
            "VRAM_limit": self.config.get("M_VRAM", 8 * 1024**3)  # 8 GB
        }

    def _init_log_file(self):
        if not self.log_path.exists():
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "w") as f:
                json.dump([], f)

    def get_memory_load(self) -> float:
        mem = psutil.virtual_memory()
        value = mem.percent / 100.0
        self.history["mem"].append(value)
        return round(value, 4)

    def get_cpu_load(self) -> float:
        value = psutil.cpu_percent(interval=0.05) / 100.0
        self.history["cpu"].append(value)
        return round(value, 4)

    def get_gpu_power(self) -> float:
        if GPU_AVAILABLE:
            try:
                watts = pynvml.nvmlDeviceGetPowerUsage(GPU_HANDLE) / 1000.0
                self.history["gpu"].append(watts)
                return round(watts, 2)
            except:
                return -1.0
        return -1.0

    def get_temperature(self) -> float:
        if GPU_AVAILABLE:
            try:
                temp = pynvml.nvmlDeviceGetTemperature(GPU_HANDLE, pynvml.NVML_TEMPERATURE_GPU)
                self.history["temp"].append(temp)
                return float(temp)
            except:
                return -1
        return 60.0 + np.random.uniform(-2.0, 2.0)

    def get_entropy(self) -> float:
        omega = (
            self.get_cpu_load() +
            max(self.get_gpu_power() / 100.0, 0) +
            self.get_memory_load()
        ) / 3.0
        kB = 1.380649e-23
        entropy = kB * np.log(1 + omega)
        normalized = entropy / self.system_limits["S_max"]
        self.history["entropy"].append(normalized)
        return round(normalized, 6)

    def get_thermal_margin(self) -> float:
        temp = self.get_temperature()
        if temp <= 0:
            return 1.0
        margin = max(0.0, 1.0 - temp / self.system_limits["T_crit"])
        return round(margin, 4)

    def get_vram_usage(self) -> float:
        if GPU_AVAILABLE:
            try:
                info = pynvml.nvmlDeviceGetMemoryInfo(GPU_HANDLE)
                usage = info.used / self.system_limits["VRAM_limit"]
                self.history["vram"].append(usage)
                return round(usage, 4)
            except:
                return -1.0
        return -1.0

    def system_snapshot(self) -> Dict[str, float]:
        snapshot = {
            "cpu_load": self.get_cpu_load(),
            "gpu_power": self.get_gpu_power(),
            "memory": self.get_memory_load(),
            "temperature": self.get_temperature(),
            "thermal_margin": self.get_thermal_margin(),
            "entropy": self.get_entropy(),
            "vram_usage": self.get_vram_usage(),
            "timestamp": time.time()
        }
        self._log_snapshot(snapshot)
        return snapshot

    def _log_snapshot(self, snapshot: Dict[str, float]):
        try:
            with open(self.log_path, "r+") as f:
                data = json.load(f)
                data.append(snapshot)
                f.seek(0)
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[Sensors] ⚠ Error al guardar snapshot: {e}")

    def print_snapshot(self):
        snap = self.system_snapshot()
        print("[SENSORS]", json.dumps(snap, indent=2))
