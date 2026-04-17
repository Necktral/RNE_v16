# thermodynamic_governor.py

import numpy as np
import psutil
import time
from collections import deque
from scipy.integrate import solve_ivp
from dataclasses import dataclass
import logging
from src.aeon_types import *

try:
    import pynvml
    pynvml.nvmlInit()
    GPU_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
    GPU_AVAILABLE = True
except Exception:
    GPU_AVAILABLE = False

@dataclass
class ThermalModelParams:
    thermal_capacitance: float = 1400.0  # J/°C
    thermal_resistance: float = 0.18     # °C/W
    ambient_temp: float = 25.0           # °C
    tdp: float = 115.0                   # Adaptado a RTX 2070 Q Max

@dataclass
class HealthStatus:
    memory: float
    energy: float
    entropy: float
    temperature: float
    stability: float
    cognitive_load: float
    temp_forecast: float
    thermal_gradient: float

class ThermodynamicGovernor:
    def __init__(self, config):
        self.config = config
        self.params = ThermalModelParams(**config.get('thermal_params', {}))
        self.critical_temp = config.get('critical_temp', 85.0)
        self.max_memory = config.get('max_memory', 0.95)

        self.current_temp = self.params.ambient_temp
        self.vfe_history = deque(maxlen=30)
        self.thermal_history = deque([self.current_temp], maxlen=100)
        self.time_history = deque([time.time()], maxlen=100)
        self.logger = logging.getLogger("ThermoGov")
        self.logger.setLevel(logging.INFO)

    def memory_usage(self):
        return psutil.virtual_memory().percent / 100.0

    def energy_consumption(self):
        try:
            if GPU_AVAILABLE:
                watts = pynvml.nvmlDeviceGetPowerUsage(GPU_HANDLE) / 1000.0
                return min(watts, self.params.tdp)
            else:
                return psutil.cpu_percent() / 100.0 * self.params.tdp
        except Exception:
            return self.params.tdp * 0.65

    def entropy_level(self):
        if len(self.vfe_history) < 5:
            return 0.0
        vfe = np.array(self.vfe_history)
        norm = vfe / (np.mean(vfe) + 1e-6)
        p = np.exp(-norm) / np.sum(np.exp(-norm))
        return float(-np.sum(p * np.log(p + 1e-8)) / np.log(len(p)))

    def update_thermal_state(self, power):
        now = time.time()
        delta = now - self.time_history[-1]
        if delta <= 0:
            # Si el delta es cero o negativo, no integramos y devolvemos la temperatura actual
            return self.current_temp

        def dT(t, T):
            return (power - (T - self.params.ambient_temp) / self.params.thermal_resistance) / self.params.thermal_capacitance

        sol = solve_ivp(dT, [0, delta], [self.current_temp], t_eval=[delta])
        if sol.y.shape[1] == 0:
            return self.current_temp
        self.current_temp = sol.y[0][-1]
        self.thermal_history.append(self.current_temp)
        self.time_history.append(now)
        return self.current_temp

    def predict_thermal_trajectory(self, seconds=5, power=None):
        power = power or self.energy_consumption()
        def dT(t, T):
            return (power - (T - self.params.ambient_temp) / self.params.thermal_resistance) / self.params.thermal_capacitance
        sol = solve_ivp(dT, [0, seconds], [self.current_temp], t_eval=[seconds])
        return sol.y[0][-1]

    def calculate_thermal_gradient(self):
        if len(self.thermal_history) < 2:
            return 0.0
        dt = np.diff(np.array(self.time_history))
        dT = np.diff(np.array(self.thermal_history))
        gradient = dT[-1] / dt[-1] if dt[-1] > 0 else 0.0
        return float(gradient)

    def assess_health(self) -> HealthStatus:
        power = self.energy_consumption()
        temp = self.update_thermal_state(power)
        forecast = self.predict_thermal_trajectory(5, power)
        return HealthStatus(
            memory=self.memory_usage(),
            energy=power / self.params.tdp,
            entropy=self.entropy_level(),
            temperature=temp / self.critical_temp,
            stability=1.0,
            cognitive_load=0.0,
            temp_forecast=forecast / self.critical_temp,
            thermal_gradient=self.calculate_thermal_gradient()
        )

    def energy_context(self, phase="default"):
        class Context:
            def __enter__(ctx):
                ctx.start = time.perf_counter()
                return ctx
            def __exit__(ctx, *args):
                elapsed = time.perf_counter() - ctx.start
                used = self.energy_consumption() * elapsed
                self.logger.info(f"[{phase}] Energía usada: {used:.2f}J en {elapsed:.3f}s")
        return Context()

    def record_vfe(self, vfe):
        self.vfe_history.append(float(vfe))

    def get_thermal_metrics(self) -> Dict[str, float]:
        self.logger.debug("[Governor] get_thermal_metrics() => valores simulados")
        return {
            "thermal_gradient": 0.0,
            "entropy_trend": 0.0,
            "cooling_rate": 0.0
        }

    def reset_thermal_model(self):
        self.logger.info("[Governor] Modelo térmico reiniciado (stub)")

    def initiate_cooling(self, intensity: float):
        self.logger.info(f"[Governor] Iniciando enfriamiento con intensidad {intensity:.2f} (stub)")

    def inject_noise(self, level: float):
        self.logger.warning(f"[Governor] Inyectando ruido térmico (nivel: {level:.2f}) (stub)")

    def activate_thermal_throttle(self):
        self.logger.critical("[THROTTLE] 🔥 Reducción de carga activada")
        time.sleep(1.0)

    def emergency_shutdown(self):
        self.logger.critical("[SHUTDOWN] 🛑 Apagado térmico de emergencia")
        time.sleep(1.5)
        raise SystemExit("🔥 Apagado crítico por temperatura")
