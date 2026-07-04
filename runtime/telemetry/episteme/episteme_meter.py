import numpy as np
import torch
import psutil
import pynvml
import logging

class EpistemeMeter:
    def __init__(self, tau_epist=30, epsilon=1e-9):
        self.tau_epist = tau_epist
        self.epsilon = epsilon
        self.ema_delta = None
        self.prev_efficiency = None
        pynvml.nvmlInit()
        self._GPU_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)

    def _mutual_information(self, posterior, prior):
        ratio = (posterior + self.epsilon) / (prior + self.epsilon)
        return np.mean(posterior * np.log(ratio))

    def _read_power(self):
        try:
            power = pynvml.nvmlDeviceGetPowerUsage(self._GPU_HANDLE) / 1000.0
        except Exception:
            power = psutil.cpu_percent() * 1.2
        return power

    def update(self, current_efficiency, fisher_info, posterior=None, prior=None):
        if self.prev_efficiency is None:
            self.prev_efficiency = current_efficiency
            delta = 0.0
        else:
            delta = current_efficiency - self.prev_efficiency
            self.prev_efficiency = current_efficiency

        alpha = 2 / (self.tau_epist + 1)
        self.ema_delta = delta if self.ema_delta is None else (1 - alpha) * self.ema_delta + alpha * delta

        fisher_density = np.sqrt(np.mean(np.square(fisher_info)))
        mutual_info = self._mutual_information(posterior, prior) if posterior is not None else 0.0
        power_estimate = self._read_power()

        return {
            "delta_epist": delta,
            "ema_delta_epist": self.ema_delta,
            "fisher_density": fisher_density,
            "mutual_info": mutual_info,
            "power_estimate": power_estimate,
            "efficiency": current_efficiency
        }

    def evaluate(self, context):
        current_efficiency = context.get("efficiency", np.random.uniform(0.5, 1.0))
        fisher_info = context.get("fisher_info", np.random.uniform(0.1, 1.0, size=10))
        posterior = context.get("posterior")
        prior = context.get("prior")
        return self.update(current_efficiency, fisher_info, posterior, prior)

    def get_global_efficiency(self) -> float:
        logger = logging.getLogger("EpistemeMeter")
        logger.debug("[Episteme] Eficiencia global retornada (stub)")
        return 1.0

    def get_accumulated_energy(self) -> float:
        logger = logging.getLogger("EpistemeMeter")
        logger.debug("[Episteme] Energía acumulada retornada (stub)")
        return 0.0

    def apply_noise(self, intensity: float):
        logger = logging.getLogger("EpistemeMeter")
        logger.warning(f"[Episteme] Ruido caótico aplicado con intensidad {intensity:.2f} (stub)")
