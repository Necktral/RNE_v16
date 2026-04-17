import numpy as np
import torch
import psutil
import pynvml
import logging
import os
import time
from src.utils.influx_logger import InfluxLogger

class EpistemeMeter:
    def __init__(self, tau_epist=30, epsilon=1e-9, influx_logger=None):
        # Gobernanza epistémica
        self.limites = {
            'delta_epist': 0.25,  # Fluctuación máxima permitida
            'ema_delta_epist': 0.15,  # Fluctuación suavizada
            'entropy': 0.95,  # Umbral de entropía
            'KL': 2.0,  # Divergencia KL máxima
            'predictive_loss': 1.5,  # Pérdida predictiva máxima
            'thermal_epsilon': 0.12,  # Inestabilidad máxima
        }
        self.veto_epistemico = False
        self.ultima_alerta = None
        self.tau_epist = tau_epist
        self.epsilon = epsilon
        self.ema_delta = None
        self.prev_efficiency = None
        pynvml.nvmlInit()
        self._GPU_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
        # InfluxLogger OO
        self.influx_logger = influx_logger or InfluxLogger(
            url=os.environ.get("INFLUXDB_URL", "http://localhost:8181"),
            token=os.environ.get("INFLUXDB_TOKEN", "<TOKEN>"),
            database=os.environ.get("INFLUXDB_BUCKET", "aeon_metrics")
        )

    def _mutual_information(self, posterior, prior):
        ratio = (posterior + self.epsilon) / (prior + self.epsilon)
        return np.mean(posterior * np.log(ratio))

    def _read_power(self):
        try:
            power = pynvml.nvmlDeviceGetPowerUsage(self._GPU_HANDLE) / 1000.0
        except Exception:
            power = psutil.cpu_percent() * 1.2
        return power

    def update(self, current_efficiency, fisher_info, posterior=None, prior=None, entropy=None, kl=None, predictive_loss=None):
        # --- Fluctuación y suavizado ---
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

        # --- Gobernanza: límites dinámicos y veto epistémico ---
        veto = False
        alertas = []
        # Límite de fluctuación instantánea
        if abs(delta) > self.limites['delta_epist']:
            veto = True
            alertas.append(f"Δ_epist ({delta:.3f}) > {self.limites['delta_epist']}")
        # Límite de fluctuación suavizada
        if abs(self.ema_delta) > self.limites['ema_delta_epist']:
            veto = True
            alertas.append(f"EMA_Δ_epist ({self.ema_delta:.3f}) > {self.limites['ema_delta_epist']}")
        # Límite de entropía
        if entropy is not None and entropy > self.limites['entropy']:
            veto = True
            alertas.append(f"Entropía ({entropy:.3f}) > {self.limites['entropy']}")
        # Límite de KL
        if kl is not None and kl > self.limites['KL']:
            veto = True
            alertas.append(f"KL ({kl:.3f}) > {self.limites['KL']}")
        # Límite de pérdida predictiva
        if predictive_loss is not None and predictive_loss > self.limites['predictive_loss']:
            veto = True
            alertas.append(f"PredictiveLoss ({predictive_loss:.3f}) > {self.limites['predictive_loss']}")
        # Alerta térmica cognitiva (inestabilidad)
        if abs(self.ema_delta) > self.limites['thermal_epsilon']:
            alertas.append(f"[ALERTA] Inestabilidad cognitiva: EMA_Δ_epist={self.ema_delta:.3f} > ϵ={self.limites['thermal_epsilon']}")

        self.veto_epistemico = veto
        self.ultima_alerta = alertas if alertas else None

        metrics = {
            "delta_epist": delta,
            "ema_delta_epist": self.ema_delta,
            "fisher_density": fisher_density,
            "mutual_info": mutual_info,
            "power_estimate": power_estimate,
            "efficiency": current_efficiency,
            "veto_epistemico": veto,
            "episteme_alertas": alertas,
        }
        # Telemetría a InfluxDB OO
        try:
            self.influx_logger.log_metric(
                name="aeon_episteme",
                value=metrics.get("efficiency", 0.0),
                tags={"type": "episteme", "veto": str(veto)},
                timestamp=int(time.time() * 1e9)
            )
        except Exception as e:
            logging.getLogger("EpistemeMeter").warning(f"No se pudo enviar métrica a InfluxDB: {e}")
        return metrics

    def evaluate(self, context):
        current_efficiency = context.get("efficiency", np.random.uniform(0.5, 1.0))
        fisher_info = context.get("fisher_info", np.random.uniform(0.1, 1.0, size=10))
        posterior = context.get("posterior")
        prior = context.get("prior")
        entropy = context.get("entropy")
        kl = context.get("KL")
        predictive_loss = context.get("predictive_loss")
        return self.update(current_efficiency, fisher_info, posterior, prior, entropy, kl, predictive_loss)

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
        # Gobernanza: límite de magnitud y consumo energético estimado
        NOISE_LIMIT = 0.2
        ENERGY_BUDGET = 1.0
        veto = False
        if abs(intensity) > NOISE_LIMIT:
            veto = True
            logger.warning(f"[Episteme] VETO: Intensidad de ruido {intensity:.2f} > límite {NOISE_LIMIT}")
        energy_est = min(abs(intensity), NOISE_LIMIT) * 0.8  # Estimación simple
        if energy_est > ENERGY_BUDGET:
            veto = True
            logger.warning(f"[Episteme] VETO: Consumo energético estimado {energy_est:.2f} > presupuesto {ENERGY_BUDGET}")
        if veto:
            logger.error(f"[Episteme] Ruido estructural VETADO (intensidad={intensity:.2f}, energía={energy_est:.2f})")
        else:
            logger.warning(f"[Episteme] Ruido caótico aplicado con intensidad {intensity:.2f} (permitido, energía={energy_est:.2f})")
