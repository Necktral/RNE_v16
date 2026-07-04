# src/aeon_fenix_delta/evolution/auto_mutator.py

import time
import math
import logging
from typing import Dict, Any, Optional, List
import numpy as np

logger = logging.getLogger(__name__)

class AutoMutator:
    def __init__(self,
                 neurogenesis_manager,
                 katana_pruner,
                 neuro_threshold: float = 0.3,
                 prune_threshold: float = -0.3,
                 max_pressure: float = 1.0,
                 min_pressure: float = -1.0,
                 pressure_decay: float = 0.95,
                 adaptive_step: float = 0.01,
                 log_events: bool = True,
                 warmup_limit: int = 5000,
                 testing: bool = False):
        self.neurogenesis = neurogenesis_manager
        self.katana = katana_pruner
        self.neuro_threshold = neuro_threshold
        self.prune_threshold = prune_threshold
        self.max_pressure = max_pressure
        self.min_pressure = min_pressure
        self.pressure_decay = pressure_decay
        self.adaptive_step = adaptive_step
        self.log_events = log_events
        self.warmup_limit = warmup_limit
        self.testing = testing

        self.pressure = 0.0
        self.pressure_history = []
        self.mutation_log = []
        self.last_event = None
        self.oscillation_counter = 0

    def evaluate_mutation_pressure(self, ctx: Dict[str, Any]) -> float:
        delta_epist = ctx.get("delta_epist", 0.0)
        eta_bayes = ctx.get("mutual_info", 0.0)
        thermal_risk = ctx.get("thermal_risk", 0.0)
        pruning_level = ctx.get("pruning_intensity", 0.0)
        growth_impact = ctx.get("neurogenesis_impact", {})
        growth_avg = np.mean(list(growth_impact.values())) if growth_impact else 0.0

        # Presión base: combinación no lineal
        pressure = (
            -math.tanh(delta_epist * 2) * 0.4 +
            math.erf(pruning_level) * 0.3 -
            thermal_risk * 0.2 +
            math.log1p(growth_avg) * 0.3 +
            1 / (1 + math.exp(-5 * (eta_bayes - 0.5))) * 0.4
        )

        # Memoria histórica amortiguada
        if self.pressure_history:
            historical = sum(p * (self.pressure_decay ** i)
                             for i, p in enumerate(reversed(self.pressure_history))) / len(self.pressure_history)
            pressure = 0.7 * pressure + 0.3 * historical

            # Oscilación: reducción si cambia de signo muchas veces
            if np.sign(self.pressure_history[-1]) != np.sign(pressure):
                self.oscillation_counter += 1
            else:
                self.oscillation_counter = max(0, self.oscillation_counter - 1)

            if self.oscillation_counter > 3:
                pressure *= 0.5
                self.oscillation_counter = 0

        self.pressure = np.clip(pressure, self.min_pressure, self.max_pressure)
        self.pressure_history.append(self.pressure)
        if len(self.pressure_history) > 10:
            self.pressure_history.pop(0)

        return self.pressure

    def adaptive_step_size(self) -> float:
        if len(self.pressure_history) < 3:
            return self.adaptive_step
        volatility = np.std(self.pressure_history[-3:])
        return np.clip(self.adaptive_step * (1 + volatility * 10), 0.001, 0.1)

    def adjust_thresholds(self):
        step = self.adaptive_step_size()
        sigmoid_factor = 1.0 / (1.0 + math.exp(-10 * (abs(self.pressure) - 0.5)))

        if self.pressure > self.neuro_threshold:
            self.neurogenesis.delta_epist_min *= (1.0 - step * sigmoid_factor)
            self.neurogenesis.mutual_info_min *= (1.0 - step * sigmoid_factor)
            if hasattr(self.katana, "base_tau"):
                self.katana.base_tau = min(0.1, self.katana.base_tau * (1.0 + step * sigmoid_factor / 2))

        elif self.pressure < self.prune_threshold:
            if hasattr(self.katana, "base_tau"):
                self.katana.base_tau = max(0.001, self.katana.base_tau * (1.0 - step * sigmoid_factor))
            self.neurogenesis.delta_epist_min *= (1.0 + step * sigmoid_factor / 2)
            self.neurogenesis.mutual_info_min *= (1.0 + step * sigmoid_factor / 2)

    def step(self, ctx: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        """
        Evaluates mutation pressure, adjusts module thresholds, and triggers adaptation actions.
        It captures adaptation payloads and returns them as a list, prioritizing neurogenesis.

        Args:
            ctx: The context dictionary with current system metrics.

        Returns:
            A list of adaptation payloads, or None if no action was taken.
        """
        # --- TESTING MODE: skip warmup and relax thresholds ---
        if self.testing:
            # Pruning: VRAM > 90%
            if ctx.get('vram_usage_gb', 0.0) / getattr(ctx, 'MAX_VRAM_GB', 1.0) >= 0.90:
                prune_payload = self.katana.step(ctx)
                self.log_mutation_event(ctx, 0, 1, None, prune_payload)
                return [prune_payload]
            # Neurogenesis: loss history rising
            hist = ctx.get('history', [])
            if len(hist) >= 4 and hist[-1] > max(hist[-4:-1]):
                neuro_payload = self.neurogenesis.step(ctx)
                self.log_mutation_event(ctx, 1, 0, neuro_payload, None)
                return [neuro_payload]

        pressure = self.evaluate_mutation_pressure(ctx)
        self.adjust_thresholds()

        neuro_intensity = np.clip((pressure - self.neuro_threshold) / 0.5, 0, 1)
        prune_intensity = np.clip((-pressure - self.prune_threshold) / 0.5, 0, 1)

        neuro_payload = None
        if neuro_intensity > 0:
            ctx["neurogenesis_intensity"] = neuro_intensity
            neuro_payload = self.neurogenesis.step(ctx)

        prune_payload = None
        if prune_intensity > 0:
            ctx["pruning_intensity"] = prune_intensity
            prune_payload = self.katana.step(ctx)

        # Prioritize neurogenesis over pruning by returning only that payload if both are generated.
        if neuro_payload:
            final_payloads = [neuro_payload]
        elif prune_payload:
            final_payloads = [prune_payload]
        else:
            final_payloads = []

        if self.log_events:
            self.log_mutation_event(ctx, neuro_intensity, prune_intensity, neuro_payload, prune_payload)

        return final_payloads if final_payloads else None

    def log_mutation_event(self, ctx: Dict[str, Any], n_intensity: float, p_intensity: float, neuro_payload: Optional[Dict], prune_payload: Optional[Dict]):
        action_taken = "none"
        prioritization_details = "N/A"

        if neuro_payload and prune_payload:
            action_taken = neuro_payload.get('action', 'neurogenesis')
            prioritization_details = f"Neurogenesis prioritized over pruning. Neurogenesis payload selected."
        elif neuro_payload:
            action_taken = neuro_payload.get('action', 'neurogenesis')
            prioritization_details = "Only neurogenesis payload was generated."
        elif prune_payload:
            action_taken = prune_payload.get('action', 'pruning')
            prioritization_details = "Only pruning payload was generated."

        event = {
            "timestamp": time.time(),
            "pressure": self.pressure,
            "trigger": action_taken,
            "neuro_intensity": n_intensity,
            "prune_intensity": p_intensity,
            "delta_epist": ctx.get("delta_epist", 0.0),
            "thermal_risk": ctx.get("thermal_risk", 0.0),
            "growth_impact": ctx.get("neurogenesis_impact", {}),
            "neuro_payload_generated": bool(neuro_payload),
            "prune_payload_generated": bool(prune_payload),
            "prioritization_details": prioritization_details,
            "thresholds": {
                "delta_min": self.neurogenesis.delta_epist_min,
                "mutual_min": self.neurogenesis.mutual_info_min,
                "pruning_tau": getattr(self.katana, "base_tau", None)
            }
        }
        self.mutation_log.append(event)
        self.last_event = event
        if len(self.mutation_log) > 1000:
            self.mutation_log = self.mutation_log[-1000:]
