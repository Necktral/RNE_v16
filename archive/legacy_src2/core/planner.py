# core/planner.py

import random
import math
from collections import deque
import numpy as np


class AEONPlanner:
    """
    Núcleo simbólico de planificación adaptativa para AEON FENIX-Δ.
    Integra múltiples estrategias, aprendizaje por recompensa, 
    y adaptación estructural basada en contexto cognitivo.
    """

    def __init__(self, config=None):
        cfg = config.get("planner", {}) if config else {}

        # Configuración de estrategias
        self.strategies = cfg.get("strategies", ["free_energy", "risk_averse", "exploratory"])
        self.strategy_weights = {
            k: cfg.get("initial_weights", {}).get(k, 1.0)
            for k in self.strategies
        }

        # Parámetros de sensibilidad
        self.epistemic_threshold = cfg.get("epistemic_threshold", 0.15)
        self.load_threshold = cfg.get("load_threshold", 0.85)
        self.min_entropy = cfg.get("min_entropy", 0.02)
        self.temperature_limit = cfg.get("temperature_limit", 0.75)

        # Aprendizaje y Q-learning simplificado
        self.learning_rate = cfg.get("learning_rate", 0.05)
        self.discount_factor = cfg.get("discount_factor", 0.95)
        self.Q = {s: 0.0 for s in self.strategies}

        # Contexto histórico
        self.history = deque(maxlen=100)
        self.last_action = None
        self.rng = np.random.default_rng(cfg.get("seed", 42))

    def decide(self, context):
        """
        Decide la próxima acción a tomar basada en múltiples estrategias y el contexto actual.
        """
        self.history.append(context)

        weighted_actions = {}

        for strategy in self.strategies:
            policy = getattr(self, f"_policy_{strategy}", None)
            if policy is None:
                continue
            action = policy(context)
            weighted_actions.setdefault(action, 0.0)
            weighted_actions[action] += self.strategy_weights[strategy] * (self.Q[strategy] + 1)

        # Selección probabilística según suma ponderada
        total = sum(weighted_actions.values())
        if total == 0:
            return "idle"
        probs = [v / total for v in weighted_actions.values()]
        chosen = self.rng.choice(list(weighted_actions.keys()), p=probs)
        self.last_action = chosen
        return chosen

    def _policy_free_energy(self, ctx):
        pe = ctx.get("prediction_error", 0.0)
        d_epist = ctx.get("delta_epist", 0.0)
        entropy = ctx.get("entropy", 0.0)
        temp = ctx.get("temperature", 0.0)
        load = ctx.get("cognitive_load", 0.0)

        if pe > 1.0 and d_epist > self.epistemic_threshold:
            return "train"
        if d_epist < self.min_entropy and pe < 0.4:
            return "rest"
        if entropy > 0.5 and temp < self.temperature_limit:
            return "prune"
        if load > self.load_threshold:
            return "pause"
        if d_epist > 0.3 and entropy < 0.1:
            return "expand"
        return "idle"

    def _policy_risk_averse(self, ctx):
        pe = ctx.get("prediction_error", 0.0)
        load = ctx.get("cognitive_load", 0.0)
        temp = ctx.get("temperature", 0.0)

        risk = (load ** 2 + temp ** 2) / 2
        if pe > 0.8 and risk < 0.4:
            return "train"
        if load > 0.9 or temp > 0.85:
            return "rest"
        if pe < 0.3 and risk > 0.6:
            return "idle"
        return "prune" if risk < 0.7 else "idle"

    def _policy_exploratory(self, _):
        recent = [c.get("action", "idle") for c in list(self.history)[-10:]]
        stagnant = len(set(recent)) < 3
        if stagnant:
            return self.rng.choice(["train", "prune", "expand", "rest"])
        return "idle"

    def update_policy(self, reward):
        """
        Actualiza los pesos de estrategias con refuerzo Q-learning simplificado.
        """
        for strategy in self.strategies:
            self.Q[strategy] = (1 - self.learning_rate) * self.Q[strategy] + \
                               self.learning_rate * (reward + self.discount_factor * self.Q[strategy])

        total = sum(self.strategy_weights.values())
        for k in self.strategy_weights:
            self.strategy_weights[k] /= total or 1.0

    def diagnose(self):
        """
        Devuelve diagnóstico completo del estado del planificador.
        """
        return {
            "strategy_weights": self.strategy_weights.copy(),
            "Q_values": self.Q.copy(),
            "last_action": self.last_action,
            "history_length": len(self.history),
            "epistemic_threshold": self.epistemic_threshold,
            "load_threshold": self.load_threshold
        }

    def get_strategy_distribution(self):
        """
        Muestra frecuencia de estrategias aplicadas en los últimos pasos.
        """
        counts = {s: 0 for s in self.strategies}
        for ctx in self.history:
            a = ctx.get("action", None)
            if a in counts:
                counts[a] += 1
        total = sum(counts.values())
        return {k: v / total for k, v in counts.items()} if total else counts

    def health_check(self):
        if not all(math.isfinite(v) for v in self.Q.values()):
            return False
        if not (0.0 < self.epistemic_threshold <= 1.0):
            return False
        if not (0.0 < self.load_threshold <= 1.0):
            return False
        return True

    def reset(self):
        self.history.clear()
        self.last_action = None
        for k in self.Q:
            self.Q[k] = 0.0


__all__ = ["AEONPlanner"]
