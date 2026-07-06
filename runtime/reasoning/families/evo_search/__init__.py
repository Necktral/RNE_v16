"""Familia EVO_SEARCH — búsqueda evolutiva sobre secuencias de intervención.

OFF (byte-idéntico): idle (stub original).
DEEP (opt-in, RNFE_EVO_SEARCH_DEEP / RNFE_REASONING_DEEP): algoritmo genético
**determinista** (RNG sembrado desde un hash del estado) que evoluciona secuencias
de intervenciones proyectadas con el modelo de efectos declarado de la firma
causal. Complementa a PLAN (BFS exacto) para espacios de acción grandes. Al ser
sembrado desde el estado, es reproducible corrida a corrida.
"""

import hashlib
import random

from runtime.reasoning.families import core_inference as ci
from runtime.reasoning.families import _deep_common as dc

FAMILY_ID = "EVO_SEARCH"

_HORIZON = 5
_POP = 12
_GENERATIONS = 8


def _seed(state, x0) -> int:
    key = f"{ci.scenario_name(state)}|{round(float(x0), 4)}"
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:12], 16)


def execute(state):
    if not ci.family_deep_enabled(FAMILY_ID):
        return {
            "family": FAMILY_ID,
            "status": "idle",
            "state_delta": {},
            "confidence": 0.0,
            "cost": 0.0,
        }

    model = ci._effect_model(state)
    mv = ci.main_variable(state)
    direction = ci.optimization_direction(state, ci.resolve_signature(state))
    obs = dc.safe_dict(state.get("observation"))
    x0 = obs.get(mv)
    md = dc.safe_dict(state.get("scenario_metadata"))
    threshold = md.get("alarm_threshold")
    threshold = float(threshold) if isinstance(threshold, (int, float)) else None

    if not isinstance(x0, (int, float)) or not model:
        return {
            "family": FAMILY_ID,
            "status": "idle",
            "state_delta": {"evo_solved": False},
            "confidence": 0.2,
            "cost": 0.0,
            "failure_mode": "no_model_or_observation",
        }

    x0 = float(x0)
    interventions = sorted(model)
    rng = random.Random(_seed(state, x0))
    sign = -1.0 if direction == "minimize" else 1.0

    def project(seq):
        x = x0
        for iv in seq:
            x = min(1.0, max(0.0, x + model[iv]))
        return x

    def value(x):  # menor es mejor en ambas direcciones
        return x if direction == "minimize" else (1.0 - x)

    def goal(x):
        if threshold is None:
            return False
        return (x < threshold) if direction == "minimize" else (x > threshold)

    def fitness(seq):
        x = project(seq)
        return -(value(x) + 0.03 * len(seq)) + (0.5 if goal(x) else 0.0)

    def rand_seq():
        return [rng.choice(interventions) for _ in range(rng.randint(1, _HORIZON))]

    pop = [rand_seq() for _ in range(_POP)]
    best, best_fit, evals = None, float("-inf"), 0
    for _ in range(_GENERATIONS):
        scored = sorted(((fitness(s), s) for s in pop), key=lambda t: t[0], reverse=True)
        evals += len(pop)
        if scored[0][0] > best_fit:
            best_fit, best = scored[0][0], list(scored[0][1])
        elite = [s for _, s in scored[: max(2, _POP // 3)]]
        nxt = [list(s) for s in elite]
        while len(nxt) < _POP:
            a, b = rng.choice(elite), rng.choice(elite)
            cut = rng.randint(0, min(len(a), len(b)))
            child = (a[:cut] + b[cut:])[: _HORIZON] or rand_seq()
            if rng.random() < 0.3:
                child[rng.randrange(len(child))] = rng.choice(interventions)
            nxt.append(child)
        pop = nxt

    terminal = project(best)
    improvement = sign * (x0 - terminal) if direction == "minimize" else sign * (terminal - x0)
    goal_reached = goal(terminal)
    confidence = dc.clamp(0.45 + (0.30 if goal_reached else 0.0) + 0.25 * dc.clamp(abs(improvement) * 5.0))
    return {
        "family": FAMILY_ID,
        "status": "ok",
        "state_delta": {
            "evo_solved": True,
            "evo_best_sequence": best,
            "evo_first_action": best[0] if best else None,
            "evo_projected_terminal": round(terminal, 6),
            "evo_goal_reached": goal_reached,
        },
        "confidence": round(confidence, 4),
        "cost": min(3.0, 0.5 + 0.02 * evals),
        "recommended_next_family": "PROB",
        "artifacts": {
            "generations": _GENERATIONS,
            "population": _POP,
            "evaluations": evals,
            "best_fitness": round(best_fit, 4),
        },
    }
