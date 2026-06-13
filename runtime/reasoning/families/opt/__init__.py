"""Familia OPT (optimización) — argmin real.

Optimiza la elección (intervención, nº de pasos) minimizando un objetivo
escalar (término de valor sobre la variable principal + coste de esfuerzo)
sobre el modelo de efectos declarado. Explicable: deja la tabla de
alternativas evaluadas. Reemplaza el stub que devolvía ``status=idle``.
"""

from runtime.reasoning.families import core_inference as ci

FAMILY_ID = "OPT"


def execute(state):
    out = ci.optimize_choice(state)
    return {
        "family": FAMILY_ID,
        "status": "ok",
        "state_delta": out["state_delta"],
        "confidence": out["confidence"],
        "cost": 1.0,
    }
