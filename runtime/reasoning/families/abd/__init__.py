"""Familia ABD (abducción) — inferencia real.

Genera y rankea hipótesis (intervención → resolución de la variable principal) a
partir de la firma causal del escenario y el estado de alarma. Reemplaza el stub
que solo fijaba `abd_hypothesis=True`.
"""

from runtime.reasoning.families import core_inference as ci

FAMILY_ID = "ABD"


def execute(state):
    out = ci.abduce(state)
    return {
        "family": FAMILY_ID,
        "status": "ok",
        "state_delta": out["state_delta"],
        "confidence": out["confidence"],
        "cost": 1.0,
    }
