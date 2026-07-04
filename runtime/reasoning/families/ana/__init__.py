"""Familia ANA (analogía) — inferencia real.

Mapea la situación actual a la memoria recuperada (por solapamiento estructural ya
puntuado) o, en su defecto, al vocabulario de proposiciones de la firma causal del
escenario. Reemplaza el stub que solo fijaba `ana_mapping=True`.
"""

from runtime.reasoning.families import core_inference as ci

FAMILY_ID = "ANA"


def execute(state):
    out = ci.analogize(state)
    return {
        "family": FAMILY_ID,
        "status": "ok",
        "state_delta": out["state_delta"],
        "confidence": out["confidence"],
        "cost": 1.0,
    }
