"""Familia PLAN (planificación) — búsqueda real.

Planifica una secuencia corta de intervenciones hacia el lado seguro del umbral
de alarma, por búsqueda hacia adelante sobre el modelo de efectos declarado de
la firma causal del escenario. Reemplaza el stub que devolvía ``status=idle``.
"""

from runtime.reasoning.families import core_inference as ci

FAMILY_ID = "PLAN"


def execute(state):
    out = ci.plan_search(state)
    return {
        "family": FAMILY_ID,
        "status": "ok",
        "state_delta": out["state_delta"],
        "confidence": out["confidence"],
        "cost": 1.1,
    }
