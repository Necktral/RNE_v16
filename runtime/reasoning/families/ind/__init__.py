"""Familia IND (inducción) — generalización real.

Induce una regularidad «alarma A & intervención X → relación R sobre la variable
principal» a partir de los episodios análogos recuperados (con conteo de soporte
y cota inferior de confianza Agresti-Coull), o *a priori* desde el modelo de
efectos de la firma causal cuando no hay ejemplos. Reemplaza el stub que devolvía
``status=idle``.
"""

from runtime.reasoning.families import core_inference as ci

FAMILY_ID = "IND"


def execute(state):
    out = ci.induce(state)
    return {
        "family": FAMILY_ID,
        "status": "ok",
        "state_delta": out["state_delta"],
        "confidence": out["confidence"],
        "cost": 0.8,
    }
