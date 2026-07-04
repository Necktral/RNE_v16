"""Familia PROB (probabilística) — calibración real.

Cierra la calibración combinando la evidencia acumulada (CAU/CTF/DED + belief +
incertidumbre) en un posterior con cota inferior de confianza (Agresti-Coull).
Núcleo determinista + aumento LLM opcional (gated, opt-in). Mantiene
`prob_calibrated` truthy (contrato) pero ahora con un posterior real.
"""

from runtime.reasoning.families import core_inference as ci

FAMILY_ID = "PROB"


def execute(state):
    out = ci.calibrate(state)
    state_delta = dict(out["state_delta"])
    cost = 0.9

    aug = ci.maybe_llm_augment(state, family=FAMILY_ID)
    state_delta["prob_llm_augmented"] = bool(aug and aug.get("ok"))
    if aug and aug.get("trigger_family") == FAMILY_ID:
        state_delta["core_reasoner_llm"] = aug
        cost += float(aug.get("latency_s") or 0.0)

    return {
        "family": FAMILY_ID,
        "status": "ok",
        "state_delta": state_delta,
        "confidence": out["confidence"],
        "cost": cost,
    }
