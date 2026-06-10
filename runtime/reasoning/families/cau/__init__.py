"""Familia CAU (causal) — inferencia real.

Infiere el enlace intervención→variable principal a partir de la transición
factual observada y la contrasta con la dirección esperada de la firma causal.
Núcleo determinista + aumento LLM opcional (gated, opt-in). Reemplaza el stub.
"""

from runtime.reasoning.families import core_inference as ci

FAMILY_ID = "CAU"


def execute(state):
    out = ci.causal_infer(state)
    state_delta = dict(out["state_delta"])
    cost = 1.0

    aug = ci.maybe_llm_augment(state, family=FAMILY_ID)
    state_delta["cau_llm_augmented"] = bool(aug and aug.get("ok"))
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
