"""Familia CTF (contrafactual) — inferencia real.

Compara la transición factual con la contrafactual sobre la variable principal
(usando la simulación real del mundo) y verifica si la intervención elegida queda
soportada, contrastando con `relation_kind`. Núcleo determinista + aumento LLM
opcional (gated, opt-in). Reemplaza el stub.
"""

from runtime.reasoning.families import core_inference as ci

FAMILY_ID = "CTF"


def execute(state):
    out = ci.counterfactual_check(state)
    state_delta = dict(out["state_delta"])
    cost = 1.0

    aug = ci.maybe_llm_augment(state, family=FAMILY_ID)
    state_delta["ctf_llm_augmented"] = bool(aug and aug.get("ok"))
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
