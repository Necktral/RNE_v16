"""Familia A12 — decisor lógico-probabilístico (no-monotonía + Bayes-factor + ACT).

OFF (byte-idéntico): idle.
DEEP (opt-in, RNFE_A12_DEEP / RNFE_REASONING_DEEP): familia TARDÍA que lee toda la
traza acumulada y produce una decisión coherente:

  - No-monotonía / defeasible: sostiene la elección reactiva por defecto y la
    RETRACTA cuando evidencia posterior la derrota (A11 predice breach, CTF/CAU en
    conflicto, DED insatisfacible).
  - Bayes-factor: adopta una alternativa SÓLO si el peso de la evidencia (producto de
    likelihood ratios, log-aditivo) supera un umbral — no por una sola señal.
  - ACT: compromete (commit) sólo con confianza suficiente; si no, se ABSTIENE
    (mantiene el default y marca que estaba derrotado) — honesto.

Compone con A11: A11 imagina el futuro; A12 decide con lógica sobre esa imaginación
+ el resto de la evidencia. Determinista. Advisory (Fase 3 lo cablea a la actuación
gated bajo RNFE_REASONING_ACTUATES).
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, Optional

from runtime.reasoning.families import core_inference as ci
from runtime.reasoning.families import _deep_common as dc

FAMILY_ID = "A12"

# Likelihood ratios: evidencia a favor de cambiar del default a una alternativa.
_LR = {
    "imagination_breach": 4.0,   # A11 predice que el default cruza la alarma (el más fuerte)
    "ctf_disagree": 2.5,
    "cau_not_help": 2.5,
    "ded_unsat": 2.0,
}
_LR_PER_WITNESS = 1.8            # por familia que recomienda la alternativa
_LR_A11_ALIGNED = 1.5           # bonus: A11 recomienda la candidata Y predice breach del default
_TAU_BF = 3.0                    # umbral de adopción (Jeffreys "sustancial")
_BF_DECISIVE = 10.0             # BF decisivo ⇒ commit aunque la posterior sea modesta
_LCB_FLOOR = 0.5                # confianza mínima para commit
_MIN_EVIDENCE = 2               # #señales mínimas para commit

_WITNESS_KEYS = {
    "opt": "opt_intervention",
    "plan": "plan_first_action",
    "ind": "ind_best_intervention",
    "abd": "abd_top_intervention",
    "heur": "heur_recommended_intervention",
    "imagination": "imagination_recommended_intervention",
}


def _idle() -> Dict[str, Any]:
    return {"family": FAMILY_ID, "status": "idle", "state_delta": {}, "confidence": 0.0, "cost": 0.0}


def decide(*, default: Optional[str], witnesses: Dict[str, Optional[str]],
           defeaters: Dict[str, bool], prob_point: float, prob_lcb: float) -> Dict[str, Any]:
    """Núcleo puro y determinista del decisor. Devuelve las claves `a12_*`."""
    active_defeaters = [name for name, present in defeaters.items() if present]
    default_defeated = len(active_defeaters) > 0

    # Candidata: mayor consenso entre testigos distintos del default.
    votes = Counter(iv for iv in witnesses.values() if iv and iv != default)
    candidate: Optional[str] = None
    candidate_support = 0
    if votes:
        candidate, candidate_support = votes.most_common(1)[0]

    # Bayes-factor (log-aditivo) para cambiar a la candidata.
    log_bf = 0.0
    if candidate is not None:
        for name in active_defeaters:
            log_bf += math.log(_LR.get(name, 1.5))
        log_bf += candidate_support * math.log(_LR_PER_WITNESS)
        if witnesses.get("imagination") == candidate and defeaters.get("imagination_breach"):
            log_bf += math.log(_LR_A11_ALIGNED)
        # Modulación por confianza de la traza (poco confiable ⇒ BF más débil).
        log_bf *= dc.clamp(0.5 + prob_point, 0.0, 1.5)
    bf = math.exp(log_bf) if candidate is not None else 1.0

    # ACT: commit si hay confianza calibrada + evidencia, si el BF es decisivo, o si
    # A11 CERTIFICA por horizonte (predijo el breach del default y recomienda la
    # candidata) — un rollout multi-paso determinista, no una conjetura probabilística.
    evidence_count = len(active_defeaters) + candidate_support
    foresight_certified = bool(
        defeaters.get("imagination_breach")
        and candidate is not None
        and witnesses.get("imagination") == candidate
    )
    confident = ((prob_lcb >= _LCB_FLOOR and evidence_count >= _MIN_EVIDENCE)
                 or (bf >= _BF_DECISIVE) or foresight_certified)
    act = "commit" if confident else "abstain"

    adopt = bool(default_defeated and candidate is not None and bf >= _TAU_BF and act == "commit")
    decision = candidate if adopt else default

    return {
        "a12_decision": decision,
        "a12_default_defeated": default_defeated,
        "a12_defeaters": active_defeaters,
        "a12_adopted_alternative": adopt,
        "a12_bayes_factor": round(bf, 4),
        "a12_log_bf": round(log_bf, 4),
        "a12_act": act,
        "a12_confidence": round(prob_lcb, 4),
        "a12_witnesses": {k: v for k, v in witnesses.items() if v},
    }


def execute(state):
    if not ci.family_deep_enabled(FAMILY_ID):
        return _idle()

    default = state.get("intervention")
    witnesses = {name: state.get(key) for name, key in _WITNESS_KEYS.items()}
    ctf = dc.safe_dict(state.get("ctf_checked"))
    cau = dc.safe_dict(state.get("cau_link"))
    defeaters = {
        "imagination_breach": state.get("imagination_chosen_breaches_at") is not None,
        "ctf_disagree": (ctf.get("supports_choice") is False)
                        or (ctf.get("agreement_with_relation_kind") is False),
        "cau_not_help": cau.get("helps_goal") is False,
        "ded_unsat": state.get("ded_status") == "unsat",
    }
    prob_point = dc.num(state.get("prob_point"), 0.5)
    prob_lcb = dc.num(state.get("prob_lcb"), 0.5)

    res = decide(default=default, witnesses=witnesses, defeaters=defeaters,
                 prob_point=prob_point, prob_lcb=prob_lcb)

    return {
        "family": FAMILY_ID,
        "status": "warn" if res["a12_default_defeated"] else "ok",
        "state_delta": res,
        "confidence": res["a12_confidence"],
        "cost": 0.5,
        "recommended_next_family": None,
        "failure_mode": None,
    }
