"""Familia DIA_ADV (dialéctica adversarial).

OFF (clásico, byte-idéntico): un flag desde contradiction_signal.
DEEP (opt-in, RNFE_DIA_ADV_DEEP / RNFE_REASONING_DEEP): desafío dialéctico real
sobre el estado acumulado — construye la **antítesis** más fuerte a la
intervención elegida (tesis), enumera **objeciones** desde la evidencia
(causal/contrafactual/dominancia por modelo de efectos) y sintetiza si la tesis
sobrevive. Determinista.

Nota: `cognitive_self_challenge.py` de este paquete es legacy AEON (numpy/random,
muta pesos) y NO participa del contrato de familia (no es determinista). Su único
consumidor es el training loop legacy (`runtime/legacy/training_loop.py`), cableado
tras guard en la reparación B9/B13.
"""

from runtime.reasoning.families import core_inference as ci
from runtime.reasoning.families import _deep_common as dc

FAMILY_ID = "DIA_ADV"


def execute(state):
    features = state.get("_meta", {}).get("features", {})
    contradiction_signal = float(features.get("contradiction_signal", 0.0))
    challenge_active = contradiction_signal >= 0.45

    if not ci.family_deep_enabled(FAMILY_ID):
        return {
            "family": FAMILY_ID,
            "status": "ok",
            "state_delta": {
                "adversarial_challenge_active": challenge_active,
                "adversarial_pressure": contradiction_signal,
            },
            "confidence": 0.72 if challenge_active else 0.58,
            "cost": 1.1,
        }

    thesis = state.get("intervention") or state.get("abd_top_intervention")
    cau = dc.safe_dict(state.get("cau_link"))
    ctf = dc.safe_dict(state.get("ctf_checked"))
    resim = dc.safe_dict(ctf.get("resim"))

    # Antítesis: la mejor alternativa a la tesis según la evidencia disponible.
    antithesis = None
    if resim.get("best_alternative") and resim.get("best_alternative") != thesis:
        antithesis = resim.get("best_alternative")
    else:
        abd_list = state.get("abd_hypotheses")
        for h in (abd_list if isinstance(abd_list, list) else []):
            iv = h.get("intervention") if isinstance(h, dict) else None
            if iv and iv != thesis:
                antithesis = iv
                break

    # Objeciones: evidencia acumulada en contra de la tesis.
    objections = []
    if cau.get("helps_goal") is False:
        objections.append("causal_evidence_against_thesis")
    if ctf.get("supports_choice") is False or ctf.get("resim_supports_choice") is False:
        objections.append("counterfactual_prefers_alternative")
    dom = resim.get("dominance_margin")
    if isinstance(dom, (int, float)) and dom < 0:
        objections.append("alternative_dominates_by_effect_model")
    if cau.get("direction_match") is False and cau.get("expected_direction"):
        objections.append("causal_direction_unconfirmed")

    pressure = dc.clamp(max(contradiction_signal, 0.25 * len(objections)))
    thesis_survives = len(objections) == 0
    challenge_active_deep = pressure >= 0.45 or bool(objections)
    confidence = 0.75 if thesis_survives else dc.clamp(0.7 - 0.12 * len(objections))
    return {
        "family": FAMILY_ID,
        "status": "ok",
        "state_delta": {
            # claves clásicas preservadas
            "adversarial_challenge_active": challenge_active_deep,
            "adversarial_pressure": round(pressure, 4),
            # capa profunda
            "dialectic_thesis": thesis,
            "dialectic_antithesis": antithesis,
            "dialectic_objections": objections,
            "dialectic_thesis_survives": thesis_survives,
        },
        "confidence": round(confidence, 4),
        "cost": 1.1,
        "recommended_next_family": "PROB" if thesis_survives else "CTF",
        "artifacts": {
            "thesis": thesis,
            "antithesis": antithesis,
            "objections": objections,
        },
    }
