"""Familia FAL_GUARD (guardia de falacias/fragilidad).

OFF (clásico, byte-idéntico): riesgo = suma ponderada de contradiction/uncertainty.
DEEP (opt-in, RNFE_FAL_GUARD_DEEP / RNFE_REASONING_DEEP): detector real de
falacias/incoherencias que inspecciona el estado de razonamiento ACUMULADO
(cau_link, ctf_checked, abd, ded, prob) y enumera defectos concretos con
severidad, en vez de un escalar opaco. Determinista.
"""

from runtime.reasoning.families import core_inference as ci
from runtime.reasoning.families import _deep_common as dc

FAMILY_ID = "FAL_GUARD"

_FALLACY_WEIGHTS = {
    "acting_against_causal_evidence": 0.90,
    "deductive_gap": 0.85,
    "counterfactual_disagreement": 0.80,
    "overconfidence": 0.70,
    "causal_direction_mismatch": 0.60,
    "hasty_abduction": 0.50,
}


def execute(state):
    features = state.get("_meta", {}).get("features", {})
    contradiction_signal = float(features.get("contradiction_signal", 0.0))
    uncertainty = float(features.get("uncertainty", 0.0))
    risk = min(1.0, (0.65 * contradiction_signal) + (0.35 * uncertainty))
    guard_clean = risk < 0.55
    status = "ok" if guard_clean else "warn"

    if not ci.family_deep_enabled(FAMILY_ID):
        return {
            "family": FAMILY_ID,
            "status": status,
            "state_delta": {"fal_guard_clean": guard_clean, "fallacy_risk": risk},
            "confidence": 0.68 if guard_clean else 0.52,
            "cost": 0.9,
        }

    cau = dc.safe_dict(state.get("cau_link"))
    ctf = dc.safe_dict(state.get("ctf_checked"))
    prob = dc.safe_dict(state.get("prob_posterior"))
    abd_list = state.get("abd_hypotheses")
    abd_list = abd_list if isinstance(abd_list, list) else []

    fallacies = []
    if cau.get("helps_goal") is False:
        fallacies.append("acting_against_causal_evidence")
    if state.get("ded_conclusion") and not state.get("ded_validated"):
        fallacies.append("deductive_gap")
    if ctf.get("agreement_with_relation_kind") is False or ctf.get("resim_supports_choice") is False:
        fallacies.append("counterfactual_disagreement")
    point = dc.num(prob.get("point"))
    lcb = dc.num(prob.get("lower_confidence_bound"))
    if point >= 0.8 and lcb < 0.5:
        fallacies.append("overconfidence")
    if cau.get("direction_match") is False and cau.get("expected_direction") and cau.get("observed_direction"):
        fallacies.append("causal_direction_mismatch")
    if len(abd_list) >= 2:
        margin = dc.num(abd_list[0].get("score")) - dc.num(abd_list[1].get("score"))
        if margin < 0.10:
            fallacies.append("hasty_abduction")

    severity = sum(_FALLACY_WEIGHTS.get(f, 0.4) for f in fallacies)
    deep_risk = dc.clamp(
        0.40 * min(1.0, severity / 2.0)
        + 0.35 * contradiction_signal
        + 0.25 * uncertainty
    )
    clean = (deep_risk < 0.55) and not fallacies
    status = "ok" if clean else "warn"
    confidence = dc.clamp(0.72 - 0.11 * len(fallacies))
    return {
        "family": FAMILY_ID,
        "status": status,
        "state_delta": {
            # claves clásicas preservadas
            "fal_guard_clean": clean,
            "fallacy_risk": round(deep_risk, 4),
            # capa profunda
            "fallacies": fallacies,
            "fallacy_count": len(fallacies),
        },
        "confidence": round(confidence, 4),
        "cost": 0.9,
        "failure_mode": None if clean else "reasoning_fallacies_detected",
        "artifacts": {
            "detected_fallacies": fallacies,
            "severity_score": round(severity, 4),
            "context_signals": {
                "contradiction": contradiction_signal,
                "uncertainty": uncertainty,
            },
        },
    }
