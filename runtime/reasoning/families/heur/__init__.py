"""Familia HEUR (heurística) — triage rápido sobre el estado acumulado.

OFF (clásico, byte-idéntico): dos flags derivados de edge_pressure/uncertainty.
DEEP (opt-in, RNFE_HEUR_DEEP / RNFE_REASONING_DEEP): capa heurística real que
sintetiza una recomendación por **voto de consenso** entre las intervenciones ya
propuestas por el core (ABD/OPT/PLAN/IND) y enruta el nivel de triage según
presión/incertidumbre. Determinista.
"""

from runtime.reasoning.families import core_inference as ci
from runtime.reasoning.families import _deep_common as dc

FAMILY_ID = "HEUR"

_NEXT_BY_TRIAGE = {"fast": "PROB", "deliberate": "CTF", "normal": "PROB"}


def execute(state):
    features = state.get("_meta", {}).get("features", {})
    edge_pressure = float(features.get("edge_pressure", 0.0))
    uncertainty = float(features.get("uncertainty", 0.0))
    triage_fast = edge_pressure >= 0.7

    if not ci.family_deep_enabled(FAMILY_ID):
        return {
            "family": FAMILY_ID,
            "status": "ok",
            "state_delta": {
                "heur_triage_fast": triage_fast,
                "heur_uncertainty_hint": uncertainty,
            },
            "confidence": 0.7 if triage_fast else 0.6,
            "cost": 0.6,
        }

    contradiction = dc.num(features.get("contradiction_signal"))
    ambiguity = dc.num(features.get("ambiguity_signal"))
    hardware = dc.num(features.get("hardware_pressure"))

    # Consenso entre las recomendaciones ya calculadas por las familias core.
    sources = {
        "abd": state.get("abd_top_intervention"),
        "opt": state.get("opt_intervention"),
        "plan": state.get("plan_first_action"),
        "ind": state.get("ind_best_intervention"),
        "chosen": state.get("intervention"),
    }
    present = [v for v in sources.values() if v]
    votes = {}
    for v in present:
        votes[v] = votes.get(v, 0) + 1
    recommended = max(votes, key=votes.get) if votes else state.get("intervention")
    agreement = (votes.get(recommended, 0) / len(present)) if present else 0.0

    if edge_pressure >= 0.7 or hardware >= 0.85:
        triage = "fast"
    elif uncertainty >= 0.6 or contradiction >= 0.5 or ambiguity >= 0.6:
        triage = "deliberate"
    else:
        triage = "normal"

    confidence = dc.clamp(0.5 + 0.4 * agreement - 0.2 * uncertainty)
    return {
        "family": FAMILY_ID,
        "status": "ok",
        "state_delta": {
            # claves clásicas preservadas (consumidores existentes intactos)
            "heur_triage_fast": triage_fast,
            "heur_uncertainty_hint": uncertainty,
            # capa profunda
            "heur_triage_level": triage,
            "heur_recommended_intervention": recommended,
            "heur_source_agreement": round(agreement, 4),
        },
        "confidence": round(confidence, 4),
        "cost": 0.6,
        "recommended_next_family": _NEXT_BY_TRIAGE[triage],
        "artifacts": {
            "vote_table": votes,
            "triage_inputs": {
                "edge_pressure": edge_pressure,
                "uncertainty": uncertainty,
                "contradiction": contradiction,
                "hardware_pressure": hardware,
            },
        },
    }
