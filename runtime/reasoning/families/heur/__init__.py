FAMILY_ID = "HEUR"


def execute(state):
    features = state.get("_meta", {}).get("features", {})
    edge_pressure = float(features.get("edge_pressure", 0.0))
    uncertainty = float(features.get("uncertainty", 0.0))
    triage_fast = edge_pressure >= 0.7
    state_delta = {
        "heur_triage_fast": triage_fast,
        "heur_uncertainty_hint": uncertainty,
    }
    return {
        "family": FAMILY_ID,
        "status": "ok",
        "state_delta": state_delta,
        "confidence": 0.7 if triage_fast else 0.6,
        "cost": 0.6,
    }
