FAMILY_ID = "DIA_ADV"


def execute(state):
    features = state.get("_meta", {}).get("features", {})
    contradiction_signal = float(features.get("contradiction_signal", 0.0))
    challenge_active = contradiction_signal >= 0.45
    state_delta = {
        "adversarial_challenge_active": challenge_active,
        "adversarial_pressure": contradiction_signal,
    }
    return {
        "family": FAMILY_ID,
        "status": "ok",
        "state_delta": state_delta,
        "confidence": 0.72 if challenge_active else 0.58,
        "cost": 1.1,
    }
