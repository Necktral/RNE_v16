FAMILY_ID = "FAL_GUARD"


def execute(state):
    features = state.get("_meta", {}).get("features", {})
    contradiction_signal = float(features.get("contradiction_signal", 0.0))
    uncertainty = float(features.get("uncertainty", 0.0))
    risk = min(1.0, (0.65 * contradiction_signal) + (0.35 * uncertainty))
    guard_clean = risk < 0.55
    status = "ok" if guard_clean else "warn"
    return {
        "family": FAMILY_ID,
        "status": status,
        "state_delta": {
            "fal_guard_clean": guard_clean,
            "fallacy_risk": risk,
        },
        "confidence": 0.68 if guard_clean else 0.52,
        "cost": 0.9,
    }
