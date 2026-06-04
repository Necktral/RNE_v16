FAMILY_ID = "ABD"


def execute(state):
    return {
        "family": FAMILY_ID,
        "status": "ok",
        "state_delta": {"abd_hypothesis": True},
        "confidence": 0.6,
        "cost": 1.0,
    }
