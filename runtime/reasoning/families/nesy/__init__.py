FAMILY_ID = "NESY"


def execute(state):
    return {
        "family": FAMILY_ID,
        "status": "idle",
        "state_delta": {},
        "confidence": 0.0,
        "cost": 0.0,
    }
