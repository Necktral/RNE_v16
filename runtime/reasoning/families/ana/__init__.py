FAMILY_ID = "ANA"


def execute(state):
    return {
        "family": FAMILY_ID,
        "status": "ok",
        "state_delta": {"ana_mapping": True},
        "confidence": 0.6,
        "cost": 1.0,
    }
