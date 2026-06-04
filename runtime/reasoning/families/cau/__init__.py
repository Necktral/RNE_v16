FAMILY_ID = "CAU"


def execute(state):
    return {
        "family": FAMILY_ID,
        "status": "ok",
        "state_delta": {"cau_link": True},
        "confidence": 0.62,
        "cost": 1.0,
    }
