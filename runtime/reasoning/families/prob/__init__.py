FAMILY_ID = "PROB"


def execute(state):
    return {"family": FAMILY_ID, "status": "ok", "state_delta": {"prob_calibrated": True}}

