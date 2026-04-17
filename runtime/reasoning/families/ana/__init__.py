FAMILY_ID = "ANA"


def execute(state):
    return {"family": FAMILY_ID, "status": "ok", "state_delta": {"ana_mapping": True}}

