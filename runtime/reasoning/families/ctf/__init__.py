FAMILY_ID = "CTF"


def execute(state):
    return {"family": FAMILY_ID, "status": "ok", "state_delta": {"ctf_checked": True}}

