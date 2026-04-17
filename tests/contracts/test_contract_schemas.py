import json
from pathlib import Path


CONTRACT_FILES = [
    "episode.schema.json",
    "proposal.schema.json",
    "certificate.schema.json",
    "rollback.schema.json",
    "telemetry_snapshot.schema.json",
    "event.schema.json",
    "tool_request.schema.json",
    "tool_result.schema.json",
    "session_bridge.schema.json",
    "reasoning_trace.schema.json",
    "artifact_index.schema.json",
    "reality_assessment.schema.json",
]


def test_contract_schemas_exist_and_parse():
    base = Path(__file__).resolve().parents[2] / "contracts"
    for name in CONTRACT_FILES:
        path = base / name
        assert path.exists(), f"Missing contract schema: {name}"
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["type"] == "object"
        assert "required" in payload and payload["required"], f"{name} must define required fields"
