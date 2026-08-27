from __future__ import annotations

import importlib.util
import json
from pathlib import Path

try:
    import jsonschema
except ImportError:  # The runtime has no dependency on JSON Schema tooling.
    jsonschema = None


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "build_experiment_registry", ROOT / "scripts" / "build_experiment_registry.py"
)
assert SPEC and SPEC.loader
registry_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(registry_module)


def _layout(tmp_path: Path) -> Path:
    development = tmp_path / "Desarrollo"
    repo = development / "RNE_v16"
    for relative in (
        "rnfe_artifacts/integral_campaigns/campaign-complete",
        "rnfe_artifacts/integral_campaigns/campaign-stale",
        "rnfe_artifacts/integral_campaigns/campaign-invalid",
        "rnfe_artifacts/integral_campaigns/campaign-quarantined",
        "rnfe_artifacts/run-a",
        "data/artifacts/run-b",
        "work/family-c",
    ):
        (repo / relative).mkdir(parents=True, exist_ok=True)
    (development / "RNE_v16_worktrees" / "detached-a").mkdir(parents=True)
    (development / "RNE_v16_analysis").mkdir()
    (development / "RNE_v16_analysis" / "evidence.md").write_text("evidence\n", encoding="utf-8")
    return development


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_registry_is_deterministic_normalized_unique_and_schema_valid(tmp_path: Path) -> None:
    development = _layout(tmp_path)
    campaign = development / "RNE_v16/rnfe_artifacts/integral_campaigns/campaign-complete"
    _write_json(campaign / "verdict.json", {"status": "completed", "commit": "a" * 40})

    first = registry_module.build_registry(development, "2026-08-07")
    second = registry_module.build_registry(development, "2026-08-07")

    assert registry_module.canonical_json(first) == registry_module.canonical_json(second)
    assert len({entry["id"] for entry in first["entries"]}) == len(first["entries"])
    assert all(entry["logical_path"].startswith("development://") for entry in first["entries"])
    assert "/home/" not in registry_module.canonical_json(first)
    schema = json.loads((ROOT / "contracts/experiment-registry.v1.schema.json").read_text())
    assert schema["$schema"].endswith("draft/2020-12/schema")
    assert schema["properties"]["schema_version"]["const"] == first["schema_version"]
    if jsonschema is not None:
        jsonschema.Draft202012Validator(schema).validate(first)


def test_invalid_incomplete_stale_and_quarantine_classification(tmp_path: Path) -> None:
    development = _layout(tmp_path)
    campaigns = development / "RNE_v16/rnfe_artifacts/integral_campaigns"
    (campaigns / "campaign-invalid" / "manifest.json").write_text("{", encoding="utf-8")
    _write_json(campaigns / "campaign-stale" / "SUPERVISOR.json", {
        "status": "running", "heartbeat_at": "2026-08-01T00:00:00Z"
    })
    _write_json(campaigns / "campaign-quarantined" / "QUARANTINE.json", {
        "status": "quarantined", "reason": "dirty_worktree"
    })
    registry = registry_module.build_registry(development, "2026-08-07")
    statuses = {
        entry["id"]: entry["status"]
        for entry in registry["entries"]
        if entry["family"] == "integral_campaign"
    }
    assert statuses["integral_campaign:campaign-invalid"] == "invalid"
    assert statuses["integral_campaign:campaign-stale"] == "stale_running"
    assert statuses["integral_campaign:campaign-quarantined"] == "failed_closed"
    assert statuses["integral_campaign:campaign-complete"] == "discovered"


def test_sensitive_values_are_excluded(tmp_path: Path) -> None:
    development = _layout(tmp_path)
    manifest = development / "RNE_v16/rnfe_artifacts/run-a/manifest.json"
    _write_json(manifest, {
        "run_id": "safe-id",
        "status": "completed",
        "dsn": "postgresql://user:password@example.invalid/db",
        "password": "do-not-publish",
        "prompt": "private prompt",
        "context": {"payload": "private payload"},
        "gates": {"semantic_gate": True, "token": "private-token"},
    })
    rendered = registry_module.canonical_json(
        registry_module.build_registry(development, "2026-08-07")
    )
    assert "safe-id" in rendered
    for forbidden in ("do-not-publish", "private prompt", "private payload", "private-token", "postgresql://"):
        assert forbidden not in rendered


def test_check_mode_has_no_side_effects_and_detects_drift(tmp_path: Path) -> None:
    development = _layout(tmp_path)
    output = tmp_path / "registry.json"
    args = [
        "--development-root", str(development),
        "--output", str(output),
        "--snapshot-date", "2026-08-07",
    ]
    assert registry_module.main(args) == 0
    original = output.read_bytes()
    assert registry_module.main([*args, "--check"]) == 0
    assert output.read_bytes() == original
    output.write_text("{}\n", encoding="utf-8")
    changed = output.read_bytes()
    assert registry_module.main([*args, "--check"]) == 1
    assert output.read_bytes() == changed
