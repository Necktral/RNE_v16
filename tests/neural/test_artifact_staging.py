from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from runtime.neural.contracts import NeuralModelManifest
from scripts.stage_neural_lab_artifacts import stage_lab_artifacts


def _source(root: Path, *, promotion_eligible: bool = False) -> None:
    target = root / "n1"
    target.mkdir(parents=True)
    artifact = target / "model.json"
    artifact.write_text('{"trained": true}', encoding="utf-8")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    manifest = NeuralModelManifest(
        organ="N1",
        capability="family_routing_proposal",
        model_id="n1-stage-test",
        version="lab",
        backend="rnfe-compact-mlp-router-v1",
        artifact_path="n1/model.json",
        artifact_sha256=digest,
        input_schema_version="n1-context-features-v1",
        output_schema_version="n1-routing-proposal-v2",
        supported_devices=("cpu",),
        parameter_count=1,
        peak_vram_gb=0.0,
        license_id="Unlicense",
        upstream_url="repo://test",
        upstream_commit="test",
        training_provenance={"promotion_eligible": promotion_eligible, "seed": 1},
    )
    (target / "manifest.json").write_text(
        json.dumps(manifest.to_dict()), encoding="utf-8"
    )
    (target / "model_card.json").write_text("{}", encoding="utf-8")


def test_stage_lab_artifacts_is_shadow_only_and_hash_preserving(tmp_path) -> None:
    source, target = tmp_path / "source", tmp_path / "target"
    _source(source)

    profile = stage_lab_artifacts(
        source_root=source, target_root=target, organs=("N1", "N5")
    )

    assert [item["organ"] for item in profile["staged"]] == ["N1"]
    assert profile["missing"] == ["N5"]
    assert profile["activation_automatic"] is False
    assert profile["training_authorized"] is False
    assert (target / "n1/model.json").read_bytes() == (
        source / "n1/model.json"
    ).read_bytes()
    assert json.loads((target / "activation_profile.json").read_text())["staged"][0][
        "environment"
    ] == {"RNFE_NEURAL_N1_MANIFEST": "n1/manifest.json"}


def test_stage_lab_artifacts_rejects_promotable_input(tmp_path) -> None:
    source = tmp_path / "source"
    _source(source, promotion_eligible=True)

    with pytest.raises(ValueError, match="must_be_non_promotable:N1"):
        stage_lab_artifacts(
            source_root=source, target_root=tmp_path / "target", organs=("N1",)
        )
