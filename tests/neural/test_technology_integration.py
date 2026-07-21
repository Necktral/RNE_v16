from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from runtime.neural.config import NeuralRuntimeConfig
from runtime.neural.contracts import NeuralMode, NeuralModelManifest
from runtime.neural.integration import SymbiosisIdentity, SymbioticNeuralCoordinator
from runtime.neural.technology_backends import (
    MAMBA2_ARTIFACT_SCHEMA,
    MAMBA2_BACKEND_ID,
    MAMBA_UPSTREAM_COMMIT,
    N3_FEATURE_NAMES,
    _compact_mamba2,
)
from runtime.storage import StorageConfig, StorageFactory
from runtime.world import ScenarioEpisodeRunner


def _storage(tmp_path: Path):
    return StorageFactory.create_facade(
        StorageConfig(
            mode="sqlite",
            sqlite_db_path=str(tmp_path / "technology.db"),
            postgres_dsn=None,
            artifact_root=tmp_path / "storage-artifacts",
            prefer_postgres_reads=False,
            strict_dual_write=False,
        )
    )


def _identity(name: str = "technology") -> SymbiosisIdentity:
    return SymbiosisIdentity(
        trace_group_id=f"trace-{name}",
        organism_id="organism-technology",
        lineage_id="lineage-technology",
        run_id="run-technology",
        episode_id=f"episode-{name}",
        scenario_id="thermal@1",
        decision_id=f"decision-{name}",
    )


def _write_mamba_artifact(root: Path, *, corrupt_hash: bool = False) -> NeuralModelManifest:
    torch = pytest.importorskip("torch")
    target = root / "n3"
    target.mkdir(parents=True)
    config = {
        "input_size": len(N3_FEATURE_NAMES),
        "d_model": 32,
        "d_state": 8,
        "nheads": 2,
        "headdim": 16,
        "block_len": 4,
        "history_size": 8,
        "output_size": 5,
    }
    model = _compact_mamba2(torch, config)
    artifact = target / "model.pt"
    torch.save(
        {
            "artifact_schema_version": MAMBA2_ARTIFACT_SCHEMA,
            "classification": "trained",
            "trained": True,
            "model_config": config,
            "state_dict": model.state_dict(),
            "training_evidence": {"dataset_sha256": "a" * 64, "seed": 7},
        },
        artifact,
    )
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    manifest = NeuralModelManifest(
        organ="N3",
        capability="temporal_reference_state",
        model_id="mamba2-test",
        version="test",
        backend=MAMBA2_BACKEND_ID,
        artifact_path="n3/model.pt",
        artifact_sha256="0" * 64 if corrupt_hash else digest,
        input_schema_version="n3-vitals-sequence-v1",
        output_schema_version="n3-temporal-proposal-v1",
        supported_devices=("cpu",),
        parameter_count=sum(item.numel() for item in model.parameters()),
        peak_vram_gb=0.0,
        license_id="Apache-2.0",
        upstream_url="https://github.com/state-spaces/mamba",
        upstream_commit=MAMBA_UPSTREAM_COMMIT,
        training_provenance={"dataset_sha256": "a" * 64, "seed": 7},
    )
    (target / "manifest.json").write_text(
        json.dumps(manifest.to_dict()), encoding="utf-8"
    )
    return manifest


def _begin(coordinator: SymbioticNeuralCoordinator, identity: SymbiosisIdentity) -> dict:
    return coordinator.begin_episode(
        identity=identity,
        observation={"temperature": 0.8, "alarm": True},
        formula="temperature > 0.5",
        proposition="temperature high",
        memory_hits=[],
        scenario_metadata={"main_variable": "temperature"},
        causal_attestation={
            "main_variable": "temperature",
            "factual_delta": -0.2,
            "counterfactual_delta": 0.1,
        },
        resources={"gpu_available": False},
    )


def test_off_does_not_open_configured_manifest(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("RNFE_ARTIFACT_ROOT", str(tmp_path / "missing-root"))
    monkeypatch.setenv("RNFE_NEURAL_N3_MANIFEST", "missing/manifest.json")
    storage = _storage(tmp_path)
    coordinator = SymbioticNeuralCoordinator(
        storage=storage, config=NeuralRuntimeConfig(mode=NeuralMode.OFF)
    )
    signals = _begin(coordinator, _identity("off"))
    assert signals["n3_temporal"]["status"] == "disabled"
    assert coordinator.runtime.registry.loaded_count == 0
    storage.close()


def test_relative_artifact_root_is_canonicalized(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RNFE_ARTIFACT_ROOT", "relative-artifacts")
    storage = _storage(tmp_path)
    coordinator = SymbioticNeuralCoordinator(
        storage=storage, config=NeuralRuntimeConfig(mode=NeuralMode.SHADOW)
    )
    assert coordinator.runtime.registry.artifact_root == (
        tmp_path / "relative-artifacts" / "neural"
    ).resolve()
    storage.close()


def test_trained_mamba2_ssd_is_shadow_and_connectomic(tmp_path: Path, monkeypatch) -> None:
    artifact_root = tmp_path / "artifacts" / "neural"
    manifest = _write_mamba_artifact(artifact_root)
    monkeypatch.setenv("RNFE_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("RNFE_NEURAL_N3_MANIFEST", "n3/manifest.json")
    storage = _storage(tmp_path)
    coordinator = SymbioticNeuralCoordinator(
        storage=storage, config=NeuralRuntimeConfig(mode=NeuralMode.PROVISIONAL)
    )
    identity = _identity("trained")
    signals = _begin(coordinator, identity)
    candidate = signals["n3_temporal"]
    assert candidate["backend"] == MAMBA2_BACKEND_ID
    assert candidate["mamba2_active"] is True
    assert candidate["authority_effect"] == "none"
    block = coordinator.certification_block(identity.episode_id)
    n3 = next(item for item in block["candidates"] if item["organ"] == "N3")
    assert n3["effective_mode"] == "shadow"
    assert n3["authority_ceiling"] == "shadow"
    assert n3["manifest_sha256"] == manifest.manifest_sha256
    assert n3["artifact_sha256"] == manifest.artifact_sha256
    assert not any(item["organ"] == "N3" for item in block["fallbacks"])
    assert block["connectome_activity"]["authority_effect"] == "none"
    storage.close()


def test_corrupt_model_hash_fails_closed_without_consuming_reference(
    tmp_path: Path, monkeypatch
) -> None:
    artifact_root = tmp_path / "artifacts" / "neural"
    _write_mamba_artifact(artifact_root, corrupt_hash=True)
    monkeypatch.setenv("RNFE_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("RNFE_NEURAL_N3_MANIFEST", "n3/manifest.json")
    storage = _storage(tmp_path)
    coordinator = SymbioticNeuralCoordinator(
        storage=storage, config=NeuralRuntimeConfig(mode=NeuralMode.SHADOW)
    )
    identity = _identity("corrupt")
    signals = _begin(coordinator, identity)
    assert signals["n3_temporal"] == {
        "status": "disabled",
        "state_key": ["organism-technology", "thermal@1", "lineage-technology"],
    }
    n3 = coordinator._session(identity.episode_id).entries["N3"]
    assert n3.candidate is None
    assert n3.candidate_hash is None
    assert n3.consumer_verdict.startswith("not_consumed:fallback:")
    assert "artifact_sha256_mismatch" in (n3.fallback_reason or "")
    assert coordinator.export_temporal_state()["entries"] == []
    block = coordinator.certification_block(identity.episode_id)
    assert any(
        item["organ"] == "N3" and "artifact_sha256_mismatch" in item["reason"]
        for item in block["fallbacks"]
    )
    assert coordinator.runtime.registry.loaded_count == 0
    storage.close()


def test_n6_consumes_connectomic_plasticity_without_self_reinforcement(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("RNFE_NEURAL_MODE", "shadow")
    for name in ("N1", "N3", "N4", "N5"):
        monkeypatch.delenv(f"RNFE_NEURAL_{name}_MANIFEST", raising=False)
    storage = _storage(tmp_path)
    runner = ScenarioEpisodeRunner(
        storage=storage, run_id="run-n6-connectome", scenario="thermal_homeostasis"
    )
    result = None
    for external_input in (0.04, 0.08, 0.12):
        result = runner.run_episode(external_input=external_input)
    assert result is not None
    n6 = next(
        item for item in result["neural_symbiosis_trace"]["organs"]
        if item["organ"] == "N6"
    )["candidate"]
    proposal = n6["proposal"]
    assert proposal["target"].startswith("connectome:")
    assert not proposal["target"].startswith("connectome:N6->")
    assert proposal["apply_authorized"] is False
    assert n6["sandbox"]["applied"] is False
    assert n6["connectome_evidence"]["graph_mutated"] is False
    storage.close()
