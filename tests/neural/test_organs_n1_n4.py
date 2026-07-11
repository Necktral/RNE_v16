from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from runtime.neural import InferenceScope, NeuralInferenceRequest, NeuralModelManifest, ResourceSnapshot
from runtime.neural.organs import (
    CausalMessagePassingBackend,
    CausalPredictionAdmission,
    CompactMLPRouterBackend,
    FamilyRouterAdmission,
    Mamba2Backend,
    ReferenceTemporalSSMBackend,
    SharedRecursiveBackend,
    SymbolicVerificationAdmission,
    TemporalMemoryAdmission,
)


def _artifact(tmp_path: Path, organ: str, payload: dict, *, backend: str = "fixture"):
    path = tmp_path / f"{organ.lower()}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    manifest = NeuralModelManifest(
        organ=organ,
        capability={"N1": "family_routing", "N2": "recursive_reasoning", "N3": "temporal_memory", "N4": "causal_prediction"}[organ],
        model_id=f"{organ.lower()}-fixture",
        version="1",
        backend=backend,
        artifact_path=path.name,
        artifact_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        input_schema_version="1",
        output_schema_version="1",
        supported_devices=("cpu",),
        license_id="Unlicense",
        upstream_url="repo://rnfe/runtime/neural",
        upstream_commit="test",
        training_provenance={"dataset": "fixture", "seed": 1},
    )
    return path, manifest


def _request(organ: str, capability: str, payload: dict):
    return NeuralInferenceRequest(
        inference_id=f"inf-{organ}",
        run_id="run-organs",
        organ=organ,
        capability=capability,
        payload=payload,
        scope=InferenceScope.LAB,
        resources=ResourceSnapshot(),
    )


def test_n1_compact_mlp_obeys_catalog_and_hard_mask(tmp_path: Path) -> None:
    payload = {
        "feature_names": ["pressure", "uncertainty"],
        "family_catalog": ["HEUR", "DIA_ADV", "FAL_GUARD", "IND", "EML_SR", "PLAN", "OPT"],
        "w1": [[1.0, 0.0], [0.0, 1.0]],
        "b1": [0.0, 0.0],
        "w2": [[1.0, 0.0], [0.0, 1.0]],
        "b2": [0.0, 0.0],
        "utility_head": [[0.1 * (i + 1), 0.05] for i in range(7)],
        "probability_head": [[0.2, 0.1 * (i + 1)] for i in range(7)],
        "temperature": 1.0,
    }
    path, manifest = _artifact(tmp_path, "N1", payload)
    backend = CompactMLPRouterBackend()
    backend.load(manifest, str(path), "cpu")
    request = _request(
        "N1",
        "family_routing",
        {"features": {"pressure": 0.4, "uncertainty": 0.8}, "allowed_families": ["IND", "PLAN"], "max_optional_families": 1},
    )
    output = backend.infer(request)
    assert {item["family"] for item in output.candidate_output["ranked"] if item["allowed"]} == {"IND", "PLAN"}
    decision = FamilyRouterAdmission()(output.candidate_output, request)
    assert decision.accepted is True
    assert len(decision.output["optional_families"]) == 1
    assert decision.output["optional_families"][0] in {"IND", "PLAN"}


def test_n2_verifies_every_candidate_with_both_symbolic_authorities(tmp_path: Path) -> None:
    payload = {
        "hidden_size": 2,
        "input_size": 2,
        "w_state": [[0.2, 0.0], [0.0, 0.2]],
        "w_input": [[1.0, 0.0], [0.0, 1.0]],
        "bias": [0.0, 0.0],
        "candidate_weight": [0.7, 0.3],
        "halt_weight": [1.0, 1.0],
        "max_iterations": 4,
        "halt_threshold": 0.99,
    }
    path, manifest = _artifact(tmp_path, "N2", payload)
    backend = SharedRecursiveBackend()
    backend.load(manifest, str(path), "cpu")
    request = _request(
        "N2",
        "recursive_reasoning",
        {"state_vector": [0.5, 0.2], "candidate_templates": ["SAFE_A", "REJECT_B"]},
    )
    output = backend.infer(request)
    calls = []

    def ded(value, _request):
        calls.append(("DED", value))
        return value == "SAFE_A"

    def lotf(value, _request):
        calls.append(("LOTF", value))
        return value == "SAFE_A"

    decision = SymbolicVerificationAdmission(ded_verifier=ded, lotf_verifier=lotf)(output.candidate_output, request)
    assert len(calls) == 2 * len(output.candidate_output["candidates"])
    if decision.accepted:
        assert all(item["ded_verified"] and item["lotf_verified"] for item in decision.output["verified"])


def test_n3_state_is_namespaced_and_never_claims_mamba(tmp_path: Path) -> None:
    payload = {
        "state_size": 2,
        "input_size": 2,
        "a": [[0.5, 0.0], [0.0, 0.5]],
        "b": [[1.0, 0.0], [0.0, 1.0]],
        "c": [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [-0.5, 0.5], [0.8, 0.2]],
    }
    path, manifest = _artifact(tmp_path, "N3", payload)
    backend = ReferenceTemporalSSMBackend()
    backend.load(manifest, str(path), "cpu")
    request = _request(
        "N3",
        "temporal_memory",
        {"organism_id": "org-1", "scenario_id": "scenario-a", "input_vector": [0.4, 0.2]},
    )
    first = backend.infer(request)
    second = backend.infer(request)
    assert first.candidate_output["state_key"] == ["org-1", "scenario-a"]
    assert first.candidate_output != second.candidate_output
    assert TemporalMemoryAdmission()(second.candidate_output, request).accepted is True

    with pytest.raises(RuntimeError, match="vendor_commit"):
        Mamba2Backend().load(manifest, str(path), "cpu")


def test_n4_message_passing_returns_predictions_without_graph_mutation(tmp_path: Path) -> None:
    payload = {
        "input_size": 2,
        "hidden_size": 2,
        "input_weight": [[1.0, 0.0], [0.0, 1.0]],
        "message_weight": [[0.5, 0.0], [0.0, 0.5]],
        "update_weight": [[0.8, 0.0], [0.0, 0.8]],
        "output_weight": [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]],
    }
    path, manifest = _artifact(tmp_path, "N4", payload)
    backend = CausalMessagePassingBackend()
    backend.load(manifest, str(path), "cpu")
    request = _request(
        "N4",
        "causal_prediction",
        {
            "nodes": [{"id": "a", "features": [1.0, 0.0]}, {"id": "b", "features": [0.0, 1.0]}],
            "edges": [{"source": "a", "target": "b", "strength": 0.7}],
        },
    )
    output = backend.infer(request)
    assert set(output.candidate_output["predictions"]) == {"a", "b"}
    assert output.candidate_output["graph_mutations"] == []
    assert CausalPredictionAdmission()(output.candidate_output, request).accepted is True
