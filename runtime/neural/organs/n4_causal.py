"""N4: message passing causal de referencia sin autoridad sobre el grafo."""

from __future__ import annotations

import json
from typing import Any, Mapping

from ..contracts import AdmissionDecision, BackendOutput, NeuralInferenceRequest, NeuralModelManifest
from ._math import matrix, matvec, sigmoid, tanh_vector, vector


class CausalMessagePassingBackend:
    def __init__(self) -> None:
        self._weights: dict[str, Any] | None = None

    def load(self, manifest: NeuralModelManifest, artifact_path: str, device: str) -> None:
        if device != "cpu" or manifest.organ != "N4":
            raise ValueError("n4_reference_backend_requires_cpu_n4_manifest")
        with open(artifact_path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
        input_size = int(raw["input_size"])
        hidden_size = int(raw["hidden_size"])
        self._weights = {
            "input_size": input_size,
            "hidden_size": hidden_size,
            "input": matrix(raw["input_weight"], columns=input_size, name="input_weight"),
            "message": matrix(raw["message_weight"], columns=hidden_size, name="message_weight"),
            "update": matrix(raw["update_weight"], columns=hidden_size, name="update_weight"),
            "output": matrix(raw["output_weight"], columns=hidden_size, name="output_weight"),
        }
        if any(len(self._weights[name]) != hidden_size for name in ("input", "message", "update")):
            raise ValueError("n4_hidden_shape_mismatch")

    def infer(self, request: NeuralInferenceRequest) -> BackendOutput:
        if self._weights is None:
            raise RuntimeError("backend_not_loaded")
        raw_nodes = request.payload.get("nodes", ())
        raw_edges = request.payload.get("edges", ())
        nodes: dict[str, list[float]] = {}
        for item in raw_nodes:
            node_id = str(item["id"])
            nodes[node_id] = tanh_vector(
                matvec(
                    self._weights["input"],
                    vector(item["features"], size=self._weights["input_size"]),
                )
            )
        if not nodes:
            raise ValueError("n4_graph_requires_nodes")
        for _ in range(3):
            incoming = {node_id: [0.0] * self._weights["hidden_size"] for node_id in nodes}
            for edge in raw_edges:
                source, target = str(edge["source"]), str(edge["target"])
                if source not in nodes or target not in nodes:
                    raise ValueError("n4_edge_references_unknown_node")
                strength = float(edge.get("strength", 1.0))
                message = matvec(self._weights["message"], nodes[source])
                incoming[target] = [a + strength * b for a, b in zip(incoming[target], message)]
            nodes = {
                node_id: tanh_vector(
                    [a + b for a, b in zip(matvec(self._weights["update"], state), incoming[node_id])]
                )
                for node_id, state in nodes.items()
            }
        predictions = {}
        for node_id, state in nodes.items():
            values = matvec(self._weights["output"], state)
            predictions[node_id] = {
                "next_state": sigmoid(values[0]),
                "intervention_effect": sigmoid(values[1]) if len(values) > 1 else 0.5,
                "edge_confidence": sigmoid(values[2]) if len(values) > 2 else 0.5,
                "uncertainty": 1.0 - sigmoid(values[2]) if len(values) > 2 else 0.5,
            }
        confidence = sum(item["edge_confidence"] for item in predictions.values()) / len(predictions)
        return BackendOutput(
            candidate_output={"predictions": predictions, "graph_mutations": []},
            confidence=confidence,
            uncertainty=1.0 - confidence,
            cost={"nodes": len(nodes), "edges": len(raw_edges), "layers": 3},
        )

    def unload(self) -> None:
        self._weights = None


class CausalPredictionAdmission:
    def __call__(self, candidate: Any, request: NeuralInferenceRequest) -> AdmissionDecision:
        if not isinstance(candidate, Mapping) or not isinstance(candidate.get("predictions"), Mapping):
            return AdmissionDecision(False, reason="n4_prediction_schema_invalid")
        if candidate.get("graph_mutations"):
            return AdmissionDecision(False, reason="n4_cannot_mutate_canonical_graph")
        return AdmissionDecision(
            True,
            output={"predictions": candidate["predictions"], "authority": "CAU+CTF+CGWM"},
            reason="n4_prediction_only",
        )
