"""N1: enrutador compacto de familias como propuesta no autoritativa."""

from __future__ import annotations

import json
from typing import Any, Mapping

from ..contracts import AdmissionDecision, BackendOutput, NeuralInferenceRequest, NeuralModelManifest
from ._math import matrix, matvec, sigmoid, silu, softmax, vector


FAMILY_CATALOG_V1 = (
    "HEUR",
    "DIA_ADV",
    "FAL_GUARD",
    "IND",
    "EML_SR",
    "PLAN",
    "OPT",
)


class CompactMLPRouterBackend:
    """Inferencia MLP desde pesos JSON; no aprende ni descarga en runtime."""

    def __init__(self) -> None:
        self._weights: dict[str, Any] | None = None

    def load(self, manifest: NeuralModelManifest, artifact_path: str, device: str) -> None:
        if device != "cpu":
            raise RuntimeError("compact_mlp_reference_backend_is_cpu_only")
        if manifest.organ != "N1":
            raise ValueError("n1_manifest_required")
        with open(artifact_path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
        features = tuple(str(item) for item in raw["feature_names"])
        catalog = tuple(str(item).upper() for item in raw["family_catalog"])
        if catalog != FAMILY_CATALOG_V1:
            raise ValueError("family_catalog_version_mismatch")
        w1 = matrix(raw["w1"], columns=len(features), name="w1")
        b1 = vector(raw["b1"], size=len(w1), name="b1")
        w2 = matrix(raw["w2"], columns=len(w1), name="w2")
        b2 = vector(raw["b2"], size=len(w2), name="b2")
        utility = matrix(raw["utility_head"], columns=len(w2), name="utility_head")
        probability = matrix(raw["probability_head"], columns=len(w2), name="probability_head")
        if len(utility) != len(catalog) or len(probability) != len(catalog):
            raise ValueError("n1_head_catalog_size_mismatch")
        self._weights = {
            "feature_names": features,
            "catalog": catalog,
            "w1": w1,
            "b1": b1,
            "w2": w2,
            "b2": b2,
            "utility": utility,
            "probability": probability,
            "temperature": float(raw.get("temperature", 1.0)),
        }

    def infer(self, request: NeuralInferenceRequest) -> BackendOutput:
        if self._weights is None:
            raise RuntimeError("backend_not_loaded")
        raw_features = request.payload.get("features", {})
        if not isinstance(raw_features, Mapping):
            raise ValueError("n1_features_must_be_mapping")
        features = [float(raw_features.get(name, 0.0)) for name in self._weights["feature_names"]]
        hidden1 = [silu(value) for value in matvec(self._weights["w1"], features, self._weights["b1"])]
        hidden2 = [silu(value) for value in matvec(self._weights["w2"], hidden1, self._weights["b2"])]
        utilities = matvec(self._weights["utility"], hidden2)
        logits = matvec(self._weights["probability"], hidden2)
        probabilities = [sigmoid(value / max(self._weights["temperature"], 1e-6)) for value in logits]
        combined = [utility * probability for utility, probability in zip(utilities, probabilities)]
        normalized = softmax(combined)
        allowed = {str(item).upper() for item in request.payload.get("allowed_families", ())}
        entries = []
        for family, score, utility, probability in zip(
            self._weights["catalog"], normalized, utilities, probabilities
        ):
            entries.append(
                {
                    "family": family,
                    "score": score if family in allowed else 0.0,
                    "expected_utility": utility,
                    "probability_positive": probability,
                    "allowed": family in allowed,
                }
            )
        entries.sort(key=lambda item: (-item["score"], item["family"]))
        selected = [item["family"] for item in entries if item["allowed"] and item["score"] > 0.0]
        uncertainty = 1.0 - max((item["probability_positive"] for item in entries), default=0.0)
        return BackendOutput(
            candidate_output={
                "catalog_version": "n1-family-catalog-v1",
                "ranked": entries,
                "optional_families": selected,
            },
            confidence=1.0 - uncertainty,
            uncertainty=uncertainty,
            cost={"parameter_count_runtime": _parameter_count(self._weights)},
        )

    def unload(self) -> None:
        self._weights = None


class FamilyRouterAdmission:
    """Valida la propuesta N1 sin ejecutar ni reordenar el backbone."""

    def __call__(self, candidate: Any, request: NeuralInferenceRequest) -> AdmissionDecision:
        if not isinstance(candidate, Mapping):
            return AdmissionDecision(False, reason="n1_candidate_not_mapping")
        if candidate.get("catalog_version") != "n1-family-catalog-v1":
            return AdmissionDecision(False, reason="n1_catalog_version_mismatch")
        allowed = {str(item).upper() for item in request.payload.get("allowed_families", ())}
        proposed = [str(item).upper() for item in candidate.get("optional_families", ())]
        if any(item not in allowed or item not in FAMILY_CATALOG_V1 for item in proposed):
            return AdmissionDecision(False, reason="n1_hard_mask_violation")
        limit = max(0, int(request.payload.get("max_optional_families", 2)))
        return AdmissionDecision(
            True,
            output={"optional_families": proposed[:limit], "catalog_version": "n1-family-catalog-v1"},
            reason="n1_bounded_routing_proposal",
        )


def _parameter_count(weights: Mapping[str, Any]) -> int:
    matrices = ("w1", "w2", "utility", "probability")
    biases = ("b1", "b2")
    return sum(len(row) for name in matrices for row in weights[name]) + sum(
        len(weights[name]) for name in biases
    )
