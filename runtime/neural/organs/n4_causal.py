"""N4 causal: ranking multitarea puro Python de hipótesis estructurales."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from ..contracts import BackendOutput, NeuralInferenceRequest


_FEATURES = (
    "empirical_gain",
    "holdout_support",
    "coverage",
    "simplicity",
    "stability",
    "invariant_safety",
)


class N4CausalRankingBackend:
    """Inferencia determinista con pesos exportados; no muta ni decide acciones."""

    def __init__(self, weights: Mapping[str, Any] | None = None) -> None:
        payload = dict(weights or {})
        self.weights = tuple(
            float(item)
            for item in payload.get("ranking_weights", (2.0, 1.5, 0.5, 0.4, 1.0, 1.5))
        )
        self.bias = float(payload.get("bias", -1.0))
        self.temperature = max(float(payload.get("temperature", 1.0)), 1e-6)
        if len(self.weights) != len(_FEATURES):
            raise ValueError("n4_ranking_weight_shape_invalid")
        if not all(math.isfinite(item) for item in (*self.weights, self.bias)):
            raise ValueError("n4_weights_must_be_finite")

    def infer(self, request: NeuralInferenceRequest) -> BackendOutput:
        candidates = request.payload.get("candidates", ())
        if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
            raise ValueError("n4_candidates_must_be_sequence")
        ranked: list[dict[str, Any]] = []
        for raw in candidates:
            if not isinstance(raw, Mapping):
                raise ValueError("n4_candidate_must_be_mapping")
            features = tuple(float(raw.get(name, 0.0)) for name in _FEATURES)
            if not all(math.isfinite(item) for item in features):
                raise ValueError("n4_feature_must_be_finite")
            logit = (self.bias + sum(a * b for a, b in zip(self.weights, features)))
            probability = 1.0 / (1.0 + math.exp(-logit / self.temperature))
            risk = min(1.0, max(0.0, 1.0 - features[-1]))
            uncertainty = min(1.0, max(0.0, 1.0 - abs(probability - 0.5) * 2.0))
            priority = probability * (1.0 - risk) * (1.0 - 0.5 * uncertainty)
            ranked.append(
                {
                    "hypothesis_id": str(raw["hypothesis_id"]),
                    "probability_valid": round(probability, 9),
                    "expected_mae_gain": round(features[0], 9),
                    "uncertainty": round(uncertainty, 9),
                    "invariant_risk": round(risk, 9),
                    "priority": round(priority, 9),
                }
            )
        ranked.sort(key=lambda item: (-item["priority"], item["hypothesis_id"]))
        confidence = (
            sum(item["probability_valid"] for item in ranked) / len(ranked)
            if ranked
            else 0.0
        )
        uncertainty = (
            sum(item["uncertainty"] for item in ranked) / len(ranked)
            if ranked
            else 1.0
        )
        return BackendOutput(
            candidate_output={"schema": "n4-ranking.v1", "rankings": ranked},
            confidence=confidence,
            uncertainty=uncertainty,
            cost={"candidates": len(ranked), "feature_count": len(_FEATURES)},
        )
