"""N4 causal: ranking multitarea puro Python de hipótesis estructurales."""

from __future__ import annotations

import hashlib
import json
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

    def __init__(
        self,
        weights: Mapping[str, Any] | None = None,
        *,
        artifact_sha256: str | None = None,
    ) -> None:
        payload = dict(weights or {})
        schema = payload.get("schema")
        if schema not in {None, "n4-ranking-artifact.v1", "n4-ranking-artifact.v2"}:
            raise ValueError("n4_artifact_schema_invalid")
        self.schema = str(schema or "n4-ranking-artifact.v1")
        self._v2 = self.schema == "n4-ranking-artifact.v2"
        self.payload = payload
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
        self.artifact_sha256 = artifact_sha256 or hashlib.sha256(
            json.dumps(
                payload,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        self.ranking_objective = str(
            payload.get(
                "ranking_objective",
                "reference_priority_v1",
            )
        )
        if self._v2:
            if payload.get("model_kind") != "trained_multihead":
                raise ValueError("n4_v2_artifact_must_be_trained_multihead")
            if tuple(payload.get("feature_order", ())) != _FEATURES:
                raise ValueError("n4_v2_feature_order_invalid")
            transform = payload.get("feature_transform") or {}
            if transform.get("kind") != "identity":
                raise ValueError("n4_v2_feature_transform_unsupported")
            model = payload.get("model") or {}
            layers = (model.get("trunk") or {}).get("layers") or ()
            if len(layers) != 1 or layers[0].get("activation") != "relu":
                raise ValueError("n4_v2_trunk_unsupported")
            self.trunk = dict(layers[0])
            self.heads = dict(model.get("heads") or {})
            if set(self.heads) != {"rank", "validity", "gain", "risk"}:
                raise ValueError("n4_v2_heads_invalid")
            calibration = payload.get("calibration") or {}
            self.platt_a = float(calibration.get("a", 1.0))
            self.platt_b = float(calibration.get("b", 0.0))
            if self.platt_a <= 0.0:
                raise ValueError("n4_v2_calibration_scale_invalid")

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
            if self._v2:
                ranked.append(self._score_v2(raw, features))
                continue
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
        order_field = "rank_score" if self._v2 else "priority"
        ranked.sort(
            key=lambda item: (-item[order_field], item["hypothesis_id"])
        )
        if self._v2:
            candidate_set_id = str(
                request.payload.get("candidate_set_id", request.inference_id)
            )
            for position, item in enumerate(ranked, 1):
                item["candidate_set_id"] = candidate_set_id
                item["rank_position"] = position
                item["within_top2_budget"] = position <= 2
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
            candidate_output={
                "schema": "n4-ranking.v2" if self._v2 else "n4-ranking.v1",
                "artifact_schema": self.schema,
                "artifact_sha256": self.artifact_sha256,
                "ranking_objective": self.ranking_objective,
                "rankings": ranked,
            },
            confidence=confidence,
            uncertainty=uncertainty,
            cost={"candidates": len(ranked), "feature_count": len(_FEATURES)},
        )

    def _score_v2(
        self, raw: Mapping[str, Any], features: tuple[float, ...]
    ) -> dict[str, Any]:
        hidden = [
            max(0.0, value)
            for value in _linear(
                features,
                self.trunk.get("weights"),
                self.trunk.get("bias"),
            )
        ]
        rank_score = _linear_head(hidden, self.heads["rank"])
        validity_logit = _linear_head(hidden, self.heads["validity"])
        gain_logit = _linear_head(hidden, self.heads["gain"])
        risk_logit = _linear_head(hidden, self.heads["risk"])
        probability = _sigmoid(self.platt_a * validity_logit + self.platt_b)
        risk = _sigmoid(risk_logit)
        if not all(
            math.isfinite(value)
            for value in (rank_score, validity_logit, probability, gain_logit, risk)
        ):
            raise ValueError("n4_v2_inference_nonfinite")
        uncertainty = min(1.0, max(0.0, 1.0 - abs(probability - 0.5) * 2.0))
        return {
            "hypothesis_id": str(raw["hypothesis_id"]),
            "candidate_source": str(
                raw.get("candidate_source", raw.get("source", "unknown"))
            ),
            "rank_score": round(rank_score, 9),
            "priority": round(rank_score, 9),
            "validity_logit": round(validity_logit, 9),
            "probability_valid": round(probability, 9),
            "validity_probability": round(probability, 9),
            "expected_mae_gain": round(math.tanh(gain_logit), 9),
            "uncertainty": round(uncertainty, 9),
            "invariant_risk": round(risk, 9),
            "artifact_schema": self.schema,
            "artifact_sha256": self.artifact_sha256,
            "ranking_objective": self.ranking_objective,
            "feature_vector": list(features),
        }


def _linear(values, weights, bias):
    rows = tuple(tuple(float(item) for item in row) for row in weights)
    offsets = tuple(float(item) for item in bias)
    if len(rows) != len(offsets) or any(len(row) != len(values) for row in rows):
        raise ValueError("n4_v2_linear_shape_invalid")
    return tuple(
        offsets[index] + sum(a * b for a, b in zip(row, values))
        for index, row in enumerate(rows)
    )


def _linear_head(values, head):
    weights = tuple(float(item) for item in head.get("weights", ()))
    bias = float(head.get("bias", 0.0))
    if len(weights) != len(values):
        raise ValueError("n4_v2_head_shape_invalid")
    return bias + sum(a * b for a, b in zip(weights, values))


def _sigmoid(value):
    return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, value))))
