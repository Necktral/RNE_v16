"""Clasificador de régimen cognitivo-operativo para MSRC v3."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal

from .contracts import ScaleEstimate, ScalePolicyState

RegimeLabel = Literal[
    "homogeneous",
    "heterogeneous",
    "viability_edge",
]


@dataclass(frozen=True)
class RegimeClassification:
    regime_label: RegimeLabel
    regime_confidence: float
    regime_tags: List[str] = field(default_factory=list)
    regime_evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "regime_label": self.regime_label,
            "regime_confidence": self.regime_confidence,
            "regime_tags": list(self.regime_tags),
            "regime_evidence": dict(self.regime_evidence),
        }


class RegimeClassifier:
    """Clasifica el estado en régimen homogéneo/heterogéneo/borde de viabilidad."""

    def __init__(
        self,
        *,
        edge_risk_threshold: float = 0.72,
        edge_low_margin_threshold: float = 0.72,
        edge_proximity_threshold: float = 0.82,
        heterogeneity_threshold: float = 0.24,
        spatial_complexity_threshold: float = 0.50,
    ):
        self.edge_risk_threshold = edge_risk_threshold
        self.edge_low_margin_threshold = edge_low_margin_threshold
        self.edge_proximity_threshold = edge_proximity_threshold
        self.heterogeneity_threshold = heterogeneity_threshold
        self.spatial_complexity_threshold = spatial_complexity_threshold

    def classify(self, *, estimate: ScaleEstimate, state: ScalePolicyState) -> RegimeClassification:
        risk_detail = (estimate.signals or {}).get("risk_detail", {}) or {}
        hetero_detail = (estimate.signals or {}).get("heterogeneity_detail", {}) or {}

        low_margin = self._clamp(float(risk_detail.get("low_margin", 0.0) or 0.0))
        proximity = self._clamp(float(risk_detail.get("proximity_to_threshold", 0.0) or 0.0))
        world_level = float(risk_detail.get("world_level", 0.0) or 0.0)

        expected_spatial_complexity = self._clamp(float(hetero_detail.get("expected_spatial_complexity", 0.0) or 0.0))
        scale_blindspot_bonus = self._clamp(float(hetero_detail.get("scale_blindspot_bonus", 0.0) or 0.0))

        edge_score = (
            0.40 * estimate.risk_score
            + 0.30 * low_margin
            + 0.30 * proximity
        )
        hetero_score = (
            0.55 * estimate.heterogeneity_score
            + 0.30 * expected_spatial_complexity
            + 0.15 * scale_blindspot_bonus
        )

        low_margin_and_proximity = (
            low_margin >= self.edge_low_margin_threshold
            and proximity >= self.edge_proximity_threshold
        )
        uniform_edge_proximity = (
            proximity >= 0.95
            and expected_spatial_complexity < self.spatial_complexity_threshold
            and estimate.risk_score >= 0.45
        )
        risk_with_required = (
            estimate.risk_score >= max(0.55, self.edge_risk_threshold - 0.10)
            and estimate.required_resolution_score >= 0.72
        )
        is_edge = (
            estimate.risk_score >= self.edge_risk_threshold
            or low_margin_and_proximity
            or uniform_edge_proximity
            or risk_with_required
        )
        is_heterogeneous = (
            estimate.heterogeneity_score >= self.heterogeneity_threshold
            or expected_spatial_complexity >= self.spatial_complexity_threshold
            or scale_blindspot_bonus >= 0.45
            or (estimate.epistemic_insufficiency_score >= 0.34 and expected_spatial_complexity >= 0.35)
        )

        if is_edge:
            label: RegimeLabel = "viability_edge"
            confidence = max(edge_score, estimate.required_resolution_score)
        elif is_heterogeneous:
            label = "heterogeneous"
            confidence = max(hetero_score, estimate.heterogeneity_score)
        else:
            label = "homogeneous"
            confidence = 1.0 - max(edge_score, hetero_score)

        tags: List[str] = []
        if estimate.risk_score >= 0.78 and estimate.required_resolution_score >= 0.70:
            tags.append("critical_high_signal")
        if (
            estimate.vram_opportunity_score >= 0.62
            and estimate.vram_pressure < 0.88
            and estimate.epistemic_insufficiency_score >= 0.22
        ):
            tags.append("probe_favorable")
        if (
            label == "homogeneous"
            and estimate.required_resolution_score < 0.26
            and estimate.heterogeneity_score < 0.12
            and estimate.epistemic_insufficiency_score < 0.15
            and state.oscillation_events == 0
        ):
            tags.append("stable_low_information")

        evidence = {
            "risk_score": estimate.risk_score,
            "low_margin": low_margin,
            "proximity_to_threshold": proximity,
            "world_level": world_level,
            "heterogeneity_score": estimate.heterogeneity_score,
            "epistemic_insufficiency_score": estimate.epistemic_insufficiency_score,
            "required_resolution_score": estimate.required_resolution_score,
            "expected_spatial_complexity": expected_spatial_complexity,
            "scale_blindspot_bonus": scale_blindspot_bonus,
            "vram_opportunity_score": estimate.vram_opportunity_score,
            "vram_pressure": estimate.vram_pressure,
            "recent_oscillation": float(state.oscillation_events) / max(state.step_index, 1),
        }

        return RegimeClassification(
            regime_label=label,
            regime_confidence=self._clamp(confidence),
            regime_tags=tags,
            regime_evidence=evidence,
        )

    def _clamp(self, value: float, lo: float = 0.0, hi: float = 1.0) -> float:
        return min(max(value, lo), hi)
