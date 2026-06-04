"""Estimador de demanda cognitiva para selección multiescala."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, List, Optional

from .contracts import ScaleEstimate
from .scale_catalog import ScaleCatalog


class ScaleEstimator:
    """Infiere demanda de resolución desde riesgo, heterogeneidad, epistemia y operación."""

    def __init__(self, *, catalog: Optional[ScaleCatalog] = None):
        self.catalog = catalog or ScaleCatalog.default()

    def estimate(
        self,
        *,
        current_scale_id: str,
        observation: Dict[str, Any],
        viability_margin: Optional[float],
        certification_verdict: Optional[str],
        metrics: Optional[Dict[str, Any]] = None,
        vram_snapshot: Optional[Dict[str, Any]] = None,
        alarm_threshold: float = 0.85,
    ) -> ScaleEstimate:
        metrics = dict(metrics or {})
        vram_snapshot = dict(vram_snapshot or {})

        expected_spatial_complexity = self._clamp(float(metrics.get("expected_spatial_complexity", 0.0) or 0.0))
        spatial_usage = self._clamp(float(metrics.get("spatial_information_usage", 0.0) or 0.0))

        heterogeneity, hetero_detail = self._compute_heterogeneity_score(observation)
        epistemic, epistemic_detail = self._compute_epistemic_insufficiency(
            current_scale_id=current_scale_id,
            observation=observation,
            metrics=metrics,
            expected_spatial_complexity=expected_spatial_complexity,
        )
        risk, risk_detail = self._compute_risk_score(
            observation=observation,
            viability_margin=viability_margin,
            certification_verdict=certification_verdict,
            alarm_threshold=alarm_threshold,
        )
        operational, op_detail = self._compute_operational_pressure(metrics=metrics)

        scale_blindspot_bonus = self._scale_blindspot_bonus(
            current_scale_id=current_scale_id,
            spatial_usage=spatial_usage,
            expected_spatial_complexity=expected_spatial_complexity,
        )
        heterogeneity_prior_bonus = 0.0
        if heterogeneity < 0.18:
            heterogeneity_prior_bonus = 0.35 * expected_spatial_complexity

        if current_scale_id == "1x1":
            heterogeneity += 0.30 * scale_blindspot_bonus
        heterogeneity += heterogeneity_prior_bonus
        heterogeneity = self._clamp(heterogeneity)

        vram_headroom = float(vram_snapshot.get("vram_headroom", 0.0))
        vram_pressure = float(vram_snapshot.get("vram_pressure", 1.0))
        vram_fragmentation_risk = float(vram_snapshot.get("vram_fragmentation_risk", 1.0))
        vram_opportunity = float(vram_snapshot.get("vram_opportunity_score", 0.0))

        required = (
            0.34 * risk
            + 0.33 * heterogeneity
            + 0.28 * epistemic
            + 0.05 * max(0.0, 1.0 - operational)
        )
        required = min(max(required, 0.0), 1.0)

        minimum_rank = self._resolution_rank_from_required(required)
        candidates = self.catalog.candidates_at_or_above(minimum_rank, executable_only=False)

        candidates_sorted = sorted(
            candidates,
            key=lambda spec: (
                -self._candidate_priority(spec, required, vram_opportunity, vram_pressure),
                spec.expected_time_cost,
            ),
        )

        recommended = [spec.scale_id for spec in candidates_sorted]
        if not recommended:
            recommended = [current_scale_id]

        hetero_detail["expected_spatial_complexity"] = expected_spatial_complexity
        hetero_detail["scale_blindspot_bonus"] = scale_blindspot_bonus
        hetero_detail["heterogeneity_prior_bonus"] = heterogeneity_prior_bonus

        signals = {
            "heterogeneity_detail": hetero_detail,
            "epistemic_detail": epistemic_detail,
            "risk_detail": risk_detail,
            "operational_detail": op_detail,
            "current_scale_id": current_scale_id,
            "minimum_rank": minimum_rank,
            "expected_spatial_complexity": expected_spatial_complexity,
            "scale_blindspot_bonus": scale_blindspot_bonus,
            "vram_snapshot": vram_snapshot,
        }

        return ScaleEstimate(
            required_resolution_score=required,
            heterogeneity_score=heterogeneity,
            epistemic_insufficiency_score=epistemic,
            risk_score=risk,
            operational_pressure_score=operational,
            vram_headroom=vram_headroom,
            vram_pressure=vram_pressure,
            vram_fragmentation_risk=vram_fragmentation_risk,
            vram_opportunity_score=vram_opportunity,
            recommended_scale_candidates=recommended,
            signals=signals,
        )

    def _compute_heterogeneity_score(self, observation: Dict[str, Any]) -> tuple[float, Dict[str, float]]:
        cell_states = observation.get("cell_states") or []
        if not isinstance(cell_states, list) or not cell_states:
            return 0.0, {
                "variance": 0.0,
                "gradient": 0.0,
                "hotspot_likelihood": 0.0,
                "anisotropy": 0.0,
                "aggregate_local_incoherence": 0.0,
            }

        temps = [float(cell.get("temperature", 0.0)) for cell in cell_states]
        mean_temp = sum(temps) / len(temps)
        variance = sum((t - mean_temp) ** 2 for t in temps) / len(temps)
        std_temp = math.sqrt(max(variance, 0.0))
        hotspot_likelihood = min(max((max(temps) - mean_temp) / 0.2, 0.0), 1.0)

        grid = self._cells_to_grid(cell_states)
        gradient = self._mean_neighbor_gradient(grid)
        anisotropy = self._anisotropy(grid)

        world_level = float(observation.get("world_level", mean_temp))
        incoherence = min(abs(world_level - mean_temp) / 0.08, 1.0)

        score = (
            0.34 * min(std_temp / 0.18, 1.0)
            + 0.30 * min(gradient / 0.16, 1.0)
            + 0.22 * hotspot_likelihood
            + 0.10 * anisotropy
            + 0.04 * incoherence
        )
        score = self._clamp(score)
        return score, {
            "variance": variance,
            "gradient": gradient,
            "hotspot_likelihood": hotspot_likelihood,
            "anisotropy": anisotropy,
            "aggregate_local_incoherence": incoherence,
        }

    def _compute_epistemic_insufficiency(
        self,
        *,
        current_scale_id: str,
        observation: Dict[str, Any],
        metrics: Dict[str, Any],
        expected_spatial_complexity: float,
    ) -> tuple[float, Dict[str, float]]:
        factual_delta = float(metrics.get("factual_delta", 0.0))
        counterfactual_delta = float(metrics.get("counterfactual_delta", 0.0))
        conflict = min(abs(factual_delta - counterfactual_delta) / 0.2, 1.0)

        contradiction_signal = self._clamp(float(metrics.get("contradiction_signal", 0.0) or 0.0))
        uncertainty = self._clamp(float(metrics.get("uncertainty", 0.0) or 0.0))
        scheduler_disagreement = self._clamp(float(metrics.get("scheduler_disagreement", 0.0) or 0.0))
        previous_probe_failure = self._clamp(float(metrics.get("previous_probe_failure", 0.0) or 0.0))
        intervention_precision = float(metrics.get("intervention_precision", 0.0) or 0.0)
        intervention_backfire = self._clamp(max(-intervention_precision, 0.0) / 0.08)

        spatial_usage = self._clamp(float(metrics.get("spatial_information_usage", 0.0) or 0.0))
        scale_blindspot_bonus = self._scale_blindspot_bonus(
            current_scale_id=current_scale_id,
            spatial_usage=spatial_usage,
            expected_spatial_complexity=expected_spatial_complexity,
        )

        sparse_props = 0.0
        props = observation.get("propositions") or []
        if isinstance(props, list) and len(props) <= 1:
            sparse_props = 0.2

        score = (
            0.30 * conflict
            + 0.18 * contradiction_signal
            + 0.14 * uncertainty
            + 0.10 * scheduler_disagreement
            + 0.08 * self._clamp(previous_probe_failure + sparse_props)
            + 0.10 * intervention_backfire
            + 0.10 * scale_blindspot_bonus
        )
        score = self._clamp(score)

        return score, {
            "factual_counterfactual_conflict": conflict,
            "contradiction_signal": contradiction_signal,
            "uncertainty": uncertainty,
            "scheduler_disagreement": scheduler_disagreement,
            "sparse_propositions_bonus": sparse_props,
            "intervention_backfire": intervention_backfire,
            "scale_blindspot_bonus": scale_blindspot_bonus,
        }

    def _compute_risk_score(
        self,
        *,
        observation: Dict[str, Any],
        viability_margin: Optional[float],
        certification_verdict: Optional[str],
        alarm_threshold: float,
    ) -> tuple[float, Dict[str, float]]:
        world_level = float(observation.get("world_level", observation.get("temperature", 0.0)))
        margin = float(viability_margin) if viability_margin is not None else 0.0
        distance_to_alarm = max(alarm_threshold - world_level, 0.0)
        proximity_to_threshold = min(max(1.0 - (distance_to_alarm / 0.25), 0.0), 1.0)
        low_margin = min(max((0.25 - margin) / 0.25, 0.0), 1.0)

        verdict_penalty = 0.0
        if certification_verdict not in {None, "passed", "certified", "PASSED", "CONDITIONALLY_PASSED"}:
            verdict_penalty = 0.5

        score = min(max((0.52 * proximity_to_threshold) + (0.38 * low_margin) + (0.10 * verdict_penalty), 0.0), 1.0)
        return score, {
            "world_level": world_level,
            "alarm_threshold": alarm_threshold,
            "proximity_to_threshold": proximity_to_threshold,
            "low_margin": low_margin,
            "verdict_penalty": verdict_penalty,
        }

    def _compute_operational_pressure(self, *, metrics: Dict[str, Any]) -> tuple[float, Dict[str, float]]:
        wall_time_ms = float(metrics.get("wall_time_ms", 0.0))
        artifact_size_bytes = float(metrics.get("artifact_size_bytes", 0.0))
        memory_pressure = float(metrics.get("memory_pressure", 0.0))
        budget_ratio = float(metrics.get("operational_budget_ratio", 0.0))
        cumulative_cost = float(metrics.get("cumulative_cost_ratio", 0.0))

        wall_component = min(max(wall_time_ms / 1200.0, 0.0), 1.0)
        artifact_component = min(max(artifact_size_bytes / 180_000.0, 0.0), 1.0)
        pressure = min(
            max(
                (0.35 * wall_component)
                + (0.30 * artifact_component)
                + (0.15 * min(max(memory_pressure, 0.0), 1.0))
                + (0.10 * min(max(budget_ratio, 0.0), 1.0))
                + (0.10 * min(max(cumulative_cost, 0.0), 1.0)),
                0.0,
            ),
            1.0,
        )

        return pressure, {
            "wall_component": wall_component,
            "artifact_component": artifact_component,
            "memory_pressure": min(max(memory_pressure, 0.0), 1.0),
            "budget_ratio": min(max(budget_ratio, 0.0), 1.0),
            "cumulative_cost_ratio": min(max(cumulative_cost, 0.0), 1.0),
        }

    def _cells_to_grid(self, cell_states: List[Dict[str, Any]]) -> List[List[float]]:
        grouped: Dict[int, Dict[int, float]] = defaultdict(dict)
        for cell in cell_states:
            row = int(cell.get("row", 0))
            col = int(cell.get("col", 0))
            grouped[row][col] = float(cell.get("temperature", 0.0))
        rows = sorted(grouped)
        grid: List[List[float]] = []
        for row in rows:
            cols = sorted(grouped[row])
            grid.append([grouped[row][col] for col in cols])
        return grid

    def _mean_neighbor_gradient(self, grid: List[List[float]]) -> float:
        if not grid:
            return 0.0
        n_rows = len(grid)
        n_cols = len(grid[0]) if grid[0] else 0
        if n_cols == 0:
            return 0.0

        total = 0.0
        count = 0
        for i in range(n_rows):
            for j in range(n_cols):
                if i + 1 < n_rows:
                    total += abs(grid[i][j] - grid[i + 1][j])
                    count += 1
                if j + 1 < n_cols:
                    total += abs(grid[i][j] - grid[i][j + 1])
                    count += 1
        return total / count if count else 0.0

    def _anisotropy(self, grid: List[List[float]]) -> float:
        if not grid or not grid[0]:
            return 0.0
        row_means = [sum(row) / len(row) for row in grid]
        col_means = [sum(grid[i][j] for i in range(len(grid))) / len(grid) for j in range(len(grid[0]))]

        row_var = self._variance(row_means)
        col_var = self._variance(col_means)
        denom = max(row_var, col_var, 1e-6)
        return min(abs(row_var - col_var) / denom, 1.0)

    def _variance(self, values: List[float]) -> float:
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        return sum((item - mean) ** 2 for item in values) / len(values)

    def _resolution_rank_from_required(self, required: float) -> int:
        if required < 0.25:
            return 1
        if required < 0.45:
            return 2
        if required < 0.60:
            return 3
        if required < 0.80:
            return 5
        if required < 0.90:
            return 10
        return 30

    def _candidate_priority(
        self,
        spec,
        required: float,
        vram_opportunity: float,
        vram_pressure: float,
    ) -> float:
        base = spec.expected_information_gain_prior
        resolution_alignment = 1.0 - min(abs(spec.resolution_rank - self._resolution_rank_from_required(required)) / 30.0, 1.0)
        vram_boost = vram_opportunity * min(spec.resolution_rank / 10.0, 1.0)

        vram_penalty = 0.0
        if vram_pressure > 0.85:
            vram_penalty = (vram_pressure - 0.85) / 0.15
        return (0.55 * base) + (0.30 * resolution_alignment) + (0.25 * vram_boost) - (0.20 * vram_penalty)

    def _scale_blindspot_bonus(
        self,
        *,
        current_scale_id: str,
        spatial_usage: float,
        expected_spatial_complexity: float,
    ) -> float:
        if current_scale_id != "1x1":
            return 0.0
        usage_component = self._clamp((spatial_usage - 0.08) / 0.42)
        return self._clamp((0.55 * usage_component) + (0.45 * expected_spatial_complexity))

    def _clamp(self, value: float, lo: float = 0.0, hi: float = 1.0) -> float:
        return min(max(value, lo), hi)
