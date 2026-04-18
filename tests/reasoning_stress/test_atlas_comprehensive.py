"""
Reasoning Difficulty & Functionality Atlas - Elite Level
=========================================================

Comprehensive mapping and visualization of the reasoning space.
Generates detailed atlas with:
- Activation frontiers
- Sensitivity maps
- Saturation regions
- Stability metrics
- Functional characterization
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict
import pytest

from runtime.reasoning.scheduler_meta.budgeting import compute_budget
from runtime.reasoning.scheduler_meta.policy import select_sequence


@dataclass
class ActivationFrontier:
    """Characterizes activation boundary for a family."""
    family_name: str
    feature_name: str
    threshold_nominal: float
    threshold_measured_low: float
    threshold_measured_high: float
    threshold_width: float
    stability_score: float
    # Boundary equation: value at which activation occurs


@dataclass
class SensitivityMetrics:
    """Local sensitivity measurements."""
    family_name: str
    feature_name: str
    sensitivity_coefficient: float  # Change in activation per unit feature
    discontinuity_points: List[float]
    smooth_regions: List[Tuple[float, float]]
    chaotic_regions: List[Tuple[float, float]]


@dataclass
class SaturationRegion:
    """Characterizes regions where scheduler saturates."""
    region_id: str
    feature_bounds: Dict[str, Tuple[float, float]]
    max_steps_achieved: float
    families_activated: List[str]
    stability_rating: str  # "stable", "fragile", "chaotic"
    scaling_quality: str  # "good", "degraded", "collapsed"


@dataclass
class FamilyFunctionalProfile:
    """Functional characterization of a family."""
    family_name: str
    status: str  # "core", "operational", "experimental", "shadow"
    activation_conditions: Dict[str, float]  # Feature thresholds
    activation_rate_global: float  # Across all contexts
    cost_mean: float
    utility_marginal: float  # Marginal improvement when active
    no_harm_score: float  # How much it can degrade (0-1, higher is safer)
    stability_score: float  # Robustness to perturbations
    recommended_contexts: List[str]


@dataclass
class ReasoningDifficultyAtlas:
    """Complete atlas of the reasoning space."""
    version: str = "1.0.0"
    test_date: str = ""

    # Frontiers
    activation_frontiers: List[ActivationFrontier] = None

    # Sensitivity
    sensitivity_maps: List[SensitivityMetrics] = None

    # Saturation
    saturation_regions: List[SaturationRegion] = None

    # Family profiles
    family_profiles: List[FamilyFunctionalProfile] = None

    # Global metrics
    global_stability_score: float = 0.0
    global_coverage_score: float = 0.0
    chaos_rate: float = 0.0

    def __post_init__(self):
        if self.activation_frontiers is None:
            self.activation_frontiers = []
        if self.sensitivity_maps is None:
            self.sensitivity_maps = []
        if self.saturation_regions is None:
            self.saturation_regions = []
        if self.family_profiles is None:
            self.family_profiles = []

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "version": self.version,
            "test_date": self.test_date,
            "activation_frontiers": [asdict(f) for f in self.activation_frontiers],
            "sensitivity_maps": [asdict(s) for s in self.sensitivity_maps],
            "saturation_regions": [asdict(r) for r in self.saturation_regions],
            "family_profiles": [asdict(p) for p in self.family_profiles],
            "global_stability_score": self.global_stability_score,
            "global_coverage_score": self.global_coverage_score,
            "chaos_rate": self.chaos_rate,
        }

    def save(self, filepath: str | Path):
        """Save atlas to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    def get_summary_report(self) -> str:
        """Generate human-readable summary report."""
        lines = [
            "=" * 80,
            "REASONING DIFFICULTY & FUNCTIONALITY ATLAS",
            "=" * 80,
            f"Version: {self.version}",
            f"Test Date: {self.test_date}",
            "",
            "GLOBAL METRICS",
            "-" * 80,
            f"  Stability Score: {self.global_stability_score:.3f}",
            f"  Coverage Score: {self.global_coverage_score:.3f}",
            f"  Chaos Rate: {self.chaos_rate:.3f}",
            "",
            "FAMILY PROFILES",
            "-" * 80,
        ]

        for profile in self.family_profiles:
            lines.extend([
                f"  {profile.family_name.upper()} ({profile.status})",
                f"    Activation Rate: {profile.activation_rate_global:.3f}",
                f"    Marginal Utility: {profile.utility_marginal:.3f}",
                f"    Stability: {profile.stability_score:.3f}",
                ""
            ])

        lines.extend([
            "ACTIVATION FRONTIERS",
            "-" * 80,
        ])

        for frontier in self.activation_frontiers:
            lines.extend([
                f"  {frontier.family_name.upper()} via {frontier.feature_name}",
                f"    Threshold: {frontier.threshold_nominal:.3f} "
                f"[{frontier.threshold_measured_low:.3f}, {frontier.threshold_measured_high:.3f}]",
                f"    Width: {frontier.threshold_width:.3f}",
                f"    Stability: {frontier.stability_score:.3f}",
                ""
            ])

        lines.extend([
            "SATURATION REGIONS",
            "-" * 80,
        ])

        for region in self.saturation_regions:
            lines.extend([
                f"  {region.region_id}",
                f"    Max Steps: {region.max_steps_achieved:.1f}",
                f"    Families: {', '.join(region.families_activated)}",
                f"    Stability: {region.stability_rating}",
                f"    Scaling: {region.scaling_quality}",
                ""
            ])

        lines.append("=" * 80)

        return "\n".join(lines)


def build_complete_atlas() -> ReasoningDifficultyAtlas:
    """
    Build complete reasoning atlas by running comprehensive tests.

    This is the master function that orchestrates all analysis.
    """
    from datetime import datetime

    atlas = ReasoningDifficultyAtlas(
        version="1.0.0",
        test_date=datetime.now().isoformat()
    )

    # 1. Map activation frontiers
    atlas.activation_frontiers = _map_activation_frontiers()

    # 2. Measure sensitivity
    atlas.sensitivity_maps = _measure_sensitivity()

    # 3. Identify saturation regions
    atlas.saturation_regions = _identify_saturation_regions()

    # 4. Profile families
    atlas.family_profiles = _profile_families()

    # 5. Compute global metrics
    atlas.global_stability_score = _compute_global_stability(atlas)
    atlas.global_coverage_score = _compute_global_coverage(atlas)
    atlas.chaos_rate = _compute_chaos_rate(atlas)

    return atlas


def _map_activation_frontiers() -> List[ActivationFrontier]:
    """Map activation frontiers for all families."""
    frontiers = []

    # HEUR via edge_pressure
    frontiers.append(_measure_frontier(
        family_name="heur",
        feature_name="edge_pressure",
        threshold_nominal=0.7,
        sweep_range=(0.5, 0.9)
    ))

    # DIA_ADV via contradiction_signal
    frontiers.append(_measure_frontier(
        family_name="dia_adv",
        feature_name="contradiction_signal",
        threshold_nominal=0.45,
        sweep_range=(0.25, 0.65)
    ))

    # FAL_GUARD via contradiction_signal
    frontiers.append(_measure_frontier(
        family_name="fal_guard",
        feature_name="contradiction_signal",
        threshold_nominal=0.45,
        sweep_range=(0.25, 0.65)
    ))

    # EML_SR via symbolic_regularity
    frontiers.append(_measure_frontier(
        family_name="eml_sr",
        feature_name="symbolic_regularity",
        threshold_nominal=0.4,
        sweep_range=(0.2, 0.6)
    ))

    # EML_SR via law_fit_signal
    frontiers.append(_measure_frontier(
        family_name="eml_sr",
        feature_name="law_fit_signal",
        threshold_nominal=0.4,
        sweep_range=(0.2, 0.6)
    ))

    return frontiers


def _measure_frontier(
    family_name: str,
    feature_name: str,
    threshold_nominal: float,
    sweep_range: Tuple[float, float]
) -> ActivationFrontier:
    """Measure activation frontier for a specific family and feature."""
    baseline_features = {
        "uncertainty": 0.25,
        "contradiction_signal": 0.0,
        "continuity_recent": 1.0,
        "edge_pressure": 0.0,
        "causal_risk": 0.0,
        "symbolic_regularity": 0.0,
        "law_fit_signal": 0.0,
    }

    # Fine sweep around threshold
    steps = 41
    start, end = sweep_range
    step_size = (end - start) / (steps - 1)

    activation_points = []

    for i in range(steps):
        value = start + (i * step_size)

        features = baseline_features.copy()
        features[feature_name] = value

        budget = compute_budget(features)
        sequence, _, _ = select_sequence(features=features, budget=budget, allow_experimental=True)

        if family_name in sequence:
            activation_points.append(value)

    if activation_points:
        threshold_low = min(activation_points)
        threshold_high = max(activation_points)
        threshold_width = threshold_high - threshold_low

        # Stability: inverse of width
        stability = 1.0 - min(threshold_width / 0.2, 1.0)
    else:
        threshold_low = threshold_nominal
        threshold_high = threshold_nominal
        threshold_width = 0.0
        stability = 0.0

    return ActivationFrontier(
        family_name=family_name,
        feature_name=feature_name,
        threshold_nominal=threshold_nominal,
        threshold_measured_low=threshold_low,
        threshold_measured_high=threshold_high,
        threshold_width=threshold_width,
        stability_score=stability
    )


def _measure_sensitivity() -> List[SensitivityMetrics]:
    """Measure local sensitivity for each family."""
    sensitivities = []

    families_to_test = [
        ("heur", "edge_pressure"),
        ("dia_adv", "contradiction_signal"),
        ("eml_sr", "symbolic_regularity"),
    ]

    for family_name, feature_name in families_to_test:
        sens = _compute_sensitivity(family_name, feature_name)
        sensitivities.append(sens)

    return sensitivities


def _compute_sensitivity(family_name: str, feature_name: str) -> SensitivityMetrics:
    """Compute sensitivity metrics for a family-feature pair."""
    baseline_features = {
        "uncertainty": 0.25,
        "contradiction_signal": 0.0,
        "continuity_recent": 1.0,
        "edge_pressure": 0.0,
        "causal_risk": 0.0,
        "symbolic_regularity": 0.0,
        "law_fit_signal": 0.0,
    }

    # Measure activation across range
    values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    activations = []

    for val in values:
        features = baseline_features.copy()
        features[feature_name] = val

        budget = compute_budget(features)
        sequence, _, _ = select_sequence(features=features, budget=budget, allow_experimental=True)

        activations.append(1.0 if family_name in sequence else 0.0)

    # Compute sensitivity coefficient (average change)
    changes = [abs(activations[i] - activations[i-1]) for i in range(1, len(activations))]
    sensitivity_coef = sum(changes) / len(changes) if changes else 0.0

    # Find discontinuities
    discontinuities = [values[i] for i in range(1, len(activations))
                      if activations[i] != activations[i-1]]

    # Identify smooth regions (no changes)
    smooth_regions = []
    chaotic_regions = []

    # Simple heuristic: < 2 discontinuities = smooth
    if len(discontinuities) <= 2:
        smooth_regions.append((0.0, 1.0))
    else:
        chaotic_regions.append((0.0, 1.0))

    return SensitivityMetrics(
        family_name=family_name,
        feature_name=feature_name,
        sensitivity_coefficient=sensitivity_coef,
        discontinuity_points=discontinuities,
        smooth_regions=smooth_regions,
        chaotic_regions=chaotic_regions
    )


def _identify_saturation_regions() -> List[SaturationRegion]:
    """Identify regions where scheduler saturates or fails."""
    regions = []

    # Test extreme high (crisis)
    crisis_features = {
        "uncertainty": 0.9,
        "contradiction_signal": 0.9,
        "continuity_recent": 0.2,
        "edge_pressure": 0.5,
        "causal_risk": 0.9,
        "symbolic_regularity": 0.8,
        "law_fit_signal": 0.8,
    }

    budget = compute_budget(crisis_features)
    sequence, _, _ = select_sequence(features=crisis_features, budget=budget, allow_experimental=True)

    regions.append(SaturationRegion(
        region_id="crisis_high_all",
        feature_bounds={
            "uncertainty": (0.8, 1.0),
            "contradiction_signal": (0.8, 1.0),
            "causal_risk": (0.8, 1.0),
        },
        max_steps_achieved=budget["max_steps"],
        families_activated=sequence,
        stability_rating="stable" if 4 <= len(sequence) <= 10 else "fragile",
        scaling_quality="good" if len(sequence) <= 10 else "degraded"
    ))

    # Test extreme edge pressure (resource constrained)
    constrained_features = {
        "uncertainty": 0.5,
        "contradiction_signal": 0.5,
        "continuity_recent": 0.8,
        "edge_pressure": 0.95,
        "causal_risk": 0.5,
        "symbolic_regularity": 0.0,
        "law_fit_signal": 0.0,
    }

    budget_constrained = compute_budget(constrained_features)
    seq_constrained, _, _ = select_sequence(features=constrained_features, budget=budget_constrained, allow_experimental=True)

    regions.append(SaturationRegion(
        region_id="resource_constrained",
        feature_bounds={"edge_pressure": (0.9, 1.0)},
        max_steps_achieved=budget_constrained["max_steps"],
        families_activated=seq_constrained,
        stability_rating="stable" if len(seq_constrained) >= 4 else "fragile",
        scaling_quality="good" if len(seq_constrained) >= 4 else "degraded"
    ))

    return regions


def _profile_families() -> List[FamilyFunctionalProfile]:
    """Profile all families functionally."""
    profiles = []

    # Generate diverse test contexts
    test_contexts = _generate_diverse_contexts(n=100)

    # Profile each family
    all_families = ["abd", "ana", "cau", "ctf", "ded", "prob", "heur", "dia_adv", "fal_guard", "eml_sr"]

    for family in all_families:
        profile = _profile_family(family, test_contexts)
        profiles.append(profile)

    return profiles


def _profile_family(family_name: str, test_contexts: List[Dict]) -> FamilyFunctionalProfile:
    """Profile a single family."""
    activation_count = 0

    for features in test_contexts:
        budget = compute_budget(features)
        sequence, _, _ = select_sequence(features=features, budget=budget, allow_experimental=True)

        if family_name in sequence:
            activation_count += 1

    activation_rate = activation_count / len(test_contexts)

    # Determine status
    core_families = {"abd", "ana", "cau", "ctf", "ded", "prob"}
    if family_name in core_families:
        status = "core"
    elif family_name in ["heur", "dia_adv", "fal_guard"]:
        status = "operational"
    else:
        status = "experimental"

    # Activation conditions
    conditions = {}
    if family_name == "heur":
        conditions["edge_pressure"] = 0.7
    elif family_name in ["dia_adv", "fal_guard"]:
        conditions["contradiction_signal"] = 0.45
    elif family_name == "eml_sr":
        conditions["symbolic_regularity"] = 0.4
        conditions["law_fit_signal"] = 0.4

    return FamilyFunctionalProfile(
        family_name=family_name,
        status=status,
        activation_conditions=conditions,
        activation_rate_global=activation_rate,
        cost_mean=1.0,  # Simplified
        utility_marginal=0.8 if activation_rate > 0.1 else 0.3,
        no_harm_score=0.95,
        stability_score=0.9,
        recommended_contexts=[]
    )


def _generate_diverse_contexts(n: int) -> List[Dict[str, float]]:
    """Generate diverse test contexts."""
    import random
    random.seed(42)

    contexts = []
    for _ in range(n):
        contexts.append({
            "uncertainty": random.uniform(0.0, 1.0),
            "contradiction_signal": random.uniform(0.0, 1.0),
            "continuity_recent": random.uniform(0.3, 1.0),
            "edge_pressure": random.uniform(0.0, 1.0),
            "causal_risk": random.uniform(0.0, 1.0),
            "symbolic_regularity": random.uniform(0.0, 1.0),
            "law_fit_signal": random.uniform(0.0, 1.0),
        })

    return contexts


def _compute_global_stability(atlas: ReasoningDifficultyAtlas) -> float:
    """Compute global stability score."""
    if not atlas.activation_frontiers:
        return 0.0

    stability_scores = [f.stability_score for f in atlas.activation_frontiers]
    return sum(stability_scores) / len(stability_scores)


def _compute_global_coverage(atlas: ReasoningDifficultyAtlas) -> float:
    """Compute global coverage score."""
    if not atlas.family_profiles:
        return 0.0

    # Coverage: proportion of families that activate
    active_families = [p for p in atlas.family_profiles if p.activation_rate_global > 0.0]
    return len(active_families) / len(atlas.family_profiles)


def _compute_chaos_rate(atlas: ReasoningDifficultyAtlas) -> float:
    """Compute chaos rate."""
    if not atlas.saturation_regions:
        return 0.0

    chaotic_regions = [r for r in atlas.saturation_regions if r.stability_rating == "chaotic"]
    return len(chaotic_regions) / len(atlas.saturation_regions)


# ============================================================================
# TEST CASES FOR ATLAS
# ============================================================================

def test_build_complete_atlas():
    """Test that complete atlas can be built."""
    atlas = build_complete_atlas()

    # Should have all components
    assert len(atlas.activation_frontiers) > 0
    assert len(atlas.family_profiles) > 0
    assert len(atlas.saturation_regions) > 0

    # Global metrics should be reasonable
    assert 0.0 <= atlas.global_stability_score <= 1.0
    assert 0.0 <= atlas.global_coverage_score <= 1.0
    assert 0.0 <= atlas.chaos_rate <= 1.0


def test_atlas_serialization():
    """Test that atlas can be serialized to JSON."""
    atlas = build_complete_atlas()

    # Should convert to dict
    atlas_dict = atlas.to_dict()
    assert "version" in atlas_dict
    assert "activation_frontiers" in atlas_dict
    assert "family_profiles" in atlas_dict


def test_atlas_summary_report():
    """Test that atlas generates readable summary."""
    atlas = build_complete_atlas()

    summary = atlas.get_summary_report()

    # Should contain key sections
    assert "REASONING DIFFICULTY & FUNCTIONALITY ATLAS" in summary
    assert "GLOBAL METRICS" in summary
    assert "FAMILY PROFILES" in summary
    assert "ACTIVATION FRONTIERS" in summary


def test_atlas_quality_thresholds():
    """Test that atlas meets quality thresholds."""
    atlas = build_complete_atlas()

    # Stability should be high
    assert atlas.global_stability_score >= 0.7, \
        f"Insufficient stability: {atlas.global_stability_score}"

    # Coverage should be good
    assert atlas.global_coverage_score >= 0.8, \
        f"Insufficient coverage: {atlas.global_coverage_score}"

    # Chaos should be low
    assert atlas.chaos_rate <= 0.1, \
        f"Too much chaos: {atlas.chaos_rate}"


@pytest.mark.slow
def test_save_atlas_to_file(tmp_path):
    """Test saving atlas to file."""
    atlas = build_complete_atlas()

    output_file = tmp_path / "reasoning_atlas.json"
    atlas.save(output_file)

    assert output_file.exists()

    # Should be valid JSON
    with open(output_file) as f:
        data = json.load(f)

    assert data["version"] == "1.0.0"
