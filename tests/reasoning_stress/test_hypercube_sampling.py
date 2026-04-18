"""
Hypercube Sampling Stress Tests - Elite Level
==============================================

Tests using Latin Hypercube Sampling and stratified sampling to map
the full reasoning space. Constructs empirical atlas of behavior across
the feature hypercube.
"""

from __future__ import annotations

import random
from typing import Dict, List, Tuple
from dataclasses import dataclass
import pytest

from runtime.reasoning.scheduler_meta.budgeting import compute_budget
from runtime.reasoning.scheduler_meta.policy import select_sequence


@dataclass
class HypercubePoint:
    """A point in the feature hypercube with measurement results."""
    features: Dict[str, float]
    sequence: List[str]
    budget: Dict[str, float]
    region: str  # Classification of this region


class HypercubeAtlas:
    """Maps the reasoning space and classifies regions."""

    def __init__(self):
        self.points: List[HypercubePoint] = []
        self.regions: Dict[str, List[HypercubePoint]] = {}

    def add_point(self, point: HypercubePoint):
        """Add a measurement point to the atlas."""
        self.points.append(point)
        if point.region not in self.regions:
            self.regions[point.region] = []
        self.regions[point.region].append(point)

    def classify_point(self, features: Dict[str, float], sequence: List[str], budget: Dict) -> str:
        """
        Classify a point into a region.

        Regions:
        - baseline: Low all features, minimal sequence
        - high_complexity: High uncertainty/contradiction, expanded sequence
        - resource_constrained: High edge_pressure, reduced budget
        - dialectical: High contradiction, dia_adv + fal_guard active
        - symbolic: High symbolic_regularity or law_fit, eml_sr active
        - crisis: Multiple high signals, maximum activation
        - chaotic: Pathological behavior (should be rare)
        - stable: Normal operation, expected behavior
        """

        # Crisis: multiple high signals
        high_signals = sum(1 for k, v in features.items() if v >= 0.7 and k != "continuity_recent")
        if high_signals >= 3:
            return "crisis"

        # Resource constrained
        if features["edge_pressure"] >= 0.7 and budget["max_steps"] <= 6:
            return "resource_constrained"

        # Dialectical
        if features["contradiction_signal"] >= 0.45 and "dia_adv" in sequence and "fal_guard" in sequence:
            return "dialectical"

        # Symbolic
        if (features.get("symbolic_regularity", 0.0) >= 0.4 or
            features.get("law_fit_signal", 0.0) >= 0.4) and "eml_sr" in sequence:
            return "symbolic"

        # High complexity
        if features["uncertainty"] >= 0.6 or features["causal_risk"] >= 0.6:
            if budget["max_steps"] >= 7:
                return "high_complexity"

        # Baseline
        if all(v <= 0.3 for k, v in features.items() if k != "continuity_recent"):
            return "baseline"

        # Chaotic: unexpected patterns
        if len(sequence) < 4 or len(sequence) > 10:
            return "chaotic"

        if "prob" not in sequence:
            return "chaotic"

        # Otherwise stable
        return "stable"

    def get_region_statistics(self) -> Dict[str, Dict]:
        """Compute statistics for each region."""
        stats = {}

        for region_name, points in self.regions.items():
            if not points:
                continue

            avg_sequence_length = sum(len(p.sequence) for p in points) / len(points)
            avg_max_steps = sum(p.budget["max_steps"] for p in points) / len(points)
            avg_risk_budget = sum(p.budget["risk_budget"] for p in points) / len(points)

            # Family activation frequency
            family_counts = {}
            for p in points:
                for fam in p.sequence:
                    family_counts[fam] = family_counts.get(fam, 0) + 1

            stats[region_name] = {
                "count": len(points),
                "avg_sequence_length": avg_sequence_length,
                "avg_max_steps": avg_max_steps,
                "avg_risk_budget": avg_risk_budget,
                "family_activation_rate": {
                    fam: count / len(points)
                    for fam, count in family_counts.items()
                }
            }

        return stats


def latin_hypercube_sample(n_samples: int, n_dimensions: int, seed: int = 42) -> List[List[float]]:
    """
    Generate Latin Hypercube samples.

    Args:
        n_samples: Number of samples to generate
        n_dimensions: Number of dimensions
        seed: Random seed for reproducibility

    Returns:
        List of sample points, each a list of floats in [0, 1]
    """
    random.seed(seed)

    # Create intervals for each dimension
    samples = []

    for _ in range(n_samples):
        sample = []
        for dim in range(n_dimensions):
            # Divide [0, 1] into n_samples intervals
            # Randomly pick a point within the allocated interval
            interval_size = 1.0 / n_samples
            interval_idx = len([s for s in samples if len(s) > dim])

            # Sample within the interval
            value = random.uniform(
                interval_idx * interval_size,
                (interval_idx + 1) * interval_size
            )
            sample.append(value)

        samples.append(sample)

    # Shuffle each dimension independently
    for dim in range(n_dimensions):
        values = [s[dim] for s in samples]
        random.shuffle(values)
        for i, sample in enumerate(samples):
            sample[dim] = values[i]

    return samples


def map_hypercube_region(n_samples: int = 100, seed: int = 42) -> HypercubeAtlas:
    """
    Map the reasoning space using Latin Hypercube Sampling.

    Args:
        n_samples: Number of samples to generate
        seed: Random seed

    Returns:
        HypercubeAtlas with classified regions
    """
    atlas = HypercubeAtlas()

    # Feature names (7 dimensions)
    feature_names = [
        "uncertainty",
        "contradiction_signal",
        "continuity_recent",
        "edge_pressure",
        "causal_risk",
        "symbolic_regularity",
        "law_fit_signal",
    ]

    # Generate samples
    samples = latin_hypercube_sample(n_samples=n_samples, n_dimensions=len(feature_names), seed=seed)

    # Evaluate each sample point
    for sample in samples:
        features = {name: value for name, value in zip(feature_names, sample)}

        budget = compute_budget(features)
        sequence, _, _ = select_sequence(
            features=features,
            budget=budget,
            allow_experimental=True
        )

        region = atlas.classify_point(features, sequence, budget)

        point = HypercubePoint(
            features=features,
            sequence=sequence,
            budget=budget,
            region=region
        )

        atlas.add_point(point)

    return atlas


# ============================================================================
# HYPERCUBE MAPPING TESTS
# ============================================================================

def test_hypercube_basic_mapping():
    """Test that hypercube mapping covers major regions."""
    atlas = map_hypercube_region(n_samples=100, seed=42)

    # Should have explored multiple regions
    assert len(atlas.regions) >= 4, f"Too few regions explored: {list(atlas.regions.keys())}"

    # Should have baseline region
    assert "baseline" in atlas.regions

    # Should have stable region
    assert "stable" in atlas.regions


def test_hypercube_no_widespread_chaos():
    """Test that chaotic region is rare."""
    atlas = map_hypercube_region(n_samples=200, seed=42)

    total_points = len(atlas.points)
    chaotic_points = len(atlas.regions.get("chaotic", []))

    chaotic_rate = chaotic_points / total_points if total_points > 0 else 0

    # Chaotic behavior should be < 5% of space
    assert chaotic_rate < 0.05, f"Too much chaos: {chaotic_rate:.1%} of space is chaotic"


def test_hypercube_crisis_handling():
    """Test that crisis region handles extreme conditions gracefully."""
    atlas = map_hypercube_region(n_samples=200, seed=42)

    crisis_points = atlas.regions.get("crisis", [])

    if crisis_points:
        # In crisis, system should still produce valid sequences
        for point in crisis_points:
            assert len(point.sequence) >= 4, f"Crisis produced too short sequence: {point.sequence}"
            assert len(point.sequence) <= 10, f"Crisis produced too long sequence: {point.sequence}"
            assert "prob" in point.sequence, "Crisis missing PROB family"
            assert point.sequence[-1] == "prob", "PROB not last in crisis"


def test_hypercube_region_statistics():
    """Test statistical properties of each region."""
    atlas = map_hypercube_region(n_samples=200, seed=42)
    stats = atlas.get_region_statistics()

    # Baseline should have shorter sequences
    if "baseline" in stats:
        assert stats["baseline"]["avg_sequence_length"] <= 7

    # High complexity should have longer sequences
    if "high_complexity" in stats:
        assert stats["high_complexity"]["avg_sequence_length"] >= 6

    # Resource constrained should have lower max_steps
    if "resource_constrained" in stats:
        assert stats["resource_constrained"]["avg_max_steps"] <= 7

    # Dialectical should have dia_adv and fal_guard
    if "dialectical" in stats:
        assert stats["dialectical"]["family_activation_rate"].get("dia_adv", 0) >= 0.8
        assert stats["dialectical"]["family_activation_rate"].get("fal_guard", 0) >= 0.8


def test_hypercube_family_coverage():
    """Test that all families are activated in some regions."""
    atlas = map_hypercube_region(n_samples=200, seed=42)

    # Collect all activated families
    all_families = set()
    for point in atlas.points:
        all_families.update(point.sequence)

    # Core families should always appear
    core_families = {"abd", "ana", "cau", "ctf", "ded", "prob"}
    assert core_families.issubset(all_families)

    # Optional families should appear in some regions
    # (heur, dia_adv, fal_guard, eml_sr)
    # At least some should be activated
    optional_families = {"heur", "dia_adv", "fal_guard", "eml_sr"}
    activated_optional = optional_families & all_families

    assert len(activated_optional) >= 2, f"Too few optional families activated: {activated_optional}"


# ============================================================================
# STRATIFIED SAMPLING TESTS
# ============================================================================

def test_stratified_boundary_regions():
    """Test specific boundary regions with stratified sampling."""
    # Define critical boundary regions
    boundary_regions = [
        # Edge of HEUR activation
        {"edge_pressure": 0.68, "uncertainty": 0.3, "contradiction_signal": 0.2},
        {"edge_pressure": 0.72, "uncertainty": 0.3, "contradiction_signal": 0.2},

        # Edge of DIA_ADV activation
        {"contradiction_signal": 0.43, "uncertainty": 0.3, "edge_pressure": 0.2},
        {"contradiction_signal": 0.47, "uncertainty": 0.3, "edge_pressure": 0.2},

        # Edge of EML_SR activation
        {"symbolic_regularity": 0.38, "uncertainty": 0.3, "contradiction_signal": 0.2},
        {"symbolic_regularity": 0.42, "uncertainty": 0.3, "contradiction_signal": 0.2},
    ]

    results = []

    for base_features in boundary_regions:
        # Fill in missing features
        features = {
            "uncertainty": 0.25,
            "contradiction_signal": 0.0,
            "continuity_recent": 1.0,
            "edge_pressure": 0.0,
            "causal_risk": 0.0,
            "symbolic_regularity": 0.0,
            "law_fit_signal": 0.0,
        }
        features.update(base_features)

        budget = compute_budget(features)
        sequence, _, _ = select_sequence(
            features=features,
            budget=budget,
            allow_experimental=True
        )

        results.append({
            "features": features,
            "sequence": sequence,
            "budget": budget
        })

    # All boundary tests should produce valid sequences
    for r in results:
        assert len(r["sequence"]) >= 4
        assert "prob" in r["sequence"]


# ============================================================================
# HIGH-DIMENSIONAL COVERAGE TESTS
# ============================================================================

def test_extreme_corners_of_hypercube():
    """Test all corners of the feature hypercube (2^7 = 128 corners)."""
    # Test a subset of extreme corners
    corners = [
        # All low
        {"uncertainty": 0.0, "contradiction_signal": 0.0, "edge_pressure": 0.0,
         "causal_risk": 0.0, "symbolic_regularity": 0.0, "law_fit_signal": 0.0},

        # All high
        {"uncertainty": 1.0, "contradiction_signal": 1.0, "edge_pressure": 1.0,
         "causal_risk": 1.0, "symbolic_regularity": 1.0, "law_fit_signal": 1.0},

        # High uncertainty, low rest
        {"uncertainty": 1.0, "contradiction_signal": 0.0, "edge_pressure": 0.0,
         "causal_risk": 0.0, "symbolic_regularity": 0.0, "law_fit_signal": 0.0},

        # High contradiction, low rest
        {"uncertainty": 0.0, "contradiction_signal": 1.0, "edge_pressure": 0.0,
         "causal_risk": 0.0, "symbolic_regularity": 0.0, "law_fit_signal": 0.0},

        # High edge_pressure, low rest (stress case)
        {"uncertainty": 0.0, "contradiction_signal": 0.0, "edge_pressure": 1.0,
         "causal_risk": 0.0, "symbolic_regularity": 0.0, "law_fit_signal": 0.0},

        # Mixed: high uncertainty + contradiction, low rest
        {"uncertainty": 1.0, "contradiction_signal": 1.0, "edge_pressure": 0.0,
         "causal_risk": 0.0, "symbolic_regularity": 0.0, "law_fit_signal": 0.0},

        # Mixed: high edge_pressure + causal_risk (conflict)
        {"uncertainty": 0.0, "contradiction_signal": 0.0, "edge_pressure": 1.0,
         "causal_risk": 1.0, "symbolic_regularity": 0.0, "law_fit_signal": 0.0},
    ]

    for corner in corners:
        # Add continuity_recent (not tested in corners)
        corner["continuity_recent"] = 1.0

        budget = compute_budget(corner)
        sequence, _, _ = select_sequence(
            features=corner,
            budget=budget,
            allow_experimental=True
        )

        # All corners should produce valid sequences
        assert 4 <= len(sequence) <= 10, f"Invalid sequence length at corner {corner}: {sequence}"
        assert "prob" in sequence, f"Missing PROB at corner {corner}"
        assert sequence[-1] == "prob", f"PROB not last at corner {corner}"


def test_hypercube_monotonicity_corridors():
    """
    Test monotonicity along single-feature corridors through the hypercube.

    For features that should increase something (e.g., uncertainty → risk_budget),
    test that behavior is monotonic along that dimension.
    """
    # Test uncertainty corridor: should increase risk_budget
    base_features = {
        "uncertainty": 0.0,  # Will vary
        "contradiction_signal": 0.2,
        "continuity_recent": 1.0,
        "edge_pressure": 0.2,
        "causal_risk": 0.2,
        "symbolic_regularity": 0.0,
        "law_fit_signal": 0.0,
    }

    uncertainty_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    risk_budgets = []

    for unc in uncertainty_values:
        features = base_features.copy()
        features["uncertainty"] = unc

        budget = compute_budget(features)
        risk_budgets.append(budget["risk_budget"])

    # Should be monotonically increasing
    for i in range(1, len(risk_budgets)):
        assert risk_budgets[i] >= risk_budgets[i-1] - 0.01, \
            f"Risk budget decreased at uncertainty {uncertainty_values[i]}: {risk_budgets}"


def test_hypercube_stability_map():
    """Create stability map across the hypercube."""
    atlas = map_hypercube_region(n_samples=150, seed=123)

    # Group by region and check stability
    region_stats = atlas.get_region_statistics()

    unstable_regions = []

    for region_name, stats in region_stats.items():
        # A region is unstable if it has high variance or chaotic behavior
        if region_name == "chaotic":
            unstable_regions.append(region_name)

    # Most regions should be stable
    stable_region_count = len(region_stats) - len(unstable_regions)
    assert stable_region_count >= len(region_stats) * 0.8, \
        f"Too many unstable regions: {unstable_regions}"


# ============================================================================
# DENSITY ANALYSIS
# ============================================================================

def test_hypercube_region_density():
    """Test that regions have reasonable density (not all points in one region)."""
    atlas = map_hypercube_region(n_samples=200, seed=42)

    total_points = len(atlas.points)
    region_counts = {name: len(points) for name, points in atlas.regions.items()}

    # No single region should dominate (> 60% of points)
    for region_name, count in region_counts.items():
        proportion = count / total_points
        assert proportion < 0.6, \
            f"Region '{region_name}' dominates with {proportion:.1%} of points"

    # Should have reasonable diversity (at least 3 regions with >5% each)
    significant_regions = [name for name, count in region_counts.items()
                          if count / total_points >= 0.05]

    assert len(significant_regions) >= 3, \
        f"Too few significant regions: {significant_regions}"
