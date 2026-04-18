"""
Boundary Sweep Tests - Elite Level
===================================

Tests for mapping activation frontiers of each reasoning family with high precision.
Measures discontinuities, stability, and threshold behavior near critical values.
"""

from __future__ import annotations

import pytest
from typing import Dict, List, Tuple
from runtime.reasoning.scheduler_meta.budgeting import compute_budget
from runtime.reasoning.scheduler_meta.policy import select_sequence
from runtime.reasoning.scheduler_meta.context_features import extract_context_features


class BoundarySweepResult:
    """Captures results from boundary sweep analysis."""

    def __init__(self, feature_name: str, threshold: float):
        self.feature_name = feature_name
        self.threshold = threshold
        self.activation_changes: List[Tuple[float, List[str]]] = []
        self.budget_changes: List[Tuple[float, Dict]] = []
        self.discontinuities: List[float] = []

    def add_measurement(self, value: float, families: List[str], budget: Dict):
        self.activation_changes.append((value, families))
        self.budget_changes.append((value, budget))

    def detect_discontinuities(self, window: float = 0.02) -> List[float]:
        """Detect sharp changes in activation patterns."""
        discontinuities = []
        for i in range(1, len(self.activation_changes)):
            prev_val, prev_families = self.activation_changes[i-1]
            curr_val, curr_families = self.activation_changes[i]

            if set(prev_families) != set(curr_families):
                if abs(curr_val - prev_val) <= window:
                    discontinuities.append(curr_val)

        self.discontinuities = discontinuities
        return discontinuities

    def measure_stability(self) -> float:
        """Measure stability as inverse of discontinuity count."""
        if not self.activation_changes:
            return 0.0
        return 1.0 - (len(self.discontinuities) / len(self.activation_changes))


def sweep_feature_boundary(
    feature_name: str,
    threshold: float,
    sweep_range: Tuple[float, float] = (-0.05, 0.05),
    steps: int = 21,
    baseline_features: Dict[str, float] | None = None
) -> BoundarySweepResult:
    """
    Perform fine-grained sweep around a threshold.

    Args:
        feature_name: Name of feature to sweep
        threshold: Threshold value to sweep around
        sweep_range: Relative range (min_offset, max_offset)
        steps: Number of measurement points
        baseline_features: Base feature values

    Returns:
        BoundarySweepResult with detailed measurements
    """
    if baseline_features is None:
        baseline_features = {
            "uncertainty": 0.25,
            "contradiction_signal": 0.0,
            "continuity_recent": 1.0,
            "edge_pressure": 0.0,
            "causal_risk": 0.0,
            "symbolic_regularity": 0.0,
            "law_fit_signal": 0.0,
        }

    result = BoundarySweepResult(feature_name, threshold)

    start = threshold + sweep_range[0]
    end = threshold + sweep_range[1]
    step_size = (end - start) / (steps - 1)

    for i in range(steps):
        value = start + (i * step_size)
        value = max(0.0, min(1.0, value))  # Clamp to [0, 1]

        # Create test context
        test_features = baseline_features.copy()
        test_features[feature_name] = value

        # Compute budget and sequence
        budget = compute_budget(test_features)
        sequence, _, _ = select_sequence(
            features=test_features,
            budget=budget,
            allow_experimental=True
        )

        result.add_measurement(value, sequence, budget)

    result.detect_discontinuities()
    return result


# ============================================================================
# EDGE PRESSURE BOUNDARY TESTS
# ============================================================================

def test_edge_pressure_heur_activation_boundary():
    """Test HEUR activation frontier at edge_pressure >= 0.7"""
    result = sweep_feature_boundary(
        feature_name="edge_pressure",
        threshold=0.7,
        sweep_range=(-0.05, 0.05),
        steps=21
    )

    # Find where HEUR enters
    heur_activation_points = [
        (val, "heur" in families)
        for val, families in result.activation_changes
    ]

    # Should activate around 0.7
    activations_below = [active for val, active in heur_activation_points if val < 0.7]
    activations_above = [active for val, active in heur_activation_points if val >= 0.7]

    # Most activations below threshold should be False
    assert sum(activations_below) / max(len(activations_below), 1) < 0.3

    # Most activations above threshold should be True
    assert sum(activations_above) / max(len(activations_above), 1) > 0.7

    # Should have exactly one major discontinuity near threshold
    assert len(result.discontinuities) >= 1
    assert any(0.65 <= d <= 0.75 for d in result.discontinuities)


def test_edge_pressure_budget_reduction():
    """Test that edge_pressure >= 0.8 reduces max_steps"""
    result_low = sweep_feature_boundary(
        feature_name="edge_pressure",
        threshold=0.8,
        sweep_range=(-0.1, -0.05),
        steps=5
    )

    result_high = sweep_feature_boundary(
        feature_name="edge_pressure",
        threshold=0.8,
        sweep_range=(0.05, 0.1),
        steps=5
    )

    avg_steps_low = sum(b["max_steps"] for _, b in result_low.budget_changes) / len(result_low.budget_changes)
    avg_steps_high = sum(b["max_steps"] for _, b in result_high.budget_changes) / len(result_high.budget_changes)

    # High edge pressure should reduce steps
    assert avg_steps_high <= avg_steps_low


def test_edge_pressure_fine_grain_stability():
    """Test stability in narrow window around 0.7 threshold"""
    result = sweep_feature_boundary(
        feature_name="edge_pressure",
        threshold=0.7,
        sweep_range=(-0.01, 0.01),
        steps=11
    )

    # Should have smooth transition, not multiple oscillations
    assert len(result.discontinuities) <= 2

    # Stability metric should be reasonable
    stability = result.measure_stability()
    assert stability >= 0.7


# ============================================================================
# CONTRADICTION SIGNAL BOUNDARY TESTS
# ============================================================================

def test_contradiction_dia_adv_activation_boundary():
    """Test DIA_ADV activation frontier at contradiction_signal >= 0.45"""
    result = sweep_feature_boundary(
        feature_name="contradiction_signal",
        threshold=0.45,
        sweep_range=(-0.05, 0.05),
        steps=21
    )

    dia_adv_points = [
        (val, "dia_adv" in families)
        for val, families in result.activation_changes
    ]

    activations_below = [active for val, active in dia_adv_points if val < 0.45]
    activations_above = [active for val, active in dia_adv_points if val >= 0.45]

    assert sum(activations_below) / max(len(activations_below), 1) < 0.3
    assert sum(activations_above) / max(len(activations_above), 1) > 0.7


def test_contradiction_fal_guard_activation_boundary():
    """Test FAL_GUARD activation frontier at contradiction_signal >= 0.45"""
    result = sweep_feature_boundary(
        feature_name="contradiction_signal",
        threshold=0.45,
        sweep_range=(-0.05, 0.05),
        steps=21
    )

    fal_guard_points = [
        (val, "fal_guard" in families)
        for val, families in result.activation_changes
    ]

    activations_below = [active for val, active in fal_guard_points if val < 0.45]
    activations_above = [active for val, active in fal_guard_points if val >= 0.45]

    assert sum(activations_below) / max(len(activations_below), 1) < 0.3
    assert sum(activations_above) / max(len(activations_above), 1) > 0.7


def test_contradiction_both_guards_activate_together():
    """Test that DIA_ADV and FAL_GUARD activate together"""
    result = sweep_feature_boundary(
        feature_name="contradiction_signal",
        threshold=0.45,
        sweep_range=(0.0, 0.1),
        steps=11
    )

    for val, families in result.activation_changes:
        if val >= 0.45:
            # If one is active, both should be active
            has_dia = "dia_adv" in families
            has_fal = "fal_guard" in families
            assert has_dia == has_fal


def test_contradiction_risk_budget_increase():
    """Test that higher contradiction increases risk_budget"""
    result = sweep_feature_boundary(
        feature_name="contradiction_signal",
        threshold=0.5,
        sweep_range=(-0.3, 0.3),
        steps=13
    )

    # Extract risk budgets
    risk_budgets = [(val, b["risk_budget"]) for val, b in result.budget_changes]

    # Should be monotonically increasing or stable
    for i in range(1, len(risk_budgets)):
        prev_val, prev_risk = risk_budgets[i-1]
        curr_val, curr_risk = risk_budgets[i]

        # Higher contradiction should not decrease risk budget
        assert curr_risk >= prev_risk - 0.01  # Allow tiny floating point error


# ============================================================================
# SYMBOLIC REGULARITY BOUNDARY TESTS
# ============================================================================

def test_symbolic_regularity_eml_sr_activation():
    """Test EML_SR activation at symbolic_regularity >= 0.4"""
    result = sweep_feature_boundary(
        feature_name="symbolic_regularity",
        threshold=0.4,
        sweep_range=(-0.05, 0.05),
        steps=21
    )

    eml_points = [
        (val, "eml_sr" in families)
        for val, families in result.activation_changes
    ]

    activations_below = [active for val, active in eml_points if val < 0.4]
    activations_above = [active for val, active in eml_points if val >= 0.4]

    # Note: EML_SR requires allow_experimental=True, which we set in sweep
    assert sum(activations_below) / max(len(activations_below), 1) < 0.3
    assert sum(activations_above) / max(len(activations_above), 1) > 0.7


def test_law_fit_signal_eml_sr_activation():
    """Test EML_SR activation at law_fit_signal >= 0.4"""
    result = sweep_feature_boundary(
        feature_name="law_fit_signal",
        threshold=0.4,
        sweep_range=(-0.05, 0.05),
        steps=21
    )

    eml_points = [
        (val, "eml_sr" in families)
        for val, families in result.activation_changes
    ]

    activations_below = [active for val, active in eml_points if val < 0.4]
    activations_above = [active for val, active in eml_points if val >= 0.4]

    assert sum(activations_below) / max(len(activations_below), 1) < 0.3
    assert sum(activations_above) / max(len(activations_above), 1) > 0.7


# ============================================================================
# UNCERTAINTY BOUNDARY TESTS
# ============================================================================

def test_uncertainty_max_steps_increase():
    """Test that uncertainty >= 0.6 increases max_steps"""
    result = sweep_feature_boundary(
        feature_name="uncertainty",
        threshold=0.6,
        sweep_range=(-0.05, 0.05),
        steps=21
    )

    steps_below = [b["max_steps"] for val, b in result.budget_changes if val < 0.6]
    steps_above = [b["max_steps"] for val, b in result.budget_changes if val >= 0.6]

    avg_below = sum(steps_below) / len(steps_below)
    avg_above = sum(steps_above) / len(steps_above)

    # Uncertainty >= 0.6 should increase budget
    assert avg_above > avg_below


def test_uncertainty_risk_budget_correlation():
    """Test that risk_budget increases monotonically with uncertainty"""
    result = sweep_feature_boundary(
        feature_name="uncertainty",
        threshold=0.5,
        sweep_range=(-0.5, 0.5),
        steps=21
    )

    # Should show positive correlation
    risk_values = [b["risk_budget"] for _, b in result.budget_changes]

    # Check monotonicity (allowing small floating point errors)
    monotonic_violations = 0
    for i in range(1, len(risk_values)):
        if risk_values[i] < risk_values[i-1] - 0.01:
            monotonic_violations += 1

    # Should be mostly monotonic
    assert monotonic_violations <= 2


# ============================================================================
# CAUSAL RISK BOUNDARY TESTS
# ============================================================================

def test_causal_risk_max_steps_increase():
    """Test that causal_risk >= 0.5 increases max_steps"""
    result = sweep_feature_boundary(
        feature_name="causal_risk",
        threshold=0.5,
        sweep_range=(-0.05, 0.05),
        steps=21
    )

    steps_below = [b["max_steps"] for val, b in result.budget_changes if val < 0.5]
    steps_above = [b["max_steps"] for val, b in result.budget_changes if val >= 0.5]

    avg_below = sum(steps_below) / len(steps_below)
    avg_above = sum(steps_above) / len(steps_above)

    assert avg_above > avg_below


# ============================================================================
# MULTI-THRESHOLD STRESS TEST
# ============================================================================

def test_all_thresholds_have_clean_boundaries():
    """Meta-test: all major thresholds should have clean activation boundaries"""

    thresholds_to_test = [
        ("edge_pressure", 0.7, "heur"),
        ("edge_pressure", 0.8, None),  # budget change
        ("contradiction_signal", 0.45, "dia_adv"),
        ("uncertainty", 0.6, None),  # budget change
        ("causal_risk", 0.5, None),  # budget change
        ("symbolic_regularity", 0.4, "eml_sr"),
        ("law_fit_signal", 0.4, "eml_sr"),
    ]

    all_clean = True
    results = []

    for feature_name, threshold, expected_family in thresholds_to_test:
        result = sweep_feature_boundary(
            feature_name=feature_name,
            threshold=threshold,
            sweep_range=(-0.05, 0.05),
            steps=21
        )

        stability = result.measure_stability()
        discontinuity_count = len(result.discontinuities)

        # A "clean" boundary has:
        # 1. High stability (>0.6)
        # 2. Few discontinuities (1-3)
        is_clean = stability >= 0.6 and discontinuity_count <= 3

        results.append({
            "feature": feature_name,
            "threshold": threshold,
            "stability": stability,
            "discontinuities": discontinuity_count,
            "clean": is_clean
        })

        if not is_clean:
            all_clean = False

    # Report all results
    for r in results:
        print(f"{r['feature']}@{r['threshold']}: stability={r['stability']:.2f}, disc={r['discontinuities']}, clean={r['clean']}")

    assert all_clean, "Some thresholds have unstable boundaries"
