"""
Temporal Hysteresis Tests - Elite Level
========================================

Tests for temporal dynamics, hysteresis effects, and sequence stability.
Measures activation/deactivation symmetry, memory effects, and oscillation
resistance.
"""

from __future__ import annotations

import pytest
from typing import Dict, List, Tuple
from dataclasses import dataclass
from runtime.reasoning.scheduler_meta.budgeting import compute_budget
from runtime.reasoning.scheduler_meta.policy import select_sequence


@dataclass
class HysteresisResult:
    """Results from a hysteresis test."""
    feature_name: str
    ascending_path: List[Tuple[float, List[str], Dict]]
    descending_path: List[Tuple[float, List[str], Dict]]
    activation_point: float | None = None
    deactivation_point: float | None = None
    hysteresis_width: float | None = None
    has_hysteresis: bool = False


def measure_hysteresis(
    feature_name: str,
    start_value: float,
    end_value: float,
    steps: int,
    target_family: str | None = None,
    baseline_features: Dict[str, float] | None = None
) -> HysteresisResult:
    """
    Measure hysteresis by ascending and descending through a feature range.

    Args:
        feature_name: Feature to vary
        start_value: Starting value (typically low)
        end_value: Ending value (typically high)
        steps: Number of steps in each direction
        target_family: Family to track activation (if None, tracks any change)
        baseline_features: Base feature values

    Returns:
        HysteresisResult with analysis
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

    result = HysteresisResult(
        feature_name=feature_name,
        ascending_path=[],
        descending_path=[]
    )

    # Ascending path
    step_size = (end_value - start_value) / (steps - 1)
    for i in range(steps):
        value = start_value + (i * step_size)
        value = max(0.0, min(1.0, value))

        features = baseline_features.copy()
        features[feature_name] = value

        budget = compute_budget(features)
        sequence, _, _ = select_sequence(
            features=features,
            budget=budget,
            allow_experimental=True
        )

        result.ascending_path.append((value, sequence, budget))

    # Descending path
    for i in range(steps):
        value = end_value - (i * step_size)
        value = max(0.0, min(1.0, value))

        features = baseline_features.copy()
        features[feature_name] = value

        budget = compute_budget(features)
        sequence, _, _ = select_sequence(
            features=features,
            budget=budget,
            allow_experimental=True
        )

        result.descending_path.append((value, sequence, budget))

    # Analyze hysteresis
    if target_family:
        # Find activation point (ascending)
        activation_point = None
        for i, (value, sequence, _) in enumerate(result.ascending_path):
            if target_family in sequence:
                if i == 0 or target_family not in result.ascending_path[i-1][1]:
                    activation_point = value
                    break

        # Find deactivation point (descending)
        deactivation_point = None
        for i, (value, sequence, _) in enumerate(result.descending_path):
            if target_family not in sequence:
                if i == 0 or target_family in result.descending_path[i-1][1]:
                    deactivation_point = value
                    break

        if activation_point is not None and deactivation_point is not None:
            result.activation_point = activation_point
            result.deactivation_point = deactivation_point
            result.hysteresis_width = abs(activation_point - deactivation_point)
            result.has_hysteresis = result.hysteresis_width > 0.02  # Significant hysteresis

    return result


# ============================================================================
# EDGE_PRESSURE HYSTERESIS TESTS
# ============================================================================

def test_edge_pressure_heur_hysteresis():
    """Test hysteresis in HEUR activation around edge_pressure 0.7"""
    result = measure_hysteresis(
        feature_name="edge_pressure",
        start_value=0.5,
        end_value=0.9,
        steps=20,
        target_family="heur"
    )

    # Should activate around 0.7 going up
    assert result.activation_point is not None
    assert 0.65 <= result.activation_point <= 0.75

    # Should deactivate around same point going down
    if result.deactivation_point is not None:
        # Hysteresis should be minimal (< 0.05)
        assert result.hysteresis_width < 0.05, \
            f"Excessive hysteresis: {result.hysteresis_width}"


def test_edge_pressure_budget_hysteresis():
    """Test that budget changes are symmetric (no hysteresis)."""
    result = measure_hysteresis(
        feature_name="edge_pressure",
        start_value=0.6,
        end_value=1.0,
        steps=15
    )

    # Compare budgets at same values in both directions
    # Create dict for ascending
    ascending_budgets = {val: budget["max_steps"] for val, _, budget in result.ascending_path}

    # Check descending matches
    for val, _, budget in result.descending_path:
        # Find closest ascending value
        closest_val = min(ascending_budgets.keys(), key=lambda v: abs(v - val))
        if abs(closest_val - val) < 0.01:  # Close enough
            ascending_steps = ascending_budgets[closest_val]
            descending_steps = budget["max_steps"]

            assert ascending_steps == descending_steps, \
                f"Budget hysteresis at {val}: ascending={ascending_steps}, descending={descending_steps}"


# ============================================================================
# CONTRADICTION SIGNAL HYSTERESIS TESTS
# ============================================================================

def test_contradiction_dia_adv_hysteresis():
    """Test hysteresis in DIA_ADV activation around contradiction 0.45"""
    result = measure_hysteresis(
        feature_name="contradiction_signal",
        start_value=0.2,
        end_value=0.7,
        steps=20,
        target_family="dia_adv"
    )

    # Should activate around 0.45
    assert result.activation_point is not None
    assert 0.43 <= result.activation_point <= 0.47

    # Hysteresis should be minimal
    if result.deactivation_point is not None:
        assert result.hysteresis_width < 0.05


def test_contradiction_fal_guard_hysteresis():
    """Test hysteresis in FAL_GUARD activation."""
    result = measure_hysteresis(
        feature_name="contradiction_signal",
        start_value=0.2,
        end_value=0.7,
        steps=20,
        target_family="fal_guard"
    )

    # Should match dia_adv behavior (they activate together)
    assert result.activation_point is not None
    assert 0.43 <= result.activation_point <= 0.47

    if result.deactivation_point is not None:
        assert result.hysteresis_width < 0.05


def test_contradiction_both_guards_symmetric():
    """Test that dia_adv and fal_guard have symmetric hysteresis."""
    result_dia = measure_hysteresis(
        feature_name="contradiction_signal",
        start_value=0.2,
        end_value=0.7,
        steps=20,
        target_family="dia_adv"
    )

    result_fal = measure_hysteresis(
        feature_name="contradiction_signal",
        start_value=0.2,
        end_value=0.7,
        steps=20,
        target_family="fal_guard"
    )

    # Activation points should match (activate together)
    if result_dia.activation_point and result_fal.activation_point:
        assert abs(result_dia.activation_point - result_fal.activation_point) < 0.01

    # Deactivation points should match
    if result_dia.deactivation_point and result_fal.deactivation_point:
        assert abs(result_dia.deactivation_point - result_fal.deactivation_point) < 0.01


# ============================================================================
# SYMBOLIC REGULARITY HYSTERESIS TESTS
# ============================================================================

def test_symbolic_regularity_eml_sr_hysteresis():
    """Test hysteresis in EML_SR activation around symbolic_regularity 0.4"""
    result = measure_hysteresis(
        feature_name="symbolic_regularity",
        start_value=0.2,
        end_value=0.7,
        steps=20,
        target_family="eml_sr"
    )

    # Should activate around 0.4
    assert result.activation_point is not None
    assert 0.38 <= result.activation_point <= 0.42

    # Minimal hysteresis
    if result.deactivation_point is not None:
        assert result.hysteresis_width < 0.05


def test_law_fit_signal_eml_sr_hysteresis():
    """Test hysteresis in EML_SR activation via law_fit_signal."""
    result = measure_hysteresis(
        feature_name="law_fit_signal",
        start_value=0.2,
        end_value=0.7,
        steps=20,
        target_family="eml_sr"
    )

    # Should activate around 0.4
    assert result.activation_point is not None
    assert 0.38 <= result.activation_point <= 0.42

    # Minimal hysteresis
    if result.deactivation_point is not None:
        assert result.hysteresis_width < 0.05


# ============================================================================
# OSCILLATION RESISTANCE TESTS
# ============================================================================

def test_no_oscillation_near_threshold():
    """Test that system doesn't oscillate when signal hovers near threshold."""
    # Simulate oscillating signal around edge_pressure 0.7
    oscillating_values = [
        0.68, 0.72, 0.69, 0.71, 0.68, 0.72, 0.70, 0.69, 0.71, 0.70
    ]

    activations = []

    for val in oscillating_values:
        features = {
            "uncertainty": 0.3,
            "contradiction_signal": 0.2,
            "continuity_recent": 1.0,
            "edge_pressure": val,
            "causal_risk": 0.2,
            "symbolic_regularity": 0.0,
            "law_fit_signal": 0.0,
        }

        budget = compute_budget(features)
        sequence, _, _ = select_sequence(features=features, budget=budget, allow_experimental=True)

        activations.append("heur" in sequence)

    # Count transitions
    transitions = sum(1 for i in range(1, len(activations)) if activations[i] != activations[i-1])

    # Should stabilize quickly, not oscillate continuously
    # Allow 1-2 transitions initially, but should settle
    assert transitions <= 4, f"Excessive oscillation: {transitions} transitions in {activations}"


def test_no_oscillation_multiple_features():
    """Test that system doesn't oscillate when multiple features vary."""
    # Simulate scenario where uncertainty and contradiction both vary near thresholds
    test_sequence = [
        {"uncertainty": 0.58, "contradiction_signal": 0.43},
        {"uncertainty": 0.62, "contradiction_signal": 0.47},
        {"uncertainty": 0.59, "contradiction_signal": 0.44},
        {"uncertainty": 0.61, "contradiction_signal": 0.46},
        {"uncertainty": 0.60, "contradiction_signal": 0.45},
    ]

    sequences = []

    for features_delta in test_sequence:
        features = {
            "uncertainty": features_delta["uncertainty"],
            "contradiction_signal": features_delta["contradiction_signal"],
            "continuity_recent": 1.0,
            "edge_pressure": 0.2,
            "causal_risk": 0.2,
            "symbolic_regularity": 0.0,
            "law_fit_signal": 0.0,
        }

        budget = compute_budget(features)
        sequence, _, _ = select_sequence(features=features, budget=budget, allow_experimental=True)

        sequences.append(sequence)

    # Count how many times sequence changes
    sequence_changes = sum(1 for i in range(1, len(sequences)) if sequences[i] != sequences[i-1])

    # Should have some stability, not change every time
    assert sequence_changes <= 3, f"Too much instability: sequences={sequences}"


# ============================================================================
# TEMPORAL CONSISTENCY TESTS
# ============================================================================

def test_gradual_increase_stability():
    """Test that gradual increases produce stable sequences."""
    # Gradually increase uncertainty from low to high
    values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    sequences = []
    budgets = []

    for unc in values:
        features = {
            "uncertainty": unc,
            "contradiction_signal": 0.2,
            "continuity_recent": 1.0,
            "edge_pressure": 0.2,
            "causal_risk": 0.2,
            "symbolic_regularity": 0.0,
            "law_fit_signal": 0.0,
        }

        budget = compute_budget(features)
        sequence, _, _ = select_sequence(features=features, budget=budget, allow_experimental=True)

        sequences.append(sequence)
        budgets.append(budget["max_steps"])

    # Budgets should increase or stay same (monotonic)
    for i in range(1, len(budgets)):
        assert budgets[i] >= budgets[i-1], \
            f"Budget decreased with increasing uncertainty: {budgets}"

    # Sequences should grow or stay same (no sudden contractions)
    for i in range(1, len(sequences)):
        prev_set = set(sequences[i-1])
        curr_set = set(sequences[i])

        # Current should contain most of previous (allowing some evolution)
        common = prev_set & curr_set
        assert len(common) >= len(prev_set) - 2, \
            f"Too much sequence instability at step {i}: {sequences[i-1]} -> {sequences[i]}"


def test_gradual_decrease_stability():
    """Test that gradual decreases produce stable sequences."""
    # Gradually decrease contradiction from high to low
    values = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

    sequences = []

    for contr in values:
        features = {
            "uncertainty": 0.3,
            "contradiction_signal": contr,
            "continuity_recent": 1.0,
            "edge_pressure": 0.2,
            "causal_risk": 0.2,
            "symbolic_regularity": 0.0,
            "law_fit_signal": 0.0,
        }

        budget = compute_budget(features)
        sequence, _, _ = select_sequence(features=features, budget=budget, allow_experimental=True)

        sequences.append(sequence)

    # Should have smooth deactivation
    # dia_adv and fal_guard should disappear at same point
    dia_adv_active = ["dia_adv" in seq for seq in sequences]
    fal_guard_active = ["fal_guard" in seq for seq in sequences]

    # Should match
    assert dia_adv_active == fal_guard_active, "Dia_adv and fal_guard deactivate asymmetrically"


# ============================================================================
# MEMORY AND INERTIA TESTS
# ============================================================================

def test_no_undesired_memory_effects():
    """Test that past evaluations don't affect current ones (stateless)."""
    # Run same features multiple times in different contexts
    test_features = {
        "uncertainty": 0.6,
        "contradiction_signal": 0.5,
        "continuity_recent": 0.8,
        "edge_pressure": 0.3,
        "causal_risk": 0.4,
        "symbolic_regularity": 0.2,
        "law_fit_signal": 0.2,
    }

    # Run after various different contexts
    contexts = [
        {"uncertainty": 0.0, "contradiction_signal": 0.0},
        {"uncertainty": 1.0, "contradiction_signal": 1.0},
        {"uncertainty": 0.5, "contradiction_signal": 0.2},
    ]

    results = []

    for _ in range(3):  # Repeat for each context
        for context in contexts:
            # Run context features
            _ = compute_budget(context)
            _ = select_sequence(features=context, budget={"max_steps": 6.0})

            # Now run test features
            budget = compute_budget(test_features)
            sequence, _, _ = select_sequence(
                features=test_features,
                budget=budget,
                allow_experimental=True
            )

            results.append((budget["max_steps"], sequence))

    # All results should be identical (stateless)
    first_result = results[0]
    for result in results[1:]:
        assert result == first_result, "System has unwanted memory/inertia"


def test_reset_after_extreme_conditions():
    """Test that system recovers from extreme conditions."""
    # Run with extreme features
    extreme_features = {
        "uncertainty": 1.0,
        "contradiction_signal": 1.0,
        "continuity_recent": 0.0,
        "edge_pressure": 1.0,
        "causal_risk": 1.0,
        "symbolic_regularity": 1.0,
        "law_fit_signal": 1.0,
    }

    budget_extreme = compute_budget(extreme_features)
    seq_extreme, _, _ = select_sequence(
        features=extreme_features,
        budget=budget_extreme,
        allow_experimental=True
    )

    # Now run with normal features
    normal_features = {
        "uncertainty": 0.3,
        "contradiction_signal": 0.2,
        "continuity_recent": 1.0,
        "edge_pressure": 0.2,
        "causal_risk": 0.2,
        "symbolic_regularity": 0.0,
        "law_fit_signal": 0.0,
    }

    budget_normal = compute_budget(normal_features)
    seq_normal, _, _ = select_sequence(
        features=normal_features,
        budget=budget_normal,
        allow_experimental=True
    )

    # Normal features should produce normal result (not affected by extreme)
    assert 4 <= len(seq_normal) <= 7
    assert "heur" not in seq_normal
    assert "dia_adv" not in seq_normal
    assert "eml_sr" not in seq_normal


# ============================================================================
# COMPREHENSIVE HYSTERESIS ANALYSIS
# ============================================================================

def test_all_thresholds_minimal_hysteresis():
    """Meta-test: all major thresholds should have minimal hysteresis."""
    threshold_tests = [
        ("edge_pressure", 0.5, 0.9, "heur"),
        ("contradiction_signal", 0.2, 0.7, "dia_adv"),
        ("symbolic_regularity", 0.2, 0.7, "eml_sr"),
        ("law_fit_signal", 0.2, 0.7, "eml_sr"),
    ]

    hysteresis_results = []

    for feature_name, start, end, target_family in threshold_tests:
        result = measure_hysteresis(
            feature_name=feature_name,
            start_value=start,
            end_value=end,
            steps=20,
            target_family=target_family
        )

        if result.has_hysteresis:
            hysteresis_results.append({
                "feature": feature_name,
                "family": target_family,
                "width": result.hysteresis_width
            })

    # Should have minimal hysteresis overall
    assert len(hysteresis_results) == 0, \
        f"Significant hysteresis detected: {hysteresis_results}"


def test_bidirectional_consistency():
    """Test that up-and-down paths are consistent for all features."""
    features_to_test = [
        "uncertainty",
        "contradiction_signal",
        "edge_pressure",
        "causal_risk",
    ]

    for feature_name in features_to_test:
        result = measure_hysteresis(
            feature_name=feature_name,
            start_value=0.2,
            end_value=0.8,
            steps=15
        )

        # Compare budgets at matching points
        ascending_dict = {val: budget for val, _, budget in result.ascending_path}
        descending_dict = {val: budget for val, _, budget in result.descending_path}

        # Find overlapping values
        for asc_val, asc_budget in ascending_dict.items():
            # Find closest descending value
            closest_desc_val = min(descending_dict.keys(), key=lambda v: abs(v - asc_val))

            if abs(closest_desc_val - asc_val) < 0.02:  # Close enough
                desc_budget = descending_dict[closest_desc_val]

                # Budgets should match
                assert asc_budget["max_steps"] == desc_budget["max_steps"], \
                    f"Budget mismatch for {feature_name} at {asc_val}: " \
                    f"asc={asc_budget['max_steps']}, desc={desc_budget['max_steps']}"
