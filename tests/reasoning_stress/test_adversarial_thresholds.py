"""
Adversarial Threshold Tests - Elite Level
==========================================

Tests designed to break the scheduler with edge cases and deceptive signals.
Validates robustness against false activations, misleading thresholds, and
conflicting policies.
"""

from __future__ import annotations

import pytest
from typing import Dict, List
from runtime.reasoning.scheduler_meta.budgeting import compute_budget
from runtime.reasoning.scheduler_meta.policy import select_sequence


# ============================================================================
# FALSE POSITIVE ADVERSARIAL TESTS
# ============================================================================

def test_adversarial_just_below_threshold_high_noise():
    """Test: high noise just below threshold shouldn't trigger activation."""
    # Contradiction at 0.44 (just below 0.45) but with other high signals
    features = {
        "uncertainty": 0.9,
        "contradiction_signal": 0.44,  # Just below threshold
        "continuity_recent": 0.2,
        "edge_pressure": 0.6,
        "causal_risk": 0.8,
        "symbolic_regularity": 0.0,
        "law_fit_signal": 0.0,
    }

    budget = compute_budget(features)
    sequence, _, _ = select_sequence(features=features, budget=budget, allow_experimental=True)

    # Should NOT activate dia_adv/fal_guard despite high noise
    assert "dia_adv" not in sequence, "False positive: dia_adv activated below threshold"
    assert "fal_guard" not in sequence, "False positive: fal_guard activated below threshold"


def test_adversarial_edge_pressure_false_activation():
    """Test: edge_pressure at 0.69 shouldn't activate HEUR."""
    features = {
        "uncertainty": 0.8,
        "contradiction_signal": 0.5,
        "continuity_recent": 0.9,
        "edge_pressure": 0.69,  # Just below 0.7
        "causal_risk": 0.5,
        "symbolic_regularity": 0.0,
        "law_fit_signal": 0.0,
    }

    budget = compute_budget(features)
    sequence, _, _ = select_sequence(features=features, budget=budget, allow_experimental=True)

    # Should NOT activate heur
    assert "heur" not in sequence, "False positive: heur activated below threshold"


def test_adversarial_symbolic_spurious_high():
    """Test: high symbolic_regularity but zero law_fit shouldn't overactivate."""
    features = {
        "uncertainty": 0.2,
        "contradiction_signal": 0.1,
        "continuity_recent": 1.0,
        "edge_pressure": 0.1,
        "causal_risk": 0.1,
        "symbolic_regularity": 0.9,  # High but artificial
        "law_fit_signal": 0.0,  # Low law fit
    }

    budget = compute_budget(features)
    sequence, _, _ = select_sequence(features=features, budget=budget, allow_experimental=True)

    # Should activate eml_sr (meets threshold)
    assert "eml_sr" in sequence

    # But shouldn't cause overactivation of other families
    assert len(sequence) <= 8


def test_adversarial_law_fit_spurious_high():
    """Test: high law_fit but zero symbolic_regularity."""
    features = {
        "uncertainty": 0.2,
        "contradiction_signal": 0.1,
        "continuity_recent": 1.0,
        "edge_pressure": 0.1,
        "causal_risk": 0.1,
        "symbolic_regularity": 0.0,  # Low symbolic
        "law_fit_signal": 0.9,  # High but artificial
    }

    budget = compute_budget(features)
    sequence, _, _ = select_sequence(features=features, budget=budget, allow_experimental=True)

    # Should activate eml_sr (meets threshold)
    assert "eml_sr" in sequence

    # Should still be reasonable
    assert len(sequence) <= 8


# ============================================================================
# BUDGET MANIPULATION ADVERSARIAL TESTS
# ============================================================================

def test_adversarial_extreme_edge_pressure_doesnt_destroy():
    """Test: extreme edge_pressure shouldn't collapse system."""
    features = {
        "uncertainty": 0.7,
        "contradiction_signal": 0.8,
        "continuity_recent": 0.5,
        "edge_pressure": 1.0,  # Maximum pressure
        "causal_risk": 0.7,
        "symbolic_regularity": 0.0,
        "law_fit_signal": 0.0,
    }

    budget = compute_budget(features)
    sequence, _, _ = select_sequence(features=features, budget=budget, allow_experimental=True)

    # Even with maximum edge pressure, should maintain minimum functionality
    assert budget["max_steps"] >= 4, "Edge pressure collapsed budget too much"
    assert len(sequence) >= 4, "Edge pressure collapsed sequence"

    # Core families should survive
    assert "abd" in sequence
    assert "prob" in sequence


def test_adversarial_conflicting_budget_signals():
    """Test: conflicting signals (edge_pressure vs causal_risk) at extremes."""
    features = {
        "uncertainty": 0.5,
        "contradiction_signal": 0.3,
        "continuity_recent": 0.8,
        "edge_pressure": 1.0,  # Wants to reduce
        "causal_risk": 1.0,    # Wants to increase
        "symbolic_regularity": 0.0,
        "law_fit_signal": 0.0,
    }

    budget = compute_budget(features)
    sequence, _, _ = select_sequence(features=features, budget=budget, allow_experimental=True)

    # Should find middle ground
    assert 4 <= budget["max_steps"] <= 8
    assert 4 <= len(sequence) <= 8

    # Should not oscillate chaotically
    # Re-run should give same result
    budget2 = compute_budget(features)
    sequence2, _, _ = select_sequence(features=features, budget=budget2, allow_experimental=True)

    assert budget["max_steps"] == budget2["max_steps"]
    assert sequence == sequence2


def test_adversarial_maximum_all_increases():
    """Test: all budget-increasing signals at maximum."""
    features = {
        "uncertainty": 1.0,  # Increases
        "contradiction_signal": 1.0,  # Increases risk
        "continuity_recent": 0.0,  # Low continuity
        "edge_pressure": 0.0,  # Low (doesn't reduce)
        "causal_risk": 1.0,  # Increases
        "symbolic_regularity": 0.0,
        "law_fit_signal": 0.0,
    }

    budget = compute_budget(features)
    sequence, _, _ = select_sequence(features=features, budget=budget, allow_experimental=True)

    # Should not exceed hard limits
    assert budget["max_steps"] <= 10, "Budget exceeded hard limit"
    assert len(sequence) <= 10, "Sequence exceeded hard limit"

    # Should activate appropriate families
    assert "dia_adv" in sequence
    assert "fal_guard" in sequence


# ============================================================================
# DECEPTIVE PATTERN ADVERSARIAL TESTS
# ============================================================================

def test_adversarial_high_coherence_low_signal():
    """Test: high continuity with contradictory low-level signals."""
    features = {
        "uncertainty": 0.8,  # High uncertainty
        "contradiction_signal": 0.0,  # But no contradiction?
        "continuity_recent": 1.0,  # High continuity
        "edge_pressure": 0.0,
        "causal_risk": 0.8,  # High causal risk
        "symbolic_regularity": 0.0,
        "law_fit_signal": 0.0,
    }

    budget = compute_budget(features)
    sequence, _, _ = select_sequence(features=features, budget=budget, allow_experimental=True)

    # Should trust the actual signals, not continuity
    # High uncertainty + causal_risk should increase budget
    assert budget["max_steps"] >= 7


def test_adversarial_oscillating_near_threshold():
    """Test: rapid oscillation near activation threshold."""
    # Test multiple points very close to threshold
    threshold_tests = [
        0.698, 0.699, 0.700, 0.701, 0.702  # Around edge_pressure 0.7
    ]

    sequences = []

    for edge_val in threshold_tests:
        features = {
            "uncertainty": 0.3,
            "contradiction_signal": 0.2,
            "continuity_recent": 1.0,
            "edge_pressure": edge_val,
            "causal_risk": 0.2,
            "symbolic_regularity": 0.0,
            "law_fit_signal": 0.0,
        }

        budget = compute_budget(features)
        sequence, _, _ = select_sequence(features=features, budget=budget, allow_experimental=True)
        sequences.append((edge_val, "heur" in sequence))

    # Should have clean transition, not oscillation
    # Count number of transitions
    transitions = 0
    for i in range(1, len(sequences)):
        if sequences[i][1] != sequences[i-1][1]:
            transitions += 1

    # Should have at most 1 transition (off -> on)
    assert transitions <= 1, f"Oscillating activation near threshold: {sequences}"


def test_adversarial_memory_contamination():
    """Test: spurious high law_fit with low actual memory quality."""
    # This simulates contaminated memory giving false law_fit signal
    features = {
        "uncertainty": 0.5,
        "contradiction_signal": 0.3,
        "continuity_recent": 0.4,  # Low continuity (poor memory)
        "edge_pressure": 0.2,
        "causal_risk": 0.3,
        "symbolic_regularity": 0.1,  # Low symbolic
        "law_fit_signal": 0.8,  # But high law fit (suspicious)
    }

    budget = compute_budget(features)
    sequence, _, _ = select_sequence(features=features, budget=budget, allow_experimental=True)

    # Should activate eml_sr (meets threshold)
    # But should maintain discipline in other areas
    assert len(sequence) <= 9


# ============================================================================
# THRESHOLD GAMING ADVERSARIAL TESTS
# ============================================================================

def test_adversarial_multiple_just_below_thresholds():
    """Test: all signals just below their activation thresholds."""
    features = {
        "uncertainty": 0.59,  # Just below 0.6
        "contradiction_signal": 0.44,  # Just below 0.45
        "continuity_recent": 1.0,
        "edge_pressure": 0.69,  # Just below 0.7
        "causal_risk": 0.49,  # Just below 0.5
        "symbolic_regularity": 0.39,  # Just below 0.4
        "law_fit_signal": 0.39,  # Just below 0.4
    }

    budget = compute_budget(features)
    sequence, _, _ = select_sequence(features=features, budget=budget, allow_experimental=True)

    # Should not activate optional families
    assert "heur" not in sequence
    assert "dia_adv" not in sequence
    assert "fal_guard" not in sequence
    assert "eml_sr" not in sequence

    # But should still have baseline sequence
    assert "abd" in sequence
    assert "prob" in sequence


def test_adversarial_multiple_just_above_thresholds():
    """Test: all signals just above their activation thresholds."""
    features = {
        "uncertainty": 0.61,  # Just above 0.6
        "contradiction_signal": 0.46,  # Just above 0.45
        "continuity_recent": 1.0,
        "edge_pressure": 0.71,  # Just above 0.7
        "causal_risk": 0.51,  # Just above 0.5
        "symbolic_regularity": 0.41,  # Just above 0.4
        "law_fit_signal": 0.41,  # Just above 0.4
    }

    budget = compute_budget(features)
    sequence, _, _ = select_sequence(features=features, budget=budget, allow_experimental=True)

    # Should activate all optional families
    assert "heur" in sequence
    assert "dia_adv" in sequence
    assert "fal_guard" in sequence
    assert "eml_sr" in sequence

    # But should not exceed hard limits
    assert len(sequence) <= 10


def test_adversarial_one_extreme_rest_zero():
    """Test: one signal at maximum, all others at zero."""
    extreme_cases = [
        ("uncertainty", 1.0),
        ("contradiction_signal", 1.0),
        ("edge_pressure", 1.0),
        ("causal_risk", 1.0),
        ("symbolic_regularity", 1.0),
        ("law_fit_signal", 1.0),
    ]

    for feature_name, extreme_value in extreme_cases:
        features = {
            "uncertainty": 0.0,
            "contradiction_signal": 0.0,
            "continuity_recent": 1.0,
            "edge_pressure": 0.0,
            "causal_risk": 0.0,
            "symbolic_regularity": 0.0,
            "law_fit_signal": 0.0,
        }
        features[feature_name] = extreme_value

        budget = compute_budget(features)
        sequence, _, _ = select_sequence(features=features, budget=budget, allow_experimental=True)

        # Should handle gracefully
        assert 4 <= len(sequence) <= 10, \
            f"Extreme {feature_name} produced invalid sequence: {sequence}"
        assert "prob" in sequence, f"Extreme {feature_name} lost PROB"


# ============================================================================
# POLICY CONFLICT ADVERSARIAL TESTS
# ============================================================================

def test_adversarial_contradictory_policy_signals():
    """Test: signals that pull policy in opposite directions."""
    features = {
        "uncertainty": 1.0,  # Pull: expand
        "contradiction_signal": 1.0,  # Pull: expand + activate guards
        "continuity_recent": 0.0,  # Pull: expand (poor continuity needs more work)
        "edge_pressure": 1.0,  # Pull: CONTRACT
        "causal_risk": 1.0,  # Pull: expand
        "symbolic_regularity": 1.0,  # Pull: activate eml_sr
        "law_fit_signal": 1.0,  # Pull: activate eml_sr
    }

    budget = compute_budget(features)
    sequence, _, _ = select_sequence(features=features, budget=budget, allow_experimental=True)

    # System should make coherent decision, not collapse
    assert 4 <= len(sequence) <= 10
    assert "prob" in sequence
    assert sequence[-1] == "prob"

    # All activations should be justified
    if "dia_adv" in sequence:
        assert "fal_guard" in sequence  # They activate together

    if "eml_sr" in sequence:
        # Justified by high symbolic or law_fit
        pass

    if "heur" in sequence:
        # Justified by high edge_pressure
        pass


def test_adversarial_budget_at_limit_with_activations():
    """Test: budget reduced to limit but families want to activate."""
    features = {
        "uncertainty": 0.0,
        "contradiction_signal": 0.8,  # Wants dia_adv + fal_guard
        "continuity_recent": 1.0,
        "edge_pressure": 0.9,  # Reduces budget
        "causal_risk": 0.0,
        "symbolic_regularity": 0.0,
        "law_fit_signal": 0.0,
    }

    budget = compute_budget(features)
    sequence, _, _ = select_sequence(features=features, budget=budget, allow_experimental=True)

    # Budget should be reduced
    assert budget["max_steps"] <= 6

    # But sequence should still fit critical families
    # Priority: base mandatory > contradiction guards > prob
    assert "abd" in sequence
    assert "prob" in sequence

    # May or may not fit dia_adv/fal_guard depending on budget
    # But if budget allows, they should be there
    if budget["max_steps"] >= 8:
        assert "dia_adv" in sequence or "fal_guard" in sequence


# ============================================================================
# DETERMINISM ADVERSARIAL TESTS
# ============================================================================

def test_adversarial_determinism_under_stress():
    """Test: system is deterministic even under extreme stress."""
    stress_features = {
        "uncertainty": 0.95,
        "contradiction_signal": 0.85,
        "continuity_recent": 0.15,
        "edge_pressure": 0.75,
        "causal_risk": 0.9,
        "symbolic_regularity": 0.7,
        "law_fit_signal": 0.65,
    }

    # Run multiple times
    results = []
    for _ in range(5):
        budget = compute_budget(stress_features)
        sequence, _, _ = select_sequence(
            features=stress_features,
            budget=budget,
            allow_experimental=True
        )
        results.append((budget["max_steps"], sequence))

    # All results should be identical
    first_result = results[0]
    for result in results[1:]:
        assert result == first_result, "Non-deterministic behavior under stress"


def test_adversarial_floating_point_precision():
    """Test: tiny floating point differences shouldn't change behavior."""
    base_features = {
        "uncertainty": 0.7,
        "contradiction_signal": 0.45,
        "continuity_recent": 1.0,
        "edge_pressure": 0.7,
        "causal_risk": 0.5,
        "symbolic_regularity": 0.4,
        "law_fit_signal": 0.4,
    }

    # Add tiny noise (floating point precision level)
    noisy_features = base_features.copy()
    noisy_features["uncertainty"] += 1e-10
    noisy_features["contradiction_signal"] += 1e-10

    budget1 = compute_budget(base_features)
    budget2 = compute_budget(noisy_features)

    seq1, _, _ = select_sequence(features=base_features, budget=budget1, allow_experimental=True)
    seq2, _, _ = select_sequence(features=noisy_features, budget=budget2, allow_experimental=True)

    # Should be identical (robust to floating point noise)
    assert budget1["max_steps"] == budget2["max_steps"]
    assert seq1 == seq2
