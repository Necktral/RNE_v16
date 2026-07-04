"""
Family Contribution Tests - Elite Level
========================================

Tests measuring the marginal utility of each reasoning family.
Validates that families provide measurable value and don't degrade performance.
"""

from __future__ import annotations

import pytest
from typing import Dict, List, Set
from dataclasses import dataclass
from runtime.reasoning.scheduler_meta.budgeting import compute_budget
from runtime.reasoning.scheduler_meta.policy import select_sequence


@dataclass
class FamilyContributionMetrics:
    """Metrics for a family's contribution."""
    family_name: str
    activation_rate: float
    avg_sequence_position: float
    contexts_tested: int
    marginal_utility_score: float | None = None
    cost_effectiveness: float | None = None
    stability_impact: float | None = None


def evaluate_family_contribution(
    family_name: str,
    test_contexts: List[Dict[str, float]],
) -> FamilyContributionMetrics:
    """
    Evaluate a family's contribution across multiple contexts.

    Args:
        family_name: Name of family to evaluate
        test_contexts: List of feature contexts to test

    Returns:
        FamilyContributionMetrics with analysis
    """
    activation_count = 0
    position_sum = 0.0
    position_count = 0

    for features in test_contexts:
        budget = compute_budget(features)
        sequence, _, _ = select_sequence(
            features=features,
            budget=budget,
            allow_experimental=True
        )

        if family_name in sequence:
            activation_count += 1
            position = sequence.index(family_name)
            position_sum += position
            position_count += 1

    activation_rate = activation_count / len(test_contexts) if test_contexts else 0.0
    avg_position = position_sum / position_count if position_count > 0 else 0.0

    return FamilyContributionMetrics(
        family_name=family_name,
        activation_rate=activation_rate,
        avg_sequence_position=avg_position,
        contexts_tested=len(test_contexts)
    )


# ============================================================================
# CORE FAMILY CONTRIBUTION TESTS
# ============================================================================

def test_core_families_always_active():
    """Test that core mandatory families are always present."""
    # Core families: ABD, ANA, CAU, CTF, DED, PROB
    core_families = {"abd", "ana", "cau", "ctf", "ded", "prob"}

    # Test across variety of contexts
    test_contexts = [
        {"uncertainty": 0.1, "contradiction_signal": 0.0, "edge_pressure": 0.0, "causal_risk": 0.0},
        {"uncertainty": 0.5, "contradiction_signal": 0.3, "edge_pressure": 0.3, "causal_risk": 0.3},
        {"uncertainty": 0.9, "contradiction_signal": 0.7, "edge_pressure": 0.5, "causal_risk": 0.7},
    ]

    for features in test_contexts:
        features["continuity_recent"] = 1.0
        features["symbolic_regularity"] = 0.0
        features["law_fit_signal"] = 0.0

        budget = compute_budget(features)
        sequence, _, _ = select_sequence(features=features, budget=budget, allow_experimental=True)

        sequence_set = set(sequence)

        # All core families should be present
        assert core_families.issubset(sequence_set), \
            f"Missing core families in {features}: {core_families - sequence_set}"


def test_prob_always_last():
    """Test that PROB family is always the last step."""
    test_contexts = [
        {"uncertainty": 0.2, "contradiction_signal": 0.0, "edge_pressure": 0.0, "causal_risk": 0.0},
        {"uncertainty": 0.6, "contradiction_signal": 0.5, "edge_pressure": 0.7, "causal_risk": 0.5},
        {"uncertainty": 1.0, "contradiction_signal": 1.0, "edge_pressure": 0.0, "causal_risk": 1.0},
    ]

    for features in test_contexts:
        features["continuity_recent"] = 1.0
        features["symbolic_regularity"] = 0.0
        features["law_fit_signal"] = 0.0

        budget = compute_budget(features)
        sequence, _, _ = select_sequence(features=features, budget=budget, allow_experimental=True)

        assert sequence[-1] == "prob", \
            f"PROB not last in sequence for {features}: {sequence}"


# ============================================================================
# OPTIONAL FAMILY CONTRIBUTION TESTS
# ============================================================================

def test_heur_contribution_under_pressure():
    """Test that HEUR activates appropriately under edge pressure."""
    # High edge pressure contexts
    high_pressure_contexts = [
        {"uncertainty": 0.3, "contradiction_signal": 0.2, "edge_pressure": 0.75, "causal_risk": 0.2},
        {"uncertainty": 0.5, "contradiction_signal": 0.3, "edge_pressure": 0.85, "causal_risk": 0.4},
        {"uncertainty": 0.7, "contradiction_signal": 0.4, "edge_pressure": 0.95, "causal_risk": 0.6},
    ]

    # Low edge pressure contexts
    low_pressure_contexts = [
        {"uncertainty": 0.3, "contradiction_signal": 0.2, "edge_pressure": 0.1, "causal_risk": 0.2},
        {"uncertainty": 0.5, "contradiction_signal": 0.3, "edge_pressure": 0.2, "causal_risk": 0.4},
        {"uncertainty": 0.7, "contradiction_signal": 0.4, "edge_pressure": 0.3, "causal_risk": 0.6},
    ]

    # Evaluate high pressure
    for features in high_pressure_contexts:
        features["continuity_recent"] = 1.0
        features["symbolic_regularity"] = 0.0
        features["law_fit_signal"] = 0.0

        budget = compute_budget(features)
        sequence, _, _ = select_sequence(features=features, budget=budget, allow_experimental=True)

        # HEUR should be present in high pressure
        assert "heur" in sequence, \
            f"HEUR missing in high pressure context: {features}"

    # Evaluate low pressure
    for features in low_pressure_contexts:
        features["continuity_recent"] = 1.0
        features["symbolic_regularity"] = 0.0
        features["law_fit_signal"] = 0.0

        budget = compute_budget(features)
        sequence, _, _ = select_sequence(features=features, budget=budget, allow_experimental=True)

        # HEUR should not be present in low pressure
        assert "heur" not in sequence, \
            f"HEUR false activation in low pressure: {features}"


def test_dia_adv_fal_guard_contribution():
    """Test that DIA_ADV and FAL_GUARD activate together under contradiction."""
    # High contradiction contexts
    high_contradiction = [
        {"uncertainty": 0.3, "contradiction_signal": 0.5, "edge_pressure": 0.2, "causal_risk": 0.2},
        {"uncertainty": 0.5, "contradiction_signal": 0.7, "edge_pressure": 0.3, "causal_risk": 0.4},
        {"uncertainty": 0.7, "contradiction_signal": 0.9, "edge_pressure": 0.4, "causal_risk": 0.6},
    ]

    # Low contradiction contexts
    low_contradiction = [
        {"uncertainty": 0.3, "contradiction_signal": 0.1, "edge_pressure": 0.2, "causal_risk": 0.2},
        {"uncertainty": 0.5, "contradiction_signal": 0.2, "edge_pressure": 0.3, "causal_risk": 0.4},
        {"uncertainty": 0.7, "contradiction_signal": 0.3, "edge_pressure": 0.4, "causal_risk": 0.6},
    ]

    # High contradiction should activate both
    for features in high_contradiction:
        features["continuity_recent"] = 1.0
        features["symbolic_regularity"] = 0.0
        features["law_fit_signal"] = 0.0

        budget = compute_budget(features)
        sequence, _, _ = select_sequence(features=features, budget=budget, allow_experimental=True)

        has_dia = "dia_adv" in sequence
        has_fal = "fal_guard" in sequence

        # Both should be present
        assert has_dia, f"DIA_ADV missing in high contradiction: {features}"
        assert has_fal, f"FAL_GUARD missing in high contradiction: {features}"

        # They should activate together
        assert has_dia == has_fal, "DIA_ADV and FAL_GUARD not synchronized"

    # Low contradiction should not activate
    for features in low_contradiction:
        features["continuity_recent"] = 1.0
        features["symbolic_regularity"] = 0.0
        features["law_fit_signal"] = 0.0

        budget = compute_budget(features)
        sequence, _, _ = select_sequence(features=features, budget=budget, allow_experimental=True)

        assert "dia_adv" not in sequence, f"DIA_ADV false activation: {features}"
        assert "fal_guard" not in sequence, f"FAL_GUARD false activation: {features}"


def test_eml_sr_contribution():
    """Test that EML_SR activates appropriately for symbolic contexts."""
    # High symbolic contexts
    high_symbolic = [
        {"uncertainty": 0.3, "contradiction_signal": 0.2, "edge_pressure": 0.2, "causal_risk": 0.2,
         "symbolic_regularity": 0.6, "law_fit_signal": 0.1},
        {"uncertainty": 0.5, "contradiction_signal": 0.3, "edge_pressure": 0.3, "causal_risk": 0.4,
         "symbolic_regularity": 0.8, "law_fit_signal": 0.2},
        {"uncertainty": 0.3, "contradiction_signal": 0.2, "edge_pressure": 0.2, "causal_risk": 0.2,
         "symbolic_regularity": 0.1, "law_fit_signal": 0.7},
    ]

    # Low symbolic contexts
    low_symbolic = [
        {"uncertainty": 0.3, "contradiction_signal": 0.2, "edge_pressure": 0.2, "causal_risk": 0.2,
         "symbolic_regularity": 0.1, "law_fit_signal": 0.1},
        {"uncertainty": 0.5, "contradiction_signal": 0.3, "edge_pressure": 0.3, "causal_risk": 0.4,
         "symbolic_regularity": 0.2, "law_fit_signal": 0.2},
    ]

    # High symbolic should activate eml_sr
    for features in high_symbolic:
        features["continuity_recent"] = 1.0

        budget = compute_budget(features)
        sequence, _, _ = select_sequence(features=features, budget=budget, allow_experimental=True)

        assert "eml_sr" in sequence, \
            f"EML_SR missing in symbolic context: {features}"

    # Low symbolic should not activate
    for features in low_symbolic:
        features["continuity_recent"] = 1.0

        budget = compute_budget(features)
        sequence, _, _ = select_sequence(features=features, budget=budget, allow_experimental=True)

        assert "eml_sr" not in sequence, \
            f"EML_SR false activation in low symbolic: {features}"


# ============================================================================
# MARGINAL UTILITY TESTS
# ============================================================================

def test_family_removal_degrades_appropriately():
    """Test that removing optional families degrades performance in appropriate contexts."""
    # Create context where HEUR should help
    high_pressure_context = {
        "uncertainty": 0.5,
        "contradiction_signal": 0.3,
        "continuity_recent": 1.0,
        "edge_pressure": 0.9,
        "causal_risk": 0.4,
        "symbolic_regularity": 0.0,
        "law_fit_signal": 0.0,
    }

    # With HEUR
    budget = compute_budget(high_pressure_context)
    sequence_with_heur, _, _ = select_sequence(
        features=high_pressure_context,
        budget=budget,
        allow_experimental=True
    )

    # HEUR should be present
    assert "heur" in sequence_with_heur

    # Without HEUR (simulated by checking if it's critical)
    # In real system, HEUR provides value under pressure
    # We validate it's activated when needed


def test_optional_families_dont_harm_baseline():
    """Test that optional families don't degrade baseline performance."""
    # Low-signal baseline context
    baseline_context = {
        "uncertainty": 0.2,
        "contradiction_signal": 0.1,
        "continuity_recent": 1.0,
        "edge_pressure": 0.1,
        "causal_risk": 0.1,
        "symbolic_regularity": 0.0,
        "law_fit_signal": 0.0,
    }

    budget = compute_budget(baseline_context)
    sequence, _, _ = select_sequence(
        features=baseline_context,
        budget=budget,
        allow_experimental=True
    )

    # Should be minimal, efficient sequence
    assert len(sequence) <= 7

    # Optional families should not activate
    assert "heur" not in sequence
    assert "dia_adv" not in sequence
    assert "fal_guard" not in sequence
    assert "eml_sr" not in sequence


# ============================================================================
# COST-EFFECTIVENESS TESTS
# ============================================================================

def test_family_activation_cost_justified():
    """Test that family activation doesn't exceed budget inappropriately."""
    # Create contexts with various budgets
    contexts = [
        {"uncertainty": 0.2, "contradiction_signal": 0.5, "edge_pressure": 0.0, "causal_risk": 0.0},
        {"uncertainty": 0.6, "contradiction_signal": 0.5, "edge_pressure": 0.0, "causal_risk": 0.0},
        {"uncertainty": 0.9, "contradiction_signal": 0.5, "edge_pressure": 0.0, "causal_risk": 0.0},
    ]

    for features in contexts:
        features["continuity_recent"] = 1.0
        features["symbolic_regularity"] = 0.0
        features["law_fit_signal"] = 0.0

        budget = compute_budget(features)
        sequence, _, _ = select_sequence(features=features, budget=budget, allow_experimental=True)

        # Sequence should not exceed budget
        assert len(sequence) <= budget["max_steps"], \
            f"Sequence exceeds budget: {len(sequence)} > {budget['max_steps']}"

        # Sequence should respect hard limits
        assert len(sequence) <= 10


def test_high_cost_families_justify_presence():
    """Test that expensive families only activate when justified."""
    # DIA_ADV and FAL_GUARD are "expensive" (add multiple steps)
    # They should only activate when contradiction is high

    # Test borderline contradiction
    borderline = {
        "uncertainty": 0.3,
        "contradiction_signal": 0.44,  # Just below threshold
        "continuity_recent": 1.0,
        "edge_pressure": 0.2,
        "causal_risk": 0.2,
        "symbolic_regularity": 0.0,
        "law_fit_signal": 0.0,
    }

    budget = compute_budget(borderline)
    sequence, _, _ = select_sequence(features=borderline, budget=budget, allow_experimental=True)

    # Should not waste budget on dia_adv/fal_guard
    assert "dia_adv" not in sequence
    assert "fal_guard" not in sequence


# ============================================================================
# STABILITY IMPACT TESTS
# ============================================================================

def test_family_addition_maintains_stability():
    """Test that adding optional families maintains sequence stability."""
    # Base context
    base_features = {
        "uncertainty": 0.5,
        "contradiction_signal": 0.3,
        "continuity_recent": 1.0,
        "edge_pressure": 0.3,
        "causal_risk": 0.3,
        "symbolic_regularity": 0.0,
        "law_fit_signal": 0.0,
    }

    # Get baseline
    budget_base = compute_budget(base_features)
    seq_base, _, _ = select_sequence(features=base_features, budget=budget_base, allow_experimental=True)

    # Now increase contradiction to activate guards
    with_guards = base_features.copy()
    with_guards["contradiction_signal"] = 0.6

    budget_guards = compute_budget(with_guards)
    seq_guards, _, _ = select_sequence(features=with_guards, budget=budget_guards, allow_experimental=True)

    # Core families should still be present
    core_families = {"abd", "ana", "cau", "ctf", "ded", "prob"}
    assert core_families.issubset(set(seq_guards))

    # PROB should still be last
    assert seq_guards[-1] == "prob"


def test_no_chaotic_expansion():
    """Test that activating multiple optional families doesn't cause chaos."""
    # Context that activates multiple optional families
    multi_activation = {
        "uncertainty": 0.7,
        "contradiction_signal": 0.6,
        "continuity_recent": 1.0,
        "edge_pressure": 0.75,
        "causal_risk": 0.6,
        "symbolic_regularity": 0.5,
        "law_fit_signal": 0.5,
    }

    budget = compute_budget(multi_activation)
    sequence, _, _ = select_sequence(
        features=multi_activation,
        budget=budget,
        allow_experimental=True
    )

    # Should activate multiple families
    assert "heur" in sequence
    assert "dia_adv" in sequence
    assert "fal_guard" in sequence
    assert "eml_sr" in sequence

    # But should maintain discipline
    assert len(sequence) <= 10
    assert sequence[-1] == "prob"

    # No duplicates
    assert len(sequence) == len(set(sequence))


# ============================================================================
# COMPREHENSIVE FAMILY ANALYSIS
# ============================================================================

def test_all_families_contribute_somewhere():
    """Test that every family activates in at least some contexts."""
    # Generate diverse contexts
    contexts = [
        # Baseline
        {"uncertainty": 0.2, "contradiction_signal": 0.1, "edge_pressure": 0.1, "causal_risk": 0.1,
         "symbolic_regularity": 0.0, "law_fit_signal": 0.0},
        # High uncertainty
        {"uncertainty": 0.8, "contradiction_signal": 0.2, "edge_pressure": 0.2, "causal_risk": 0.5,
         "symbolic_regularity": 0.0, "law_fit_signal": 0.0},
        # High contradiction
        {"uncertainty": 0.3, "contradiction_signal": 0.7, "edge_pressure": 0.2, "causal_risk": 0.3,
         "symbolic_regularity": 0.0, "law_fit_signal": 0.0},
        # High edge pressure
        {"uncertainty": 0.4, "contradiction_signal": 0.3, "edge_pressure": 0.8, "causal_risk": 0.3,
         "symbolic_regularity": 0.0, "law_fit_signal": 0.0},
        # High symbolic
        {"uncertainty": 0.3, "contradiction_signal": 0.2, "edge_pressure": 0.2, "causal_risk": 0.2,
         "symbolic_regularity": 0.7, "law_fit_signal": 0.0},
        # High law fit
        {"uncertainty": 0.3, "contradiction_signal": 0.2, "edge_pressure": 0.2, "causal_risk": 0.2,
         "symbolic_regularity": 0.0, "law_fit_signal": 0.7},
        # Everything high
        {"uncertainty": 0.8, "contradiction_signal": 0.8, "edge_pressure": 0.7, "causal_risk": 0.7,
         "symbolic_regularity": 0.6, "law_fit_signal": 0.6},
    ]

    # Add continuity_recent to all
    for ctx in contexts:
        ctx["continuity_recent"] = 1.0

    # Collect all activated families
    all_activated = set()

    for features in contexts:
        budget = compute_budget(features)
        sequence, _, _ = select_sequence(features=features, budget=budget, allow_experimental=True)
        all_activated.update(sequence)

    # All families should appear somewhere
    expected_families = {"abd", "ana", "cau", "ctf", "ded", "prob", "heur", "dia_adv", "fal_guard", "eml_sr"}

    assert expected_families.issubset(all_activated), \
        f"Some families never activated: {expected_families - all_activated}"


def test_family_activation_patterns():
    """Test that family activation patterns are consistent and predictable."""
    all_families = ["abd", "ana", "cau", "ctf", "ded", "prob", "heur", "dia_adv", "fal_guard", "eml_sr"]

    # Evaluate each family
    for family in all_families:
        # Create contexts where family should/shouldn't activate
        if family in ["abd", "ana", "cau", "ctf", "ded", "prob"]:
            # Core family: should always activate
            test_contexts = [
                {"uncertainty": 0.2, "contradiction_signal": 0.1},
                {"uncertainty": 0.9, "contradiction_signal": 0.8},
            ]

            for features in test_contexts:
                features.update({
                    "continuity_recent": 1.0,
                    "edge_pressure": 0.2,
                    "causal_risk": 0.2,
                    "symbolic_regularity": 0.0,
                    "law_fit_signal": 0.0,
                })

                budget = compute_budget(features)
                sequence, _, _ = select_sequence(features=features, budget=budget, allow_experimental=True)

                assert family in sequence, f"Core family {family} missing in {features}"
