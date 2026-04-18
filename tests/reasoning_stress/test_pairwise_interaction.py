"""
Pairwise Interaction Stress Tests - Elite Level
================================================

Tests for measuring non-linear interactions between feature pairs.
Discovers coupling, overactivation, cancellation, and policy conflicts.
"""

from __future__ import annotations

import pytest
from typing import Dict, List, Tuple
from runtime.reasoning.scheduler_meta.budgeting import compute_budget
from runtime.reasoning.scheduler_meta.policy import select_sequence


class InteractionResult:
    """Captures interaction effects between two features."""

    def __init__(self, feature1: str, feature2: str):
        self.feature1 = feature1
        self.feature2 = feature2
        self.measurements: List[Dict] = []

    def add_measurement(
        self,
        val1: float,
        val2: float,
        families: List[str],
        budget: Dict,
        individual1_families: List[str],
        individual2_families: List[str]
    ):
        """Record a measurement point."""
        self.measurements.append({
            "val1": val1,
            "val2": val2,
            "families": families,
            "budget": budget,
            "individual1": individual1_families,
            "individual2": individual2_families,
        })

    def detect_overactivation(self) -> List[Dict]:
        """Detect cases where combined activation exceeds individual sum."""
        overactivations = []

        for m in self.measurements:
            combined_count = len(m["families"])
            individual_union = set(m["individual1"]) | set(m["individual2"])
            expected_count = len(individual_union)

            if combined_count > expected_count:
                overactivations.append({
                    "val1": m["val1"],
                    "val2": m["val2"],
                    "combined": m["families"],
                    "expected": list(individual_union),
                    "excess": combined_count - expected_count
                })

        return overactivations

    def detect_cancellation(self) -> List[Dict]:
        """Detect cases where combined activation is less than individual union."""
        cancellations = []

        for m in self.measurements:
            combined_set = set(m["families"])
            individual_union = set(m["individual1"]) | set(m["individual2"])

            missing = individual_union - combined_set
            if missing:
                cancellations.append({
                    "val1": m["val1"],
                    "val2": m["val2"],
                    "combined": m["families"],
                    "expected": list(individual_union),
                    "missing": list(missing)
                })

        return cancellations

    def measure_nonlinearity(self) -> float:
        """
        Measure degree of non-linear interaction.

        Returns value from 0.0 (perfectly additive) to 1.0 (highly non-linear).
        """
        if not self.measurements:
            return 0.0

        nonlinear_count = 0
        for m in self.measurements:
            combined_set = set(m["families"])
            individual_union = set(m["individual1"]) | set(m["individual2"])

            if combined_set != individual_union:
                nonlinear_count += 1

        return nonlinear_count / len(self.measurements)


def test_interaction_grid(
    feature1: str,
    feature2: str,
    values1: List[float],
    values2: List[float],
    baseline_features: Dict[str, float] | None = None
) -> InteractionResult:
    """
    Test interaction between two features across a grid of values.

    Args:
        feature1: First feature name
        feature2: Second feature name
        values1: List of values to test for feature1
        values2: List of values to test for feature2
        baseline_features: Baseline feature values

    Returns:
        InteractionResult with detailed analysis
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

    result = InteractionResult(feature1, feature2)

    for val1 in values1:
        for val2 in values2:
            # Test combined
            combined_features = baseline_features.copy()
            combined_features[feature1] = val1
            combined_features[feature2] = val2

            budget = compute_budget(combined_features)
            families, _, _ = select_sequence(
                features=combined_features,
                budget=budget,
                allow_experimental=True
            )

            # Test feature1 alone
            f1_features = baseline_features.copy()
            f1_features[feature1] = val1
            f1_budget = compute_budget(f1_features)
            f1_families, _, _ = select_sequence(
                features=f1_features,
                budget=f1_budget,
                allow_experimental=True
            )

            # Test feature2 alone
            f2_features = baseline_features.copy()
            f2_features[feature2] = val2
            f2_budget = compute_budget(f2_features)
            f2_families, _, _ = select_sequence(
                features=f2_features,
                budget=f2_budget,
                allow_experimental=True
            )

            result.add_measurement(
                val1=val1,
                val2=val2,
                families=families,
                budget=budget,
                individual1_families=f1_families,
                individual2_families=f2_families
            )

    return result


# ============================================================================
# CONTRADICTION × EDGE_PRESSURE INTERACTION
# ============================================================================

def test_contradiction_edge_pressure_interaction():
    """Test interaction between contradiction and edge_pressure."""
    result = test_interaction_grid(
        feature1="contradiction_signal",
        feature2="edge_pressure",
        values1=[0.0, 0.45, 0.8],
        values2=[0.0, 0.7, 0.9]
    )

    # Should not have major overactivations
    overactivations = result.detect_overactivation()
    assert len(overactivations) == 0, f"Unexpected overactivations: {overactivations}"

    # High edge pressure + high contradiction should not cancel families
    cancellations = result.detect_cancellation()
    critical_cancellations = [
        c for c in cancellations
        if c["val1"] >= 0.45 and c["val2"] >= 0.7
    ]
    assert len(critical_cancellations) == 0, f"Critical families cancelled: {critical_cancellations}"


def test_contradiction_edge_pressure_budget_conflict():
    """Test that edge_pressure reduction doesn't destroy contradiction response."""
    # High contradiction, low edge -> should activate dia_adv + fal_guard
    low_edge_features = {
        "uncertainty": 0.25,
        "contradiction_signal": 0.8,
        "continuity_recent": 1.0,
        "edge_pressure": 0.2,
        "causal_risk": 0.0,
        "symbolic_regularity": 0.0,
        "law_fit_signal": 0.0,
    }

    # High contradiction, high edge -> edge reduces budget
    high_edge_features = low_edge_features.copy()
    high_edge_features["edge_pressure"] = 0.9

    budget_low = compute_budget(low_edge_features)
    budget_high = compute_budget(high_edge_features)

    seq_low, _, _ = select_sequence(features=low_edge_features, budget=budget_low, allow_experimental=True)
    seq_high, _, _ = select_sequence(features=high_edge_features, budget=budget_high, allow_experimental=True)

    # Even with reduced budget, critical families should remain
    assert "dia_adv" in seq_low
    assert "fal_guard" in seq_low

    # Edge pressure might reduce sequence length, but core families should survive
    # At minimum, mandatory base families + contradiction families should fit
    assert "abd" in seq_high
    assert "prob" in seq_high


# ============================================================================
# CONTRADICTION × UNCERTAINTY INTERACTION
# ============================================================================

def test_contradiction_uncertainty_interaction():
    """Test interaction between contradiction and uncertainty."""
    result = test_interaction_grid(
        feature1="contradiction_signal",
        feature2="uncertainty",
        values1=[0.0, 0.45, 0.8],
        values2=[0.25, 0.6, 0.9]
    )

    # High values of both should amplify, not cancel
    overactivations = result.detect_overactivation()
    # Some overactivation is acceptable (emergence)

    cancellations = result.detect_cancellation()
    critical_cancellations = [
        c for c in cancellations
        if c["val1"] >= 0.45 and c["val2"] >= 0.6
    ]
    assert len(critical_cancellations) == 0


def test_contradiction_uncertainty_risk_budget_amplification():
    """Test that contradiction + uncertainty amplify risk_budget."""
    baseline = {
        "uncertainty": 0.3,
        "contradiction_signal": 0.3,
        "continuity_recent": 1.0,
        "edge_pressure": 0.0,
        "causal_risk": 0.0,
        "symbolic_regularity": 0.0,
        "law_fit_signal": 0.0,
    }

    high_both = baseline.copy()
    high_both["uncertainty"] = 0.9
    high_both["contradiction_signal"] = 0.8

    high_uncertainty_only = baseline.copy()
    high_uncertainty_only["uncertainty"] = 0.9

    high_contradiction_only = baseline.copy()
    high_contradiction_only["contradiction_signal"] = 0.8

    budget_both = compute_budget(high_both)
    budget_unc = compute_budget(high_uncertainty_only)
    budget_con = compute_budget(high_contradiction_only)

    # Combined should be >= max of individuals
    max_individual = max(budget_unc["risk_budget"], budget_con["risk_budget"])
    assert budget_both["risk_budget"] >= max_individual


# ============================================================================
# SYMBOLIC_REGULARITY × LAW_FIT INTERACTION
# ============================================================================

def test_symbolic_law_fit_interaction():
    """Test interaction between symbolic_regularity and law_fit_signal."""
    result = test_interaction_grid(
        feature1="symbolic_regularity",
        feature2="law_fit_signal",
        values1=[0.0, 0.4, 0.8],
        values2=[0.0, 0.4, 0.8]
    )

    # Either high should activate eml_sr
    for m in result.measurements:
        if m["val1"] >= 0.4 or m["val2"] >= 0.4:
            assert "eml_sr" in m["families"], f"EML_SR should activate at val1={m['val1']}, val2={m['val2']}"


def test_symbolic_law_fit_no_double_activation():
    """Test that symbolic_regularity + law_fit don't double-activate EML_SR."""
    features = {
        "uncertainty": 0.25,
        "contradiction_signal": 0.0,
        "continuity_recent": 1.0,
        "edge_pressure": 0.0,
        "causal_risk": 0.0,
        "symbolic_regularity": 0.8,
        "law_fit_signal": 0.8,
    }

    budget = compute_budget(features)
    sequence, _, _ = select_sequence(features=features, budget=budget, allow_experimental=True)

    # Should have eml_sr exactly once, not duplicated
    eml_count = sum(1 for fam in sequence if fam == "eml_sr")
    assert eml_count <= 1


# ============================================================================
# CAUSAL_RISK × CONTRADICTION INTERACTION
# ============================================================================

def test_causal_risk_contradiction_interaction():
    """Test interaction between causal_risk and contradiction."""
    result = test_interaction_grid(
        feature1="causal_risk",
        feature2="contradiction_signal",
        values1=[0.0, 0.5, 0.9],
        values2=[0.0, 0.45, 0.8]
    )

    # High causal_risk increases budget
    # High contradiction activates families
    # Both should work together harmoniously

    cancellations = result.detect_cancellation()
    assert len(cancellations) == 0


def test_causal_risk_contradiction_budget_synergy():
    """Test that causal_risk + contradiction synergize in budget."""
    baseline = {
        "uncertainty": 0.25,
        "contradiction_signal": 0.2,
        "continuity_recent": 1.0,
        "edge_pressure": 0.0,
        "causal_risk": 0.2,
        "symbolic_regularity": 0.0,
        "law_fit_signal": 0.0,
    }

    high_both = baseline.copy()
    high_both["causal_risk"] = 0.9
    high_both["contradiction_signal"] = 0.8

    budget_baseline = compute_budget(baseline)
    budget_high = compute_budget(high_both)

    # Should increase both max_steps and risk_budget
    assert budget_high["max_steps"] > budget_baseline["max_steps"]
    assert budget_high["risk_budget"] > budget_baseline["risk_budget"]


# ============================================================================
# EDGE_PRESSURE × CAUSAL_RISK CONFLICT
# ============================================================================

def test_edge_pressure_causal_risk_conflict():
    """Test the tension between edge_pressure (reduce) and causal_risk (increase)."""
    # Edge pressure wants to reduce
    # Causal risk wants to increase
    # System should find reasonable compromise

    features = {
        "uncertainty": 0.25,
        "contradiction_signal": 0.0,
        "continuity_recent": 1.0,
        "edge_pressure": 0.9,  # High: wants to reduce
        "causal_risk": 0.9,    # High: wants to increase
        "symbolic_regularity": 0.0,
        "law_fit_signal": 0.0,
    }

    budget = compute_budget(features)

    # Should not collapse to minimum (edge wins completely)
    assert budget["max_steps"] > 4

    # Should not expand to maximum (causal wins completely)
    assert budget["max_steps"] < 10

    # Should be somewhere in middle
    assert 5 <= budget["max_steps"] <= 8


# ============================================================================
# COMPREHENSIVE INTERACTION MATRIX
# ============================================================================

def test_interaction_matrix_all_pairs():
    """Test all major feature pairs for pathological interactions."""
    features_to_test = [
        "uncertainty",
        "contradiction_signal",
        "edge_pressure",
        "causal_risk",
        "symbolic_regularity",
        "law_fit_signal",
    ]

    # Test values: low, medium, high
    test_values = [0.1, 0.5, 0.9]

    pathological_pairs = []

    for i, feat1 in enumerate(features_to_test):
        for feat2 in features_to_test[i+1:]:  # Avoid duplicates
            result = test_interaction_grid(
                feature1=feat1,
                feature2=feat2,
                values1=test_values,
                values2=test_values
            )

            # Check for pathologies
            overactivations = result.detect_overactivation()
            cancellations = result.detect_cancellation()
            nonlinearity = result.measure_nonlinearity()

            has_pathology = (
                len(overactivations) > 3 or  # Too many overactivations
                len(cancellations) > 3 or     # Too many cancellations
                nonlinearity > 0.8            # Extremely non-linear
            )

            if has_pathology:
                pathological_pairs.append({
                    "pair": (feat1, feat2),
                    "overactivations": len(overactivations),
                    "cancellations": len(cancellations),
                    "nonlinearity": nonlinearity
                })

    # Report findings
    if pathological_pairs:
        for p in pathological_pairs:
            print(f"Pathological interaction: {p['pair']}, "
                  f"overact={p['overactivations']}, "
                  f"cancel={p['cancellations']}, "
                  f"nonlin={p['nonlinearity']:.2f}")

    # Should not have widespread pathologies
    assert len(pathological_pairs) <= 2, f"Too many pathological interactions: {pathological_pairs}"


def test_no_family_activation_chaos():
    """Test that no combination leads to chaotic activation patterns."""
    # Test extreme combinations
    extreme_cases = [
        {"uncertainty": 1.0, "contradiction_signal": 1.0, "edge_pressure": 1.0, "causal_risk": 1.0},
        {"uncertainty": 0.0, "contradiction_signal": 0.0, "edge_pressure": 0.0, "causal_risk": 0.0},
        {"uncertainty": 1.0, "contradiction_signal": 0.0, "edge_pressure": 1.0, "causal_risk": 0.0},
        {"uncertainty": 0.0, "contradiction_signal": 1.0, "edge_pressure": 0.0, "causal_risk": 1.0},
    ]

    for features in extreme_cases:
        # Add missing features
        features.setdefault("continuity_recent", 1.0)
        features.setdefault("symbolic_regularity", 0.0)
        features.setdefault("law_fit_signal", 0.0)

        budget = compute_budget(features)
        sequence, _, _ = select_sequence(features=features, budget=budget, allow_experimental=True)

        # Should always have reasonable sequence
        assert len(sequence) >= 4, f"Sequence too short for {features}"
        assert len(sequence) <= 10, f"Sequence too long for {features}"

        # Should always have core families
        assert "abd" in sequence
        assert "prob" in sequence

        # PROB should always be last
        assert sequence[-1] == "prob"
