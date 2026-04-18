"""
Temporal Multiscale Cascade Tests - Fractal Stress Level
=========================================================

Tests temporal behavior at multiple time scales to detect:
- Temporal self-similarity vs fragility
- Scale-dependent memory effects
- Cascade propagation patterns
- Recovery dynamics
"""

from __future__ import annotations

import pytest
from tests.reasoning_stress.fractal_utils import (
    measure_temporal_cascades,
    TemporalCascadeMetrics
)


# ============================================================================
# INCREASING PERTURBATION TESTS
# ============================================================================

def test_uncertainty_increasing_temporal_cascade():
    """Test cascades from gradually increasing uncertainty."""
    result = measure_temporal_cascades(
        feature_name="uncertainty",
        perturbation_pattern="increasing",
        time_scales=[1, 2, 4, 8, 16]
    )

    # Should show some self-similarity
    assert result.scale_invariance_error < 0.5, \
        f"Poor scale invariance: error={result.scale_invariance_error:.3f}"

    # Memory fragility should be low for smooth increasing signal
    assert result.temporal_memory_fragility < 0.3, \
        f"High temporal memory fragility: {result.temporal_memory_fragility:.3f}"


def test_contradiction_increasing_temporal_cascade():
    """Test cascades from increasing contradiction signal."""
    result = measure_temporal_cascades(
        feature_name="contradiction_signal",
        perturbation_pattern="increasing"
    )

    # Increasing contradiction should trigger dia_adv/fal_guard at some point
    # But behavior should be scale-consistent
    assert result.scale_invariance_error < 0.6, \
        f"Inconsistent cascade scaling: {result.scale_invariance_error:.3f}"


def test_edge_pressure_increasing_cascade():
    """Test cascades from increasing edge pressure."""
    result = measure_temporal_cascades(
        feature_name="edge_pressure",
        perturbation_pattern="increasing"
    )

    # Edge pressure increase may reduce budget - check consistency
    assert result.scale_invariance_error < 0.5


# ============================================================================
# DECREASING PERTURBATION TESTS
# ============================================================================

def test_uncertainty_decreasing_temporal_cascade():
    """Test cascades from decreasing uncertainty."""
    result = measure_temporal_cascades(
        feature_name="uncertainty",
        perturbation_pattern="decreasing"
    )

    # Decreasing should be symmetric to increasing
    assert result.scale_invariance_error < 0.5

    # Fragility should still be low
    assert result.temporal_memory_fragility < 0.3


def test_contradiction_decreasing_cascade():
    """Test deactivation cascades from decreasing contradiction."""
    result = measure_temporal_cascades(
        feature_name="contradiction_signal",
        perturbation_pattern="decreasing"
    )

    # Should see symmetric behavior to increasing
    assert result.scale_invariance_error < 0.6


# ============================================================================
# OSCILLATING PERTURBATION TESTS
# ============================================================================

def test_uncertainty_oscillating_cascade():
    """Test response to oscillating uncertainty signal."""
    result = measure_temporal_cascades(
        feature_name="uncertainty",
        perturbation_pattern="oscillating",
        time_scales=[4, 8, 16]  # Oscillation needs multiple steps
    )

    # Oscillating signals may have higher fragility
    assert result.temporal_memory_fragility < 0.5, \
        f"Excessive memory effects from oscillation: {result.temporal_memory_fragility:.3f}"

    # Should not show excessive scale dependence
    assert result.scale_invariance_error < 0.7


def test_edge_pressure_oscillating_cascade():
    """Test response to oscillating edge pressure."""
    result = measure_temporal_cascades(
        feature_name="edge_pressure",
        perturbation_pattern="oscillating",
        time_scales=[4, 8, 16]
    )

    # Edge pressure oscillation shouldn't cause chaos
    assert result.temporal_memory_fragility < 0.5


def test_contradiction_oscillating_cascade():
    """Test response to oscillating contradiction signal."""
    result = measure_temporal_cascades(
        feature_name="contradiction_signal",
        perturbation_pattern="oscillating",
        time_scales=[4, 8, 16]
    )

    # Oscillation around threshold may trigger activation/deactivation
    # But should be controlled
    assert result.temporal_memory_fragility < 0.6, \
        "Contradiction oscillation causes excessive memory effects"


# ============================================================================
# PULSED PERTURBATION TESTS
# ============================================================================

def test_uncertainty_pulsed_cascade():
    """Test response to pulsed (burst) uncertainty."""
    result = measure_temporal_cascades(
        feature_name="uncertainty",
        perturbation_pattern="pulsed",
        time_scales=[4, 8, 16]
    )

    # Pulsed signals test recovery
    # Recovery should be consistent across scales
    assert result.scale_invariance_error < 0.7


def test_causal_risk_pulsed_cascade():
    """Test response to pulsed causal risk."""
    result = measure_temporal_cascades(
        feature_name="causal_risk",
        perturbation_pattern="pulsed"
    )

    # Should show controlled response
    assert result.scale_invariance_error < 0.7


# ============================================================================
# SELF-SIMILARITY TESTS
# ============================================================================

def test_temporal_self_similarity_across_features():
    """Test that temporal behavior shows consistent self-similarity."""
    features_to_test = [
        "uncertainty",
        "contradiction_signal",
        "edge_pressure",
        "causal_risk"
    ]

    self_similar_count = 0

    for feature in features_to_test:
        result = measure_temporal_cascades(
            feature_name=feature,
            perturbation_pattern="increasing"
        )

        if result.self_similar:
            self_similar_count += 1

    # Most features should show self-similar temporal behavior
    assert self_similar_count >= len(features_to_test) * 0.6, \
        f"Only {self_similar_count}/{len(features_to_test)} features show self-similarity"


def test_scale_invariance_errors_bounded():
    """Test that scale invariance errors are reasonable."""
    features = ["uncertainty", "contradiction_signal", "edge_pressure"]
    patterns = ["increasing", "decreasing"]

    max_error = 0.0
    errors = []

    for feature in features:
        for pattern in patterns:
            result = measure_temporal_cascades(
                feature_name=feature,
                perturbation_pattern=pattern
            )

            errors.append(result.scale_invariance_error)
            max_error = max(max_error, result.scale_invariance_error)

    # No error should be excessive
    assert max_error < 0.8, \
        f"Excessive scale invariance error: {max_error:.3f}"

    # Mean error should be reasonable
    mean_error = sum(errors) / len(errors)
    assert mean_error < 0.6, \
        f"Mean scale invariance error too high: {mean_error:.3f}"


# ============================================================================
# MEMORY FRAGILITY TESTS
# ============================================================================

def test_temporal_memory_fragility_bounded():
    """Test that temporal memory effects don't grow spuriously with scale."""
    result_inc = measure_temporal_cascades(
        feature_name="uncertainty",
        perturbation_pattern="increasing",
        time_scales=[1, 2, 4, 8, 16, 32]  # Extended scales
    )

    # Fragility should not grow with more scales
    assert result_inc.temporal_memory_fragility < 0.4, \
        f"High temporal memory fragility: {result_inc.temporal_memory_fragility:.3f}"


def test_no_spurious_memory_in_oscillation():
    """Test that oscillation doesn't create spurious sticky states."""
    result = measure_temporal_cascades(
        feature_name="contradiction_signal",
        perturbation_pattern="oscillating",
        time_scales=[8, 16]  # Enough for multiple cycles
    )

    # Should recover from oscillations without permanent state changes
    assert result.temporal_memory_fragility < 0.6, \
        "Oscillation creates spurious memory"


# ============================================================================
# RECOVERY DYNAMICS TESTS
# ============================================================================

def test_recovery_time_scales_reasonably():
    """Test that recovery times scale appropriately with perturbation scale."""
    result = measure_temporal_cascades(
        feature_name="uncertainty",
        perturbation_pattern="pulsed",
        time_scales=[4, 8, 16]
    )

    # Recovery times should increase with scale but stay bounded
    for i in range(1, len(result.recovery_times)):
        ratio = result.recovery_times[i] / max(result.recovery_times[i-1], 0.1)
        assert ratio < 3.0, \
            f"Recovery time grows too fast: {result.recovery_times}"


def test_activation_persistence_patterns():
    """Test that activation persistence behaves consistently."""
    result = measure_temporal_cascades(
        feature_name="contradiction_signal",
        perturbation_pattern="increasing"
    )

    # Persistence should be relatively stable across scales
    persistence_variance = sum(
        (result.activation_persistence[i] - result.activation_persistence[0]) ** 2
        for i in range(len(result.activation_persistence))
    ) / len(result.activation_persistence)

    assert persistence_variance < 0.2, \
        f"High persistence variance: {persistence_variance:.3f}"


# ============================================================================
# CASCADE SIZE TESTS
# ============================================================================

def test_family_change_cascade_size():
    """Test that family changes cascade in controlled manner."""
    result = measure_temporal_cascades(
        feature_name="contradiction_signal",
        perturbation_pattern="increasing",
        time_scales=[8, 16]
    )

    # Family changes should be bounded
    max_changes = max(result.family_change_counts)
    assert max_changes <= 10, \
        f"Too many family changes in cascade: {max_changes}"


def test_budget_change_cascade_size():
    """Test that budget changes cascade appropriately."""
    result = measure_temporal_cascades(
        feature_name="uncertainty",
        perturbation_pattern="increasing",
        time_scales=[8, 16]
    )

    # max_steps changes should be bounded
    max_changes = max(result.max_steps_changes)
    assert max_changes <= 10, \
        f"Excessive max_steps changes: {max_changes}"


# ============================================================================
# PATTERN COMPARISON TESTS
# ============================================================================

def test_increasing_vs_decreasing_symmetry():
    """Test that increasing and decreasing patterns show symmetric behavior."""
    result_inc = measure_temporal_cascades(
        feature_name="uncertainty",
        perturbation_pattern="increasing"
    )

    result_dec = measure_temporal_cascades(
        feature_name="uncertainty",
        perturbation_pattern="decreasing"
    )

    # Scale invariance errors should be similar
    error_diff = abs(result_inc.scale_invariance_error - result_dec.scale_invariance_error)
    assert error_diff < 0.3, \
        f"Asymmetric temporal behavior: inc_error={result_inc.scale_invariance_error:.3f}, " \
        f"dec_error={result_dec.scale_invariance_error:.3f}"

    # Memory fragility should be similar
    fragility_diff = abs(result_inc.temporal_memory_fragility - result_dec.temporal_memory_fragility)
    assert fragility_diff < 0.2, \
        f"Asymmetric memory effects: diff={fragility_diff:.3f}"


# ============================================================================
# COMPREHENSIVE CASCADE ANALYSIS
# ============================================================================

@pytest.mark.parametrize("feature,pattern", [
    ("uncertainty", "increasing"),
    ("uncertainty", "decreasing"),
    ("contradiction_signal", "increasing"),
    ("contradiction_signal", "decreasing"),
    ("edge_pressure", "increasing"),
    ("causal_risk", "increasing"),
])
def test_no_pathological_cascades(feature: str, pattern: str):
    """Parametrized test to ensure no cascades are pathological."""
    result = measure_temporal_cascades(
        feature_name=feature,
        perturbation_pattern=pattern
    )

    # No excessive scale variance
    assert result.scale_invariance_error < 0.8, \
        f"{feature}/{pattern}: Excessive scale variance {result.scale_invariance_error:.3f}"

    # No excessive memory fragility
    assert result.temporal_memory_fragility < 0.7, \
        f"{feature}/{pattern}: Excessive memory fragility {result.temporal_memory_fragility:.3f}"


# ============================================================================
# ARTIFACT GENERATION
# ============================================================================

@pytest.mark.slow
def test_generate_temporal_cascade_report(tmp_path):
    """Generate comprehensive temporal cascade analysis report."""
    import json
    from datetime import datetime

    configurations = [
        ("uncertainty", "increasing"),
        ("uncertainty", "decreasing"),
        ("uncertainty", "oscillating"),
        ("uncertainty", "pulsed"),
        ("contradiction_signal", "increasing"),
        ("contradiction_signal", "decreasing"),
        ("contradiction_signal", "oscillating"),
        ("edge_pressure", "increasing"),
        ("edge_pressure", "decreasing"),
        ("causal_risk", "increasing"),
        ("causal_risk", "pulsed"),
    ]

    report = {
        "test_date": datetime.now().isoformat(),
        "test_type": "temporal_multiscale_cascade",
        "version": "1.0.0",
        "cascades": []
    }

    for feature, pattern in configurations:
        result = measure_temporal_cascades(
            feature_name=feature,
            perturbation_pattern=pattern
        )

        report["cascades"].append({
            "feature": feature,
            "pattern": pattern,
            "self_similar": result.self_similar,
            "scale_invariance_error": result.scale_invariance_error,
            "temporal_memory_fragility": result.temporal_memory_fragility,
            "time_scales": result.time_scales,
            "family_change_counts": result.family_change_counts,
            "max_steps_changes": result.max_steps_changes,
            "recommendation_changes": result.recommendation_changes,
            "activation_persistence": result.activation_persistence,
            "recovery_times": result.recovery_times,
            "hysteresis_widths": result.hysteresis_widths
        })

    # Save report
    output_file = tmp_path / "temporal_cascade_report.json"
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)

    assert output_file.exists()

    # Generate summary
    summary_lines = [
        "="*80,
        "TEMPORAL MULTISCALE CASCADE REPORT",
        "="*80,
        f"Test Date: {report['test_date']}",
        f"Cascade Patterns Tested: {len(report['cascades'])}",
        "",
        "SELF-SIMILARITY SUMMARY:",
        "-"*80
    ]

    self_similar = sum(1 for c in report["cascades"] if c["self_similar"])
    mean_scale_error = sum(c["scale_invariance_error"] for c in report["cascades"]) / len(report["cascades"])
    mean_fragility = sum(c["temporal_memory_fragility"] for c in report["cascades"]) / len(report["cascades"])

    summary_lines.extend([
        f"  Self-Similar: {self_similar}/{len(report['cascades'])}",
        f"  Mean Scale Invariance Error: {mean_scale_error:.3f}",
        f"  Mean Memory Fragility: {mean_fragility:.3f}",
        "",
        "CASCADE DETAILS:",
        "-"*80
    ])

    for c in report["cascades"]:
        summary_lines.extend([
            f"{c['feature']} - {c['pattern']}",
            f"  Self-Similar: {c['self_similar']}",
            f"  Scale Error: {c['scale_invariance_error']:.3f}",
            f"  Memory Fragility: {c['temporal_memory_fragility']:.3f}",
            f"  Max Family Changes: {max(c['family_change_counts'])}",
            ""
        ])

    summary_lines.append("="*80)

    summary_file = tmp_path / "temporal_cascade_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("\n".join(summary_lines))

    assert summary_file.exists()

    print("\n" + "\n".join(summary_lines))

    # Validate quality
    assert mean_scale_error < 0.6, \
        f"Mean scale invariance error too high: {mean_scale_error:.3f}"

    assert mean_fragility < 0.5, \
        f"Mean memory fragility too high: {mean_fragility:.3f}"
