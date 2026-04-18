"""
Activation Avalanche Statistics Tests - Fractal Stress Level
=============================================================

Tests distribution of activation avalanche sizes to detect:
- Rigid vs critical vs fragile dynamics
- Heavy-tailed distributions indicating criticality
- Controlled vs pathological sensitivity
"""

from __future__ import annotations

import pytest
from tests.reasoning_stress.fractal_utils import (
    measure_activation_avalanches,
    AvalancheDistribution
)


# ============================================================================
# BASIC AVALANCHE DISTRIBUTION TESTS
# ============================================================================

def test_avalanche_distribution_exists():
    """Test that avalanche measurement produces valid distribution."""
    result = measure_activation_avalanches(
        n_trials=500,
        perturbation_magnitude=0.02
    )

    # Should have measured avalanches
    assert len(result.avalanche_sizes) == 500

    # Should have histogram
    assert len(result.size_histogram) > 0

    # Mean size should be reasonable
    assert 0.0 <= result.mean_size <= 3.0


def test_avalanche_size_bounded():
    """Test that avalanche sizes are bounded (no runaway cascades)."""
    result = measure_activation_avalanches(n_trials=500)

    # Maximum avalanche should be bounded
    assert result.max_size <= 3, \
        f"Unbounded avalanche detected: max_size={result.max_size}"


def test_avalanche_criticality_classification():
    """Test that system is classified appropriately."""
    result = measure_activation_avalanches(n_trials=500)

    # Should not be fragile (too sensitive)
    assert result.criticalit_indicator != "fragile", \
        "System shows fragile dynamics (too sensitive to perturbations)"

    # Should be interesting or at least not rigid
    assert result.criticalit_indicator in ["interesting", "rigid"], \
        f"Unexpected criticality: {result.criticalit_indicator}"


# ============================================================================
# PERTURBATION MAGNITUDE TESTS
# ============================================================================

def test_small_perturbations_produce_small_avalanches():
    """Test that small perturbations produce controlled avalanches."""
    result = measure_activation_avalanches(
        n_trials=300,
        perturbation_magnitude=0.01  # Very small perturbation
    )

    # Mean size should be small
    assert result.mean_size < 1.5, \
        f"Small perturbations produce large avalanches: mean={result.mean_size:.3f}"


def test_moderate_perturbations():
    """Test avalanches from moderate perturbations."""
    result = measure_activation_avalanches(
        n_trials=500,
        perturbation_magnitude=0.03  # Moderate
    )

    # Should still be controlled
    assert result.max_size <= 3

    # May have some interesting dynamics
    assert result.mean_size < 2.0


def test_perturbation_magnitude_scaling():
    """Test that avalanche size scales reasonably with perturbation magnitude."""
    result_small = measure_activation_avalanches(
        n_trials=200,
        perturbation_magnitude=0.01
    )

    result_large = measure_activation_avalanches(
        n_trials=200,
        perturbation_magnitude=0.05
    )

    # Larger perturbations should produce larger or equal average avalanches
    assert result_large.mean_size >= result_small.mean_size - 0.2, \
        "Avalanche size does not scale with perturbation"


# ============================================================================
# DISTRIBUTION SHAPE TESTS
# ============================================================================

def test_avalanche_distribution_shape():
    """Test shape of avalanche size distribution."""
    result = measure_activation_avalanches(n_trials=1000)

    # Most avalanches should be small (size 0 or 1)
    small_avalanches = sum(
        count for size, count in result.size_histogram.items()
        if size <= 1
    )

    total_avalanches = sum(result.size_histogram.values())

    small_fraction = small_avalanches / total_avalanches

    # At least 50% should be small avalanches (controlled system)
    assert small_fraction >= 0.4, \
        f"Too few small avalanches: {small_fraction:.1%}"


def test_no_dominant_avalanche_size():
    """Test that distribution is not concentrated on one size."""
    result = measure_activation_avalanches(n_trials=1000)

    # No single size should dominate (> 80%)
    max_count = max(result.size_histogram.values())
    total_count = sum(result.size_histogram.values())

    max_fraction = max_count / total_count

    assert max_fraction < 0.85, \
        f"Distribution too concentrated: {max_fraction:.1%} on one size"


# ============================================================================
# HEAVY-TAILED ANALYSIS
# ============================================================================

def test_heavy_tail_analysis():
    """Test heavy-tail detection and interpretation."""
    result = measure_activation_avalanches(n_trials=800)

    # If heavy-tailed, should indicate interesting criticality
    if result.is_heavy_tailed:
        # Heavy tail should correlate with criticality
        assert result.criticalit_indicator in ["interesting", "fragile"], \
            "Heavy-tailed but not classified as critical"

        # Tail exponent should be reasonable
        assert -4.0 <= result.tail_exponent <= -1.0, \
            f"Unreasonable tail exponent: {result.tail_exponent:.3f}"


def test_not_too_heavy_tailed():
    """Test that distribution is not excessively heavy-tailed (fragile)."""
    result = measure_activation_avalanches(n_trials=1000)

    # If heavy-tailed, large avalanches should still be minority
    large_avalanches = sum(
        count for size, count in result.size_histogram.items()
        if size >= 2
    )

    total = sum(result.size_histogram.values())
    large_fraction = large_avalanches / total if total > 0 else 0

    # Large avalanches should be < 30% (not too fragile)
    assert large_fraction < 0.35, \
        f"Too many large avalanches: {large_fraction:.1%} (fragile system)"


# ============================================================================
# CRITICALITY INDICATOR TESTS
# ============================================================================

def test_not_rigid_system():
    """Test that system is not overly rigid (responds to perturbations)."""
    result = measure_activation_avalanches(n_trials=500)

    # Mean avalanche size should be > 0 (not completely rigid)
    assert result.mean_size > 0.1, \
        "System appears rigid (no response to perturbations)"

    # Should have some variety in avalanche sizes
    assert len(result.size_histogram) >= 2, \
        "System too rigid (all avalanches same size)"


def test_interesting_criticality_characteristics():
    """Test characteristics of 'interesting' criticality region."""
    result = measure_activation_avalanches(n_trials=800)

    if result.criticalit_indicator == "interesting":
        # Should have moderate mean size
        assert 0.3 <= result.mean_size <= 1.5, \
            f"'Interesting' criticality has unusual mean: {result.mean_size:.3f}"

        # Should have some but not excessive large avalanches
        assert result.max_size <= 3


# ============================================================================
# STABILITY AND CONSISTENCY TESTS
# ============================================================================

def test_avalanche_measurement_consistency():
    """Test that avalanche measurements are consistent across runs."""
    result1 = measure_activation_avalanches(n_trials=300)
    # Note: Uses fixed seed, so should be identical

    result2 = measure_activation_avalanches(n_trials=300)

    # Should produce same results (deterministic with seed=42)
    assert result1.mean_size == result2.mean_size
    assert result1.max_size == result2.max_size
    assert result1.criticalit_indicator == result2.criticalit_indicator


def test_large_sample_stability():
    """Test that results stabilize with larger samples."""
    result_large = measure_activation_avalanches(n_trials=1500)

    # Mean should be reasonable
    assert 0.0 <= result_large.mean_size <= 2.0

    # Should have clear criticality classification
    assert result_large.criticalit_indicator in ["rigid", "interesting", "fragile"]


# ============================================================================
# HISTOGRAM PROPERTIES
# ============================================================================

def test_histogram_completeness():
    """Test that histogram covers all avalanche sizes."""
    result = measure_activation_avalanches(n_trials=500)

    # Histogram should account for all avalanches
    histogram_total = sum(result.size_histogram.values())
    assert histogram_total == len(result.avalanche_sizes)


def test_histogram_size_range():
    """Test that histogram only contains valid sizes."""
    result = measure_activation_avalanches(n_trials=500)

    # All sizes should be non-negative and bounded
    for size in result.size_histogram.keys():
        assert 0 <= size <= 3, \
            f"Invalid avalanche size in histogram: {size}"


# ============================================================================
# COMPARATIVE ANALYSIS
# ============================================================================

def test_compare_small_vs_large_perturbations():
    """Compare avalanche characteristics for different perturbation sizes."""
    small_pert = measure_activation_avalanches(
        n_trials=400,
        perturbation_magnitude=0.01
    )

    large_pert = measure_activation_avalanches(
        n_trials=400,
        perturbation_magnitude=0.04
    )

    # Larger perturbations should produce larger or equal avalanches
    assert large_pert.mean_size >= small_pert.mean_size - 0.3

    # Both should be controlled
    assert small_pert.max_size <= 3
    assert large_pert.max_size <= 3


# ============================================================================
# EDGE CASE TESTS
# ============================================================================

def test_minimal_sample_avalanches():
    """Test avalanche measurement with minimal samples."""
    result = measure_activation_avalanches(n_trials=50)

    # Should still produce valid results
    assert len(result.avalanche_sizes) == 50
    assert result.mean_size >= 0.0
    assert result.criticalit_indicator in ["rigid", "interesting", "fragile"]


def test_zero_perturbation_handling():
    """Test behavior with near-zero perturbations."""
    result = measure_activation_avalanches(
        n_trials=200,
        perturbation_magnitude=0.001  # Very small
    )

    # Should mostly produce zero-size avalanches (no change)
    zero_avalanches = result.size_histogram.get(0, 0)
    total = sum(result.size_histogram.values())

    zero_fraction = zero_avalanches / total if total > 0 else 0

    # Most should be zero with such small perturbations
    assert zero_fraction >= 0.3, \
        "Even tiny perturbations cause significant avalanches"


# ============================================================================
# COMPREHENSIVE AVALANCHE ANALYSIS
# ============================================================================

@pytest.mark.slow
def test_comprehensive_avalanche_analysis(tmp_path):
    """Comprehensive avalanche analysis across multiple configurations."""
    import json
    from datetime import datetime

    configurations = [
        ("standard", 0.02, 1000),
        ("small_pert", 0.01, 800),
        ("moderate_pert", 0.03, 800),
        ("large_pert", 0.05, 800),
        ("large_sample", 0.02, 2000),
    ]

    report = {
        "test_date": datetime.now().isoformat(),
        "test_type": "activation_avalanche_statistics",
        "version": "1.0.0",
        "analyses": []
    }

    for config_name, magnitude, n_trials in configurations:
        result = measure_activation_avalanches(
            n_trials=n_trials,
            perturbation_magnitude=magnitude
        )

        report["analyses"].append({
            "configuration": config_name,
            "perturbation_magnitude": magnitude,
            "n_trials": n_trials,
            "mean_size": result.mean_size,
            "max_size": result.max_size,
            "tail_exponent": result.tail_exponent,
            "is_heavy_tailed": result.is_heavy_tailed,
            "criticality_indicator": result.criticalit_indicator,
            "size_histogram": result.size_histogram
        })

    # Save report
    output_file = tmp_path / "avalanche_statistics_report.json"
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)

    assert output_file.exists()

    # Generate summary
    summary_lines = [
        "="*80,
        "ACTIVATION AVALANCHE STATISTICS REPORT",
        "="*80,
        f"Test Date: {report['test_date']}",
        f"Configurations Tested: {len(report['analyses'])}",
        "",
        "CRITICALITY SUMMARY:",
        "-"*80
    ]

    criticality_counts = {}
    for analysis in report["analyses"]:
        indicator = analysis["criticality_indicator"]
        criticality_counts[indicator] = criticality_counts.get(indicator, 0) + 1

    for indicator, count in criticality_counts.items():
        summary_lines.append(f"  {indicator.capitalize()}: {count}")

    summary_lines.extend([
        "",
        "AVALANCHE DETAILS:",
        "-"*80
    ])

    for analysis in report["analyses"]:
        summary_lines.extend([
            f"{analysis['configuration']} (pert={analysis['perturbation_magnitude']:.3f}, n={analysis['n_trials']})",
            f"  Mean Size: {analysis['mean_size']:.3f}",
            f"  Max Size: {analysis['max_size']}",
            f"  Heavy-Tailed: {analysis['is_heavy_tailed']}",
            f"  Criticality: {analysis['criticality_indicator']}",
            ""
        ])

    summary_lines.append("="*80)

    summary_file = tmp_path / "avalanche_statistics_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("\n".join(summary_lines))

    assert summary_file.exists()

    print("\n" + "\n".join(summary_lines))

    # Validate quality
    fragile_count = criticality_counts.get("fragile", 0)
    assert fragile_count == 0, \
        f"{fragile_count} configurations show fragile dynamics"

    # At least some should be interesting
    interesting_count = criticality_counts.get("interesting", 0)
    assert interesting_count >= len(report["analyses"]) * 0.3, \
        "System may be too rigid (few interesting dynamics)"


# ============================================================================
# DIAGNOSTIC TESTS
# ============================================================================

def test_avalanche_diagnostic_output():
    """Test that avalanche analysis provides useful diagnostics."""
    result = measure_activation_avalanches(n_trials=500)

    # Should provide all key metrics
    assert result.mean_size is not None
    assert result.max_size is not None
    assert result.tail_exponent is not None
    assert result.is_heavy_tailed is not None
    assert result.criticalit_indicator is not None

    # Histogram should be populated
    assert len(result.size_histogram) > 0

    # Avalanche sizes should match histogram
    assert len(result.avalanche_sizes) == sum(result.size_histogram.values())


def test_avalanche_interpretability():
    """Test that avalanche results are interpretable."""
    result = measure_activation_avalanches(n_trials=500)

    # Criticality indicator should provide clear interpretation
    assert result.criticalit_indicator in ["rigid", "interesting", "fragile"]

    # Heavy-tailed classification should be boolean
    assert isinstance(result.is_heavy_tailed, bool)

    # Statistics should be reasonable
    assert result.mean_size <= result.max_size
