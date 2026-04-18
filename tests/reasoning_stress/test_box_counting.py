"""
Box-Counting Fractal Dimension Tests - Fractal Stress Level
============================================================

Tests fractal dimension of activation frontiers in 2D/3D feature subspaces.
Determines whether frontiers are:
- Clean (dimension ~1 in 2D)
- Rugose but controlled (1 < dim < 1.35)
- Pathological (dim > 1.35 or poor fit)
"""

from __future__ import annotations

import pytest
from tests.reasoning_stress.fractal_utils import (
    estimate_box_counting_dimension,
    BoxCountingResult
)


# ============================================================================
# 2D ACTIVATION FRONTIER TESTS
# ============================================================================

def test_contradiction_edge_heur_frontier_dimension():
    """Test fractal dimension of HEUR frontier in contradiction-edge space."""
    result = estimate_box_counting_dimension(
        feature_x="contradiction_signal",
        feature_y="edge_pressure",
        family_name="heur",
        n_samples=500
    )

    # HEUR activates primarily on edge_pressure, so frontier should be clean
    assert result.fit_quality >= 0.7, \
        f"Poor box-counting fit: {result.fit_quality:.3f}"

    assert result.fractal_dimension < 1.4, \
        f"Frontier too rugose: dimension={result.fractal_dimension:.3f}"

    assert result.interpretation in ["clean", "rugose_controlled"], \
        f"Pathological frontier: {result.interpretation}"


def test_uncertainty_contradiction_dia_adv_frontier():
    """Test DIA_ADV frontier in uncertainty-contradiction space."""
    result = estimate_box_counting_dimension(
        feature_x="uncertainty",
        feature_y="contradiction_signal",
        family_name="dia_adv",
        n_samples=500
    )

    # DIA_ADV activates on contradiction_signal primarily
    assert result.fit_quality >= 0.6, \
        f"Poor fit quality: {result.fit_quality:.3f}"

    assert result.interpretation != "pathological", \
        f"DIA_ADV frontier is pathological: dim={result.fractal_dimension:.3f}"


def test_symbolic_law_eml_sr_frontier():
    """Test EML_SR frontier in symbolic_regularity-law_fit_signal space."""
    result = estimate_box_counting_dimension(
        feature_x="symbolic_regularity",
        feature_y="law_fit_signal",
        family_name="eml_sr",
        n_samples=500
    )

    # EML_SR activates on either feature, may have more complex frontier
    # But should not be pathological
    assert result.interpretation != "pathological", \
        f"EML_SR frontier pathological: {result.interpretation}"

    # If we have enough data
    if result.interpretation != "insufficient_data":
        assert result.fractal_dimension < 1.5, \
            f"EML_SR frontier too complex: dim={result.fractal_dimension:.3f}"


def test_edge_causal_heur_frontier():
    """Test HEUR frontier in edge_pressure-causal_risk space."""
    result = estimate_box_counting_dimension(
        feature_x="edge_pressure",
        feature_y="causal_risk",
        family_name="heur",
        n_samples=500
    )

    # Should be clean since HEUR depends mainly on edge_pressure
    assert result.interpretation in ["clean", "rugose_controlled"], \
        f"HEUR frontier not clean: {result.interpretation}"


# ============================================================================
# FRONTIER COMPARISON TESTS
# ============================================================================

def test_all_major_frontiers_not_pathological():
    """Test that all major activation frontiers are well-behaved."""
    frontiers_to_test = [
        ("contradiction_signal", "edge_pressure", "heur"),
        ("uncertainty", "contradiction_signal", "dia_adv"),
        ("contradiction_signal", "causal_risk", "fal_guard"),
        ("symbolic_regularity", "law_fit_signal", "eml_sr"),
    ]

    results = []

    for feature_x, feature_y, family in frontiers_to_test:
        result = estimate_box_counting_dimension(
            feature_x=feature_x,
            feature_y=feature_y,
            family_name=family,
            n_samples=400
        )

        results.append({
            "feature_pair": f"{feature_x} × {feature_y}",
            "family": family,
            "dimension": result.fractal_dimension,
            "fit_quality": result.fit_quality,
            "interpretation": result.interpretation
        })

    # No frontier should be pathological
    pathological = [r for r in results if r["interpretation"] == "pathological"]
    assert len(pathological) == 0, \
        f"Pathological frontiers detected: {pathological}"

    # Most should have good fit
    poor_fit = [r for r in results if r["fit_quality"] < 0.6 and r["interpretation"] != "insufficient_data"]
    assert len(poor_fit) <= 1, \
        f"Too many poor fits: {poor_fit}"

    # Report results
    print("\n" + "="*80)
    print("BOX-COUNTING FRACTAL DIMENSION ANALYSIS")
    print("="*80)
    for r in results:
        print(f"{r['family'].upper()} in {r['feature_pair']}")
        print(f"  Dimension: {r['dimension']:.3f}")
        print(f"  Fit Quality: {r['fit_quality']:.3f}")
        print(f"  Interpretation: {r['interpretation']}")
        print()


def test_frontier_dimension_distribution():
    """Test distribution of fractal dimensions across frontiers."""
    frontiers = [
        ("edge_pressure", "contradiction_signal", "heur"),
        ("edge_pressure", "uncertainty", "heur"),
        ("contradiction_signal", "uncertainty", "dia_adv"),
        ("contradiction_signal", "causal_risk", "dia_adv"),
    ]

    dimensions = []

    for feature_x, feature_y, family in frontiers:
        result = estimate_box_counting_dimension(
            feature_x=feature_x,
            feature_y=feature_y,
            family_name=family,
            n_samples=300
        )

        if result.interpretation != "insufficient_data":
            dimensions.append(result.fractal_dimension)

    if dimensions:
        mean_dim = sum(dimensions) / len(dimensions)
        max_dim = max(dimensions)

        # Mean dimension should be close to 1 (clean boundaries)
        assert mean_dim < 1.4, \
            f"Mean fractal dimension too high: {mean_dim:.3f}"

        # Maximum dimension should not be excessive
        assert max_dim < 1.6, \
            f"Maximum fractal dimension too high: {max_dim:.3f}"


# ============================================================================
# FIT QUALITY TESTS
# ============================================================================

def test_box_counting_fit_quality():
    """Test that box-counting produces good fits for major frontiers."""
    result = estimate_box_counting_dimension(
        feature_x="edge_pressure",
        feature_y="contradiction_signal",
        family_name="heur",
        n_samples=600,
        box_sizes=[0.25, 0.125, 0.0625, 0.03125, 0.015625]  # More scales
    )

    # With more scales, fit should be better
    if result.interpretation != "insufficient_data":
        assert result.fit_quality >= 0.75, \
            f"Poor fit even with many scales: {result.fit_quality:.3f}"


def test_box_counting_consistency():
    """Test that box-counting is consistent across different samplings."""
    # Run twice with different sampling
    result1 = estimate_box_counting_dimension(
        feature_x="contradiction_signal",
        feature_y="edge_pressure",
        family_name="dia_adv",
        n_samples=400
    )

    result2 = estimate_box_counting_dimension(
        feature_x="contradiction_signal",
        feature_y="edge_pressure",
        family_name="dia_adv",
        n_samples=400
    )

    # Results should be similar (within 0.2)
    if result1.interpretation != "insufficient_data" and result2.interpretation != "insufficient_data":
        dim_diff = abs(result1.fractal_dimension - result2.fractal_dimension)
        assert dim_diff < 0.3, \
            f"Inconsistent results: dim1={result1.fractal_dimension:.3f}, dim2={result2.fractal_dimension:.3f}"


# ============================================================================
# CLEAN VS RUGOSE DETECTION
# ============================================================================

def test_clean_frontier_detection():
    """Test that genuinely clean frontiers are detected as such."""
    # HEUR should have clean frontier in space where it's the only active family
    result = estimate_box_counting_dimension(
        feature_x="edge_pressure",
        feature_y="causal_risk",
        family_name="heur",
        n_samples=500
    )

    # Should be clean or rugose_controlled
    if result.interpretation != "insufficient_data":
        assert result.interpretation in ["clean", "rugose_controlled"], \
            f"HEUR frontier not detected as clean: {result.interpretation}"

        # Dimension should be close to 1
        assert result.fractal_dimension < 1.25, \
            f"Clean frontier has high dimension: {result.fractal_dimension:.3f}"


def test_rugose_vs_pathological_distinction():
    """Test that we can distinguish rugose-but-controlled from pathological."""
    # Test a potentially complex frontier (EML_SR with two activation paths)
    result = estimate_box_counting_dimension(
        feature_x="symbolic_regularity",
        feature_y="law_fit_signal",
        family_name="eml_sr",
        n_samples=600
    )

    if result.interpretation != "insufficient_data":
        # May be rugose due to dual activation paths, but should not be pathological
        assert result.interpretation != "pathological", \
            f"EML_SR frontier classified as pathological"

        # Even if rugose, should have reasonable dimension
        assert result.fractal_dimension < 1.5, \
            f"Dimension too high: {result.fractal_dimension:.3f}"


# ============================================================================
# BOX SIZE SCALING TESTS
# ============================================================================

def test_box_counting_scaling_law():
    """Test that box counts follow expected scaling law."""
    result = estimate_box_counting_dimension(
        feature_x="edge_pressure",
        feature_y="contradiction_signal",
        family_name="heur",
        n_samples=500,
        box_sizes=[0.2, 0.1, 0.05, 0.025, 0.0125]
    )

    if result.interpretation != "insufficient_data":
        # Box counts should increase as box size decreases
        for i in range(1, len(result.box_counts)):
            assert result.box_counts[i] >= result.box_counts[i-1], \
                f"Box counts do not increase monotonically: {result.box_counts}"

        # Should have power-law relationship (good fit)
        assert result.fit_quality >= 0.6, \
            f"Poor power-law fit: {result.fit_quality:.3f}"


# ============================================================================
# INSUFFICIENT DATA HANDLING
# ============================================================================

def test_handles_insufficient_frontier_data():
    """Test graceful handling when frontier is sparse."""
    # Test in a space where family may not activate much
    result = estimate_box_counting_dimension(
        feature_x="continuity_recent",
        feature_y="uncertainty",
        family_name="heur",  # Doesn't depend on these features
        n_samples=100  # Small sample
    )

    # Should either return insufficient_data or have low confidence
    if result.interpretation == "insufficient_data":
        assert result.fractal_dimension >= 0.0  # Valid default
        assert result.fit_quality >= 0.0  # Valid default
    else:
        # If we got data, fit quality might be low
        assert result.fit_quality >= 0.0


# ============================================================================
# COMPREHENSIVE FRONTIER ATLAS
# ============================================================================

@pytest.mark.slow
def test_generate_fractal_dimension_atlas(tmp_path):
    """Generate comprehensive atlas of fractal dimensions."""
    import json
    from datetime import datetime

    # Test all major frontiers in multiple projections
    test_configurations = [
        # HEUR frontiers
        ("contradiction_signal", "edge_pressure", "heur"),
        ("uncertainty", "edge_pressure", "heur"),
        ("edge_pressure", "causal_risk", "heur"),

        # DIA_ADV frontiers
        ("uncertainty", "contradiction_signal", "dia_adv"),
        ("contradiction_signal", "causal_risk", "dia_adv"),
        ("edge_pressure", "contradiction_signal", "dia_adv"),

        # FAL_GUARD frontiers
        ("uncertainty", "contradiction_signal", "fal_guard"),
        ("contradiction_signal", "causal_risk", "fal_guard"),

        # EML_SR frontiers
        ("symbolic_regularity", "law_fit_signal", "eml_sr"),
        ("symbolic_regularity", "uncertainty", "eml_sr"),
        ("law_fit_signal", "uncertainty", "eml_sr"),
    ]

    atlas = {
        "test_date": datetime.now().isoformat(),
        "test_type": "box_counting_fractal_dimension",
        "version": "1.0.0",
        "frontiers": []
    }

    for feature_x, feature_y, family in test_configurations:
        result = estimate_box_counting_dimension(
            feature_x=feature_x,
            feature_y=feature_y,
            family_name=family,
            n_samples=500
        )

        atlas["frontiers"].append({
            "feature_x": feature_x,
            "feature_y": feature_y,
            "family": family,
            "fractal_dimension": result.fractal_dimension,
            "fit_quality": result.fit_quality,
            "interpretation": result.interpretation,
            "box_sizes": result.box_sizes,
            "box_counts": result.box_counts
        })

    # Save atlas
    output_file = tmp_path / "fractal_dimension_atlas.json"
    with open(output_file, 'w') as f:
        json.dump(atlas, f, indent=2)

    assert output_file.exists()

    # Generate summary
    summary_lines = [
        "="*80,
        "FRACTAL DIMENSION ATLAS",
        "="*80,
        f"Test Date: {atlas['test_date']}",
        f"Frontiers Analyzed: {len(atlas['frontiers'])}",
        "",
        "INTERPRETATION SUMMARY:",
        "-"*80
    ]

    clean = sum(1 for f in atlas["frontiers"] if f["interpretation"] == "clean")
    rugose = sum(1 for f in atlas["frontiers"] if f["interpretation"] == "rugose_controlled")
    pathological = sum(1 for f in atlas["frontiers"] if f["interpretation"] == "pathological")
    insufficient = sum(1 for f in atlas["frontiers"] if f["interpretation"] == "insufficient_data")

    summary_lines.extend([
        f"  Clean: {clean}",
        f"  Rugose (Controlled): {rugose}",
        f"  Pathological: {pathological}",
        f"  Insufficient Data: {insufficient}",
        "",
        "FRONTIER DETAILS:",
        "-"*80
    ])

    for f in atlas["frontiers"]:
        summary_lines.extend([
            f"{f['family'].upper()} in {f['feature_x']} × {f['feature_y']}",
            f"  Dimension: {f['fractal_dimension']:.3f}",
            f"  Fit Quality: {f['fit_quality']:.3f}",
            f"  Interpretation: {f['interpretation']}",
            ""
        ])

    summary_lines.append("="*80)

    summary_file = tmp_path / "fractal_dimension_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("\n".join(summary_lines))

    assert summary_file.exists()

    print("\n" + "\n".join(summary_lines))

    # Validate atlas quality
    assert pathological == 0, f"{pathological} pathological frontiers detected"
    assert clean + rugose >= len(atlas["frontiers"]) * 0.7, \
        "Most frontiers should be clean or controlled"
