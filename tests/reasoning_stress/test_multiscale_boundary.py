"""
Multiscale Boundary Refinement Tests - Fractal Stress Level
============================================================

Tests boundary behavior at multiple resolution scales to detect:
- Convergence vs divergence as resolution refines
- Boundary roughness and self-similarity
- Scale-dependent hysteresis
- Disciplined vs pathological thresholds
"""

from __future__ import annotations

import pytest
from typing import Dict
from tests.reasoning_stress.fractal_utils import (
    measure_multiscale_boundary,
    MultiscaleBoundaryMetrics
)


# ============================================================================
# EDGE PRESSURE MULTISCALE TESTS
# ============================================================================

def test_edge_pressure_heur_multiscale_convergence():
    """Test that HEUR activation boundary converges at finer resolutions."""
    result = measure_multiscale_boundary(
        feature_name="edge_pressure",
        threshold=0.7,
        family_name="heur",
        resolutions=[0.10, 0.05, 0.02, 0.01, 0.005]
    )

    # Should converge as resolution refines
    assert result.converges, \
        f"HEUR boundary does not converge: activation_points={result.activation_points}"

    # Roughness should be low for disciplined boundary
    assert result.roughness_exponent < 0.5, \
        f"Excessive roughness: {result.roughness_exponent}"

    # Should be classified as disciplined
    assert result.discipline == "disciplined", \
        f"HEUR boundary is {result.discipline}, not disciplined"


def test_edge_pressure_heur_hysteresis_scaling():
    """Test that hysteresis width behaves predictably across scales."""
    result = measure_multiscale_boundary(
        feature_name="edge_pressure",
        threshold=0.7,
        family_name="heur"
    )

    # Hysteresis should not increase with finer resolution (no spurious memory)
    for i in range(1, len(result.hysteresis_widths)):
        assert result.hysteresis_widths[i] <= result.hysteresis_widths[i-1] * 1.5, \
            f"Hysteresis grows with resolution: {result.hysteresis_widths}"


def test_edge_pressure_budget_reduction_multiscale():
    """Test budget reduction threshold at multiple scales."""
    result = measure_multiscale_boundary(
        feature_name="edge_pressure",
        threshold=0.8,
        family_name=None  # Track generic budget changes
    )

    # Should have clean convergence
    assert result.converges, "Budget reduction boundary does not converge"

    # Discontinuity counts should stabilize or decrease
    for i in range(1, len(result.discontinuity_counts)):
        assert result.discontinuity_counts[i] <= result.discontinuity_counts[i-1] + 2, \
            f"Discontinuities increase with resolution: {result.discontinuity_counts}"


# ============================================================================
# CONTRADICTION SIGNAL MULTISCALE TESTS
# ============================================================================

def test_contradiction_dia_adv_multiscale_convergence():
    """Test DIA_ADV activation boundary convergence."""
    result = measure_multiscale_boundary(
        feature_name="contradiction_signal",
        threshold=0.45,
        family_name="dia_adv"
    )

    # Should converge
    assert result.converges, \
        f"DIA_ADV boundary does not converge: {result.activation_points}"

    # Should be disciplined or critical, not pathological
    assert result.discipline in ["disciplined", "critical"], \
        f"DIA_ADV boundary is {result.discipline}"


def test_contradiction_fal_guard_multiscale_convergence():
    """Test FAL_GUARD activation boundary convergence."""
    result = measure_multiscale_boundary(
        feature_name="contradiction_signal",
        threshold=0.45,
        family_name="fal_guard"
    )

    assert result.converges
    assert result.discipline in ["disciplined", "critical"]


def test_contradiction_guards_multiscale_consistency():
    """Test that DIA_ADV and FAL_GUARD have consistent multiscale behavior."""
    result_dia = measure_multiscale_boundary(
        feature_name="contradiction_signal",
        threshold=0.45,
        family_name="dia_adv"
    )

    result_fal = measure_multiscale_boundary(
        feature_name="contradiction_signal",
        threshold=0.45,
        family_name="fal_guard"
    )

    # Both should have similar convergence behavior
    assert result_dia.converges == result_fal.converges

    # Activation points should be similar across resolutions
    for i in range(len(result_dia.activation_points)):
        diff = abs(result_dia.activation_points[i] - result_fal.activation_points[i])
        assert diff < 0.03, \
            f"DIA_ADV and FAL_GUARD diverge at resolution {result_dia.resolutions[i]}: diff={diff}"


# ============================================================================
# SYMBOLIC REGULARITY MULTISCALE TESTS
# ============================================================================

def test_symbolic_regularity_eml_sr_multiscale():
    """Test EML_SR activation via symbolic_regularity at multiple scales."""
    result = measure_multiscale_boundary(
        feature_name="symbolic_regularity",
        threshold=0.4,
        family_name="eml_sr"
    )

    # Should converge
    assert result.converges, \
        f"EML_SR/symbolic boundary does not converge: {result.activation_points}"

    # Experimental feature may be critical rather than disciplined
    assert result.discipline in ["disciplined", "critical"], \
        f"EML_SR/symbolic is {result.discipline}"


def test_law_fit_signal_eml_sr_multiscale():
    """Test EML_SR activation via law_fit_signal at multiple scales."""
    result = measure_multiscale_boundary(
        feature_name="law_fit_signal",
        threshold=0.4,
        family_name="eml_sr"
    )

    assert result.converges
    assert result.discipline in ["disciplined", "critical"]


# ============================================================================
# UNCERTAINTY MULTISCALE TESTS
# ============================================================================

def test_uncertainty_budget_increase_multiscale():
    """Test uncertainty budget increase at multiple scales."""
    result = measure_multiscale_boundary(
        feature_name="uncertainty",
        threshold=0.6,
        family_name=None  # Track budget changes
    )

    # Should converge
    assert result.converges, "Uncertainty budget boundary does not converge"

    # Convergence rate should be reasonable
    assert result.convergence_rate < 0.8, \
        f"Poor convergence rate: {result.convergence_rate}"


# ============================================================================
# CAUSAL RISK MULTISCALE TESTS
# ============================================================================

def test_causal_risk_budget_increase_multiscale():
    """Test causal_risk budget increase at multiple scales."""
    result = measure_multiscale_boundary(
        feature_name="causal_risk",
        threshold=0.5,
        family_name=None
    )

    assert result.converges
    assert result.discipline in ["disciplined", "critical"]


# ============================================================================
# COMPREHENSIVE MULTISCALE ANALYSIS
# ============================================================================

def test_all_major_thresholds_converge():
    """Meta-test: all major thresholds should converge at finer resolutions."""
    thresholds_to_test = [
        ("edge_pressure", 0.7, "heur"),
        ("edge_pressure", 0.8, None),
        ("contradiction_signal", 0.45, "dia_adv"),
        ("contradiction_signal", 0.45, "fal_guard"),
        ("symbolic_regularity", 0.4, "eml_sr"),
        ("law_fit_signal", 0.4, "eml_sr"),
        ("uncertainty", 0.6, None),
        ("causal_risk", 0.5, None),
    ]

    results = []

    for feature_name, threshold, family_name in thresholds_to_test:
        result = measure_multiscale_boundary(
            feature_name=feature_name,
            threshold=threshold,
            family_name=family_name
        )

        results.append({
            "feature": feature_name,
            "threshold": threshold,
            "family": family_name,
            "converges": result.converges,
            "discipline": result.discipline,
            "roughness": result.roughness_exponent
        })

    # All should converge
    non_convergent = [r for r in results if not r["converges"]]
    assert len(non_convergent) == 0, \
        f"Non-convergent boundaries: {non_convergent}"

    # Most should be disciplined or critical
    pathological = [r for r in results if r["discipline"] == "pathological"]
    assert len(pathological) == 0, \
        f"Pathological boundaries detected: {pathological}"

    # Report all results
    print("\n" + "="*80)
    print("MULTISCALE BOUNDARY CONVERGENCE ANALYSIS")
    print("="*80)
    for r in results:
        print(f"{r['feature']}@{r['threshold']} ({r['family'] or 'budget'})")
        print(f"  Converges: {r['converges']}")
        print(f"  Discipline: {r['discipline']}")
        print(f"  Roughness: {r['roughness']:.3f}")
        print()


def test_boundary_roughness_distribution():
    """Test that boundary roughness is concentrated in acceptable range."""
    thresholds_to_test = [
        ("edge_pressure", 0.7, "heur"),
        ("contradiction_signal", 0.45, "dia_adv"),
        ("symbolic_regularity", 0.4, "eml_sr"),
    ]

    roughness_values = []

    for feature_name, threshold, family_name in thresholds_to_test:
        result = measure_multiscale_boundary(
            feature_name=feature_name,
            threshold=threshold,
            family_name=family_name
        )
        roughness_values.append(result.roughness_exponent)

    # Mean roughness should be low
    mean_roughness = sum(roughness_values) / len(roughness_values)
    assert mean_roughness < 0.4, \
        f"Mean boundary roughness too high: {mean_roughness:.3f}"

    # No boundary should have excessive roughness
    assert all(r < 0.7 for r in roughness_values), \
        f"Excessive roughness detected: {roughness_values}"


def test_hysteresis_scale_independence():
    """Test that hysteresis does not grow spuriously with finer resolution."""
    thresholds_to_test = [
        ("edge_pressure", 0.7, "heur"),
        ("contradiction_signal", 0.45, "dia_adv"),
    ]

    for feature_name, threshold, family_name in thresholds_to_test:
        result = measure_multiscale_boundary(
            feature_name=feature_name,
            threshold=threshold,
            family_name=family_name
        )

        # Hysteresis at finest resolution should not exceed coarsest by much
        finest_hysteresis = result.hysteresis_widths[-1]
        coarsest_hysteresis = result.hysteresis_widths[0]

        # Allow some variance but not growth
        assert finest_hysteresis <= coarsest_hysteresis * 2.0, \
            f"{feature_name}@{threshold}: Hysteresis grows with resolution: " \
            f"{coarsest_hysteresis:.4f} -> {finest_hysteresis:.4f}"


# ============================================================================
# RESOLUTION-SPECIFIC DIAGNOSTICS
# ============================================================================

def test_finest_resolution_stability():
    """Test behavior at finest resolution (0.005)."""
    result = measure_multiscale_boundary(
        feature_name="edge_pressure",
        threshold=0.7,
        family_name="heur",
        resolutions=[0.005]  # Only finest resolution
    )

    # At finest resolution, should have minimal discontinuities
    assert result.discontinuity_counts[0] <= 5, \
        f"Too many discontinuities at finest resolution: {result.discontinuity_counts[0]}"

    # Boundary variance should be low
    assert result.boundary_variances[0] < 0.001, \
        f"High boundary variance at finest resolution: {result.boundary_variances[0]}"


def test_coarsest_resolution_approximation():
    """Test that coarse resolution gives reasonable approximation."""
    # Measure at coarse resolution
    result_coarse = measure_multiscale_boundary(
        feature_name="contradiction_signal",
        threshold=0.45,
        family_name="dia_adv",
        resolutions=[0.10]
    )

    # Measure at fine resolution
    result_fine = measure_multiscale_boundary(
        feature_name="contradiction_signal",
        threshold=0.45,
        family_name="dia_adv",
        resolutions=[0.01]
    )

    # Coarse should be within reasonable error of fine
    error = abs(result_coarse.activation_points[0] - result_fine.activation_points[0])
    assert error < 0.05, \
        f"Coarse resolution error too large: {error:.4f}"


# ============================================================================
# PATHOLOGICAL CASE DETECTION
# ============================================================================

@pytest.mark.parametrize("feature,threshold,family", [
    ("edge_pressure", 0.7, "heur"),
    ("contradiction_signal", 0.45, "dia_adv"),
    ("symbolic_regularity", 0.4, "eml_sr"),
])
def test_no_pathological_boundaries(feature: str, threshold: float, family: str):
    """Parametrized test to ensure no boundaries are pathological."""
    result = measure_multiscale_boundary(
        feature_name=feature,
        threshold=threshold,
        family_name=family
    )

    assert result.discipline != "pathological", \
        f"{feature}@{threshold}/{family} has pathological boundary: " \
        f"converges={result.converges}, roughness={result.roughness_exponent:.3f}"


# ============================================================================
# CONVERGENCE RATE ANALYSIS
# ============================================================================

def test_convergence_rates_are_acceptable():
    """Test that convergence rates indicate stable refinement."""
    thresholds = [
        ("edge_pressure", 0.7, "heur"),
        ("contradiction_signal", 0.45, "dia_adv"),
        ("uncertainty", 0.6, None),
    ]

    for feature, threshold, family in thresholds:
        result = measure_multiscale_boundary(
            feature_name=feature,
            threshold=threshold,
            family_name=family
        )

        # Convergence rate should indicate refinement (< 1.0)
        assert result.convergence_rate < 1.0, \
            f"{feature}@{threshold}: Poor convergence rate {result.convergence_rate:.3f}"

        # Should converge reasonably fast
        assert result.convergence_rate < 0.7, \
            f"{feature}@{threshold}: Slow convergence {result.convergence_rate:.3f}"


# ============================================================================
# ARTIFACT GENERATION
# ============================================================================

@pytest.mark.slow
def test_generate_multiscale_report(tmp_path):
    """Generate comprehensive multiscale analysis report."""
    import json
    from datetime import datetime

    thresholds = [
        ("edge_pressure", 0.7, "heur"),
        ("edge_pressure", 0.8, None),
        ("contradiction_signal", 0.45, "dia_adv"),
        ("contradiction_signal", 0.45, "fal_guard"),
        ("symbolic_regularity", 0.4, "eml_sr"),
        ("law_fit_signal", 0.4, "eml_sr"),
        ("uncertainty", 0.6, None),
        ("causal_risk", 0.5, None),
    ]

    report = {
        "test_date": datetime.now().isoformat(),
        "test_type": "multiscale_boundary_refinement",
        "version": "1.0.0",
        "boundaries": []
    }

    for feature, threshold, family in thresholds:
        result = measure_multiscale_boundary(
            feature_name=feature,
            threshold=threshold,
            family_name=family
        )

        report["boundaries"].append({
            "feature": feature,
            "threshold": threshold,
            "family": family or "budget",
            "converges": result.converges,
            "convergence_rate": result.convergence_rate,
            "roughness_exponent": result.roughness_exponent,
            "discipline": result.discipline,
            "resolutions": result.resolutions,
            "activation_points": result.activation_points,
            "hysteresis_widths": result.hysteresis_widths,
            "discontinuity_counts": result.discontinuity_counts
        })

    # Save report
    output_file = tmp_path / "multiscale_boundary_report.json"
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)

    assert output_file.exists()

    # Generate summary
    summary_lines = [
        "="*80,
        "MULTISCALE BOUNDARY REFINEMENT REPORT",
        "="*80,
        f"Test Date: {report['test_date']}",
        f"Boundaries Tested: {len(report['boundaries'])}",
        "",
        "CONVERGENCE SUMMARY:",
        "-"*80
    ]

    convergent = sum(1 for b in report["boundaries"] if b["converges"])
    disciplined = sum(1 for b in report["boundaries"] if b["discipline"] == "disciplined")
    critical = sum(1 for b in report["boundaries"] if b["discipline"] == "critical")
    pathological = sum(1 for b in report["boundaries"] if b["discipline"] == "pathological")

    summary_lines.extend([
        f"  Convergent: {convergent}/{len(report['boundaries'])}",
        f"  Disciplined: {disciplined}",
        f"  Critical: {critical}",
        f"  Pathological: {pathological}",
        "",
        "BOUNDARY DETAILS:",
        "-"*80
    ])

    for b in report["boundaries"]:
        summary_lines.extend([
            f"{b['feature']}@{b['threshold']} ({b['family']})",
            f"  Converges: {b['converges']}",
            f"  Discipline: {b['discipline']}",
            f"  Roughness: {b['roughness_exponent']:.3f}",
            f"  Convergence Rate: {b['convergence_rate']:.3f}",
            ""
        ])

    summary_lines.append("="*80)

    summary_file = tmp_path / "multiscale_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("\n".join(summary_lines))

    assert summary_file.exists()

    print("\n" + "\n".join(summary_lines))
