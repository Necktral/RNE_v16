"""
Fractal Atlas Integration - Elite Level
========================================

Integrates fractal analysis metrics into comprehensive reasoning atlas.
Provides unified view of:
- Multiscale boundary behavior
- Fractal dimension measurements
- Temporal cascade dynamics
- Avalanche statistics
- Overall system discipline vs criticality vs fragility
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass, asdict
from datetime import datetime
import pytest

from tests.reasoning_stress.fractal_utils import (
    measure_multiscale_boundary,
    estimate_box_counting_dimension,
    measure_temporal_cascades,
    measure_activation_avalanches,
    MultiscaleBoundaryMetrics,
    BoxCountingResult,
    TemporalCascadeMetrics,
    AvalancheDistribution
)


@dataclass
class FractalAtlasMetrics:
    """Fractal characterization of the reasoning scheduler."""
    version: str = "1.0.0"
    test_date: str = ""

    # Multiscale boundary analysis
    multiscale_boundaries: List[Dict] = None

    # Box-counting fractal dimensions
    fractal_dimensions: List[Dict] = None

    # Temporal cascade dynamics
    temporal_cascades: List[Dict] = None

    # Avalanche statistics
    avalanche_distribution: Dict | None = None

    # Overall classification
    system_discipline: str = ""  # "disciplined", "critical", "fragile", "pathological"
    scale_invariance_quality: str = ""  # "excellent", "good", "poor"
    boundary_roughness_class: str = ""  # "smooth", "moderately_rough", "highly_rough"
    temporal_stability_class: str = ""  # "stable", "metastable", "unstable"
    criticality_regime: str = ""  # "subcritical", "critical", "supercritical"

    # Aggregate scores
    mean_convergence_rate: float = 0.0
    mean_roughness_exponent: float = 0.0
    mean_fractal_dimension: float = 0.0
    mean_scale_invariance_error: float = 0.0
    avalanche_criticality_score: float = 0.0

    def __post_init__(self):
        if self.multiscale_boundaries is None:
            self.multiscale_boundaries = []
        if self.fractal_dimensions is None:
            self.fractal_dimensions = []
        if self.temporal_cascades is None:
            self.temporal_cascades = []

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "version": self.version,
            "test_date": self.test_date,
            "multiscale_boundaries": self.multiscale_boundaries,
            "fractal_dimensions": self.fractal_dimensions,
            "temporal_cascades": self.temporal_cascades,
            "avalanche_distribution": self.avalanche_distribution,
            "system_discipline": self.system_discipline,
            "scale_invariance_quality": self.scale_invariance_quality,
            "boundary_roughness_class": self.boundary_roughness_class,
            "temporal_stability_class": self.temporal_stability_class,
            "criticality_regime": self.criticality_regime,
            "mean_convergence_rate": self.mean_convergence_rate,
            "mean_roughness_exponent": self.mean_roughness_exponent,
            "mean_fractal_dimension": self.mean_fractal_dimension,
            "mean_scale_invariance_error": self.mean_scale_invariance_error,
            "avalanche_criticality_score": self.avalanche_criticality_score
        }

    def save(self, filepath: str | Path):
        """Save fractal atlas to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    def get_summary_report(self) -> str:
        """Generate human-readable summary report."""
        lines = [
            "=" * 80,
            "FRACTAL REASONING SCHEDULER ATLAS",
            "=" * 80,
            f"Version: {self.version}",
            f"Test Date: {self.test_date}",
            "",
            "OVERALL CLASSIFICATION",
            "-" * 80,
            f"  System Discipline: {self.system_discipline.upper()}",
            f"  Scale Invariance Quality: {self.scale_invariance_quality}",
            f"  Boundary Roughness: {self.boundary_roughness_class}",
            f"  Temporal Stability: {self.temporal_stability_class}",
            f"  Criticality Regime: {self.criticality_regime}",
            "",
            "AGGREGATE METRICS",
            "-" * 80,
            f"  Mean Convergence Rate: {self.mean_convergence_rate:.3f}",
            f"  Mean Roughness Exponent: {self.mean_roughness_exponent:.3f}",
            f"  Mean Fractal Dimension: {self.mean_fractal_dimension:.3f}",
            f"  Mean Scale Invariance Error: {self.mean_scale_invariance_error:.3f}",
            f"  Avalanche Criticality Score: {self.avalanche_criticality_score:.3f}",
            "",
            "MULTISCALE BOUNDARY CONVERGENCE",
            "-" * 80,
            f"  Boundaries Tested: {len(self.multiscale_boundaries)}",
        ]

        convergent = sum(1 for b in self.multiscale_boundaries if b.get("converges", False))
        disciplined = sum(1 for b in self.multiscale_boundaries if b.get("discipline") == "disciplined")
        critical = sum(1 for b in self.multiscale_boundaries if b.get("discipline") == "critical")
        pathological = sum(1 for b in self.multiscale_boundaries if b.get("discipline") == "pathological")

        lines.extend([
            f"  Convergent: {convergent}/{len(self.multiscale_boundaries)}",
            f"  Disciplined: {disciplined}",
            f"  Critical: {critical}",
            f"  Pathological: {pathological}",
            "",
            "FRACTAL DIMENSION ANALYSIS",
            "-" * 80,
            f"  Frontiers Analyzed: {len(self.fractal_dimensions)}",
        ])

        clean = sum(1 for f in self.fractal_dimensions if f.get("interpretation") == "clean")
        rugose = sum(1 for f in self.fractal_dimensions if f.get("interpretation") == "rugose_controlled")
        pathological_fd = sum(1 for f in self.fractal_dimensions if f.get("interpretation") == "pathological")

        lines.extend([
            f"  Clean Frontiers: {clean}",
            f"  Rugose (Controlled): {rugose}",
            f"  Pathological: {pathological_fd}",
            "",
            "TEMPORAL CASCADE DYNAMICS",
            "-" * 80,
            f"  Cascade Patterns Tested: {len(self.temporal_cascades)}",
        ])

        self_similar = sum(1 for c in self.temporal_cascades if c.get("self_similar", False))

        lines.extend([
            f"  Self-Similar: {self_similar}/{len(self.temporal_cascades)}",
            "",
            "AVALANCHE STATISTICS",
            "-" * 80,
        ])

        if self.avalanche_distribution:
            lines.extend([
                f"  Mean Avalanche Size: {self.avalanche_distribution.get('mean_size', 0):.3f}",
                f"  Max Avalanche Size: {self.avalanche_distribution.get('max_size', 0)}",
                f"  Heavy-Tailed: {self.avalanche_distribution.get('is_heavy_tailed', False)}",
                f"  Criticality: {self.avalanche_distribution.get('criticality_indicator', 'unknown')}",
            ])

        lines.extend([
            "",
            "=" * 80,
            "",
            "INTERPRETATION:",
            "-" * 80,
        ])

        # Add interpretation based on classification
        if self.system_discipline == "disciplined":
            lines.append("  ✓ System exhibits DISCIPLINED behavior:")
            lines.append("    - Boundaries converge cleanly at finer resolutions")
            lines.append("    - Minimal fractal complexity")
            lines.append("    - Stable temporal dynamics")
            lines.append("    - Controlled sensitivity to perturbations")
        elif self.system_discipline == "critical":
            lines.append("  ⚠ System exhibits CRITICAL behavior:")
            lines.append("    - Rich multiscale structure")
            lines.append("    - Some boundary roughness but controlled")
            lines.append("    - Interesting temporal dynamics")
            lines.append("    - Balanced sensitivity")
        elif self.system_discipline == "fragile":
            lines.append("  ⚠ System exhibits FRAGILE behavior:")
            lines.append("    - Excessive sensitivity to small perturbations")
            lines.append("    - Large avalanches common")
            lines.append("    - May indicate need for recalibration")
        elif self.system_discipline == "pathological":
            lines.append("  ✗ System exhibits PATHOLOGICAL behavior:")
            lines.append("    - Non-convergent boundaries")
            lines.append("    - Excessive fractal complexity")
            lines.append("    - Unstable temporal dynamics")
            lines.append("    - REQUIRES immediate attention")

        lines.append("=" * 80)

        return "\n".join(lines)


def build_fractal_atlas() -> FractalAtlasMetrics:
    """
    Build comprehensive fractal atlas of reasoning scheduler.

    Returns:
        FractalAtlasMetrics with complete fractal characterization
    """
    atlas = FractalAtlasMetrics(
        version="1.0.0",
        test_date=datetime.now().isoformat()
    )

    # 1. Multiscale boundary analysis
    atlas.multiscale_boundaries = _analyze_multiscale_boundaries()

    # 2. Fractal dimension analysis
    atlas.fractal_dimensions = _analyze_fractal_dimensions()

    # 3. Temporal cascade analysis
    atlas.temporal_cascades = _analyze_temporal_cascades()

    # 4. Avalanche statistics
    atlas.avalanche_distribution = _analyze_avalanche_statistics()

    # 5. Compute aggregate metrics
    _compute_aggregate_metrics(atlas)

    # 6. Classify system
    _classify_system(atlas)

    return atlas


def _analyze_multiscale_boundaries() -> List[Dict]:
    """Analyze multiscale boundary behavior for all major thresholds."""
    boundaries_to_test = [
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

    for feature_name, threshold, family_name in boundaries_to_test:
        result = measure_multiscale_boundary(
            feature_name=feature_name,
            threshold=threshold,
            family_name=family_name
        )

        results.append({
            "feature": feature_name,
            "threshold": threshold,
            "family": family_name or "budget",
            "converges": result.converges,
            "convergence_rate": result.convergence_rate,
            "roughness_exponent": result.roughness_exponent,
            "discipline": result.discipline,
            "mean_hysteresis": sum(result.hysteresis_widths) / len(result.hysteresis_widths)
        })

    return results


def _analyze_fractal_dimensions() -> List[Dict]:
    """Analyze fractal dimensions of activation frontiers."""
    frontiers_to_test = [
        ("contradiction_signal", "edge_pressure", "heur"),
        ("uncertainty", "contradiction_signal", "dia_adv"),
        ("contradiction_signal", "causal_risk", "fal_guard"),
        ("symbolic_regularity", "law_fit_signal", "eml_sr"),
        ("edge_pressure", "causal_risk", "heur"),
        ("uncertainty", "edge_pressure", "heur"),
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
            "feature_x": feature_x,
            "feature_y": feature_y,
            "family": family,
            "fractal_dimension": result.fractal_dimension,
            "fit_quality": result.fit_quality,
            "interpretation": result.interpretation
        })

    return results


def _analyze_temporal_cascades() -> List[Dict]:
    """Analyze temporal cascade behavior."""
    configurations = [
        ("uncertainty", "increasing"),
        ("uncertainty", "decreasing"),
        ("contradiction_signal", "increasing"),
        ("contradiction_signal", "decreasing"),
        ("edge_pressure", "increasing"),
        ("causal_risk", "increasing"),
    ]

    results = []

    for feature, pattern in configurations:
        result = measure_temporal_cascades(
            feature_name=feature,
            perturbation_pattern=pattern
        )

        results.append({
            "feature": feature,
            "pattern": pattern,
            "self_similar": result.self_similar,
            "scale_invariance_error": result.scale_invariance_error,
            "temporal_memory_fragility": result.temporal_memory_fragility
        })

    return results


def _analyze_avalanche_statistics() -> Dict:
    """Analyze activation avalanche distribution."""
    result = measure_activation_avalanches(n_trials=1000)

    return {
        "mean_size": result.mean_size,
        "max_size": result.max_size,
        "tail_exponent": result.tail_exponent,
        "is_heavy_tailed": result.is_heavy_tailed,
        "criticality_indicator": result.criticalit_indicator,
        "size_histogram": result.size_histogram
    }


def _compute_aggregate_metrics(atlas: FractalAtlasMetrics):
    """Compute aggregate metrics from detailed analyses."""
    # Mean convergence rate
    if atlas.multiscale_boundaries:
        atlas.mean_convergence_rate = sum(
            b["convergence_rate"] for b in atlas.multiscale_boundaries
        ) / len(atlas.multiscale_boundaries)

        atlas.mean_roughness_exponent = sum(
            b["roughness_exponent"] for b in atlas.multiscale_boundaries
        ) / len(atlas.multiscale_boundaries)

    # Mean fractal dimension
    valid_dimensions = [
        f["fractal_dimension"] for f in atlas.fractal_dimensions
        if f["interpretation"] != "insufficient_data"
    ]
    if valid_dimensions:
        atlas.mean_fractal_dimension = sum(valid_dimensions) / len(valid_dimensions)

    # Mean scale invariance error
    if atlas.temporal_cascades:
        atlas.mean_scale_invariance_error = sum(
            c["scale_invariance_error"] for c in atlas.temporal_cascades
        ) / len(atlas.temporal_cascades)

    # Avalanche criticality score
    if atlas.avalanche_distribution:
        mean_size = atlas.avalanche_distribution["mean_size"]
        max_size = atlas.avalanche_distribution["max_size"]

        # Score: 0 = rigid, 0.5 = interesting, 1.0 = fragile
        if mean_size < 0.3:
            atlas.avalanche_criticality_score = 0.0  # Rigid
        elif mean_size > 1.5:
            atlas.avalanche_criticality_score = 1.0  # Fragile
        else:
            atlas.avalanche_criticality_score = 0.5  # Interesting


def _classify_system(atlas: FractalAtlasMetrics):
    """Classify overall system discipline and criticality."""
    # System discipline
    pathological_boundaries = sum(
        1 for b in atlas.multiscale_boundaries
        if b["discipline"] == "pathological"
    )

    disciplined_boundaries = sum(
        1 for b in atlas.multiscale_boundaries
        if b["discipline"] == "disciplined"
    )

    critical_boundaries = sum(
        1 for b in atlas.multiscale_boundaries
        if b["discipline"] == "critical"
    )

    if pathological_boundaries > 0:
        atlas.system_discipline = "pathological"
    elif disciplined_boundaries >= len(atlas.multiscale_boundaries) * 0.7:
        atlas.system_discipline = "disciplined"
    elif critical_boundaries >= len(atlas.multiscale_boundaries) * 0.5:
        atlas.system_discipline = "critical"
    elif atlas.avalanche_distribution and atlas.avalanche_distribution["criticality_indicator"] == "fragile":
        atlas.system_discipline = "fragile"
    else:
        atlas.system_discipline = "critical"

    # Scale invariance quality
    if atlas.mean_scale_invariance_error < 0.3:
        atlas.scale_invariance_quality = "excellent"
    elif atlas.mean_scale_invariance_error < 0.5:
        atlas.scale_invariance_quality = "good"
    else:
        atlas.scale_invariance_quality = "poor"

    # Boundary roughness class
    if atlas.mean_roughness_exponent < 0.3:
        atlas.boundary_roughness_class = "smooth"
    elif atlas.mean_roughness_exponent < 0.6:
        atlas.boundary_roughness_class = "moderately_rough"
    else:
        atlas.boundary_roughness_class = "highly_rough"

    # Temporal stability class
    mean_fragility = sum(
        c["temporal_memory_fragility"] for c in atlas.temporal_cascades
    ) / len(atlas.temporal_cascades) if atlas.temporal_cascades else 0.0

    if mean_fragility < 0.3:
        atlas.temporal_stability_class = "stable"
    elif mean_fragility < 0.5:
        atlas.temporal_stability_class = "metastable"
    else:
        atlas.temporal_stability_class = "unstable"

    # Criticality regime
    if atlas.avalanche_distribution:
        if atlas.avalanche_distribution["criticality_indicator"] == "rigid":
            atlas.criticality_regime = "subcritical"
        elif atlas.avalanche_distribution["criticality_indicator"] == "interesting":
            atlas.criticality_regime = "critical"
        else:
            atlas.criticality_regime = "supercritical"
    else:
        atlas.criticality_regime = "unknown"


# ============================================================================
# TEST CASES
# ============================================================================

def test_build_fractal_atlas():
    """Test that fractal atlas can be built."""
    atlas = build_fractal_atlas()

    # Should have all components
    assert len(atlas.multiscale_boundaries) > 0
    assert len(atlas.fractal_dimensions) > 0
    assert len(atlas.temporal_cascades) > 0
    assert atlas.avalanche_distribution is not None

    # Should have classifications
    assert atlas.system_discipline in ["disciplined", "critical", "fragile", "pathological"]
    assert atlas.scale_invariance_quality in ["excellent", "good", "poor"]


def test_fractal_atlas_serialization():
    """Test that fractal atlas can be serialized."""
    atlas = build_fractal_atlas()

    atlas_dict = atlas.to_dict()

    assert "version" in atlas_dict
    assert "system_discipline" in atlas_dict
    assert "multiscale_boundaries" in atlas_dict


def test_fractal_atlas_summary_report():
    """Test that fractal atlas generates readable summary."""
    atlas = build_fractal_atlas()

    summary = atlas.get_summary_report()

    assert "FRACTAL REASONING SCHEDULER ATLAS" in summary
    assert "OVERALL CLASSIFICATION" in summary
    assert "MULTISCALE BOUNDARY CONVERGENCE" in summary
    assert "INTERPRETATION" in summary


@pytest.mark.slow
def test_save_fractal_atlas(tmp_path):
    """Test saving fractal atlas to file."""
    atlas = build_fractal_atlas()

    output_file = tmp_path / "fractal_atlas.json"
    atlas.save(output_file)

    assert output_file.exists()

    # Should be valid JSON
    with open(output_file) as f:
        data = json.load(f)

    assert data["version"] == "1.0.0"

    # Save summary report
    summary_file = tmp_path / "fractal_atlas_summary.txt"
    with open(summary_file, 'w') as f:
        f.write(atlas.get_summary_report())

    assert summary_file.exists()

    print(atlas.get_summary_report())


def test_fractal_atlas_quality_thresholds():
    """Test that fractal atlas meets quality thresholds."""
    atlas = build_fractal_atlas()

    # System should not be pathological
    assert atlas.system_discipline != "pathological", \
        f"System is pathological: {atlas.get_summary_report()}"

    # Scale invariance should be at least good
    assert atlas.scale_invariance_quality in ["excellent", "good"], \
        f"Poor scale invariance: {atlas.scale_invariance_quality}"

    # Boundaries should not be highly rough
    assert atlas.boundary_roughness_class != "highly_rough", \
        f"Boundaries too rough: {atlas.boundary_roughness_class}"

    # Temporal stability should not be unstable
    assert atlas.temporal_stability_class != "unstable", \
        f"Temporal dynamics unstable: {atlas.temporal_stability_class}"
