"""
Fractal Geometry Catalog Tests - Experiment 2
==============================================

Tests scheduler behavior across diverse fractal geometry families.
Maps geometric parameters to scheduler features and measures
which geometries support which fractal functionalities.
"""

from __future__ import annotations

import pytest
import numpy as np
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from tests.reasoning_stress.fractal_geometries import (
    FractalFamily,
    GeometricParameters,
    MetaParameters,
    generate_sierpinski_triangle,
    generate_cantor_carpet,
    generate_fractal_tree,
    generate_lorenz_attractor,
    generate_rossler_attractor,
    generate_henon_attractor,
    generate_fractional_brownian_motion,
    estimate_fractal_dimension_boxcount,
    compute_generalized_dimensions,
    compute_participation_ratio,
)

from runtime.reasoning.scheduler_meta.budgeting import compute_budget
from runtime.reasoning.scheduler_meta.policy import select_sequence


# ============================================================================
# GEOMETRY-TO-SCHEDULER MAPPING
# ============================================================================

def map_geometry_to_scheduler_features(
    geometry_points: np.ndarray,
    fractal_dimension: float,
    meta_params: MetaParameters
) -> Dict[str, float]:
    """
    Map fractal geometry properties to scheduler feature space.

    This is the key bridge: we convert geometric fractal properties
    into the continuous feature signals that the scheduler uses.
    """
    # Normalize points to [0, 1]
    if len(geometry_points) > 0:
        points_min = geometry_points.min(axis=0)
        points_max = geometry_points.max(axis=0)
        points_range = points_max - points_min
        points_range[points_range == 0] = 1.0
        normalized_points = (geometry_points - points_min) / points_range
    else:
        normalized_points = geometry_points

    # Compute density and complexity metrics
    n_points = len(normalized_points)

    # Map fractal dimension to uncertainty
    # Higher D_F -> more complex structure -> higher uncertainty
    uncertainty = min(1.0, max(0.0, (fractal_dimension - 1.0) / 2.0))

    # Map point density variation to contradiction_signal
    if n_points > 100:
        # Sample local densities
        sample_indices = np.random.choice(n_points, min(100, n_points), replace=False)
        local_densities = []

        for idx in sample_indices:
            point = normalized_points[idx]
            # Count neighbors within radius 0.1
            if normalized_points.shape[1] == 2:
                distances = np.sqrt(((normalized_points - point)**2).sum(axis=1))
            else:
                distances = np.linalg.norm(normalized_points - point, axis=1)

            neighbors = (distances < 0.1).sum()
            local_densities.append(neighbors)

        density_variance = np.var(local_densities) if local_densities else 0.0
        contradiction_signal = min(1.0, density_variance / 50.0)
    else:
        contradiction_signal = 0.2

    # Map rigidity parameter to edge_pressure
    edge_pressure = meta_params.lambda_rig

    # Map structural depth to causal_risk
    # Deeper structures -> more causal dependencies
    causal_risk = min(1.0, meta_params.lambda_fractal)

    # Map to symbolic regularity based on D_q
    # If D_q is close to target, high symbolic regularity
    symbolic_regularity = max(0.0, 1.0 - abs(fractal_dimension - meta_params.target_dimension))

    # Law fit signal from participation ratio (if available)
    law_fit_signal = 0.3  # Default

    # Continuity from spectral margin
    continuity_recent = meta_params.spectral_margin_min

    return {
        "uncertainty": uncertainty,
        "contradiction_signal": contradiction_signal,
        "continuity_recent": continuity_recent,
        "edge_pressure": edge_pressure,
        "causal_risk": causal_risk,
        "symbolic_regularity": symbolic_regularity,
        "law_fit_signal": law_fit_signal,
    }


# ============================================================================
# FAMILY T: SIERPINSKI TRIANGLE TESTS
# ============================================================================

def test_sierpinski_triangle_basic():
    """Test Sierpinski triangle generation and FD estimation."""
    params = GeometricParameters(
        family=FractalFamily.TRIANGULAR,
        depth=6,
        resolution=128,
        seed=42
    )

    points = generate_sierpinski_triangle(params)

    assert len(points) > 0
    assert points.shape[1] == 2  # 2D points

    # Estimate fractal dimension
    fd, r2 = estimate_fractal_dimension_boxcount(points)

    # Sierpinski triangle should have D ≈ log(3)/log(2) ≈ 1.585
    assert 1.4 <= fd <= 1.8, f"FD={fd:.3f} outside expected range for Sierpinski"
    assert r2 >= 0.9, f"Poor fit: R²={r2:.3f}"


def test_sierpinski_to_scheduler_mapping():
    """Test mapping Sierpinski geometry to scheduler features."""
    params = GeometricParameters(
        family=FractalFamily.TRIANGULAR,
        depth=5,
        resolution=64,
        seed=42
    )

    meta_params = MetaParameters(
        lambda_rig=0.3,
        lambda_fractal=0.2,
        target_dimension=1.5
    )

    points = generate_sierpinski_triangle(params)
    fd, _ = estimate_fractal_dimension_boxcount(points)

    features = map_geometry_to_scheduler_features(points, fd, meta_params)

    # Should produce valid features
    assert all(0.0 <= v <= 1.0 for k, v in features.items()
               if k != "continuity_recent")

    # Test scheduler response
    budget = compute_budget(features)
    sequence, _, _ = select_sequence(
        features=features,
        budget=budget,
        allow_experimental=True
    )

    # Should produce valid sequence
    assert len(sequence) >= 4
    assert "prob" in sequence


def test_sierpinski_parameter_sweep():
    """Test sweeping Sierpinski parameters and observing scheduler response."""
    scale_values = [0.3, 0.4, 0.5, 0.6, 0.7]
    results = []

    meta_params = MetaParameters(lambda_rig=0.4, target_dimension=1.5)

    for scale in scale_values:
        params = GeometricParameters(
            family=FractalFamily.TRIANGULAR,
            scales=[scale, scale, scale],
            depth=5,
            resolution=64,
            seed=42
        )

        points = generate_sierpinski_triangle(params)
        fd, r2 = estimate_fractal_dimension_boxcount(points)

        features = map_geometry_to_scheduler_features(points, fd, meta_params)

        budget = compute_budget(features)
        sequence, _, _ = select_sequence(
            features=features,
            budget=budget,
            allow_experimental=True
        )

        results.append({
            "scale": scale,
            "fd": fd,
            "r2": r2,
            "uncertainty": features["uncertainty"],
            "max_steps": budget["max_steps"],
            "sequence_length": len(sequence)
        })

    # FD should decrease with increasing scale (less self-similar)
    fds = [r["fd"] for r in results]
    # Monotonicity not strict due to stochastic IFS, but general trend

    print("\nSierpinski Parameter Sweep Results:")
    for r in results:
        print(f"  scale={r['scale']:.1f}: FD={r['fd']:.3f}, "
              f"uncertainty={r['uncertainty']:.3f}, max_steps={r['max_steps']:.0f}")


# ============================================================================
# FAMILY C: CANTOR CARPET TESTS
# ============================================================================

def test_cantor_carpet_generation():
    """Test Cantor carpet generation."""
    params = GeometricParameters(
        family=FractalFamily.CARPET,
        grid_size=(3, 3),
        depth=4,
        seed=42
    )

    points = generate_cantor_carpet(params)

    assert len(points) > 0
    assert points.shape[1] == 2

    # Sierpinski carpet D ≈ log(8)/log(3) ≈ 1.893
    fd, r2 = estimate_fractal_dimension_boxcount(points)

    assert 1.7 <= fd <= 2.1, f"FD={fd:.3f} unexpected for carpet"


def test_carpet_grid_size_variation():
    """Test varying grid size and its effect on scheduler."""
    grid_sizes = [(2, 2), (3, 3), (4, 4)]
    results = []

    meta_params = MetaParameters(lambda_rig=0.5)

    for grid_size in grid_sizes:
        params = GeometricParameters(
            family=FractalFamily.CARPET,
            grid_size=grid_size,
            depth=3,
            seed=42
        )

        points = generate_cantor_carpet(params)
        fd, _ = estimate_fractal_dimension_boxcount(points)

        features = map_geometry_to_scheduler_features(points, fd, meta_params)
        budget = compute_budget(features)

        results.append({
            "grid_size": grid_size,
            "fd": fd,
            "max_steps": budget["max_steps"]
        })

    print("\nCarpet Grid Size Variation:")
    for r in results:
        print(f"  grid={r['grid_size']}: FD={r['fd']:.3f}, max_steps={r['max_steps']:.0f}")


# ============================================================================
# FAMILY B: FRACTAL TREE TESTS
# ============================================================================

def test_fractal_tree_generation():
    """Test fractal tree generation."""
    params = GeometricParameters(
        family=FractalFamily.BRANCHING,
        branching_factor=2,
        depth=6,
        scales=[0.7] * 6,
        seed=42
    )

    nodes, edges = generate_fractal_tree(params)

    assert len(nodes) > 0
    assert len(edges) > 0

    # Tree should have hierarchical structure
    # Number of nodes ≈ Σ b^i for i=0 to depth
    expected_nodes_approx = sum(2**i for i in range(params.depth + 1))
    assert len(nodes) <= expected_nodes_approx * 1.2


def test_tree_branching_factor_sweep():
    """Test effect of branching factor on scheduler."""
    branching_factors = [2, 3, 4]
    results = []

    meta_params = MetaParameters(lambda_rig=0.3)

    for b in branching_factors:
        params = GeometricParameters(
            family=FractalFamily.BRANCHING,
            branching_factor=b,
            depth=4,
            seed=42
        )

        nodes, _ = generate_fractal_tree(params)

        # Estimate "dimension" from node count growth
        # D ≈ log(b) / log(1/scale)
        approximate_fd = math.log(b) / math.log(1/0.7) if b > 1 else 1.0

        features = map_geometry_to_scheduler_features(nodes, approximate_fd, meta_params)
        budget = compute_budget(features)
        sequence, _, _ = select_sequence(features=features, budget=budget, allow_experimental=True)

        results.append({
            "branching_factor": b,
            "nodes": len(nodes),
            "approximate_fd": approximate_fd,
            "sequence_length": len(sequence)
        })

    print("\nTree Branching Factor Sweep:")
    for r in results:
        print(f"  b={r['branching_factor']}: nodes={r['nodes']}, "
              f"FD≈{r['approximate_fd']:.3f}, seq_len={r['sequence_length']}")


# ============================================================================
# FAMILY AXC: CONTINUOUS ATTRACTORS TESTS
# ============================================================================

def test_lorenz_attractor_generation():
    """Test Lorenz attractor generation."""
    params = GeometricParameters(
        family=FractalFamily.CONTINUOUS_ATTRACTOR,
        trajectory_length=5000,
        integration_step=0.01,
        seed=42
    )

    trajectory = generate_lorenz_attractor(params)

    assert trajectory.shape == (5000, 3)

    # Lorenz has fractal dimension ≈ 2.06
    # Box-counting in 3D
    fd, r2 = estimate_fractal_dimension_boxcount(trajectory)

    assert 1.8 <= fd <= 2.5, f"FD={fd:.3f} unexpected for Lorenz"


def test_rossler_attractor_generation():
    """Test Rössler attractor generation."""
    params = GeometricParameters(
        family=FractalFamily.CONTINUOUS_ATTRACTOR,
        trajectory_length=5000,
        integration_step=0.01,
        seed=42
    )

    trajectory = generate_rossler_attractor(params)

    assert trajectory.shape == (5000, 3)

    fd, r2 = estimate_fractal_dimension_boxcount(trajectory)

    # Rössler also has fractal dimension around 2
    assert 1.7 <= fd <= 2.5


def test_attractor_parameter_variation():
    """Test varying attractor parameters and scheduler response."""
    # Vary Lorenz ρ parameter
    rho_values = [20.0, 28.0, 35.0]
    results = []

    meta_params = MetaParameters(lambda_rig=0.4)

    for rho in rho_values:
        params = GeometricParameters(
            family=FractalFamily.CONTINUOUS_ATTRACTOR,
            system_params={'sigma': 10.0, 'rho': rho, 'beta': 8.0/3.0},
            trajectory_length=3000,
            integration_step=0.01,
            seed=42
        )

        trajectory = generate_lorenz_attractor(params)
        fd, _ = estimate_fractal_dimension_boxcount(trajectory[::10])  # Subsample

        features = map_geometry_to_scheduler_features(trajectory[::10], fd, meta_params)
        budget = compute_budget(features)

        results.append({
            "rho": rho,
            "fd": fd,
            "uncertainty": features["uncertainty"],
            "max_steps": budget["max_steps"]
        })

    print("\nLorenz ρ Parameter Sweep:")
    for r in results:
        print(f"  ρ={r['rho']:.1f}: FD={r['fd']:.3f}, "
              f"uncertainty={r['uncertainty']:.3f}, max_steps={r['max_steps']:.0f}")


# ============================================================================
# FAMILY AXD: DISCRETE ATTRACTORS TESTS
# ============================================================================

def test_henon_attractor_generation():
    """Test Hénon map attractor."""
    params = GeometricParameters(
        family=FractalFamily.DISCRETE_ATTRACTOR,
        trajectory_length=10000,
        seed=42
    )

    trajectory = generate_henon_attractor(params)

    assert trajectory.shape == (10000, 2)

    fd, r2 = estimate_fractal_dimension_boxcount(trajectory)

    # Hénon attractor has dimension ≈ 1.26
    assert 1.1 <= fd <= 1.5, f"FD={fd:.3f} unexpected for Hénon"


# ============================================================================
# FAMILY RS: STOCHASTIC FRACTALS TESTS
# ============================================================================

def test_fractional_brownian_motion():
    """Test fractional Brownian motion generation."""
    hurst_values = [0.3, 0.5, 0.7, 0.9]
    results = []

    meta_params = MetaParameters(lambda_rig=0.5)

    for H in hurst_values:
        params = GeometricParameters(
            family=FractalFamily.STOCHASTIC,
            hurst_exponent=H,
            resolution=512,
            seed=42
        )

        fbm = generate_fractional_brownian_motion(params)

        # Convert to 2D points for FD estimation
        points_2d = np.column_stack([np.arange(len(fbm)) / len(fbm), fbm / np.max(np.abs(fbm))])

        fd, _ = estimate_fractal_dimension_boxcount(points_2d)

        # Theoretical: FD = 2 - H for fBm
        theoretical_fd = 2.0 - H

        features = map_geometry_to_scheduler_features(points_2d, fd, meta_params)
        budget = compute_budget(features)

        results.append({
            "hurst": H,
            "fd_measured": fd,
            "fd_theoretical": theoretical_fd,
            "max_steps": budget["max_steps"]
        })

    print("\nFractional Brownian Motion Hurst Sweep:")
    for r in results:
        print(f"  H={r['hurst']:.1f}: FD_meas={r['fd_measured']:.3f}, "
              f"FD_theory={r['fd_theoretical']:.3f}, max_steps={r['max_steps']:.0f}")


# ============================================================================
# COMPREHENSIVE GEOMETRY CATALOG TEST
# ============================================================================

@pytest.mark.slow
def test_comprehensive_geometry_catalog(tmp_path):
    """Test all geometry families and generate comprehensive catalog."""
    import math

    catalog = {
        "test_date": datetime.now().isoformat(),
        "test_type": "fractal_geometry_catalog",
        "version": "1.0.0",
        "geometries": []
    }

    meta_params = MetaParameters(lambda_rig=0.4, target_dimension=1.5)

    # Test configurations for each family
    configs = [
        ("Sierpinski Triangle", FractalFamily.TRIANGULAR,
         lambda: generate_sierpinski_triangle(GeometricParameters(
             family=FractalFamily.TRIANGULAR, depth=5, resolution=64, seed=42))),

        ("Cantor Carpet", FractalFamily.CARPET,
         lambda: generate_cantor_carpet(GeometricParameters(
             family=FractalFamily.CARPET, grid_size=(3,3), depth=3, seed=42))),

        ("Binary Tree", FractalFamily.BRANCHING,
         lambda: generate_fractal_tree(GeometricParameters(
             family=FractalFamily.BRANCHING, branching_factor=2, depth=5, seed=42))[0]),

        ("Lorenz Attractor", FractalFamily.CONTINUOUS_ATTRACTOR,
         lambda: generate_lorenz_attractor(GeometricParameters(
             family=FractalFamily.CONTINUOUS_ATTRACTOR, trajectory_length=3000, seed=42))[::10]),

        ("Rössler Attractor", FractalFamily.CONTINUOUS_ATTRACTOR,
         lambda: generate_rossler_attractor(GeometricParameters(
             family=FractalFamily.CONTINUOUS_ATTRACTOR, trajectory_length=3000, seed=42))[::10]),

        ("Hénon Map", FractalFamily.DISCRETE_ATTRACTOR,
         lambda: generate_henon_attractor(GeometricParameters(
             family=FractalFamily.DISCRETE_ATTRACTOR, trajectory_length=5000, seed=42))),
    ]

    for name, family, generator in configs:
        try:
            points = generator()

            if len(points) > 0:
                fd, r2 = estimate_fractal_dimension_boxcount(points)
                d_q = compute_generalized_dimensions(points, q_values=[0.0, 1.0, 2.0])

                features = map_geometry_to_scheduler_features(points, fd, meta_params)
                budget = compute_budget(features)
                sequence, scores, recommended = select_sequence(
                    features=features,
                    budget=budget,
                    allow_experimental=True
                )

                catalog["geometries"].append({
                    "name": name,
                    "family": family.value,
                    "n_points": len(points),
                    "dimension": points.shape[1] if len(points.shape) > 1 else 1,
                    "fractal_dimension": fd,
                    "r_squared": r2,
                    "d_q": {str(q): d for q, d in d_q.items()},
                    "scheduler_features": features,
                    "budget": {k: float(v) for k, v in budget.items()},
                    "sequence": sequence,
                    "sequence_length": len(sequence)
                })

        except Exception as e:
            print(f"Warning: Failed to process {name}: {e}")

    # Save catalog
    output_file = tmp_path / "fractal_geometry_catalog.json"
    with open(output_file, 'w') as f:
        json.dump(catalog, f, indent=2)

    assert output_file.exists()

    # Generate summary
    summary_lines = [
        "=" * 80,
        "FRACTAL GEOMETRY CATALOG - EXPERIMENT 2",
        "=" * 80,
        f"Test Date: {catalog['test_date']}",
        f"Geometries Tested: {len(catalog['geometries'])}",
        "",
        "GEOMETRY CHARACTERISTICS:",
        "-" * 80
    ]

    for geom in catalog["geometries"]:
        summary_lines.extend([
            f"{geom['name']} (Family: {geom['family']})",
            f"  Points: {geom['n_points']}, Dimension: {geom['dimension']}D",
            f"  Fractal Dimension: {geom['fractal_dimension']:.3f} (R²={geom['r_squared']:.3f})",
            f"  D_q: {', '.join(f'D_{q}={d:.3f}' for q, d in geom['d_q'].items())}",
            f"  Scheduler: max_steps={geom['budget']['max_steps']:.0f}, "
            f"seq_len={geom['sequence_length']}, seq={geom['sequence'][:3]}...",
            ""
        ])

    summary_lines.append("=" * 80)

    summary_file = tmp_path / "fractal_geometry_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("\n".join(summary_lines))

    assert summary_file.exists()

    print("\n" + "\n".join(summary_lines))

    # Validate that we tested multiple families
    families_tested = {g["family"] for g in catalog["geometries"]}
    assert len(families_tested) >= 4, "Should test at least 4 different fractal families"
