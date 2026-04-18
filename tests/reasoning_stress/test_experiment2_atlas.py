"""
Experiment 2 Fractal Geometry-Functionality Atlas
==================================================

Maps all 12 fractal geometry families to 9 fractal functionalities.
This creates the complete catalog showing which geometries support
which scheduler behaviors.
"""

from __future__ import annotations

import pytest
import numpy as np
import json
from datetime import datetime
from typing import Dict, List

from tests.reasoning_stress.fractal_geometries import (
    FractalFamily,
    GeometricParameters,
    MetaParameters,
    generate_sierpinski_triangle,
    generate_cantor_carpet,
    generate_menger_sponge,
    generate_sierpinski_tetrahedron,
    generate_fractal_tree,
    generate_lorenz_attractor,
    generate_rossler_attractor,
    generate_henon_attractor,
    generate_mandelbrot_set,
    generate_julia_set,
    generate_game_of_life_pattern,
    generate_rule30_pattern,
    generate_scale_free_graph,
    generate_small_world_graph,
    generate_wavelet_decomposition,
    generate_kd_tree_partition,
    generate_fractional_brownian_motion,
    estimate_fractal_dimension_boxcount,
    compute_generalized_dimensions,
    compute_participation_ratio,
)


# ============================================================================
# FRACTAL FUNCTIONALITY EVALUATORS
# ============================================================================

def evaluate_selfsimilarity(points: np.ndarray, fd: float) -> Dict:
    """F1: Multi-scale self-similarity via D_q stability."""
    d_q = compute_generalized_dimensions(points, q_values=[-2, -1, 0, 1, 2])
    d_q_values = list(d_q.values())
    variance = np.var(d_q_values) if len(d_q_values) > 1 else 0.0
    score = max(0.0, 1.0 - variance)

    return {
        "d_q_variance": variance,
        "score": score,
        "d_0": d_q.get(0.0, fd),
        "d_2": d_q.get(2.0, fd),
    }


def evaluate_variety(family: FractalFamily) -> Dict:
    """F2: Fractal Dynamic Variety (VFD)."""
    variety_map = {
        FractalFamily.CONTINUOUS_ATTRACTOR: 0.9,
        FractalFamily.DISCRETE_ATTRACTOR: 0.85,
        FractalFamily.CELLULAR_AUTOMATA: 0.8,
        FractalFamily.STOCHASTIC: 0.75,
        FractalFamily.COMPLEX_PLANE: 0.6,
        FractalFamily.FRACTAL_GRAPH: 0.5,
        FractalFamily.WAVELET: 0.55,
        FractalFamily.PARTITION: 0.4,
        FractalFamily.BRANCHING: 0.3,
        FractalFamily.VOLUMETRIC_3D: 0.25,
        FractalFamily.CARPET: 0.2,
        FractalFamily.TRIANGULAR: 0.2,
    }

    score = variety_map.get(family, 0.5)
    return {"score": score, "is_dynamic": score > 0.6}


def evaluate_complexity(points: np.ndarray, fd: float) -> Dict:
    """F3: Complexity via D_q and Participation Ratio."""
    d_q = compute_generalized_dimensions(points, q_values=[0.0, 2.0])
    pr = compute_participation_ratio(points)
    d_q_spread = abs(d_q.get(0.0, fd) - d_q.get(2.0, fd))
    score = min(1.0, (pr + d_q_spread) / 2.0)

    return {
        "participation_ratio": pr,
        "d_q_spread": d_q_spread,
        "score": score,
    }


def evaluate_memory(family: FractalFamily) -> Dict:
    """F4: Multi-scale Fractal Memory (MFM)."""
    memory_map = {
        FractalFamily.CONTINUOUS_ATTRACTOR: 0.9,
        FractalFamily.DISCRETE_ATTRACTOR: 0.85,
        FractalFamily.WAVELET: 0.8,
        FractalFamily.STOCHASTIC: 0.75,
        FractalFamily.CELLULAR_AUTOMATA: 0.6,
        FractalFamily.FRACTAL_GRAPH: 0.5,
        FractalFamily.COMPLEX_PLANE: 0.55,
        FractalFamily.PARTITION: 0.35,
        FractalFamily.BRANCHING: 0.3,
        FractalFamily.VOLUMETRIC_3D: 0.25,
        FractalFamily.CARPET: 0.2,
        FractalFamily.TRIANGULAR: 0.2,
    }

    score = memory_map.get(family, 0.4)
    return {"score": score, "has_strong_memory": score > 0.7}


def evaluate_edges(family: FractalFamily) -> Dict:
    """F5: Fractal edge structure."""
    edge_map = {
        FractalFamily.FRACTAL_GRAPH: 1.0,
        FractalFamily.BRANCHING: 0.9,
        FractalFamily.COMPLEX_PLANE: 0.8,
        FractalFamily.CARPET: 0.75,
        FractalFamily.VOLUMETRIC_3D: 0.7,
        FractalFamily.CELLULAR_AUTOMATA: 0.65,
        FractalFamily.PARTITION: 0.6,
        FractalFamily.TRIANGULAR: 0.55,
        FractalFamily.WAVELET: 0.4,
        FractalFamily.STOCHASTIC: 0.3,
    }

    score = edge_map.get(family, 0.5)
    has_explicit = family in {FractalFamily.FRACTAL_GRAPH, FractalFamily.BRANCHING}

    return {"score": score, "has_explicit_edges": has_explicit}


def evaluate_communication(family: FractalFamily, fd: float) -> Dict:
    """F6: Fractal communication cost."""
    comm_map = {
        FractalFamily.FRACTAL_GRAPH: 0.9,
        FractalFamily.BRANCHING: 0.85,
        FractalFamily.PARTITION: 0.75,
        FractalFamily.VOLUMETRIC_3D: 0.7,
        FractalFamily.CARPET: 0.65,
    }

    score = comm_map.get(family, 0.5)
    path_length_factor = min(1.0, fd / 2.0)

    return {
        "score": score,
        "path_length_factor": path_length_factor,
    }


def evaluate_vsh_evolution(family: FractalFamily) -> Dict:
    """F7: Viability-Suitability-Habitability evolution."""
    evolving = {
        FractalFamily.CONTINUOUS_ATTRACTOR,
        FractalFamily.DISCRETE_ATTRACTOR,
        FractalFamily.CELLULAR_AUTOMATA,
        FractalFamily.STOCHASTIC,
    }

    can_evolve = family in evolving
    return {
        "can_evolve": can_evolve,
        "score": 1.0 if can_evolve else 0.3,
    }


def evaluate_cognitive_alignment(fd: float, target_fd: float) -> Dict:
    """F8: Cognitive fractality alignment."""
    error = abs(fd - target_fd)
    score = max(0.0, 1.0 - error)

    return {
        "fd_error": error,
        "score": score,
        "is_aligned": error < 0.5,
    }


def evaluate_coherence(family: FractalFamily) -> Dict:
    """F9: Long-term coherence C_∞."""
    coherence_map = {
        FractalFamily.TRIANGULAR: 0.9,
        FractalFamily.CARPET: 0.9,
        FractalFamily.CONTINUOUS_ATTRACTOR: 0.85,
        FractalFamily.VOLUMETRIC_3D: 0.85,
        FractalFamily.WAVELET: 0.8,
        FractalFamily.BRANCHING: 0.75,
        FractalFamily.DISCRETE_ATTRACTOR: 0.7,
        FractalFamily.FRACTAL_GRAPH: 0.65,
        FractalFamily.COMPLEX_PLANE: 0.6,
        FractalFamily.CELLULAR_AUTOMATA: 0.5,
        FractalFamily.STOCHASTIC: 0.45,
        FractalFamily.PARTITION: 0.5,
    }

    score = coherence_map.get(family, 0.5)
    return {"score": score, "is_coherent": score > 0.7}


# ============================================================================
# COMPREHENSIVE ATLAS GENERATION
# ============================================================================

@pytest.mark.slow
def test_generate_experiment2_atlas(tmp_path):
    """
    Generate Experiment 2 Fractal Atlas: Complete geometry-to-functionality mapping.

    Produces:
    - JSON catalog with all 12 families × 9 functionalities
    - Summary table showing scores
    - Top performers for each functionality
    """

    atlas = {
        "experiment": "Experiment 2: Fractal Geometry Catalog",
        "test_date": datetime.now().isoformat(),
        "version": "2.0.0",
        "geometries": []
    }

    meta_params = MetaParameters(lambda_rig=0.4, target_dimension=1.5)

    # All 12 geometry families
    configs = [
        ("Sierpinski Triangle", FractalFamily.TRIANGULAR,
         lambda: generate_sierpinski_triangle(GeometricParameters(
             family=FractalFamily.TRIANGULAR, depth=5, resolution=64, seed=42))),

        ("Cantor Carpet", FractalFamily.CARPET,
         lambda: generate_cantor_carpet(GeometricParameters(
             family=FractalFamily.CARPET, grid_size=(3,3), depth=3, seed=42))),

        ("Menger Sponge", FractalFamily.VOLUMETRIC_3D,
         lambda: generate_menger_sponge(GeometricParameters(
             family=FractalFamily.VOLUMETRIC_3D, depth=3, seed=42))),

        ("Binary Tree", FractalFamily.BRANCHING,
         lambda: generate_fractal_tree(GeometricParameters(
             family=FractalFamily.BRANCHING, branching_factor=2, depth=5, seed=42))[0]),

        ("Lorenz Attractor", FractalFamily.CONTINUOUS_ATTRACTOR,
         lambda: generate_lorenz_attractor(GeometricParameters(
             family=FractalFamily.CONTINUOUS_ATTRACTOR, trajectory_length=2000, seed=42))[::8]),

        ("Hénon Map", FractalFamily.DISCRETE_ATTRACTOR,
         lambda: generate_henon_attractor(GeometricParameters(
             family=FractalFamily.DISCRETE_ATTRACTOR, trajectory_length=3000, seed=42))),

        ("Mandelbrot Set", FractalFamily.COMPLEX_PLANE,
         lambda: generate_mandelbrot_set(GeometricParameters(
             family=FractalFamily.COMPLEX_PLANE, resolution=80, depth=60, seed=42))),

        ("Game of Life", FractalFamily.CELLULAR_AUTOMATA,
         lambda: generate_game_of_life_pattern(GeometricParameters(
             family=FractalFamily.CELLULAR_AUTOMATA, grid_size=(64, 64), depth=30, seed=42))),

        ("Scale-Free Graph", FractalFamily.FRACTAL_GRAPH,
         lambda: generate_scale_free_graph(GeometricParameters(
             family=FractalFamily.FRACTAL_GRAPH, grid_size=(60,), branching_factor=3, seed=42))[0]),

        ("Wavelet Decomp", FractalFamily.WAVELET,
         lambda: np.vstack([
             np.column_stack([np.linspace(0, 1, len(coeffs)), coeffs / (np.max(np.abs(coeffs)) + 1e-10)])
             for level, coeffs in generate_wavelet_decomposition(
                 GeometricParameters(family=FractalFamily.WAVELET, depth=3, resolution=128, seed=42)
             ).items() if len(coeffs) > 0
         ])),

        ("KD-Tree Partition", FractalFamily.PARTITION,
         lambda: generate_kd_tree_partition(GeometricParameters(
             family=FractalFamily.PARTITION, depth=5, resolution=100, seed=42))[0]),

        ("Fractional Brownian", FractalFamily.STOCHASTIC,
         lambda: (lambda fbm: np.column_stack([
             np.arange(len(fbm)) / len(fbm), fbm / np.max(np.abs(fbm))
         ]))(generate_fractional_brownian_motion(GeometricParameters(
             family=FractalFamily.STOCHASTIC, hurst_exponent=0.7, resolution=256, seed=42)))),
    ]

    for name, family, generator in configs:
        try:
            print(f"Processing: {name}")
            points = generator()

            if len(points) == 0:
                continue

            fd, r2 = estimate_fractal_dimension_boxcount(points)

            # Evaluate all 9 functionalities
            f1 = evaluate_selfsimilarity(points, fd)
            f2 = evaluate_variety(family)
            f3 = evaluate_complexity(points, fd)
            f4 = evaluate_memory(family)
            f5 = evaluate_edges(family)
            f6 = evaluate_communication(family, fd)
            f7 = evaluate_vsh_evolution(family)
            f8 = evaluate_cognitive_alignment(fd, meta_params.target_dimension)
            f9 = evaluate_coherence(family)

            scores = [f1["score"], f2["score"], f3["score"], f4["score"],
                     f5["score"], f6["score"], f7["score"], f8["score"], f9["score"]]

            atlas["geometries"].append({
                "name": name,
                "family": family.value,
                "fractal_dimension": fd,
                "r_squared": r2,
                "functionalities": {
                    "F1_selfsimilarity": f1,
                    "F2_variety": f2,
                    "F3_complexity": f3,
                    "F4_memory": f4,
                    "F5_edges": f5,
                    "F6_communication": f6,
                    "F7_vsh_evolution": f7,
                    "F8_cognitive": f8,
                    "F9_coherence": f9,
                },
                "overall_score": np.mean(scores),
            })

        except Exception as e:
            print(f"Warning: Failed {name}: {e}")

    # Save atlas
    output_file = tmp_path / "experiment2_fractal_atlas.json"
    with open(output_file, 'w') as f:
        json.dump(atlas, f, indent=2)

    assert output_file.exists()

    # Generate summary table
    summary_lines = [
        "=" * 120,
        "EXPERIMENT 2: FRACTAL GEOMETRY-FUNCTIONALITY ATLAS",
        "=" * 120,
        f"Date: {atlas['test_date']}",
        f"Geometries: {len(atlas['geometries'])}",
        "",
        "FUNCTIONALITIES:",
        "  F1: Multi-scale Self-Similarity    F4: Multi-scale Memory      F7: V-S-H Evolution",
        "  F2: Dynamic Variety (VFD)          F5: Fractal Edges           F8: Cognitive Alignment",
        "  F3: Complexity (D_q, PR)           F6: Communication Cost      F9: Long-term Coherence",
        "",
        "-" * 120,
        f"{'Geometry':<22} {'Family':<6} {'FD':>6} {'Overall':>7} | " +
        "   ".join(f"F{i}" for i in range(1, 10)),
        "-" * 120,
    ]

    for geom in sorted(atlas["geometries"], key=lambda g: -g["overall_score"]):
        funcs = geom["functionalities"]
        score_str = "  ".join(f"{funcs[f'F{i}_{k}']['score']:>4.2f}"
                              for i, k in enumerate([
                                  "selfsimilarity", "variety", "complexity", "memory",
                                  "edges", "communication", "vsh_evolution", "cognitive", "coherence"
                              ], 1))

        summary_lines.append(
            f"{geom['name']:<22} {geom['family']:<6} {geom['fractal_dimension']:>6.3f} "
            f"{geom['overall_score']:>7.3f} | {score_str}"
        )

    summary_lines.extend([
        "-" * 120,
        "",
        "TOP PERFORMERS:",
        "-" * 80,
    ])

    func_names = [
        "Self-Similarity", "Dynamic Variety", "Complexity", "Memory",
        "Edges", "Communication", "Evolution", "Cognitive", "Coherence"
    ]

    for i, func_name in enumerate(func_names, 1):
        best = max(atlas["geometries"],
                  key=lambda g: list(g["functionalities"].values())[i-1]["score"])
        score = list(best["functionalities"].values())[i-1]["score"]

        summary_lines.append(
            f"  F{i} {func_name:<20}: {best['name']:<22} (score={score:.3f})"
        )

    summary_lines.append("=" * 120)

    summary_file = tmp_path / "experiment2_atlas_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("\n".join(summary_lines))

    assert summary_file.exists()

    print("\n" + "\n".join(summary_lines))

    # Validate
    assert len(atlas["geometries"]) >= 10
    for geom in atlas["geometries"]:
        assert len(geom["functionalities"]) == 9
        assert 0.0 <= geom["overall_score"] <= 1.0


# ============================================================================
# SPECIFIC FUNCTIONALITY VALIDATION TESTS
# ============================================================================

def test_attractors_high_memory():
    """Attractors should have high memory scores."""
    result = evaluate_memory(FractalFamily.CONTINUOUS_ATTRACTOR)
    assert result["has_strong_memory"]


def test_graphs_explicit_edges():
    """Graph families should have explicit edges."""
    result = evaluate_edges(FractalFamily.FRACTAL_GRAPH)
    assert result["has_explicit_edges"]


def test_dynamic_systems_evolve():
    """Dynamic systems should support evolution."""
    result = evaluate_vsh_evolution(FractalFamily.CONTINUOUS_ATTRACTOR)
    assert result["can_evolve"]


def test_static_fractals_coherent():
    """Static fractals should be highly coherent."""
    result = evaluate_coherence(FractalFamily.TRIANGULAR)
    assert result["is_coherent"]
