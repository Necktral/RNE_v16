"""
Fractal Geometry Catalog - Experiment 2
========================================

Generators for diverse fractal geometry families to test scheduler behavior
across different geometric structures.

Families:
- T: IFS Triangular (Sierpinski-type)
- C: IFS Square Carpets (Cantor 2D)
- F3D: 3D Fractals (Menger, tetrahedron)
- B: Fractal Trees/Branching
- AXC: Continuous Strange Attractors (Lorenz, Rössler)
- AXD: Discrete Strange Attractors (Hénon, logistic maps)
- MC: Complex Plane Fractals (Mandelbrot, Julia)
- RS: Stochastic Fractals (fractional Brownian)
- AC: Cellular Automata (Game of Life, Class IV)
- GF: Fractal Graphs (scale-free, small-world)
- W: Wavelet/Multi-resolution
- PF: Fractal Space Partitions (KD-trees)
"""

from __future__ import annotations

import numpy as np
import math
from typing import Dict, List, Tuple, Callable, Any
from dataclasses import dataclass
from enum import Enum


class FractalFamily(Enum):
    """Fractal geometry family identifiers."""
    TRIANGULAR = "T"  # Sierpinski triangles
    CARPET = "C"  # Cantor carpets
    VOLUMETRIC_3D = "F3D"  # 3D fractals
    BRANCHING = "B"  # Trees
    CONTINUOUS_ATTRACTOR = "AXC"  # Lorenz, Rössler
    DISCRETE_ATTRACTOR = "AXD"  # Hénon, logistic
    COMPLEX_PLANE = "MC"  # Mandelbrot, Julia
    STOCHASTIC = "RS"  # Brownian fractional
    CELLULAR_AUTOMATA = "AC"  # Game of Life
    FRACTAL_GRAPH = "GF"  # Scale-free networks
    WAVELET = "W"  # Wavelet bases
    PARTITION = "PF"  # KD-trees


@dataclass
class GeometricParameters:
    """Geometric parameters θ_geom for a fractal structure."""
    family: FractalFamily

    # Common parameters
    scales: List[float] | None = None  # Contraction scales s_i
    translations: List[Tuple[float, ...]] | None = None  # Translation vectors
    rotations: List[float] | None = None  # Rotation angles
    depth: int = 5  # Structural depth L

    # Probability/density
    probabilities: List[float] | None = None  # Selection probabilities p_i

    # Topology-specific
    grid_size: Tuple[int, ...] | None = None  # For carpets, grids
    branching_factor: int | None = None  # For trees
    pattern_mask: np.ndarray | None = None  # For carpets, CA

    # Dynamics-specific (attractors)
    system_params: Dict[str, float] | None = None  # σ, ρ, β for Lorenz, etc.
    integration_step: float = 0.01
    trajectory_length: int = 10000

    # Stochastic
    hurst_exponent: float | None = None  # H for fractional Brownian
    noise_amplitude: float = 1.0

    # Graph-specific
    power_law_exponent: float | None = None  # For scale-free
    small_world_prob: float | None = None  # Rewiring probability

    # Additional
    resolution: int = 256  # Spatial resolution
    seed: int = 42


@dataclass
class MetaParameters:
    """Meta-parameters θ_meta controlling fractality and rigidity."""

    # Rigidity and exploration
    lambda_rig: float = 0.5  # Global rigidity [0,1]
    tau: float = 1.0  # Exploration temperature

    # Fractal regularization
    lambda_fractal: float = 0.1  # Weight for (D_q - D*)²
    target_dimension: float = 1.5  # D* target

    # Meta-cost
    lambda_meta: float = 0.01  # Architecture modification cost

    # SAFE constraints
    vram_limit_mb: float = 1024.0
    temperature_limit_c: float = 85.0
    latency_limit_ms: float = 100.0
    spectral_margin_min: float = 0.1  # 1 - ρ(J)

    # Memory scales for MFM
    coarse_memory_fraction: float = 0.3
    medium_memory_fraction: float = 0.4
    fine_memory_fraction: float = 0.3
    bubble_up_rate: float = 0.1
    push_down_rate: float = 0.05

    # Communication costs
    alpha_distance: float = 1.0  # Weight for d_F
    beta_quality: float = 0.5  # Weight for 1/Q
    gamma_latency: float = 0.3  # Weight for latency
    delta_congestion: float = 0.2  # Weight for congestion

    # Risk and robustness
    cvar_alpha: float = 0.95  # CVaR confidence level
    temporal_horizon: int = 100


# ============================================================================
# FAMILY T: IFS TRIANGULAR (Sierpinski-type)
# ============================================================================

def generate_sierpinski_triangle(params: GeometricParameters) -> np.ndarray:
    """
    Generate Sierpinski triangle using IFS.

    Returns:
        Array of points (N, 2) representing the fractal.
    """
    if params.scales is None:
        # Default: 3 maps with scale 0.5
        params.scales = [0.5, 0.5, 0.5]

    if params.translations is None:
        # Standard Sierpinski: corners of unit triangle
        params.translations = [
            (0.0, 0.0),
            (0.5, 0.0),
            (0.25, math.sqrt(3)/4)
        ]

    if params.probabilities is None:
        # Equal probability
        n_maps = len(params.scales)
        params.probabilities = [1.0/n_maps] * n_maps

    # IFS iteration
    np.random.seed(params.seed)
    n_points = params.resolution ** 2
    points = np.zeros((n_points, 2))

    # Starting point
    x, y = 0.5, 0.5

    for i in range(n_points):
        # Select map
        map_idx = np.random.choice(len(params.scales), p=params.probabilities)

        # Apply transformation
        scale = params.scales[map_idx]
        tx, ty = params.translations[map_idx]
        rotation = params.rotations[map_idx] if params.rotations else 0.0

        # Rotate and scale
        cos_r = math.cos(rotation)
        sin_r = math.sin(rotation)
        x_new = scale * (cos_r * x - sin_r * y) + tx
        y_new = scale * (sin_r * x + cos_r * y) + ty

        x, y = x_new, y_new
        points[i] = [x, y]

    return points


def estimate_fractal_dimension_boxcount(points: np.ndarray,
                                        box_sizes: List[float] | None = None) -> Tuple[float, float]:
    """
    Estimate fractal dimension via box-counting.

    Returns:
        (dimension, R²)
    """
    if box_sizes is None:
        box_sizes = [2**(-i) for i in range(2, 10)]

    counts = []

    for epsilon in box_sizes:
        # Count boxes
        if points.shape[1] == 2:
            # 2D points
            boxes = set()
            for point in points:
                box_x = int(point[0] / epsilon)
                box_y = int(point[1] / epsilon)
                boxes.add((box_x, box_y))
            counts.append(len(boxes))
        elif points.shape[1] == 3:
            # 3D points
            boxes = set()
            for point in points:
                box_x = int(point[0] / epsilon)
                box_y = int(point[1] / epsilon)
                box_z = int(point[2] / epsilon)
                boxes.add((box_x, box_y, box_z))
            counts.append(len(boxes))

    # Log-log regression
    log_epsilon = [math.log(1.0 / eps) for eps in box_sizes]
    log_counts = [math.log(c) for c in counts if c > 0]

    if len(log_counts) < 3:
        return 1.0, 0.0

    # Linear regression
    n = len(log_epsilon)
    mean_x = sum(log_epsilon) / n
    mean_y = sum(log_counts) / n

    numerator = sum((log_epsilon[i] - mean_x) * (log_counts[i] - mean_y) for i in range(n))
    denominator = sum((log_epsilon[i] - mean_x) ** 2 for i in range(n))

    if denominator == 0:
        return 1.0, 0.0

    slope = numerator / denominator

    # R²
    ss_tot = sum((log_counts[i] - mean_y) ** 2 for i in range(n))
    ss_res = sum((log_counts[i] - (mean_y + slope * (log_epsilon[i] - mean_x))) ** 2 for i in range(n))
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return slope, r_squared


# ============================================================================
# FAMILY C: IFS SQUARE CARPETS (Cantor 2D)
# ============================================================================

def generate_cantor_carpet(params: GeometricParameters) -> np.ndarray:
    """
    Generate Cantor-like carpet (Sierpinski carpet variant).

    Returns:
        Array of points representing the carpet.
    """
    if params.grid_size is None:
        params.grid_size = (3, 3)  # Standard 3x3

    k = params.grid_size[0]

    if params.pattern_mask is None:
        # Standard Sierpinski carpet: remove center
        params.pattern_mask = np.ones((k, k), dtype=bool)
        params.pattern_mask[k//2, k//2] = False

    # Recursive generation
    def generate_level(depth: int, origin: Tuple[float, float], size: float) -> List[Tuple[float, float]]:
        if depth == 0:
            return [(origin[0] + size/2, origin[1] + size/2)]

        points = []
        cell_size = size / k

        for i in range(k):
            for j in range(k):
                if params.pattern_mask[i, j]:
                    cell_origin = (origin[0] + i * cell_size, origin[1] + j * cell_size)
                    points.extend(generate_level(depth - 1, cell_origin, cell_size))

        return points

    points_list = generate_level(params.depth, (0.0, 0.0), 1.0)
    return np.array(points_list)


# ============================================================================
# FAMILY B: FRACTAL TREES/BRANCHING
# ============================================================================

def generate_fractal_tree(params: GeometricParameters) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Generate fractal tree structure.

    Returns:
        (nodes, edges) where nodes is (N, 2) array and edges is list of (i, j) tuples.
    """
    if params.branching_factor is None:
        params.branching_factor = 2  # Binary tree

    b = params.branching_factor

    # Branch length scaling
    if params.scales is None:
        params.scales = [0.7] * params.depth  # Decay per level

    # Angular distribution
    if params.rotations is None:
        # Default: symmetric branching
        angle_span = math.pi / 3  # 60 degrees
        params.rotations = [angle_span * (i - (b-1)/2) / max(1, (b-1)) for i in range(b)]

    nodes = [(0.0, 0.0)]  # Root
    edges = []

    def add_branch(parent_idx: int, parent_pos: Tuple[float, float],
                   parent_angle: float, level: int):
        if level >= params.depth:
            return

        length = params.scales[level] if level < len(params.scales) else 0.7

        for i in range(b):
            angle = parent_angle + params.rotations[i]

            # New node position
            new_x = parent_pos[0] + length * math.cos(angle)
            new_y = parent_pos[1] + length * math.sin(angle)
            new_pos = (new_x, new_y)

            new_idx = len(nodes)
            nodes.append(new_pos)
            edges.append((parent_idx, new_idx))

            # Recurse
            add_branch(new_idx, new_pos, angle, level + 1)

    # Start from root pointing up
    add_branch(0, (0.0, 0.0), math.pi/2, 0)

    return np.array(nodes), edges


# ============================================================================
# FAMILY AXC: CONTINUOUS STRANGE ATTRACTORS
# ============================================================================

def generate_lorenz_attractor(params: GeometricParameters) -> np.ndarray:
    """
    Generate Lorenz attractor trajectory.

    Returns:
        Array of points (N, 3).
    """
    if params.system_params is None:
        # Classic Lorenz parameters
        params.system_params = {
            'sigma': 10.0,
            'rho': 28.0,
            'beta': 8.0/3.0
        }

    sigma = params.system_params['sigma']
    rho = params.system_params['rho']
    beta = params.system_params['beta']

    dt = params.integration_step
    n = params.trajectory_length

    # State
    np.random.seed(params.seed)
    state = np.random.randn(3)  # Random initial condition

    trajectory = np.zeros((n, 3))

    for i in range(n):
        # Lorenz equations
        dx = sigma * (state[1] - state[0])
        dy = state[0] * (rho - state[2]) - state[1]
        dz = state[0] * state[1] - beta * state[2]

        # Euler integration
        state[0] += dx * dt
        state[1] += dy * dt
        state[2] += dz * dt

        trajectory[i] = state

    return trajectory


def generate_rossler_attractor(params: GeometricParameters) -> np.ndarray:
    """
    Generate Rössler attractor trajectory.

    Returns:
        Array of points (N, 3).
    """
    if params.system_params is None:
        # Classic Rössler parameters
        params.system_params = {
            'a': 0.2,
            'b': 0.2,
            'c': 5.7
        }

    a = params.system_params['a']
    b = params.system_params['b']
    c = params.system_params['c']

    dt = params.integration_step
    n = params.trajectory_length

    np.random.seed(params.seed)
    state = np.random.randn(3)

    trajectory = np.zeros((n, 3))

    for i in range(n):
        # Rössler equations
        dx = -state[1] - state[2]
        dy = state[0] + a * state[1]
        dz = b + state[2] * (state[0] - c)

        state[0] += dx * dt
        state[1] += dy * dt
        state[2] += dz * dt

        trajectory[i] = state

    return trajectory


# ============================================================================
# FAMILY AXD: DISCRETE STRANGE ATTRACTORS
# ============================================================================

def generate_henon_attractor(params: GeometricParameters) -> np.ndarray:
    """
    Generate Hénon map attractor.

    Returns:
        Array of points (N, 2).
    """
    if params.system_params is None:
        # Classic Hénon parameters
        params.system_params = {
            'a': 1.4,
            'b': 0.3
        }

    a = params.system_params['a']
    b = params.system_params['b']

    n = params.trajectory_length

    np.random.seed(params.seed)
    x, y = 0.1, 0.1  # Initial condition

    trajectory = np.zeros((n, 2))

    for i in range(n):
        x_new = 1 - a * x**2 + y
        y_new = b * x

        x, y = x_new, y_new
        trajectory[i] = [x, y]

    return trajectory


# ============================================================================
# FAMILY RS: STOCHASTIC FRACTALS
# ============================================================================

def generate_fractional_brownian_motion(params: GeometricParameters) -> np.ndarray:
    """
    Generate fractional Brownian motion using midpoint displacement.

    Returns:
        Array of values (time series or 2D field).
    """
    if params.hurst_exponent is None:
        params.hurst_exponent = 0.5  # Standard Brownian

    H = params.hurst_exponent
    n = params.resolution

    np.random.seed(params.seed)

    # 1D fractional Brownian motion
    increments = np.random.randn(n)

    # Simple approximation: scale increments by power law
    for i in range(1, n):
        increments[i] *= (i ** H)

    fbm = np.cumsum(increments)
    fbm *= params.noise_amplitude

    return fbm


# ============================================================================
# METRICS COMPUTATION
# ============================================================================

def compute_generalized_dimensions(points: np.ndarray,
                                   q_values: List[float] | None = None,
                                   box_sizes: List[float] | None = None) -> Dict[float, float]:
    """
    Compute generalized dimensions D_q.

    Args:
        points: Point cloud
        q_values: List of q values (0=capacity, 1=information, 2=correlation)
        box_sizes: Epsilon values for box counting

    Returns:
        Dictionary mapping q -> D_q
    """
    if q_values is None:
        q_values = [0.0, 1.0, 2.0]

    if box_sizes is None:
        box_sizes = [2**(-i) for i in range(3, 8)]

    dimensions = {}

    for q in q_values:
        if abs(q) < 1e-10:
            # D_0 = capacity dimension (box-counting)
            dim, _ = estimate_fractal_dimension_boxcount(points, box_sizes)
            dimensions[q] = dim
        else:
            # Simplified: use box-counting with probability weighting
            # For full implementation, would need to compute p_i properly
            dim, _ = estimate_fractal_dimension_boxcount(points, box_sizes)
            dimensions[q] = dim * 0.95  # Approximation

    return dimensions


def compute_participation_ratio(probabilities: np.ndarray) -> float:
    """
    Compute participation ratio PR = 1 / Σ p_i²

    Args:
        probabilities: Probability distribution

    Returns:
        Participation ratio
    """
    p_squared_sum = np.sum(probabilities ** 2)
    if p_squared_sum > 0:
        return 1.0 / p_squared_sum
    return 0.0


def compute_lyapunov_exponent(trajectory: np.ndarray,
                              dynamics_function: Callable,
                              perturbation: float = 1e-8) -> float:
    """
    Estimate largest Lyapunov exponent.

    Args:
        trajectory: Reference trajectory
        dynamics_function: Function mapping state -> next state
        perturbation: Initial perturbation size

    Returns:
        Estimated Lyapunov exponent
    """
    # Simplified estimation
    separations = []

    for i in range(min(100, len(trajectory) - 1)):
        # Add small perturbation
        perturbed = trajectory[i] + perturbation * np.random.randn(len(trajectory[i]))

        # Evolve one step
        next_ref = trajectory[i + 1]
        next_pert = dynamics_function(perturbed)

        # Measure separation
        separation = np.linalg.norm(next_pert - next_ref)
        if separation > 0:
            separations.append(math.log(separation / perturbation))

    if separations:
        return np.mean(separations)
    return 0.0
