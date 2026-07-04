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

    # Normalizar al cubo unitario [0,1]^d: el box-counting con box_sizes ∈ (0,1]
    # solo es válido sobre coordenadas normalizadas. Sin esto, atractores con rango
    # grande (p.ej. Lorenz, coords ±20) ponen cada punto en su propia caja para todo
    # epsilon → conteo saturado constante → pendiente≈0 → FD≈0 (bug histórico).
    pts = np.asarray(points, dtype=float)
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    span = np.where(maxs > mins, maxs - mins, 1.0)
    points = (pts - mins) / span

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

    # Log-log regression sobre la REGIÓN DE ESCALA. La pendiente sólo estima la
    # dimensión donde el conteo crece linealmente en log-log; las cajas saturadas
    # (conteo ≈ nº de puntos) y las triviales (≤4) aplanan la pendiente y subestiman
    # la dimensión. Se emparejan (log(1/eps), log(count)) y se filtra a la región
    # válida (antes se filtraba sólo counts>0 sin alinear log_epsilon → desajuste).
    n_points = len(points)
    pairs = [
        (math.log(1.0 / eps), math.log(c))
        for eps, c in zip(box_sizes, counts)
        if 4 < c < 0.9 * n_points
    ]
    if len(pairs) < 3:
        # Fallback: cualquier caja con conteo positivo.
        pairs = [
            (math.log(1.0 / eps), math.log(c))
            for eps, c in zip(box_sizes, counts)
            if c > 0
        ]
    if len(pairs) < 3:
        return 1.0, 0.0

    log_epsilon = [p[0] for p in pairs]
    log_counts = [p[1] for p in pairs]

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


# ============================================================================
# FAMILY F3D: 3D FRACTALS (Menger, Sierpinski tetrahedron)
# ============================================================================

def generate_menger_sponge(params: GeometricParameters) -> np.ndarray:
    """
    Generate Menger sponge fractal in 3D.

    Returns:
        Array of points (N, 3) representing occupied voxels.
    """
    if params.depth is None:
        params.depth = 3

    def subdivide_cube(origin: Tuple[float, float, float], size: float, level: int) -> List[Tuple[float, float, float]]:
        """Recursively subdivide cube, removing center cross pattern."""
        if level == 0:
            return [origin]

        points = []
        step = size / 3

        # Menger sponge: remove center of each face and center of cube
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    # Skip center cross pattern
                    center_count = (1 if i == 1 else 0) + (1 if j == 1 else 0) + (1 if k == 1 else 0)
                    if center_count >= 2:  # Remove if on center cross
                        continue

                    sub_origin = (
                        origin[0] + i * step,
                        origin[1] + j * step,
                        origin[2] + k * step
                    )
                    points.extend(subdivide_cube(sub_origin, step, level - 1))

        return points

    voxels = subdivide_cube((0.0, 0.0, 0.0), 1.0, params.depth)
    return np.array(voxels)


def generate_sierpinski_tetrahedron(params: GeometricParameters) -> np.ndarray:
    """
    Generate Sierpinski tetrahedron (3D Sierpinski triangle).

    Returns:
        Array of points (N, 3).
    """
    if params.scales is None:
        params.scales = [0.5] * 4  # 4 vertices

    if params.translations is None:
        # Regular tetrahedron vertices
        sqrt2 = math.sqrt(2)
        sqrt6 = math.sqrt(6)
        params.translations = [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.5, sqrt6/3, 0.0),
            (0.5, sqrt6/6, sqrt2/sqrt6)
        ]

    if params.probabilities is None:
        params.probabilities = [0.25] * 4

    # IFS iteration in 3D
    np.random.seed(params.seed)
    n_points = params.resolution ** 2

    points = np.zeros((n_points, 3))
    x, y, z = 0.5, 0.5, 0.5

    for i in range(n_points):
        map_idx = np.random.choice(len(params.scales), p=params.probabilities)
        scale = params.scales[map_idx]
        tx, ty, tz = params.translations[map_idx]

        x = scale * x + tx
        y = scale * y + ty
        z = scale * z + tz

        points[i] = [x, y, z]

    return points


# ============================================================================
# FAMILY MC: COMPLEX PLANE FRACTALS (Mandelbrot, Julia)
# ============================================================================

def generate_mandelbrot_set(params: GeometricParameters) -> np.ndarray:
    """
    Generate Mandelbrot set boundary points.

    Returns:
        Array of points (N, 2) on/near the boundary.
    """
    if params.resolution is None:
        params.resolution = 256

    max_iter = 100
    escape_radius = 2.0

    # Complex plane region
    x_min, x_max = -2.5, 1.0
    y_min, y_max = -1.25, 1.25

    res = params.resolution
    boundary_points = []

    for i in range(res):
        for j in range(res):
            cx = x_min + (x_max - x_min) * i / res
            cy = y_min + (y_max - y_min) * j / res
            c = complex(cx, cy)

            z = 0
            iteration = 0

            while abs(z) <= escape_radius and iteration < max_iter:
                z = z*z + c
                iteration += 1

            # Points near boundary (escape at intermediate iterations)
            if 10 < iteration < max_iter - 10:
                boundary_points.append([cx, cy])

    return np.array(boundary_points) if boundary_points else np.array([[0, 0]])


def generate_julia_set(params: GeometricParameters) -> np.ndarray:
    """
    Generate Julia set for a given complex parameter c.

    Returns:
        Array of points (N, 2) on/near the boundary.
    """
    if params.system_params is None:
        # Classic Julia set parameter
        params.system_params = {'c_real': -0.7, 'c_imag': 0.27015}

    c = complex(params.system_params['c_real'], params.system_params['c_imag'])

    max_iter = 100
    escape_radius = 2.0
    res = params.resolution if params.resolution else 256

    x_min, x_max = -1.5, 1.5
    y_min, y_max = -1.5, 1.5

    boundary_points = []

    for i in range(res):
        for j in range(res):
            zx = x_min + (x_max - x_min) * i / res
            zy = y_min + (y_max - y_min) * j / res
            z = complex(zx, zy)

            iteration = 0
            while abs(z) <= escape_radius and iteration < max_iter:
                z = z*z + c
                iteration += 1

            if 10 < iteration < max_iter - 10:
                boundary_points.append([zx, zy])

    return np.array(boundary_points) if boundary_points else np.array([[0, 0]])


# ============================================================================
# FAMILY AC: CELLULAR AUTOMATA
# ============================================================================

def generate_game_of_life_pattern(params: GeometricParameters) -> np.ndarray:
    """
    Generate pattern from Conway's Game of Life evolution.

    Returns:
        Grid array showing final state.
    """
    if params.grid_size is None:
        params.grid_size = (64, 64)

    if params.trajectory_length is None:
        params.trajectory_length = 100  # Number of steps

    rows, cols = params.grid_size
    np.random.seed(params.seed)

    # Random initial state
    grid = np.random.choice([0, 1], size=(rows, cols), p=[0.7, 0.3])

    def step(g):
        """One step of Game of Life."""
        new_grid = np.zeros_like(g)

        for i in range(rows):
            for j in range(cols):
                # Count neighbors
                neighbors = 0
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        ni = (i + di) % rows
                        nj = (j + dj) % cols
                        neighbors += g[ni, nj]

                # Rules
                if g[i, j] == 1:
                    new_grid[i, j] = 1 if neighbors in [2, 3] else 0
                else:
                    new_grid[i, j] = 1 if neighbors == 3 else 0

        return new_grid

    # Evolve
    for _ in range(params.trajectory_length):
        grid = step(grid)

    # Extract live cell positions
    live_cells = np.argwhere(grid == 1)
    return live_cells.astype(float) / max(rows, cols)  # Normalize


def generate_rule30_pattern(params: GeometricParameters) -> np.ndarray:
    """
    Generate pattern from Rule 30 cellular automaton (Wolfram Class III).

    Returns:
        Grid showing evolution over time.
    """
    if params.grid_size is None:
        params.grid_size = (128, 128)

    width, height = params.grid_size
    np.random.seed(params.seed)

    grid = np.zeros((height, width), dtype=int)

    # Initial condition: single cell in center
    grid[0, width // 2] = 1

    def apply_rule30(left, center, right):
        """Rule 30: 00011110 in binary."""
        state = (left << 2) | (center << 1) | right
        rule = 30
        return (rule >> state) & 1

    # Evolve
    for t in range(1, height):
        for x in range(width):
            left = grid[t-1, (x-1) % width]
            center = grid[t-1, x]
            right = grid[t-1, (x+1) % width]
            grid[t, x] = apply_rule30(left, center, right)

    # Extract active cells
    active_cells = np.argwhere(grid == 1)
    return active_cells.astype(float) / max(width, height)


# ============================================================================
# FAMILY GF: FRACTAL GRAPHS (Scale-free, Small-world)
# ============================================================================

def generate_scale_free_graph(params: GeometricParameters) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Generate scale-free graph using preferential attachment (Barabási-Albert).

    Returns:
        (node_positions, edges)
    """
    if params.resolution is None:
        n_nodes = 100
    else:
        n_nodes = params.resolution

    if params.branching_factor is None:
        m = 3  # Edges to attach from new node
    else:
        m = params.branching_factor

    np.random.seed(params.seed)

    edges = []
    degrees = np.zeros(n_nodes)

    # Start with small complete graph
    for i in range(m):
        for j in range(i+1, m):
            edges.append((i, j))
            degrees[i] += 1
            degrees[j] += 1

    # Preferential attachment
    for new_node in range(m, n_nodes):
        # Select m nodes to connect to, with probability proportional to degree
        if degrees[:new_node].sum() > 0:
            probs = degrees[:new_node] / degrees[:new_node].sum()
            targets = np.random.choice(new_node, size=min(m, new_node), replace=False, p=probs)

            for target in targets:
                edges.append((new_node, target))
                degrees[new_node] += 1
                degrees[target] += 1

    # Layout nodes in 2D (spring layout approximation)
    positions = np.random.randn(n_nodes, 2)

    return positions, edges


def generate_small_world_graph(params: GeometricParameters) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Generate small-world graph (Watts-Strogatz model).

    Returns:
        (node_positions, edges)
    """
    if params.resolution is None:
        n_nodes = 50
    else:
        n_nodes = params.resolution

    k = 4  # Each node connected to k nearest neighbors
    if params.small_world_prob is None:
        p = 0.1  # Rewiring probability
    else:
        p = params.small_world_prob

    np.random.seed(params.seed)

    # Place nodes on circle
    angles = np.linspace(0, 2*np.pi, n_nodes, endpoint=False)
    positions = np.column_stack([np.cos(angles), np.sin(angles)])

    edges = []

    # Create ring lattice
    for i in range(n_nodes):
        for j in range(1, k//2 + 1):
            neighbor = (i + j) % n_nodes
            edges.append((i, neighbor))

    # Rewire with probability p
    rewired_edges = []
    for i, j in edges:
        if np.random.random() < p:
            # Rewire to random node
            new_j = np.random.randint(0, n_nodes)
            while new_j == i or (i, new_j) in rewired_edges:
                new_j = np.random.randint(0, n_nodes)
            rewired_edges.append((i, new_j))
        else:
            rewired_edges.append((i, j))

    return positions, rewired_edges


# ============================================================================
# FAMILY W: WAVELET/MULTI-RESOLUTION
# ============================================================================

def generate_wavelet_decomposition(params: GeometricParameters) -> Dict[int, np.ndarray]:
    """
    Generate multi-scale wavelet decomposition (Haar wavelet).

    Returns:
        Dictionary mapping scale -> coefficients
    """
    if params.resolution is None:
        n = 256
    else:
        n = params.resolution

    # Ensure power of 2
    n = 2 ** int(np.log2(n))

    np.random.seed(params.seed)

    # Generate signal with multi-scale structure
    signal = np.zeros(n)
    for scale in range(1, int(np.log2(n))):
        freq = 2 ** scale
        signal += np.sin(2 * np.pi * freq * np.arange(n) / n) / freq

    signal += params.noise_amplitude * np.random.randn(n) if hasattr(params, 'noise_amplitude') else 0

    # Haar wavelet decomposition
    def haar_decomposition(s):
        levels = {}
        current = s.copy()
        level = 0

        while len(current) > 1:
            # Average and difference
            avg = (current[::2] + current[1::2]) / 2
            diff = (current[::2] - current[1::2]) / 2

            levels[level] = diff
            current = avg
            level += 1

        levels[level] = current  # Final approximation

        return levels

    return haar_decomposition(signal)


# ============================================================================
# FAMILY PF: FRACTAL SPACE PARTITIONS (KD-trees)
# ============================================================================

def generate_kd_tree_partition(params: GeometricParameters) -> Tuple[np.ndarray, List]:
    """
    Generate fractal KD-tree partition of space.

    Returns:
        (split_points, partition_structure)
    """
    if params.depth is None:
        params.depth = 6

    if params.resolution is None:
        n_points = 1000
    else:
        n_points = params.resolution

    np.random.seed(params.seed)

    # Generate random points in 2D
    points = np.random.rand(n_points, 2)

    def build_kdtree(pts, depth=0, max_depth=params.depth):
        """Recursively build KD-tree."""
        if depth >= max_depth or len(pts) < 2:
            return {"leaf": True, "points": pts}

        # Split dimension (alternate x, y)
        axis = depth % 2

        # Find median
        sorted_indices = np.argsort(pts[:, axis])
        median_idx = len(pts) // 2

        median_value = pts[sorted_indices[median_idx], axis]

        # Split points
        left_pts = pts[pts[:, axis] < median_value]
        right_pts = pts[pts[:, axis] >= median_value]

        return {
            "leaf": False,
            "axis": axis,
            "split_value": median_value,
            "left": build_kdtree(left_pts, depth + 1, max_depth),
            "right": build_kdtree(right_pts, depth + 1, max_depth)
        }

    tree = build_kdtree(points)

    # Extract split points for visualization
    def extract_splits(node, splits=[]):
        if node.get("leaf"):
            return splits

        splits.append((node["axis"], node["split_value"]))
        extract_splits(node.get("left", {}), splits)
        extract_splits(node.get("right", {}), splits)
        return splits

    splits = extract_splits(tree)

    return points, splits


# ============================================================================
# EXTENDED METRICS - Phase 3
# ============================================================================

def compute_spectral_radius(jacobian: np.ndarray) -> float:
    """
    Compute spectral radius ρ(J) of Jacobian matrix.

    Args:
        jacobian: Jacobian matrix

    Returns:
        Spectral radius (largest absolute eigenvalue)
    """
    eigenvalues = np.linalg.eigvals(jacobian)
    return np.max(np.abs(eigenvalues))


def compute_route_entropy(routes: List[List[int]], probabilities: np.ndarray | None = None) -> float:
    """
    Compute route entropy H_route = -Σ p(π) log p(π)

    Args:
        routes: List of routes (each route is list of node indices)
        probabilities: Optional probability for each route

    Returns:
        Route entropy
    """
    if probabilities is None:
        # Uniform distribution
        probabilities = np.ones(len(routes)) / len(routes)

    entropy = 0.0
    for p in probabilities:
        if p > 0:
            entropy -= p * np.log(p)

    return entropy


def compute_channel_entropy(channel_usage: Dict[Any, float]) -> float:
    """
    Compute channel entropy from usage distribution.

    Args:
        channel_usage: Dictionary mapping channel -> usage count/probability

    Returns:
        Channel entropy
    """
    total = sum(channel_usage.values())
    if total == 0:
        return 0.0

    entropy = 0.0
    for count in channel_usage.values():
        p = count / total
        if p > 0:
            entropy -= p * np.log(p)

    return entropy


def compute_cvar(losses: np.ndarray, alpha: float = 0.95) -> float:
    """
    Compute Conditional Value at Risk (CVaR).

    CVaR_α[L] = E[L | L >= VaR_α]

    Args:
        losses: Array of loss values
        alpha: Confidence level (e.g., 0.95 for 95%)

    Returns:
        CVaR value
    """
    if len(losses) == 0:
        return 0.0

    # Find VaR threshold
    var_threshold = np.percentile(losses, alpha * 100)

    # Compute mean of losses exceeding VaR
    tail_losses = losses[losses >= var_threshold]

    if len(tail_losses) == 0:
        return var_threshold

    return np.mean(tail_losses)


def compute_informational_action(
    information_gain: float,
    structural_cost: float,
    meta_cost: float,
    fractal_penalty: float,
    route_entropy: float,
    lambda_mdl: float = 0.1,
    lambda_meta: float = 0.01,
    lambda_fractal: float = 0.1
) -> float:
    """
    Compute informational action A[Θ].

    A[Θ] = -İ + λ_MDL·C_struct + λ_meta·C_meta + λ_fractal·(D_q - D*)² + H_route

    Args:
        information_gain: İ (information gain rate)
        structural_cost: C_struct (structural complexity)
        meta_cost: C_meta (meta-architecture cost)
        fractal_penalty: (D_q - D*)² term
        route_entropy: H_route
        lambda_mdl: Weight for structural cost
        lambda_meta: Weight for meta cost
        lambda_fractal: Weight for fractal penalty

    Returns:
        Action value
    """
    action = (
        -information_gain
        + lambda_mdl * structural_cost
        + lambda_meta * meta_cost
        + lambda_fractal * fractal_penalty
        + route_entropy
    )

    return action


def check_safe_constraints(
    vram_usage_mb: float,
    temperature_c: float,
    latency_ms: float,
    spectral_margin: float,
    safe_params: MetaParameters
) -> Dict[str, bool]:
    """
    Check SAFE (Safety-Aware Fractal Evolution) constraints.

    Args:
        vram_usage_mb: Current VRAM usage
        temperature_c: Current temperature
        latency_ms: Current latency
        spectral_margin: Current spectral margin 1 - ρ(J)
        safe_params: MetaParameters with SAFE limits

    Returns:
        Dictionary indicating which constraints are satisfied
    """
    return {
        "vram_ok": vram_usage_mb <= safe_params.vram_limit_mb,
        "temperature_ok": temperature_c <= safe_params.temperature_limit_c,
        "latency_ok": latency_ms <= safe_params.latency_limit_ms,
        "spectral_ok": spectral_margin >= safe_params.spectral_margin_min,
        "all_safe": (
            vram_usage_mb <= safe_params.vram_limit_mb
            and temperature_c <= safe_params.temperature_limit_c
            and latency_ms <= safe_params.latency_limit_ms
            and spectral_margin >= safe_params.spectral_margin_min
        )
    }


def compute_coherence_infinity(
    loss_variances: np.ndarray,
    env_gaps: np.ndarray,
    gradient_violations: np.ndarray,
    safe_violations: np.ndarray,
    weights: Tuple[float, float, float, float] = (0.4, 0.3, 0.2, 0.1)
) -> float:
    """
    Compute long-term fractal coherence C_∞.

    C_∞ = ∫ [var(L) + gap_env + non_inv_grad + violations_SAFE] dt

    Args:
        loss_variances: Variance of losses over time
        env_gaps: Performance gaps between environments
        gradient_violations: Non-invariance of gradients
        safe_violations: Accumulated SAFE constraint violations
        weights: Weights for each component

    Returns:
        C_∞ coherence measure
    """
    w_loss, w_gap, w_grad, w_safe = weights

    coherence = (
        w_loss * np.sum(loss_variances)
        + w_gap * np.sum(env_gaps)
        + w_grad * np.sum(gradient_violations)
        + w_safe * np.sum(safe_violations)
    )

    return coherence


def compute_multiscale_memory_metrics(
    memory_distribution: Dict[str, np.ndarray],
    scales: List[str] = ["coarse", "medium", "fine"]
) -> Dict[str, float]:
    """
    Compute Multi-scale Fractal Memory (MFM) metrics.

    Args:
        memory_distribution: Dict mapping scale -> memory weights
        scales: Scale names

    Returns:
        Dictionary of MFM metrics
    """
    metrics = {}

    # Entropy fractal H_f
    all_weights = np.concatenate([memory_distribution.get(s, np.array([])) for s in scales])
    if len(all_weights) > 0:
        probs = all_weights / (all_weights.sum() + 1e-10)
        h_f = -np.sum(probs * np.log(probs + 1e-10))
        metrics["entropy_fractal"] = h_f

    # Drift fractal δ_d (simplified: variance across scales)
    if len(scales) > 1:
        scale_means = [memory_distribution.get(s, np.array([0])).mean() for s in scales]
        metrics["drift_fractal"] = np.var(scale_means)
    else:
        metrics["drift_fractal"] = 0.0

    # Scale balance
    for scale in scales:
        if scale in memory_distribution:
            metrics[f"weight_{scale}"] = memory_distribution[scale].sum()

    return metrics


def compute_fractal_communication_cost(
    distance_fractal: float,
    quality: float,
    latency: float,
    congestion: float,
    weights: Tuple[float, float, float, float] = (1.0, 0.5, 0.3, 0.2)
) -> float:
    """
    Compute fractal communication cost for an edge.

    w_t(e) = α·d_F(e;t) + β/Q(e;t) + γ·lat(e;t) + δ·cong(e;t)

    Args:
        distance_fractal: Fractal distance d_F
        quality: Channel quality Q
        latency: Latency
        congestion: Congestion level
        weights: (α, β, γ, δ)

    Returns:
        Communication cost
    """
    alpha, beta, gamma, delta = weights

    cost = (
        alpha * distance_fractal
        + beta / (quality + 1e-10)
        + gamma * latency
        + delta * congestion
    )

    return cost


def estimate_renormalization_smoothness(
    geometry_points: np.ndarray,
    levels: List[int] = [0, 1, 2]
) -> float:
    """
    Measure renormalization smoothness across fractal levels.

    Args:
        geometry_points: Point cloud
        levels: Levels to compare

    Returns:
        Smoothness score (lower = more self-similar)
    """
    if len(geometry_points) < 10:
        return 0.0

    # Subsample at different scales
    level_densities = []

    for level in levels:
        stride = 2 ** level
        subsampled = geometry_points[::stride]

        if len(subsampled) > 1:
            # Estimate local density
            from scipy.spatial import distance_matrix
            if len(subsampled) > 100:
                sample_idx = np.random.choice(len(subsampled), 100, replace=False)
                sample = subsampled[sample_idx]
            else:
                sample = subsampled

            # Mean nearest neighbor distance
            try:
                dists = distance_matrix(sample, sample)
                np.fill_diagonal(dists, np.inf)
                mean_nn_dist = np.mean(np.min(dists, axis=1))
                level_densities.append(1.0 / (mean_nn_dist + 1e-10))
            except:
                level_densities.append(1.0)

    # Smoothness = variance of log-densities
    if len(level_densities) > 1:
        log_densities = np.log(np.array(level_densities) + 1e-10)
        return np.var(log_densities)

    return 0.0
