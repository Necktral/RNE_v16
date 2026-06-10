"""
Fractal Analysis Utilities for Reasoning Scheduler Stress Testing
==================================================================

Provides utilities for measuring:
- Multiscale boundary refinement
- Box-counting fractal dimension
- Temporal cascade analysis
- Activation avalanche statistics
- Scale invariance metrics
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple, Callable
from dataclasses import dataclass
from runtime.reasoning.scheduler_meta.budgeting import compute_budget
from runtime.reasoning.scheduler_meta.policy import select_sequence


@dataclass
class MultiscaleBoundaryMetrics:
    """Metrics for a boundary at multiple resolutions."""
    feature_name: str
    threshold_nominal: float
    family_name: str | None

    # Per-resolution measurements
    resolutions: List[float]  # e.g., [0.10, 0.05, 0.02, 0.01, 0.005]
    activation_points: List[float]  # Activation threshold per resolution
    deactivation_points: List[float]  # Deactivation threshold per resolution
    hysteresis_widths: List[float]  # Width of hysteresis per resolution
    discontinuity_counts: List[int]  # Number of discontinuities per resolution
    local_slopes: List[float]  # Average local slope per resolution
    boundary_variances: List[float]  # Variance of boundary location per resolution

    # Convergence metrics
    converges: bool  # Whether boundary converges as resolution refines
    convergence_rate: float  # How fast it converges
    roughness_exponent: float  # How rough the boundary is

    # Classification
    discipline: str  # "disciplined", "critical", "pathological"


@dataclass
class BoxCountingResult:
    """Result of box-counting fractal dimension analysis."""
    feature_pair: Tuple[str, str]
    family_name: str
    box_sizes: List[float]  # Sizes of boxes used
    box_counts: List[int]  # Number of boxes needed at each size
    fractal_dimension: float  # Estimated fractal dimension
    fit_quality: float  # R² of log-log regression
    interpretation: str  # "clean", "rugose_controlled", "pathological"


@dataclass
class TemporalCascadeMetrics:
    """Metrics for temporal cascade behavior."""
    perturbation_feature: str
    time_scales: List[int]  # [1, 2, 4, 8, 16] steps

    # Per time scale
    family_change_counts: List[int]
    max_steps_changes: List[int]
    recommendation_changes: List[int]
    activation_persistence: List[float]
    recovery_times: List[float]
    hysteresis_widths: List[float]

    # Auto-similarity metrics
    self_similar: bool
    scale_invariance_error: float
    temporal_memory_fragility: float


@dataclass
class AvalancheDistribution:
    """Distribution of activation avalanche sizes."""
    avalanche_sizes: List[int]  # Size of each avalanche
    size_histogram: Dict[int, int]  # Histogram of sizes
    mean_size: float
    max_size: int
    tail_exponent: float  # Power law exponent if heavy-tailed
    is_heavy_tailed: bool
    criticalit_indicator: str  # "rigid", "interesting", "fragile"


def measure_multiscale_boundary(
    feature_name: str,
    threshold: float,
    family_name: str | None = None,
    resolutions: List[float] | None = None,
    baseline_features: Dict[str, float] | None = None
) -> MultiscaleBoundaryMetrics:
    """
    Measure boundary behavior at multiple resolutions.

    Args:
        feature_name: Feature to sweep
        threshold: Nominal threshold value
        family_name: Family to track (None = track any activation change)
        resolutions: Resolution levels to test
        baseline_features: Base feature values

    Returns:
        MultiscaleBoundaryMetrics with convergence analysis
    """
    if resolutions is None:
        resolutions = [0.10, 0.05, 0.02, 0.01, 0.005]

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

    activation_points = []
    deactivation_points = []
    hysteresis_widths = []
    discontinuity_counts = []
    local_slopes = []
    boundary_variances = []

    for resolution in resolutions:
        # Measure at this resolution
        metrics = _measure_boundary_at_resolution(
            feature_name=feature_name,
            threshold=threshold,
            resolution=resolution,
            family_name=family_name,
            baseline_features=baseline_features
        )

        activation_points.append(metrics['activation_point'])
        deactivation_points.append(metrics['deactivation_point'])
        hysteresis_widths.append(metrics['hysteresis_width'])
        discontinuity_counts.append(metrics['discontinuity_count'])
        local_slopes.append(metrics['local_slope'])
        boundary_variances.append(metrics['boundary_variance'])

    # Analyze convergence
    converges, convergence_rate, roughness = _analyze_convergence(
        activation_points, resolutions
    )

    # Classify discipline
    discipline = _classify_boundary_discipline(
        converges, roughness, hysteresis_widths
    )

    return MultiscaleBoundaryMetrics(
        feature_name=feature_name,
        threshold_nominal=threshold,
        family_name=family_name,
        resolutions=resolutions,
        activation_points=activation_points,
        deactivation_points=deactivation_points,
        hysteresis_widths=hysteresis_widths,
        discontinuity_counts=discontinuity_counts,
        local_slopes=local_slopes,
        boundary_variances=boundary_variances,
        converges=converges,
        convergence_rate=convergence_rate,
        roughness_exponent=roughness,
        discipline=discipline
    )


def _measure_boundary_at_resolution(
    feature_name: str,
    threshold: float,
    resolution: float,
    family_name: str | None,
    baseline_features: Dict[str, float]
) -> Dict:
    """Measure boundary at a specific resolution."""
    sweep_range = (-resolution, resolution)
    steps = max(11, int(2 * resolution / 0.005))  # Adaptive step count

    start = threshold + sweep_range[0]
    end = threshold + sweep_range[1]
    step_size = (end - start) / (steps - 1)

    # Ascending sweep
    ascending_activations = []
    for i in range(steps):
        value = start + (i * step_size)
        value = max(0.0, min(1.0, value))

        features = baseline_features.copy()
        features[feature_name] = value

        budget = compute_budget(features)
        sequence, _, _ = select_sequence(
            features=features,
            budget=budget,
            allow_experimental=True
        )

        if family_name:
            is_active = family_name in sequence
        else:
            is_active = len(sequence) > 6  # Generic activation

        ascending_activations.append((value, is_active))

    # Descending sweep
    descending_activations = []
    for i in range(steps):
        value = end - (i * step_size)
        value = max(0.0, min(1.0, value))

        features = baseline_features.copy()
        features[feature_name] = value

        budget = compute_budget(features)
        sequence, _, _ = select_sequence(
            features=features,
            budget=budget,
            allow_experimental=True
        )

        if family_name:
            is_active = family_name in sequence
        else:
            is_active = len(sequence) > 6

        descending_activations.append((value, is_active))

    # Find activation point (ascending)
    activation_point = None
    for i, (value, active) in enumerate(ascending_activations):
        if active and i > 0 and not ascending_activations[i-1][1]:
            activation_point = value
            break

    # Find deactivation point (descending)
    deactivation_point = None
    for i, (value, active) in enumerate(descending_activations):
        if not active and i > 0 and descending_activations[i-1][1]:
            deactivation_point = value
            break

    # Hysteresis width
    if activation_point and deactivation_point:
        hysteresis_width = abs(activation_point - deactivation_point)
    else:
        hysteresis_width = 0.0

    # Count discontinuities
    discontinuity_count = sum(
        1 for i in range(1, len(ascending_activations))
        if ascending_activations[i][1] != ascending_activations[i-1][1]
    )

    # Local slope (rate of change)
    local_slope = discontinuity_count / steps if steps > 0 else 0.0

    # Boundary variance
    transition_points = [
        value for i, (value, active) in enumerate(ascending_activations)
        if i > 0 and ascending_activations[i][1] != ascending_activations[i-1][1]
    ]
    if len(transition_points) > 1:
        mean_transition = sum(transition_points) / len(transition_points)
        boundary_variance = sum((p - mean_transition) ** 2 for p in transition_points) / len(transition_points)
    else:
        boundary_variance = 0.0

    return {
        'activation_point': activation_point or threshold,
        'deactivation_point': deactivation_point or threshold,
        'hysteresis_width': hysteresis_width,
        'discontinuity_count': discontinuity_count,
        'local_slope': local_slope,
        'boundary_variance': boundary_variance
    }


def _analyze_convergence(
    activation_points: List[float],
    resolutions: List[float]
) -> Tuple[bool, float, float]:
    """Analyze if boundary converges as resolution refines."""
    if len(activation_points) < 2:
        return False, 0.0, 0.0

    # Check if points are converging
    differences = [
        abs(activation_points[i] - activation_points[i-1])
        for i in range(1, len(activation_points))
    ]

    # Converge si la frontera se ESTABILIZA al refinar la resolución. El criterio
    # original "diferencias monótonamente decrecientes" daba falsos negativos en
    # fronteras perfectamente disciplinadas por el jitter discreto del sweep (el
    # punto de activación detectado salta de un step a otro). Se acepta también que
    # la mitad final de los puntos de activación esté concentrada (tail estable).
    monotone = all(
        differences[i] <= differences[i-1] * 1.2  # Allow some noise
        for i in range(1, len(differences))
    ) if len(differences) > 1 else True
    tail = activation_points[len(activation_points) // 2:]
    mean_tail = sum(tail) / len(tail)
    tail_spread = max((abs(p - mean_tail) for p in tail), default=0.0)
    converges = monotone or tail_spread <= 0.05

    # Convergence rate (how fast differences decrease)
    if len(differences) > 1:
        convergence_rate = differences[-1] / differences[0] if differences[0] > 0 else 0.0
    else:
        convergence_rate = 1.0

    # Rugosidad: dispersión (desviación estándar) de los puntos de activación a través
    # de resoluciones. Frontera estable/convergente → ~0; oscilante → mayor (acotada a
    # [0, ~0.5] en el espacio de features [0,1]). La fórmula anterior
    # log(var)/log(ratio_resoluciones) divergía a ≈7.7 cuando var→0 (por el clamp 1e-10),
    # clasificando como "pathological" fronteras perfectamente estables.
    mean_ap = sum(activation_points) / len(activation_points)
    roughness = math.sqrt(
        sum((p - mean_ap) ** 2 for p in activation_points) / len(activation_points)
    )

    return converges, convergence_rate, roughness


def _classify_boundary_discipline(
    converges: bool,
    roughness: float,
    hysteresis_widths: List[float]
) -> str:
    """Classify boundary as disciplined, critical, or pathological."""
    avg_hysteresis = sum(hysteresis_widths) / len(hysteresis_widths) if hysteresis_widths else 0.0

    if converges and roughness < 0.3 and avg_hysteresis < 0.02:
        return "disciplined"
    elif not converges or roughness > 0.7 or avg_hysteresis > 0.05:
        return "pathological"
    else:
        return "critical"


def estimate_box_counting_dimension(
    feature_x: str,
    feature_y: str,
    family_name: str,
    box_sizes: List[float] | None = None,
    n_samples: int = 500,
    baseline_features: Dict[str, float] | None = None
) -> BoxCountingResult:
    """
    Estimate fractal dimension of activation frontier using box-counting.

    Args:
        feature_x: First feature dimension
        feature_y: Second feature dimension
        family_name: Family whose frontier to analyze
        box_sizes: Sizes of boxes to use
        n_samples: Number of sample points
        baseline_features: Base feature values

    Returns:
        BoxCountingResult with dimension estimate
    """
    if box_sizes is None:
        box_sizes = [0.2, 0.1, 0.05, 0.025, 0.0125]

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

    # Sample the 2D space to find frontier points
    frontier_points = _find_frontier_points(
        feature_x, feature_y, family_name, n_samples, baseline_features
    )

    if len(frontier_points) < 10:
        # Not enough frontier points
        return BoxCountingResult(
            feature_pair=(feature_x, feature_y),
            family_name=family_name,
            box_sizes=box_sizes,
            box_counts=[0] * len(box_sizes),
            fractal_dimension=1.0,
            fit_quality=0.0,
            interpretation="insufficient_data"
        )

    # Count boxes at each scale
    box_counts = []
    for box_size in box_sizes:
        count = _count_boxes_covering_points(frontier_points, box_size)
        box_counts.append(count)

    # Estimate dimension via log-log regression
    dimension, fit_quality = _estimate_dimension_from_box_counts(
        box_sizes, box_counts
    )

    # Interpret result
    interpretation = _interpret_fractal_dimension(dimension, fit_quality)

    return BoxCountingResult(
        feature_pair=(feature_x, feature_y),
        family_name=family_name,
        box_sizes=box_sizes,
        box_counts=box_counts,
        fractal_dimension=dimension,
        fit_quality=fit_quality,
        interpretation=interpretation
    )


def _find_frontier_points(
    feature_x: str,
    feature_y: str,
    family_name: str,
    n_samples: int,
    baseline_features: Dict[str, float]
) -> List[Tuple[float, float]]:
    """Find points on the activation frontier."""
    import random
    random.seed(42)

    frontier_points = []

    # Grid sampling with random perturbations
    grid_size = int(math.sqrt(n_samples))

    for i in range(grid_size):
        for j in range(grid_size):
            x = (i + random.random()) / grid_size
            y = (j + random.random()) / grid_size

            features = baseline_features.copy()
            features[feature_x] = x
            features[feature_y] = y

            budget = compute_budget(features)
            sequence, _, _ = select_sequence(
                features=features,
                budget=budget,
                allow_experimental=True
            )

            is_active = family_name in sequence

            # Check if this is a frontier point (near boundary)
            # Sample neighbors
            for dx, dy in [(-0.02, 0), (0.02, 0), (0, -0.02), (0, 0.02)]:
                nx = max(0.0, min(1.0, x + dx))
                ny = max(0.0, min(1.0, y + dy))

                neighbor_features = baseline_features.copy()
                neighbor_features[feature_x] = nx
                neighbor_features[feature_y] = ny

                neighbor_budget = compute_budget(neighbor_features)
                neighbor_sequence, _, _ = select_sequence(
                    features=neighbor_features,
                    budget=neighbor_budget,
                    allow_experimental=True
                )

                neighbor_active = family_name in neighbor_sequence

                if is_active != neighbor_active:
                    # This is a frontier point
                    frontier_points.append((x, y))
                    break

    return frontier_points


def _count_boxes_covering_points(
    points: List[Tuple[float, float]],
    box_size: float
) -> int:
    """Count number of boxes of given size needed to cover points."""
    if not points:
        return 0

    # Discretize points into boxes
    boxes = set()
    for x, y in points:
        box_x = int(x / box_size)
        box_y = int(y / box_size)
        boxes.add((box_x, box_y))

    return len(boxes)


def _estimate_dimension_from_box_counts(
    box_sizes: List[float],
    box_counts: List[int]
) -> Tuple[float, float]:
    """Estimate fractal dimension from box counts using log-log regression."""
    if len(box_sizes) < 3 or any(c == 0 for c in box_counts):
        return 1.0, 0.0

    # Región de escala: mantener el prefijo estrictamente creciente del conteo (hasta
    # el primer plateau). Una vez que la frontera del scheduler satura (es finita/
    # dispersa, p.ej. [5, 10, 18, 18, 18]), el conteo deja de crecer; incluir el
    # plateau aplana la pendiente y hunde el R² → falsos "poor_fit"/"pathological" en
    # fronteras que en realidad son limpias (~1D).
    kept = [(box_sizes[0], box_counts[0])]
    for i in range(1, len(box_counts)):
        if box_counts[i] > box_counts[i - 1]:
            kept.append((box_sizes[i], box_counts[i]))
        else:
            break
    if len(kept) < 3:
        kept = list(zip(box_sizes, box_counts))  # fallback: usar todas las cajas
    log_epsilon = [math.log(1.0 / s) for s, _ in kept]
    log_counts = [math.log(c) for _, c in kept]

    # Linear regression
    n = len(log_epsilon)
    mean_x = sum(log_epsilon) / n
    mean_y = sum(log_counts) / n

    numerator = sum((log_epsilon[i] - mean_x) * (log_counts[i] - mean_y) for i in range(n))
    denominator = sum((log_epsilon[i] - mean_x) ** 2 for i in range(n))

    if denominator == 0:
        return 1.0, 0.0

    slope = numerator / denominator

    # R² for fit quality
    ss_tot = sum((log_counts[i] - mean_y) ** 2 for i in range(n))
    ss_res = sum((log_counts[i] - (mean_y + slope * (log_epsilon[i] - mean_x))) ** 2 for i in range(n))

    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return slope, r_squared


def _interpret_fractal_dimension(dimension: float, fit_quality: float) -> str:
    """Interpret fractal dimension result."""
    if fit_quality < 0.8:
        return "poor_fit"

    if dimension < 1.15:
        return "clean"  # Clean, nearly 1D boundary
    elif dimension < 1.35:
        return "rugose_controlled"  # Some roughness but controlled
    else:
        return "pathological"  # Highly fragmented boundary


def measure_temporal_cascades(
    feature_name: str,
    perturbation_pattern: str,  # "increasing", "decreasing", "oscillating", "pulsed"
    time_scales: List[int] | None = None,
    baseline_features: Dict[str, float] | None = None
) -> TemporalCascadeMetrics:
    """
    Measure cascade behavior at different temporal scales.

    Args:
        feature_name: Feature to perturb
        perturbation_pattern: Type of perturbation
        time_scales: Time scales to test (in steps)
        baseline_features: Base feature values

    Returns:
        TemporalCascadeMetrics
    """
    if time_scales is None:
        time_scales = [1, 2, 4, 8, 16]

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

    family_change_counts = []
    max_steps_changes = []
    recommendation_changes = []
    activation_persistence = []
    recovery_times = []
    hysteresis_widths = []

    for scale in time_scales:
        metrics = _measure_cascade_at_scale(
            feature_name, perturbation_pattern, scale, baseline_features
        )

        family_change_counts.append(metrics['family_changes'])
        max_steps_changes.append(metrics['max_steps_changes'])
        recommendation_changes.append(metrics['recommendation_changes'])
        activation_persistence.append(metrics['persistence'])
        recovery_times.append(metrics['recovery_time'])
        hysteresis_widths.append(metrics['hysteresis_width'])

    # Analyze self-similarity
    self_similar, scale_error = _analyze_temporal_self_similarity(
        family_change_counts, time_scales
    )

    # Measure memory fragility
    fragility = _measure_temporal_memory_fragility(hysteresis_widths, time_scales)

    return TemporalCascadeMetrics(
        perturbation_feature=feature_name,
        time_scales=time_scales,
        family_change_counts=family_change_counts,
        max_steps_changes=max_steps_changes,
        recommendation_changes=recommendation_changes,
        activation_persistence=activation_persistence,
        recovery_times=recovery_times,
        hysteresis_widths=hysteresis_widths,
        self_similar=self_similar,
        scale_invariance_error=scale_error,
        temporal_memory_fragility=fragility
    )


def _measure_cascade_at_scale(
    feature_name: str,
    pattern: str,
    scale: int,
    baseline_features: Dict[str, float]
) -> Dict:
    """Measure cascade metrics at a specific time scale."""
    # Generate perturbation sequence
    values = _generate_perturbation_sequence(pattern, scale)

    previous_sequence = None
    previous_budget = None
    previous_recommendation = None

    family_changes = 0
    max_steps_changes = 0
    recommendation_changes = 0

    for value in values:
        features = baseline_features.copy()
        features[feature_name] = value

        budget = compute_budget(features)
        sequence, _, recommendation = select_sequence(
            features=features,
            budget=budget,
            allow_experimental=True
        )

        if previous_sequence is not None:
            if set(sequence) != set(previous_sequence):
                family_changes += 1

        if previous_budget is not None:
            if budget['max_steps'] != previous_budget['max_steps']:
                max_steps_changes += 1

        if previous_recommendation is not None:
            if recommendation != previous_recommendation:
                recommendation_changes += 1

        previous_sequence = sequence
        previous_budget = budget
        previous_recommendation = recommendation

    # Persistence: how long activations last
    persistence = 1.0 - (family_changes / max(len(values) - 1, 1))

    # Recovery time: estimate (simplified)
    recovery_time = family_changes * 0.5  # Proportional to changes

    # Hysteresis width (simplified)
    hysteresis_width = max_steps_changes / max(len(values), 1)

    return {
        'family_changes': family_changes,
        'max_steps_changes': max_steps_changes,
        'recommendation_changes': recommendation_changes,
        'persistence': persistence,
        'recovery_time': recovery_time,
        'hysteresis_width': hysteresis_width
    }


def _generate_perturbation_sequence(pattern: str, length: int) -> List[float]:
    """Generate a perturbation sequence."""
    if pattern == "increasing":
        return [i / length for i in range(length)]
    elif pattern == "decreasing":
        return [(length - i) / length for i in range(length)]
    elif pattern == "oscillating":
        return [0.5 + 0.3 * math.sin(2 * math.pi * i / length) for i in range(length)]
    elif pattern == "pulsed":
        return [0.8 if i % 4 == 0 else 0.2 for i in range(length)]
    else:
        return [0.5] * length


def _analyze_temporal_self_similarity(
    change_counts: List[int],
    scales: List[int]
) -> Tuple[bool, float]:
    """Analyze if temporal patterns are self-similar."""
    if len(change_counts) < 3:
        return False, 0.0

    # Normalize by scale
    normalized = [change_counts[i] / scales[i] for i in range(len(change_counts))]

    # Check if normalized values are similar
    mean_norm = sum(normalized) / len(normalized)
    variance = sum((n - mean_norm) ** 2 for n in normalized) / len(normalized)

    scale_error = math.sqrt(variance) / (mean_norm + 1e-10)

    # Self-similar if error is low
    self_similar = scale_error < 0.3

    return self_similar, scale_error


def _measure_temporal_memory_fragility(
    hysteresis_widths: List[float],
    scales: List[int]
) -> float:
    """Measure if hysteresis depends spuriously on scale."""
    if len(hysteresis_widths) < 2:
        return 0.0

    # High fragility if hysteresis varies a lot with scale
    variance = sum((hysteresis_widths[i] - hysteresis_widths[0]) ** 2
                   for i in range(len(hysteresis_widths))) / len(hysteresis_widths)

    return math.sqrt(variance)


def measure_activation_avalanches(
    n_trials: int = 500,
    perturbation_magnitude: float = 0.02,
    baseline_features: Dict[str, float] | None = None
) -> AvalancheDistribution:
    """
    Measure distribution of activation avalanche sizes.

    An avalanche is defined as a small perturbation that triggers cascading changes.

    Args:
        n_trials: Number of random perturbations to test
        perturbation_magnitude: Size of perturbation
        baseline_features: Base feature values

    Returns:
        AvalancheDistribution
    """
    import random
    random.seed(42)

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

    avalanche_sizes = []

    feature_names = list(baseline_features.keys())

    for _ in range(n_trials):
        # Random starting point
        features = {
            name: random.uniform(0.0, 1.0) for name in feature_names
        }
        features["continuity_recent"] = random.uniform(0.5, 1.0)

        # Compute baseline
        budget_before = compute_budget(features)
        sequence_before, _, rec_before = select_sequence(
            features=features,
            budget=budget_before,
            allow_experimental=True
        )

        # Small perturbation
        perturbed_feature = random.choice(feature_names)
        features[perturbed_feature] = max(0.0, min(1.0,
            features[perturbed_feature] + random.uniform(-perturbation_magnitude, perturbation_magnitude)
        ))

        # Compute after perturbation
        budget_after = compute_budget(features)
        sequence_after, _, rec_after = select_sequence(
            features=features,
            budget=budget_after,
            allow_experimental=True
        )

        # Count changes (avalanche size)
        avalanche_size = 0

        # Family activations changed
        if set(sequence_before) != set(sequence_after):
            avalanche_size += 1

        # Budget changed non-trivially
        if abs(budget_after['max_steps'] - budget_before['max_steps']) >= 1:
            avalanche_size += 1

        # Recommendation changed
        if rec_before != rec_after:
            avalanche_size += 1

        avalanche_sizes.append(avalanche_size)

    # Build histogram
    size_histogram = {}
    for size in avalanche_sizes:
        size_histogram[size] = size_histogram.get(size, 0) + 1

    mean_size = sum(avalanche_sizes) / len(avalanche_sizes) if avalanche_sizes else 0.0
    max_size = max(avalanche_sizes) if avalanche_sizes else 0

    # Estimate tail exponent (simplified power law fit)
    tail_exponent, is_heavy_tailed = _estimate_tail_exponent(size_histogram)

    # Classify criticality
    criticality = _classify_criticality(mean_size, is_heavy_tailed, size_histogram)

    return AvalancheDistribution(
        avalanche_sizes=avalanche_sizes,
        size_histogram=size_histogram,
        mean_size=mean_size,
        max_size=max_size,
        tail_exponent=tail_exponent,
        is_heavy_tailed=is_heavy_tailed,
        criticalit_indicator=criticality
    )


def _estimate_tail_exponent(histogram: Dict[int, int]) -> Tuple[float, bool]:
    """Estimate power law exponent of tail."""
    if not histogram or len(histogram) < 3:
        return 1.0, False

    # Simple check: if large avalanches are common, it's heavy-tailed
    total = sum(histogram.values())
    large_avalanches = sum(count for size, count in histogram.items() if size >= 2)

    fraction_large = large_avalanches / total if total > 0 else 0.0

    is_heavy_tailed = fraction_large > 0.15

    # Rough exponent estimate
    if is_heavy_tailed:
        exponent = -2.0  # Typical for critical systems
    else:
        exponent = -4.0  # Steep decay

    return exponent, is_heavy_tailed


def _classify_criticality(
    mean_size: float,
    is_heavy_tailed: bool,
    histogram: Dict[int, int]
) -> str:
    """Classify system criticality."""
    if mean_size < 0.3 and not is_heavy_tailed:
        return "rigid"  # Not very responsive
    elif is_heavy_tailed and mean_size > 0.5:
        return "fragile"  # Too sensitive
    else:
        return "interesting"  # Good balance
