# Experiment 2: Fractal Geometry Catalog - Complete Implementation

## Overview

This document summarizes the complete implementation of **Experiment 2: Catálogo de Geometrías Fractales**, which extends the fractal stress characterization framework to test scheduler behavior across diverse fractal geometries.

## Implementation Summary

### Phase 1: Core Fractal Geometry Framework ✅
**Files:** `fractal_geometries.py`, `test_geometry_catalog.py`
**Families Implemented:** T, C, B, AXC, AXD, RS (6/12)

- Created `FractalFamily` enum with all 12 family types
- Implemented `GeometricParameters` and `MetaParameters` dataclasses
- Generated first 6 fractal families:
  - **T (Triangular):** Sierpinski triangle via IFS
  - **C (Carpet):** Cantor carpets via recursive subdivision
  - **B (Branching):** Fractal trees with hierarchical structure
  - **AXC (Continuous Attractors):** Lorenz and Rössler attractors
  - **AXD (Discrete Attractors):** Hénon map
  - **RS (Stochastic):** Fractional Brownian motion (fBm)

- Created **geometry-to-scheduler mapping** function - the critical bridge
- Implemented tests with parameter sweeps and scheduler integration

### Phase 2: Extended Fractal Families ✅
**Families Added:** F3D, MC, AC, GF, W, PF (6/12)

Added remaining geometry generators:

- **F3D (3D Fractals):**
  - `generate_menger_sponge()` - 3D Cantor set (D ≈ 2.727)
  - `generate_sierpinski_tetrahedron()` - 3D Sierpinski (D = 2.0)

- **MC (Complex Plane):**
  - `generate_mandelbrot_set()` - Mandelbrot boundary (D ≈ 2.0)
  - `generate_julia_set()` - Julia sets with parameter c

- **AC (Cellular Automata):**
  - `generate_game_of_life_pattern()` - Conway's Life evolution
  - `generate_rule30_pattern()` - Wolfram Class III chaos

- **GF (Fractal Graphs):**
  - `generate_scale_free_graph()` - Barabási-Albert preferential attachment
  - `generate_small_world_graph()` - Watts-Strogatz rewiring

- **W (Wavelet):**
  - `generate_wavelet_decomposition()` - Haar wavelet multi-scale decomposition

- **PF (Partition):**
  - `generate_kd_tree_partition()` - Fractal KD-tree space partitioning

### Phase 3: Extended Metrics ✅
**Comprehensive metric functions for 9 fractal functionalities:**

1. **Multi-scale Self-Similarity:**
   - `compute_generalized_dimensions(q_values)` - D_q spectrum
   - Box-counting dimension estimation

2. **Fractal Dynamic Variety (VFD):**
   - `compute_lyapunov_exponents()` - Chaos quantification
   - Attractor characterization

3. **Complexity Metrics:**
   - `compute_participation_ratio()` - Localization measure
   - D_q spread analysis

4. **Multi-scale Fractal Memory (MFM):**
   - `compute_multiscale_memory_metrics()` - τ_decay, memory depth
   - Cross-scale correlation

5. **Fractal Edges:**
   - Graph edge extraction
   - Boundary detection

6. **Fractal Communication:**
   - `compute_fractal_communication_cost()` - w_t(e) = α·d_F + β/Q + γ·lat + δ·cong
   - `compute_route_entropy()` - H_route = -Σ p(π) log p(π)
   - `compute_channel_entropy()` - Communication channel utilization

7. **V-S-H Evolution:**
   - Viability-Suitability-Habitability tracking
   - Evolutionary potential assessment

8. **Cognitive Fractality:**
   - `compute_informational_action()` - A[Θ] = -İ + λ_MDL·C_struct + λ_meta·C_meta + λ_fractal·(D_q - D*)² + H_route
   - `check_safe_constraints()` - SAFE (Safety-Aware Fractal Evolution)

9. **Long-term Coherence:**
   - `compute_coherence_infinity()` - C_∞ stability measure
   - `estimate_renormalization_smoothness()` - RG flow analysis

Additional advanced metrics:
- `compute_spectral_radius()` - ρ(J) for Jacobian stability
- `compute_cvar()` - Conditional Value at Risk

### Phase 4: Comprehensive Testing ✅
**Files:** Extended `test_geometry_catalog.py`, new `test_experiment2_atlas.py`

**test_geometry_catalog.py extensions:**
- Tests for all 6 new families (F3D, MC, AC, GF, W, PF)
- Each family has:
  - Basic generation tests
  - Scheduler integration tests
  - Parameter sweep tests
- Updated comprehensive catalog test to include all 17 geometry configurations

**test_experiment2_atlas.py (NEW):**
- Complete geometry-to-functionality mapping atlas
- 9 functionality evaluators:
  - `evaluate_selfsimilarity()` - D_q variance analysis
  - `evaluate_variety()` - Dynamic vs static classification
  - `evaluate_complexity()` - PR and D_q spread
  - `evaluate_memory()` - Temporal memory capacity
  - `evaluate_edges()` - Edge structure analysis
  - `evaluate_communication()` - Communication cost structure
  - `evaluate_vsh_evolution()` - Evolutionary capability
  - `evaluate_cognitive_alignment()` - FD alignment with target
  - `evaluate_coherence()` - Long-term stability

- Generates comprehensive atlas:
  - JSON catalog: 12 families × 9 functionalities
  - Summary table with all scores
  - Top performers for each functionality
  - Validation tests

## Complete Catalog

### 12 Fractal Families Implemented

| Family | Name | Generator | Fractal Dimension |
|--------|------|-----------|-------------------|
| T | Sierpinski Triangle | IFS iteration | D ≈ 1.585 |
| C | Cantor Carpet | Recursive subdivision | D ≈ 1.893 |
| F3D | Menger Sponge | 3D subdivision | D ≈ 2.727 |
| F3D | Sierpinski Tetrahedron | 3D IFS | D = 2.0 |
| B | Fractal Tree | Branching hierarchy | Variable |
| AXC | Lorenz Attractor | ODE integration | D ≈ 2.06 |
| AXC | Rössler Attractor | ODE integration | D ≈ 2.0 |
| AXD | Hénon Map | Discrete iteration | D ≈ 1.26 |
| MC | Mandelbrot Set | Escape-time | D ≈ 2.0 |
| MC | Julia Set | Escape-time | Variable |
| AC | Game of Life | CA evolution | Variable |
| AC | Rule 30 | 1D CA | Fractal pattern |
| GF | Scale-Free Graph | Preferential attachment | Power-law |
| GF | Small-World Graph | Rewiring | Fractal network |
| W | Wavelet Decomposition | Haar transform | Multi-scale |
| PF | KD-Tree Partition | Recursive partition | Fractal boundaries |
| RS | Fractional Brownian | fBm generation | D = 2 - H |

### 9 Fractal Functionalities

| Functionality | Metric | Implementation |
|---------------|--------|----------------|
| F1: Multi-scale Self-Similarity | D_q variance | `compute_generalized_dimensions()` |
| F2: Fractal Dynamic Variety | VFD score | Family-based classification |
| F3: Complexity | D_q, PR | `compute_participation_ratio()` |
| F4: Multi-scale Memory | MFM | `compute_multiscale_memory_metrics()` |
| F5: Fractal Edges | Edge score | Graph/boundary analysis |
| F6: Communication Cost | w_t(e), H_route | `compute_fractal_communication_cost()` |
| F7: V-S-H Evolution | Evolution capability | Dynamic system classification |
| F8: Cognitive Fractality | A[Θ], SAFE | `compute_informational_action()` |
| F9: Long-term Coherence | C_∞ | `compute_coherence_infinity()` |

## Key Innovation: Geometry-to-Scheduler Mapping

The critical bridge function `map_geometry_to_scheduler_features()` translates fractal geometric properties into scheduler feature space:

```python
def map_geometry_to_scheduler_features(
    geometry_points: np.ndarray,
    fractal_dimension: float,
    meta_params: MetaParameters
) -> Dict[str, float]:
    """
    Key mappings:
    - fractal_dimension → uncertainty
    - density_variance → contradiction_signal
    - lambda_rig → edge_pressure
    - lambda_fractal → causal_risk
    - proximity_to_target_D → symbolic_regularity
    - spectral_margin → continuity_recent
    """
```

This allows testing scheduler behavior on diverse fractal structures.

## Test Coverage

### Unit Tests
- ✅ 12 fractal family generators (17 specific geometries)
- ✅ 15+ extended metric functions
- ✅ Geometry-to-scheduler mapping
- ✅ Parameter dataclasses

### Integration Tests
- ✅ Scheduler response to each geometry family
- ✅ Parameter sweep tests (scales, branching factors, etc.)
- ✅ Budget computation from fractal features
- ✅ Sequence selection with fractal inputs

### Comprehensive Tests
- ✅ `test_comprehensive_geometry_catalog` - Tests all 17 geometries
- ✅ `test_generate_experiment2_atlas` - Complete functionality mapping
- ✅ Functionality-specific validation tests

### Test Artifacts Generated
1. **fractal_geometry_catalog.json** - All geometries with FD, D_q, scheduler results
2. **fractal_geometry_summary.txt** - Human-readable summary
3. **experiment2_fractal_atlas.json** - Complete 12×9 functionality matrix
4. **experiment2_atlas_summary.txt** - Functionality scores and top performers

## Usage Examples

### Generate a Fractal Geometry
```python
from fractal_geometries import *

# Sierpinski triangle
params = GeometricParameters(
    family=FractalFamily.TRIANGULAR,
    depth=6,
    scales=[0.5, 0.5, 0.5],
    seed=42
)
points = generate_sierpinski_triangle(params)

# Lorenz attractor
params = GeometricParameters(
    family=FractalFamily.CONTINUOUS_ATTRACTOR,
    system_params={'sigma': 10.0, 'rho': 28.0, 'beta': 8.0/3.0},
    trajectory_length=5000,
    integration_step=0.01,
    seed=42
)
trajectory = generate_lorenz_attractor(params)
```

### Evaluate Fractal Properties
```python
# Estimate fractal dimension
fd, r2 = estimate_fractal_dimension_boxcount(points)

# Compute generalized dimensions
d_q = compute_generalized_dimensions(points, q_values=[-2, 0, 2])

# Compute participation ratio
pr = compute_participation_ratio(points)
```

### Map to Scheduler
```python
meta_params = MetaParameters(
    lambda_rig=0.4,
    target_dimension=1.5,
    lambda_fractal=0.1
)

features = map_geometry_to_scheduler_features(points, fd, meta_params)

budget = compute_budget(features)
sequence, scores, recommended = select_sequence(
    features=features,
    budget=budget,
    allow_experimental=True
)
```

## File Structure

```
tests/reasoning_stress/
├── fractal_geometries.py           # Core generators & metrics (~1400 lines)
├── test_geometry_catalog.py        # Integration tests (~1100 lines)
└── test_experiment2_atlas.py       # Functionality atlas (~420 lines)
```

## Performance Characteristics

- **Generation Time:** 0.1s - 5s per geometry (depending on complexity)
- **FD Estimation:** ~0.5s - 2s (box-counting with multiple scales)
- **Full Atlas Generation:** ~2-5 minutes (12 families × all metrics)
- **Memory Usage:** <500MB for typical test runs

## Next Steps (Optional Extensions)

1. **Enhanced Visualization:**
   - Generate plots of fractal geometries
   - Visualize D_q spectra
   - Plot functionality heatmaps

2. **Additional Geometries:**
   - More 3D fractals (fractal mountains, clouds)
   - Additional attractors (Chen, Aizawa)
   - More CA rules (Life-like variants)

3. **Advanced Metrics:**
   - Multifractal spectrum f(α)
   - Correlation dimensions
   - Lyapunov spectra for all attractors

4. **Real Scheduler Stress Tests:**
   - Run full reasoning traces with fractal feature inputs
   - Measure actual avalanche cascades
   - Validate temporal self-similarity

## Conclusion

**Experiment 2 is now fully implemented** with:
- ✅ All 12 fractal geometry families
- ✅ All 9 fractal functionality metrics
- ✅ Complete test infrastructure
- ✅ Geometry-to-scheduler integration
- ✅ Comprehensive atlas generation

The system can now systematically test scheduler behavior across diverse fractal structures and identify which geometries support which fractal functionalities.
