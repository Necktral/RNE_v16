"""
Fractal Reasoning Stress Testing Suite
=======================================

This module provides fractal-level stress characterization of the dynamic reasoning scheduler,
measuring multiscale geometric structure to determine whether the system exhibits:

1. **Disciplined behavior**: Clean boundaries, stable scaling, controlled complexity
2. **Critical behavior**: Rich multiscale structure with interesting dynamics
3. **Fragile behavior**: Excessive sensitivity, large cascades, instability
4. **Pathological behavior**: Non-convergent boundaries, fractal instability

## Motivation

The reasoning scheduler uses dynamic budgeting and contextual family activation based on continuous
feature signals. This creates a **dynamic surface** rather than fixed rules. To understand its
behavior limits before recalibration, we need to characterize:

- **Multiscale geometry** of decision boundaries
- **Fractal dimension** of activation frontiers
- **Temporal cascade** patterns
- **Activation avalanche** distributions

## Test Modules

### 1. `test_multiscale_boundary.py`
Measures activation boundaries at multiple resolutions (0.10, 0.05, 0.02, 0.01, 0.005).

**Tests for**:
- Convergence vs divergence as resolution refines
- Boundary roughness exponents
- Scale-dependent hysteresis
- Disciplined vs pathological thresholds

**Key metrics**:
- `converges`: Whether boundary location stabilizes
- `convergence_rate`: How fast it stabilizes
- `roughness_exponent`: Fractal roughness measure
- `discipline`: "disciplined" | "critical" | "pathological"

### 2. `test_box_counting.py`
Estimates fractal dimension of activation frontiers in 2D feature subspaces using box-counting.

**Tests for**:
- Clean boundaries (dimension ~1)
- Rugose but controlled boundaries (1 < dim < 1.35)
- Pathological fragmentation (dim > 1.35)

**Key metrics**:
- `fractal_dimension`: Estimated via log-log regression
- `fit_quality`: R² of power-law fit
- `interpretation`: "clean" | "rugose_controlled" | "pathological"

### 3. `test_temporal_cascade.py`
Measures temporal cascade behavior at multiple time scales (1, 2, 4, 8, 16 steps).

**Tests for**:
- Temporal self-similarity
- Scale-dependent memory effects
- Cascade propagation patterns
- Recovery dynamics

**Key metrics**:
- `self_similar`: Whether patterns preserve form across scales
- `scale_invariance_error`: How much behavior varies with scale
- `temporal_memory_fragility`: Spurious hysteresis effects

### 4. `test_activation_avalanche.py`
Measures distribution of activation avalanche sizes from small perturbations.

**Tests for**:
- Rigid dynamics (no response)
- Interesting critical dynamics (balanced)
- Fragile dynamics (excessive cascades)

**Key metrics**:
- `mean_size`: Average avalanche size
- `is_heavy_tailed`: Whether distribution has power-law tail
- `criticality_indicator`: "rigid" | "interesting" | "fragile"

### 5. `test_fractal_atlas.py`
Integrates all fractal analyses into comprehensive atlas with overall classification.

**Provides**:
- Aggregate fractal metrics
- System-level classification
- Diagnostic report
- JSON export for analysis

## Usage

### Run full fractal stress suite:
```bash
pytest tests/reasoning_stress/test_multiscale_boundary.py -v
pytest tests/reasoning_stress/test_box_counting.py -v
pytest tests/reasoning_stress/test_temporal_cascade.py -v
pytest tests/reasoning_stress/test_activation_avalanche.py -v
pytest tests/reasoning_stress/test_fractal_atlas.py -v
```

### Run specific tests:
```bash
# Test edge_pressure/HEUR boundary convergence
pytest tests/reasoning_stress/test_multiscale_boundary.py::test_edge_pressure_heur_multiscale_convergence -v

# Test fractal dimension of DIA_ADV frontier
pytest tests/reasoning_stress/test_box_counting.py::test_uncertainty_contradiction_dia_adv_frontier -v

# Test temporal self-similarity
pytest tests/reasoning_stress/test_temporal_cascade.py::test_temporal_self_similarity_across_features -v

# Test avalanche criticality
pytest tests/reasoning_stress/test_activation_avalanche.py::test_avalanche_criticality_classification -v
```

### Generate comprehensive fractal atlas:
```bash
pytest tests/reasoning_stress/test_fractal_atlas.py::test_save_fractal_atlas -v -s
```

This produces:
- `fractal_atlas.json`: Complete quantitative data
- `fractal_atlas_summary.txt`: Human-readable report
- Console output with interpretation

## Expected Results

### Healthy Scheduler (Disciplined)
- All boundaries converge (convergence_rate < 0.7)
- Low roughness (roughness_exponent < 0.5)
- Clean fractal dimensions (dim < 1.25)
- Stable temporal dynamics (scale_error < 0.5)
- Controlled avalanches (criticality = "interesting")
- **Classification**: `system_discipline = "disciplined"`

### Rich but Stable (Critical)
- Most boundaries converge
- Moderate roughness (0.3 < roughness < 0.6)
- Some rugose frontiers (1.15 < dim < 1.35)
- Self-similar temporal patterns
- Interesting avalanche distribution
- **Classification**: `system_discipline = "critical"`

### Needs Attention (Fragile/Pathological)
- Non-convergent boundaries
- High roughness (roughness > 0.7)
- Pathological fractal dimensions (dim > 1.35)
- Unstable temporal dynamics (scale_error > 0.7)
- Excessive avalanches (criticality = "fragile")
- **Classification**: `system_discipline = "pathological"` or `"fragile"`

## Interpretation Guide

### Multiscale Boundaries
- **Converges + Low roughness** → Threshold is well-calibrated
- **Converges + Moderate roughness** → Complex but stable
- **Non-convergent** → Threshold may be poorly defined or coupled to other features

### Fractal Dimensions
- **~1.0 in 2D** → Clean, 1D-like boundary (good)
- **1.15-1.35** → Some complexity but controlled (acceptable)
- **>1.4** → Highly fragmented, potential coupling issues (investigate)

### Temporal Cascades
- **Self-similar + Low scale error** → Stable dynamics
- **Non-self-similar** → Scale-dependent behavior (may indicate artifacts)
- **High memory fragility** → Spurious hysteresis effects (review policy logic)

### Avalanche Statistics
- **Rigid** (mean < 0.3) → System too insensitive, may underrespond
- **Interesting** (0.3 < mean < 1.5) → Balanced responsiveness
- **Fragile** (mean > 1.5) → Overly sensitive, may need dampening

## Integration with Existing Suite

This fractal stress layer complements the existing elite stress tests:

1. **Boundary Sweep** (existing) → Identifies thresholds
2. **Multiscale Boundary** (new) → Verifies threshold stability across scales
3. **Pairwise Interaction** (existing) → Tests 2-feature coupling
4. **Box-Counting** (new) → Quantifies frontier complexity
5. **Temporal Hysteresis** (existing) → Tests activation/deactivation symmetry
6. **Temporal Cascade** (new) → Tests multiscale temporal patterns
7. **Hypercube Sampling** (existing) → Maps full 7D space
8. **Activation Avalanche** (new) → Characterizes perturbation response
9. **Atlas Comprehensive** (existing) → Generates behavioral atlas
10. **Fractal Atlas** (new) → Adds fractal characterization layer

## What This Enables

### Before Recalibration
- Identify which thresholds are well-behaved vs problematic
- Understand coupling between features
- Detect regions of instability
- Measure actual vs intended complexity

### During Development
- Verify new features don't introduce pathological boundaries
- Test robustness of policy changes
- Validate budgeting logic scaling

### For Documentation
- Provide quantitative evidence of system stability
- Demonstrate understanding of edge cases
- Support architectural decisions with geometric analysis

## References

The fractal analysis approach is inspired by:
- Critical phenomena in complex systems
- Multiscale analysis in dynamical systems
- Self-organized criticality
- Geometric measure theory

Applied specifically to characterize the reasoning scheduler's behavioral landscape.

---

**Author**: Claude Code (Anthropic)
**Date**: 2026-04-18
**Version**: 1.0.0
**Status**: Production-ready fractal stress characterization
"""
