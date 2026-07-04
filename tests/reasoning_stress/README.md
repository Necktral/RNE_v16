# Elite Reasoning Stress Test Suite

## Overview

This test suite implements **elite-level stress testing** for the dynamic reasoning scheduler. It goes far beyond basic unit tests to comprehensively map the reasoning space, identify failure boundaries, measure sensitivities, and validate robustness under extreme conditions.

## Motivation

The reasoning scheduler is a **dynamic system** with contextual thresholds, not a static taxonomy of difficulty levels. The system defines a **surface of activation and budgeting** that responds to multiple interacting signals:

- `uncertainty`, `contradiction_signal`, `causal_risk`, `edge_pressure`
- `symbolic_regularity`, `law_fit_signal`, `continuity_recent`

Simple threshold tests are insufficient. We need to map the **complete behavioral landscape**.

## Test Categories

### 1. Boundary Sweep Tests (`test_boundary_sweep.py`)

**Purpose**: Map activation frontiers with high precision.

**Methodology**:
- Fine-grained sweeps around known thresholds (±0.05, 21 steps)
- Measure activation/deactivation points
- Detect discontinuities and stability

**Key Tests**:
- `test_edge_pressure_heur_activation_boundary()`: HEUR activation at 0.7
- `test_contradiction_dia_adv_activation_boundary()`: DIA_ADV at 0.45
- `test_symbolic_regularity_eml_sr_activation()`: EML_SR at 0.4
- `test_all_thresholds_have_clean_boundaries()`: Meta-test for all thresholds

**Metrics**:
- Activation point precision
- Boundary stability
- Discontinuity count

### 2. Pairwise Interaction Tests (`test_pairwise_interaction.py`)

**Purpose**: Discover non-linear interactions between feature pairs.

**Methodology**:
- Test all major feature pairs across grids of values
- Detect overactivation, cancellation, and coupling
- Measure nonlinearity

**Key Tests**:
- `test_contradiction_edge_pressure_interaction()`: Conflict resolution
- `test_contradiction_uncertainty_risk_budget_amplification()`: Synergy
- `test_symbolic_law_fit_interaction()`: OR-gate activation
- `test_interaction_matrix_all_pairs()`: Comprehensive pairwise analysis

**Metrics**:
- Overactivation count
- Cancellation count
- Nonlinearity score (0-1)

### 3. Hypercube Sampling Tests (`test_hypercube_sampling.py`)

**Purpose**: Map the full 7-dimensional feature space systematically.

**Methodology**:
- Latin Hypercube Sampling (LHS) for efficient coverage
- Classify points into behavioral regions
- Build empirical atlas of reasoning space

**Key Tests**:
- `test_hypercube_basic_mapping()`: Region coverage
- `test_hypercube_no_widespread_chaos()`: Chaos rate < 5%
- `test_hypercube_crisis_handling()`: Extreme condition stability
- `test_extreme_corners_of_hypercube()`: 2^7 corner cases

**Regions Identified**:
- `baseline`: Low signals, minimal sequence
- `high_complexity`: High uncertainty/contradiction
- `resource_constrained`: High edge_pressure
- `dialectical`: Contradiction guards active
- `symbolic`: EML_SR active
- `crisis`: Multiple high signals
- `chaotic`: Pathological behavior (should be rare)
- `stable`: Normal operation

### 4. Adversarial Threshold Tests (`test_adversarial_thresholds.py`)

**Purpose**: Break the scheduler with deceptive and extreme inputs.

**Methodology**:
- Test just-below and just-above thresholds
- Conflicting policy signals
- Spurious high signals
- Budget manipulation

**Key Tests**:
- `test_adversarial_just_below_threshold_high_noise()`: False positive resistance
- `test_adversarial_extreme_edge_pressure_doesnt_destroy()`: Survival under pressure
- `test_adversarial_conflicting_budget_signals()`: Coherent conflict resolution
- `test_adversarial_determinism_under_stress()`: Reproducibility

**Adversarial Scenarios**:
- High noise below threshold
- Spurious symbolic regularity
- Conflicting budget signals
- Floating-point precision attacks
- Multiple-just-below/above thresholds

### 5. Temporal Hysteresis Tests (`test_temporal_hysteresis.py`)

**Purpose**: Measure temporal dynamics and memory effects.

**Methodology**:
- Sweep features up and down
- Measure activation vs deactivation points
- Detect hysteresis width
- Test oscillation resistance

**Key Tests**:
- `test_edge_pressure_heur_hysteresis()`: Symmetric activation
- `test_contradiction_both_guards_symmetric()`: DIA_ADV/FAL_GUARD sync
- `test_no_oscillation_near_threshold()`: Stability near boundaries
- `test_bidirectional_consistency()`: Up/down path equivalence

**Metrics**:
- Hysteresis width (should be < 0.05)
- Oscillation count (should be minimal)
- Transition smoothness

### 6. Family Contribution Tests (`test_family_contribution.py`)

**Purpose**: Validate that each family provides measurable value.

**Methodology**:
- Measure activation rates across contexts
- Compare with/without optional families
- Validate cost-effectiveness
- Measure stability impact

**Key Tests**:
- `test_core_families_always_active()`: Core presence
- `test_heur_contribution_under_pressure()`: Context-appropriate activation
- `test_dia_adv_fal_guard_contribution()`: Synchronized activation
- `test_all_families_contribute_somewhere()`: Universal coverage

**Family Profiles**:
- Core families: Always present
- Operational families: Context-dependent, proven utility
- Experimental families: Conditional activation, shadow mode

### 7. Comprehensive Atlas (`test_atlas_comprehensive.py`)

**Purpose**: Build complete empirical atlas of reasoning space.

**Components**:

#### Activation Frontiers
- Measured thresholds with confidence bounds
- Stability scores
- Boundary equations

#### Sensitivity Maps
- Local sensitivity coefficients
- Discontinuity points
- Smooth/chaotic regions

#### Saturation Regions
- Crisis handling
- Resource constraints
- Scaling quality

#### Family Functional Profiles
- Status: core/operational/experimental/shadow
- Activation conditions
- Marginal utility
- No-harm score
- Stability score

**Atlas Output**:
- JSON serialization
- Human-readable report
- Quality thresholds validation

## Running the Tests

### Run all reasoning stress tests:
```bash
pytest tests/reasoning_stress/ -v
```

### Run specific category:
```bash
pytest tests/reasoning_stress/test_boundary_sweep.py -v
pytest tests/reasoning_stress/test_adversarial_thresholds.py -v
```

### Run with coverage:
```bash
pytest tests/reasoning_stress/ --cov=runtime.reasoning.scheduler_meta
```

### Generate atlas:
```bash
pytest tests/reasoning_stress/test_atlas_comprehensive.py::test_build_complete_atlas -v
```

## Quality Thresholds

The test suite enforces these quality requirements:

### Stability
- Global stability score ≥ 0.7
- Individual frontier stability ≥ 0.6
- Hysteresis width < 0.05

### Coverage
- Global coverage score ≥ 0.8
- All families activate somewhere
- All thresholds have clean boundaries

### Robustness
- Chaos rate < 5%
- No pathological interactions
- Deterministic under stress

### Precision
- Threshold precision ±0.02
- Boundary discontinuities ≤ 3
- Oscillation transitions ≤ 4

## Interpreting Results

### Successful Run
```
✓ All thresholds stable
✓ No widespread chaos
✓ Families contribute appropriately
✓ Atlas quality thresholds met
```

### Warning Signs
```
⚠ Hysteresis width > 0.05 → Unstable activation
⚠ Chaos rate > 5% → Fragile regions
⚠ Excessive oscillation → Threshold instability
⚠ Overactivation detected → Policy conflict
```

### Failure Modes
```
✗ Core families missing → Broken mandatory sequence
✗ Determinism violated → Non-reproducible behavior
✗ Budget exceeded → Hard limit violation
✗ Chaotic expansion → Loss of discipline
```

## Advanced Analysis

### Generate Full Atlas Report
```python
from tests.reasoning_stress.test_atlas_comprehensive import build_complete_atlas

atlas = build_complete_atlas()
print(atlas.get_summary_report())
atlas.save("reasoning_atlas.json")
```

### Custom Hypercube Analysis
```python
from tests.reasoning_stress.test_hypercube_sampling import map_hypercube_region

atlas = map_hypercube_region(n_samples=500, seed=42)
stats = atlas.get_region_statistics()
```

### Measure Specific Interaction
```python
from tests.reasoning_stress.test_pairwise_interaction import test_interaction_grid

result = test_interaction_grid(
    feature1="contradiction_signal",
    feature2="uncertainty",
    values1=[0.0, 0.5, 1.0],
    values2=[0.0, 0.5, 1.0]
)

print(f"Nonlinearity: {result.measure_nonlinearity():.3f}")
```

## Architecture Principles

These tests embody elite testing principles:

1. **Map, Don't Just Verify**: Build empirical map of behavior space
2. **Stress, Don't Just Validate**: Test at limits, not just happy paths
3. **Measure, Don't Just Assert**: Quantify quality metrics
4. **Explore, Don't Just Execute**: Discover unknown failure modes
5. **Document, Don't Just Test**: Generate actionable atlas

## Contributing

When adding new tests:

1. Follow the established pattern (Result class + analysis)
2. Add to appropriate category or create new one
3. Update atlas generation if adding family/feature
4. Document in this README
5. Ensure quality thresholds are validated

## Future Enhancements

Planned improvements:

- [ ] Temporal sequence analysis (multi-episode trajectories)
- [ ] Adaptive threshold learning from atlas
- [ ] Real-time monitoring dashboard
- [ ] Comparative analysis across versions
- [ ] Automated regression detection

## References

This test suite implements concepts from:

- Latin Hypercube Sampling (McKay et al., 1979)
- Sensitivity Analysis (Saltelli et al., 2008)
- Adversarial Testing (Goodfellow et al., 2014)
- Hysteresis Analysis (Mayergoyz, 2003)

## License

Part of RNFE-v15 reasoning system.
