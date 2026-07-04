# Campaña Fuerte de Validación Arquitectónica

- campaign_id: `archval_20260420_strong`
- duración_total_s: `1290.80`
- db: `aeon_event_log.db`

## A. Estado del Gate A
| Test | Passed | Duración (s) | Log |
|---|---:|---:|---|
| `tests/benchmarks/test_metrics_consistency.py` | `True` | `0.62` | `data/benchmarks/arch_validation/archval_20260420_strong/logs/gateA_contrato.log` |
| `tests/benchmarks/test_benchmark_pipeline_resilience.py` | `True` | `1.72` | `data/benchmarks/arch_validation/archval_20260420_strong/logs/gateA_persistencia_analisis.log` |
| `tests/world/test_grid_thermal_scenario.py` | `True` | `0.80` | `data/benchmarks/arch_validation/archval_20260420_strong/logs/gateA_world_semantics.log` |
| `tests/integration/test_grid_5x5_episode.py` | `True` | `9.95` | `data/benchmarks/arch_validation/archval_20260420_strong/logs/gateA_integration_5x5.log` |
| `tests/benchmarks/test_1x1_vs_5x5_benchmark.py` | `True` | `20.50` | `data/benchmarks/arch_validation/archval_20260420_strong/logs/gateA_benchmark_sanity.log` |
- Gate A passed: `True`

## B. Resultados de la Matriz A
| Nivel | Net Gain | Precision Δ% | Success Δpp | Wall Ratio | Artifact Ratio |
|---|---:|---:|---:|---:|---:|
| SAFE | -0.0364 | -96.00 | 0.00 | 1.038 | 2.191 |
| ELEVATED | -0.0317 | -96.00 | 0.00 | 1.007 | 2.195 |
| WARNING | -0.0344 | -96.00 | 0.00 | 1.026 | 2.185 |
| CRITICAL | 0.0317 | 128.00 | 0.00 | 1.021 | 2.217 |
- Gate B passed: `True`

## C. Resultados de la Matriz B
- topologías_con_señal: `[]`
- topologías_sin_señal: `['checkerboard', 'gradient_ew', 'gradient_ns', 'hotspot_center']`

## D. Resultados de la Matriz C
| Condición | Contraste | Net Gain | Top Failure (right) |
|---|---|---:|---|
| tight_margin | uniform_vs_1x1 | -0.0331 | None |
| tight_margin | hotspot_vs_1x1 | 0.3805 | None |
| weak_cooling | uniform_vs_1x1 | -0.0364 | None |
| weak_cooling | hotspot_vs_1x1 | 0.3326 | None |
| near_alarm | uniform_vs_1x1 | -0.0197 | None |
| near_alarm | hotspot_vs_1x1 | 0.3676 | None |
| high_temp_tight | uniform_vs_1x1 | 0.3138 | None |
| high_temp_tight | hotspot_vs_1x1 | 0.3215 | None |

## E. Resultados de la Matriz D
| Cell | Source Record | Source Gain | Repeat Gain | Stable |
|---|---|---:|---:|---:|
| D-cell-00 | B-SAFE-hotspot_center-vs-uniform | 0.4284 | 0.4068 | True |
| D-cell-01 | B-ELEVATED-gradient_ns-vs-uniform | 0.4040 | 0.4227 | True |
| D-cell-02 | B-WARNING-gradient_ns-vs-uniform | 0.4031 | 0.4265 | True |
| D-cell-03 | B-CRITICAL-hotspot_center-vs-uniform | 0.0013 | 0.3961 | True |
| D-cell-04 | B-CRITICAL-checkerboard-vs-uniform | -0.0034 | 0.4141 | True |
| D-cell-05 | B-SAFE-gradient_ns-vs-uniform | -0.0048 | 0.4155 | True |
| D-cell-06 | B-SAFE-gradient_ew-vs-uniform | -0.0054 | 0.4289 | True |
| D-cell-07 | C-weak_cooling-uniform-vs-1x1 | -0.0364 | -0.0341 | True |
| D-cell-08 | A-SAFE | -0.0364 | -0.0356 | True |
| D-cell-09 | A-WARNING | -0.0344 | -0.0344 | True |
- Gate D passed: `True` (stable=10/10)

## F. Dictamen final
- `avance parcial con señal condicionada`

## G. Riesgos residuales
- semánticos: success_rate/viability_margin siguen siendo proxies explícitos.
- numéricos: robustez validada en escenarios degenerados, mantener vigilancia en campañas mayores.
- metodológicos: señal puede ser local por topología/nivel; dictamen depende de repetibilidad.
