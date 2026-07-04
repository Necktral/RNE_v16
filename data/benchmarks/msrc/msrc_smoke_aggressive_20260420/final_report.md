# MSRC Policy Benchmark Report

- campaign_id: `msrc_smoke_aggressive_20260420`
- total_runs: `80`

## Policy Summary
| Policy | Runs | Mean Success | Mean IVC-R | Mean IntPrecision | Mean Wall(ms) | Mean Artifact(B) | Mean ResCost | Upgrade Rate | Probe Rate | Mean Oscillation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| adaptive_msrc | 16 | 1.0000 | 0.1615 | -0.0247 | 208.02 | 10409.8 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| adaptive_msrc_aggressive | 16 | 1.0000 | 0.1616 | -0.0247 | 206.66 | 10409.9 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| always_1x1 | 16 | 1.0000 | 0.1610 | -0.0247 | 213.97 | 10409.6 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| always_5x5 | 16 | 1.0000 | 0.3496 | 0.0532 | 220.07 | 21366.6 | 2.2000 | 0.0000 | 0.0000 | 0.0000 |
| probe_before_switch | 16 | 1.0000 | 0.1612 | -0.0247 | 213.13 | 10410.2 | 1.0000 | 0.0000 | 0.0000 | 0.0000 |

## Notes
- Los proxies se mantienen explícitos: success_rate y viability_margin.
- MSRC prioriza suficiencia cognitiva y viabilidad antes de costo meta.