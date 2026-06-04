# External Reasoner Conflict Repeatability Report

- campaign_id: `conflict-repeatability-gated-v1-4x4`
- regime: `causal_counterfactual_conflict`
- profiles: `core_only`, `core_plus_external_reasoner_gated_v1`
- blocks: `4`
- episodes_per_block: `4`
- wall_clock: `1535.74 sec`
- dictamen: `conflict_resolver_repetible`

## Estado Operativo

- llamadas_externas: `16`
- external_reasoner_ok_rate: `1.000`
- schema_validated_rate: `1.000`
- guard_pass_rate: `0.875`
- guard_reject_rate: `0.125`
- latency_mean: `96.115 s`
- latency_p95: `98.953 s`
- generation_tps_mean: `44.138`
- invalid_intervention_accepted: `0`

## Core Vs Gated

| Perfil | ivc_r | ivc_r_std | precision | viability | success | closure | latency | calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| core_only | 0.000691 | 0.000000 | -0.001818 | -0.031600 | 0.000 | 0.000 | 0.000 | 0 |
| core_plus_external_reasoner_gated_v1 | 0.231401 | 0.087386 | 0.067784 | 0.029650 | 0.875 | 0.875 | 96.115 | 16 |

## Deltas

- ivc_r_delta: `0.230710`
- intervention_precision_delta: `0.069602`
- viability_margin_delta: `0.061250`
- success_rate_delta: `0.875000`
- closure_stable_rate_delta: `0.875000`

## Repetibilidad Por Bloque

| Bloque | delta_ivc_r | delta_precision | delta_viability | delta_success | delta_closure | guard_pass | corrected | dictamen |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 0.198022 | 0.059659 | 0.052500 | 0.750 | 0.750 | 0.750 | 3 | bloque_replicado |
| 1 | 0.262732 | 0.079545 | 0.070000 | 1.000 | 1.000 | 1.000 | 4 | bloque_replicado |
| 2 | 0.197833 | 0.059659 | 0.052500 | 0.750 | 0.750 | 0.750 | 3 | bloque_replicado |
| 3 | 0.264253 | 0.079545 | 0.070000 | 1.000 | 1.000 | 1.000 | 4 | bloque_replicado |

## Coste Por Correccion

- llamadas_externas_totales: `16`
- fallos_core_corregidos: `14`
- segundos_por_fallo_corregido: `109.846`

## Stopping Rules

- should_abort: `False`
- violations: `[]`
