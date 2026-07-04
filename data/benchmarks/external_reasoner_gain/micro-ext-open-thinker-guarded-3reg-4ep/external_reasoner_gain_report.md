# External Reasoner Gain Report

- output_dir: `data/benchmarks/external_reasoner_gain/micro-ext-open-thinker-guarded-3reg-4ep`
- dictamen: `external_reasoner_aporta_ganancia_cognitiva`
- microbenchmark_verdict: `ganancia_cognitiva_inicial`

| profile | regime | episodes | ivc_r | precision | viability | success | closure | ext_ok | schema | guard_pass | guard_reject | changed | ext_tps | contribution |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| core_only | causal_counterfactual_conflict | 4 | 0.000691 | -0.001818 | -0.031600 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000000 |
| core_only | heterogeneous_warning | 4 | 0.838898 | 0.066004 | 0.023600 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000000 |
| core_only | viability_edge | 4 | 0.742290 | 0.041830 | 0.020400 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000000 |
| core_plus_external_reasoner_guarded | causal_counterfactual_conflict | 4 | 0.263833 | 0.077727 | 0.038400 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 42.600 | 0.145525 |
| core_plus_external_reasoner_guarded | heterogeneous_warning | 4 | 0.277590 | 0.066004 | 0.023600 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 42.000 | -0.252588 |
| core_plus_external_reasoner_guarded | viability_edge | 4 | 0.241491 | 0.041830 | 0.020400 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 44.150 | -0.225360 |

## Deltas Vs Core

| profile | regime | delta_ivc_r | delta_precision | delta_viability | closure | ext_ok |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| core_plus_external_reasoner_guarded | heterogeneous_warning | -0.561307 | 0.000000 | 0.000000 | 1.000 | 1.000 |
| core_plus_external_reasoner_guarded | viability_edge | -0.500799 | 0.000000 | 0.000000 | 1.000 | 1.000 |
| core_plus_external_reasoner_guarded | causal_counterfactual_conflict | 0.263141 | 0.079545 | 0.070000 | 1.000 | 1.000 |
