# External Reasoner Gain Report

- output_dir: `data/benchmarks/external_reasoner_gain/gating-v1-3reg-4ep`
- dictamen: `external_reasoner_aporta_ganancia_cognitiva`
- microbenchmark_verdict: `sin_ganancia`
- gating_verdict: `gated_external_reasoner_util_condicionado`

| profile | regime | episodes | call_rate | ivc_r | precision | viability | success | closure | ext_ok | schema | guard_pass | guard_reject | changed | latency | ext_tps | contribution |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| core_only | causal_counterfactual_conflict | 4 | 0.000 | 0.000691 | -0.001818 | -0.031600 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000000 |
| core_only | heterogeneous_warning | 4 | 0.000 | 0.838898 | 0.066004 | 0.023600 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000000 |
| core_only | viability_edge | 4 | 0.000 | 0.742290 | 0.041830 | 0.020400 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000000 |
| core_plus_external_reasoner_gated_v1 | causal_counterfactual_conflict | 4 | 1.000 | 0.272617 | 0.077727 | 0.038400 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 71.477 | 43.975 | 0.149478 |
| core_plus_external_reasoner_gated_v1 | heterogeneous_warning | 4 | 0.000 | 0.838898 | 0.066004 | 0.023600 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000000 |
| core_plus_external_reasoner_gated_v1 | viability_edge | 4 | 0.000 | 0.742290 | 0.041830 | 0.020400 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000000 |
| core_plus_external_reasoner_guarded_always_in_target_regimes | causal_counterfactual_conflict | 4 | 1.000 | 0.202261 | 0.057841 | 0.020900 | 0.750 | 0.750 | 1.000 | 1.000 | 0.750 | 0.250 | 0.750 | 71.998 | 45.400 | 0.111040 |
| core_plus_external_reasoner_guarded_always_in_target_regimes | heterogeneous_warning | 4 | 1.000 | 0.275746 | 0.066004 | 0.023600 | 1.000 | 1.000 | 0.750 | 0.750 | 0.750 | 0.250 | 0.000 | 67.940 | 43.050 | -0.253418 |
| core_plus_external_reasoner_guarded_always_in_target_regimes | viability_edge | 4 | 1.000 | 0.245255 | 0.041830 | 0.020400 | 1.000 | 1.000 | 0.750 | 0.750 | 0.750 | 0.250 | 0.000 | 64.520 | 46.650 | -0.223666 |

## Deltas Vs Core

| profile | regime | delta_ivc_r | delta_precision | delta_viability | closure | ext_ok |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| core_plus_external_reasoner_guarded_always_in_target_regimes | heterogeneous_warning | -0.563152 | 0.000000 | 0.000000 | 1.000 | 0.750 |
| core_plus_external_reasoner_guarded_always_in_target_regimes | viability_edge | -0.497035 | 0.000000 | 0.000000 | 1.000 | 0.750 |
| core_plus_external_reasoner_guarded_always_in_target_regimes | causal_counterfactual_conflict | 0.201570 | 0.059659 | 0.052500 | 0.750 | 1.000 |
| core_plus_external_reasoner_gated_v1 | heterogeneous_warning | 0.000000 | 0.000000 | 0.000000 | 1.000 | 0.000 |
| core_plus_external_reasoner_gated_v1 | viability_edge | 0.000000 | 0.000000 | 0.000000 | 1.000 | 0.000 |
| core_plus_external_reasoner_gated_v1 | causal_counterfactual_conflict | 0.271926 | 0.079545 | 0.070000 | 1.000 | 1.000 |
