# External Reasoner Gain Report

- output_dir: `data/benchmarks/external_reasoner_gain/conflict-repeatability-gated-v1-4x4`
- dictamen: `external_reasoner_aporta_solo_estructura`
- microbenchmark_verdict: `sin_ganancia`
- gating_verdict: `gated_external_reasoner_util_condicionado`

| profile | regime | episodes | call_rate | ivc_r | precision | viability | success | closure | ext_ok | schema | guard_pass | guard_reject | changed | latency | ext_tps | contribution |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| core_only | causal_counterfactual_conflict | 16 | 0.000 | 0.000691 | -0.001818 | -0.031600 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000000 |
| core_plus_external_reasoner_gated_v1 | causal_counterfactual_conflict | 16 | 1.000 | 0.231401 | 0.067784 | 0.029650 | 0.875 | 0.875 | 1.000 | 1.000 | 0.875 | 0.125 | 0.875 | 96.115 | 44.138 | 0.127542 |

## Deltas Vs Core

| profile | regime | delta_ivc_r | delta_precision | delta_viability | closure | ext_ok |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| core_plus_external_reasoner_gated_v1 | causal_counterfactual_conflict | 0.230710 | 0.069602 | 0.061250 | 0.875 | 1.000 |
