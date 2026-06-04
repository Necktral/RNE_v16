# External Reasoner Latency Report

- campaign_id: `latency-gated-v1-causal-4ep-abort-unsafe`
- profile: `core_plus_external_reasoner_gated_v1`
- regime: `causal_counterfactual_conflict`
- dictamen: `latency_optimized_without_cognitive_loss`
- baseline_latency_mean_s: `96.115`
- baseline_corrected_core_failure_rate: `0.875`

| Variante | max_tokens | ctx | params | latency_mean | ok_rate | schema_rate | guard_pass | corrected_rate | ivc_r | dictamen |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| tokens_256_standard | 256 |  | prompt=standard,no_warmup=true | 60.714 | 1.000 | 1.000 | 1.000 | 1.000 | 0.275608 | passes |
| tokens_192_standard | 192 |  | prompt=standard,no_warmup=true | 56.593 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000231 | fails:ok_rate_lt_0.95,schema_rate_lt_0.95,guard_pass_lt_0.80,corrected_rate_lt_0.80,delta_ivc_r_not_positive |
| tokens_128_standard | 128 |  | prompt=standard,no_warmup=true | 61.603 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000229 | fails:ok_rate_lt_0.95,schema_rate_lt_0.95,guard_pass_lt_0.80,corrected_rate_lt_0.80,delta_ivc_r_not_positive |
| tokens_96_standard | 96 |  | prompt=standard,no_warmup=true | 54.086 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000232 | fails:ok_rate_lt_0.95,schema_rate_lt_0.95,guard_pass_lt_0.80,corrected_rate_lt_0.80,delta_ivc_r_not_positive |
| tokens_128_compact_ctx1024 | 128 | 1024 | prompt=compact,no_warmup=true,ctx=1024,batch=128,ubatch=64 | 51.241 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000234 | fails:ok_rate_lt_0.95,schema_rate_lt_0.95,guard_pass_lt_0.80,corrected_rate_lt_0.80,delta_ivc_r_not_positive |
| tokens_96_compact_ctx1024 | 96 | 1024 | prompt=compact,no_warmup=true,ctx=1024,batch=128,ubatch=64 | 56.189 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000232 | fails:ok_rate_lt_0.95,schema_rate_lt_0.95,guard_pass_lt_0.80,corrected_rate_lt_0.80,delta_ivc_r_not_positive |

## Latency Profile

```json
{
  "available_timings": [
    "elapsed",
    "prompt_tps",
    "generation_tps",
    "prompt_bytes"
  ],
  "baseline_latency_mean_s": 96.115,
  "best_generation_tps_mean": 49.275,
  "best_latency_drop_fraction": 0.36831631825157396,
  "best_latency_drop_s": 35.40072292875003,
  "best_latency_mean_s": 60.71427707124997,
  "best_prompt_bytes_mean": 1215.0,
  "best_prompt_tps_mean": 923.325,
  "dominant_cost_inference": "generation_prompt_params_helped"
}
```

## Best Variant

```json
{
  "abort_reason": null,
  "aborted": false,
  "call_rate": 1.0,
  "closure": 1.0,
  "corrected_core_failure_rate": 1.0,
  "cost_per_corrected_failure": 60.71427707124997,
  "ctx": null,
  "delta_ivc_r_vs_core_mean": 0.2749163657078451,
  "dictamen": "passes",
  "episodes": 4,
  "external_reasoner_ok_rate": 1.0,
  "generation_tps_mean": 49.275,
  "guard_pass_rate": 1.0,
  "guard_reject_rate": 0.0,
  "invalid_accepted_count": 0,
  "ivc_r": 0.27560758868804514,
  "latency_mean": 60.71427707124997,
  "latency_p95": 76.73096632500028,
  "max_tokens": 256,
  "params": "prompt=standard,no_warmup=true",
  "precision": 0.07772727272727287,
  "prompt_bytes_mean": 1215.0,
  "prompt_tps_mean": 923.325,
  "schema_validated_rate": 1.0,
  "success": 1.0,
  "variant": "tokens_256_standard",
  "viability": 0.0384000000000001
}
```
