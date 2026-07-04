# External Reasoner Latency Report

- campaign_id: `latency-gated-v1-causal-4ep-incremental`
- profile: `core_plus_external_reasoner_gated_v1`
- regime: `causal_counterfactual_conflict`
- dictamen: `server_mode_required`
- baseline_latency_mean_s: `96.115`
- baseline_corrected_core_failure_rate: `0.875`

| Variante | max_tokens | ctx | params | latency_mean | ok_rate | schema_rate | guard_pass | corrected_rate | ivc_r | dictamen |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| tokens_256_standard | 256 |  | prompt=standard,no_warmup=true | 80.902 | 1.000 | 1.000 | 1.000 | 1.000 | 0.267619 | fails:latency_reduction_lt_20pct |
| tokens_192_standard | 192 |  | prompt=standard,no_warmup=true | 80.296 | 0.500 | 0.500 | 0.250 | 0.250 | 0.067418 | fails:latency_reduction_lt_20pct,ok_rate_lt_0.95,schema_rate_lt_0.95,guard_pass_lt_0.80,corrected_rate_lt_0.80 |

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
  "best_generation_tps_mean": 48.87499999999999,
  "best_latency_drop_fraction": 0.15828237233522402,
  "best_latency_drop_s": 15.213310217000057,
  "best_latency_mean_s": 80.90168978299994,
  "best_prompt_bytes_mean": 1215.0,
  "best_prompt_tps_mean": 760.3249999999999,
  "dominant_cost_inference": "subprocess_model_load_or_process_start_likely"
}
```

## Best Variant

```json
{
  "call_rate": 1.0,
  "closure": 1.0,
  "corrected_core_failure_rate": 1.0,
  "cost_per_corrected_failure": 80.90168978299994,
  "ctx": null,
  "delta_ivc_r_vs_core_mean": 0.2669274017964599,
  "dictamen": "fails:latency_reduction_lt_20pct",
  "episodes": 4,
  "external_reasoner_ok_rate": 1.0,
  "generation_tps_mean": 48.87499999999999,
  "guard_pass_rate": 1.0,
  "guard_reject_rate": 0.0,
  "invalid_accepted_count": 0,
  "ivc_r": 0.26761862477665993,
  "latency_mean": 80.90168978299994,
  "latency_p95": 99.98947139799975,
  "max_tokens": 256,
  "params": "prompt=standard,no_warmup=true",
  "precision": 0.07772727272727287,
  "prompt_bytes_mean": 1215.0,
  "prompt_tps_mean": 760.3249999999999,
  "schema_validated_rate": 1.0,
  "success": 1.0,
  "variant": "tokens_256_standard",
  "viability": 0.0384000000000001
}
```
