from runtime.reasoning.scheduler_meta.meta_scheduler import MetaScheduler


def _contexts():
    return [
        {
            "run_id": "run-adapt-1",
            "uncertainty": 0.8,
            "contradiction_signal": 0.75,
            "counterfactual_gap": 0.7,
            "edge_pressure": 0.2,
        },
        {
            "run_id": "run-adapt-2",
            "uncertainty": 0.3,
            "contradiction_signal": 0.1,
            "counterfactual_gap": 0.2,
            "edge_pressure": 0.6,
        },
        {
            "run_id": "run-adapt-3",
            "uncertainty": 0.7,
            "contradiction_signal": 0.55,
            "counterfactual_gap": 0.4,
            "edge_pressure": 0.4,
        },
        {
            "run_id": "run-adapt-4",
            "uncertainty": 0.2,
            "contradiction_signal": 0.0,
            "counterfactual_gap": 0.1,
            "edge_pressure": 0.9,
        },
    ]


def _run_batch(mode: str):
    scheduler = MetaScheduler(mode=mode)
    outputs = []
    for context in _contexts():
        outputs.append((context, scheduler.run(dict(context))))
    return outputs


def test_meta_scheduler_adaptive_beats_fixed_in_correction_rate():
    fixed_outputs = _run_batch("fixed")
    adaptive_outputs = _run_batch("adaptive")

    fixed_success = 0
    adaptive_success = 0
    fixed_corrections = 0
    adaptive_corrections = 0
    contradicted_cases = 0

    for (ctx_f, out_fixed), (ctx_a, out_adaptive) in zip(fixed_outputs, adaptive_outputs):
        assert ctx_f["run_id"] == ctx_a["run_id"]
        if out_fixed["state"].get("prob_calibrated"):
            fixed_success += 1
        if out_adaptive["state"].get("prob_calibrated"):
            adaptive_success += 1

        if ctx_f["contradiction_signal"] >= 0.5:
            contradicted_cases += 1
            fixed_corrections += int("fallacy_risk" in out_fixed["state"])
            adaptive_corrections += int("fallacy_risk" in out_adaptive["state"])

    success_rate_fixed = fixed_success / len(fixed_outputs)
    success_rate_adaptive = adaptive_success / len(adaptive_outputs)
    correction_rate_fixed = fixed_corrections / max(1, contradicted_cases)
    correction_rate_adaptive = adaptive_corrections / max(1, contradicted_cases)

    assert success_rate_adaptive >= success_rate_fixed
    assert correction_rate_adaptive >= correction_rate_fixed + 0.05


def test_meta_scheduler_adaptive_trace_contains_selection_metadata():
    scheduler = MetaScheduler(mode="adaptive")
    result = scheduler.run(
        {
            "run_id": "run-adapt-meta",
            "uncertainty": 0.7,
            "contradiction_signal": 0.8,
            "counterfactual_gap": 0.6,
            "edge_pressure": 0.2,
        }
    )
    assert result["meta_family"] == "META"
    assert result["mode"] == "adaptive"
    assert result["sequence"]
    first = result["trace"][0]
    assert "selection_reason" in first["detail"]
    assert "budget_used" in first["detail"]
    assert "confidence" in first["detail"]
    assert "recommended_next_family" in first["detail"]
    assert "early_stop" in first["detail"]
    assert any(step["family"] == "DIA_ADV" for step in result["trace"])
    assert any(step["family"] == "FAL_GUARD" for step in result["trace"])
