from runtime.control.msrc import ScaleCatalog, ScaleEstimator


def _uniform_observation(temp: float = 0.75):
    return {
        "world_level": temp,
        "cell_states": [
            {"row": i, "col": j, "temperature": temp}
            for i in range(5)
            for j in range(5)
        ],
        "propositions": ["TEMP_NORMAL"],
    }


def _hotspot_observation(base: float = 0.72, hot: float = 0.95):
    cells = []
    for i in range(5):
        for j in range(5):
            temp = hot if abs(i - 2) <= 1 and abs(j - 2) <= 1 else base
            cells.append({"row": i, "col": j, "temperature": temp})
    return {
        "world_level": sum(c["temperature"] for c in cells) / len(cells),
        "cell_states": cells,
        "propositions": ["TEMP_HIGH", "HOTSPOT_DETECTED"],
    }


def test_estimator_low_heterogeneity_for_uniform_state():
    estimator = ScaleEstimator(catalog=ScaleCatalog.default())
    estimate = estimator.estimate(
        current_scale_id="1x1",
        observation=_uniform_observation(),
        viability_margin=0.4,
        certification_verdict="passed",
        metrics={"wall_time_ms": 120.0, "artifact_size_bytes": 32000},
        vram_snapshot={
            "vram_headroom": 0.6,
            "vram_pressure": 0.4,
            "vram_fragmentation_risk": 0.2,
            "vram_opportunity_score": 0.7,
        },
    )
    assert estimate.heterogeneity_score < 0.2
    assert estimate.required_resolution_score < 0.6


def test_estimator_detects_hotspot_and_recommends_higher_scale():
    estimator = ScaleEstimator(catalog=ScaleCatalog.default())
    estimate = estimator.estimate(
        current_scale_id="1x1",
        observation=_hotspot_observation(),
        viability_margin=0.05,
        certification_verdict="failed",
        metrics={
            "wall_time_ms": 410.0,
            "artifact_size_bytes": 92000,
            "contradiction_signal": 0.7,
            "uncertainty": 0.8,
            "factual_delta": -0.04,
            "counterfactual_delta": 0.09,
        },
        vram_snapshot={
            "vram_headroom": 0.55,
            "vram_pressure": 0.45,
            "vram_fragmentation_risk": 0.15,
            "vram_opportunity_score": 0.82,
        },
    )
    assert estimate.heterogeneity_score > 0.35
    assert estimate.risk_score > 0.5
    assert "5x5" in estimate.recommended_scale_candidates


def test_estimator_exposes_vram_signals_in_output():
    estimator = ScaleEstimator(catalog=ScaleCatalog.default())
    estimate = estimator.estimate(
        current_scale_id="5x5",
        observation=_uniform_observation(),
        viability_margin=0.3,
        certification_verdict="passed",
        metrics={},
        vram_snapshot={
            "vram_headroom": 0.2,
            "vram_pressure": 0.8,
            "vram_fragmentation_risk": 0.4,
            "vram_opportunity_score": 0.3,
        },
    )
    assert estimate.vram_headroom == 0.2
    assert estimate.vram_pressure == 0.8
    assert estimate.vram_fragmentation_risk == 0.4
    assert estimate.vram_opportunity_score == 0.3


def test_estimator_accounts_for_intervention_backfire_and_scale_blindspot():
    estimator = ScaleEstimator(catalog=ScaleCatalog.default())
    base = estimator.estimate(
        current_scale_id="1x1",
        observation=_uniform_observation(temp=0.82),
        viability_margin=0.2,
        certification_verdict="passed",
        metrics={
            "intervention_precision": 0.01,
            "spatial_information_usage": 0.05,
            "expected_spatial_complexity": 0.0,
        },
        vram_snapshot={"vram_headroom": 0.7, "vram_pressure": 0.3, "vram_fragmentation_risk": 0.1, "vram_opportunity_score": 0.8},
    )
    boosted = estimator.estimate(
        current_scale_id="1x1",
        observation=_uniform_observation(temp=0.82),
        viability_margin=0.2,
        certification_verdict="passed",
        metrics={
            "intervention_precision": -0.05,
            "spatial_information_usage": 0.35,
            "expected_spatial_complexity": 0.8,
        },
        vram_snapshot={"vram_headroom": 0.7, "vram_pressure": 0.3, "vram_fragmentation_risk": 0.1, "vram_opportunity_score": 0.8},
    )

    assert boosted.signals["epistemic_detail"]["intervention_backfire"] > 0.0
    assert boosted.signals["epistemic_detail"]["scale_blindspot_bonus"] > 0.0
    assert boosted.heterogeneity_score > base.heterogeneity_score
    assert boosted.epistemic_insufficiency_score > base.epistemic_insufficiency_score


def test_expected_spatial_complexity_prior_raises_heterogeneity_in_1x1():
    estimator = ScaleEstimator(catalog=ScaleCatalog.default())
    low_prior = estimator.estimate(
        current_scale_id="1x1",
        observation=_uniform_observation(temp=0.78),
        viability_margin=0.3,
        certification_verdict="passed",
        metrics={"expected_spatial_complexity": 0.0, "spatial_information_usage": 0.0},
        vram_snapshot={"vram_headroom": 0.6, "vram_pressure": 0.4, "vram_fragmentation_risk": 0.2, "vram_opportunity_score": 0.7},
    )
    high_prior = estimator.estimate(
        current_scale_id="1x1",
        observation=_uniform_observation(temp=0.78),
        viability_margin=0.3,
        certification_verdict="passed",
        metrics={"expected_spatial_complexity": 0.85, "spatial_information_usage": 0.2},
        vram_snapshot={"vram_headroom": 0.6, "vram_pressure": 0.4, "vram_fragmentation_risk": 0.2, "vram_opportunity_score": 0.7},
    )

    assert high_prior.signals["heterogeneity_detail"]["heterogeneity_prior_bonus"] > 0.0
    assert high_prior.heterogeneity_score > low_prior.heterogeneity_score
