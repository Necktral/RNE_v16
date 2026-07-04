from runtime.control.msrc import RegimeClassifier, ScaleEstimate, ScalePolicyState


def _estimate(
    *,
    required: float,
    heterogeneity: float,
    epistemic: float,
    risk: float,
    vram_opportunity: float = 0.5,
    vram_pressure: float = 0.5,
    low_margin: float = 0.0,
    proximity: float = 0.0,
    world_level: float = 0.7,
    expected_spatial_complexity: float = 0.0,
    scale_blindspot_bonus: float = 0.0,
):
    return ScaleEstimate(
        required_resolution_score=required,
        heterogeneity_score=heterogeneity,
        epistemic_insufficiency_score=epistemic,
        risk_score=risk,
        operational_pressure_score=0.2,
        vram_headroom=max(0.0, 1.0 - vram_pressure),
        vram_pressure=vram_pressure,
        vram_fragmentation_risk=0.2,
        vram_opportunity_score=vram_opportunity,
        recommended_scale_candidates=["1x1", "5x5"],
        signals={
            "risk_detail": {
                "low_margin": low_margin,
                "proximity_to_threshold": proximity,
                "world_level": world_level,
            },
            "heterogeneity_detail": {
                "expected_spatial_complexity": expected_spatial_complexity,
                "scale_blindspot_bonus": scale_blindspot_bonus,
            },
        },
    )


def test_regime_classifier_marks_viability_edge_when_risk_is_high():
    classifier = RegimeClassifier()
    state = ScalePolicyState(current_scale_id="1x1")
    estimate = _estimate(
        required=0.72,
        heterogeneity=0.35,
        epistemic=0.24,
        risk=0.78,
        low_margin=0.82,
        proximity=0.86,
    )

    classification = classifier.classify(estimate=estimate, state=state)

    assert classification.regime_label == "viability_edge"
    assert classification.regime_confidence >= 0.70


def test_regime_classifier_marks_heterogeneous_when_spatial_signal_is_high():
    classifier = RegimeClassifier()
    state = ScalePolicyState(current_scale_id="1x1")
    estimate = _estimate(
        required=0.44,
        heterogeneity=0.33,
        epistemic=0.25,
        risk=0.28,
        expected_spatial_complexity=0.74,
        scale_blindspot_bonus=0.56,
    )

    classification = classifier.classify(estimate=estimate, state=state)

    assert classification.regime_label == "heterogeneous"
    assert classification.regime_confidence >= 0.33


def test_regime_classifier_marks_homogeneous_when_signals_are_low():
    classifier = RegimeClassifier()
    state = ScalePolicyState(current_scale_id="1x1")
    estimate = _estimate(
        required=0.18,
        heterogeneity=0.05,
        epistemic=0.08,
        risk=0.16,
        expected_spatial_complexity=0.1,
        scale_blindspot_bonus=0.0,
    )

    classification = classifier.classify(estimate=estimate, state=state)

    assert classification.regime_label == "homogeneous"
    assert classification.regime_confidence > 0.55


def test_regime_classifier_prioritizes_viability_edge_over_heterogeneous():
    classifier = RegimeClassifier()
    state = ScalePolicyState(current_scale_id="1x1")
    estimate = _estimate(
        required=0.78,
        heterogeneity=0.72,
        epistemic=0.40,
        risk=0.70,
        low_margin=0.76,
        proximity=0.80,
        expected_spatial_complexity=0.88,
        scale_blindspot_bonus=0.65,
    )

    classification = classifier.classify(estimate=estimate, state=state)

    assert classification.regime_label == "viability_edge"
    assert "risk_score" in classification.regime_evidence
