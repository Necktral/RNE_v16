from __future__ import annotations

import json
from pathlib import Path

import pytest

from runtime.reasoning.scheduler_meta.family_profiles import resolve_family_profile
from scripts.intelligence_campaign_lib import (
    classify_regime_gain,
    compute_family_synergy_delta,
    compute_ioc_proxy_gain,
    family_redundancy_flag,
    render_intelligence_verdict,
)


def test_combined_profiles_resolve_expected_optional_families() -> None:
    assert resolve_family_profile("core_plus_heur_guard").optional_families == ["heur", "fal_guard"]
    assert resolve_family_profile("core_plus_heur_dialectic").optional_families == ["heur", "dia_adv"]
    assert resolve_family_profile("core_plus_guard_dialectic").optional_families == ["fal_guard", "dia_adv"]
    assert resolve_family_profile("core_plus_triple_optional").optional_families == [
        "heur",
        "dia_adv",
        "fal_guard",
    ]


def test_ioc_proxy_gain_renormalizes_when_spatial_is_not_applicable() -> None:
    value = compute_ioc_proxy_gain(
        ivc_r=1.0,
        intervention_precision=0.5,
        viability_margin=0.25,
        spatial_information_usage=None,
        spatial_applicable=False,
    )
    expected = (0.45 * 1.0 + 0.25 * 0.5 + 0.20 * 0.25) / (0.45 + 0.25 + 0.20)
    assert value == pytest.approx(expected)


def test_classify_regime_gain_distinguishes_strong_conditioned_marginal_and_none() -> None:
    strong = classify_regime_gain(
        candidate={
            "closure_stable": True,
            "success_rate": 1.0,
            "closure_break_rate": 0.0,
            "backbone_floor_satisfied_rate": 1.0,
            "regime": "heterogeneous_warning",
        },
        baseline={"closure_stable": True},
        comparison={
            "delta_ivc_r": 0.10,
            "block_median_delta_ivc_r": 0.08,
            "ivc_r_ci_lower": 0.01,
            "ivc_r_ci_upper": 0.12,
            "repeatability_blocks_positive": {"ivc_r": {"rate": 0.75}},
            "secondary_positive_repeatable_metrics": ["intervention_precision"],
        },
    )
    assert strong == "ganancia cognitiva fuerte"

    conditioned = classify_regime_gain(
        candidate={
            "closure_stable": True,
            "success_rate": 1.0,
            "closure_break_rate": 0.0,
            "backbone_floor_satisfied_rate": 1.0,
            "regime": "heterogeneous_warning",
        },
        baseline={"closure_stable": True},
        comparison={
            "delta_ivc_r": 0.03,
            "block_median_delta_ivc_r": 0.01,
            "ivc_r_ci_lower": -0.01,
            "ivc_r_ci_upper": 0.07,
            "repeatability_blocks_positive": {"ivc_r": {"rate": 0.50}},
            "secondary_positive_repeatable_metrics": [],
        },
    )
    assert conditioned == "ganancia cognitiva condicionada"

    marginal = classify_regime_gain(
        candidate={
            "closure_stable": True,
            "success_rate": 1.0,
            "closure_break_rate": 0.0,
            "backbone_floor_satisfied_rate": 1.0,
            "regime": "heterogeneous_warning",
        },
        baseline={"closure_stable": True},
        comparison={
            "delta_ivc_r": 0.01,
            "block_median_delta_ivc_r": 0.0,
            "ivc_r_ci_lower": -0.03,
            "ivc_r_ci_upper": 0.02,
            "repeatability_blocks_positive": {"ivc_r": {"rate": 0.125}},
            "secondary_positive_repeatable_metrics": [],
        },
    )
    assert marginal == "ganancia marginal"

    none = classify_regime_gain(
        candidate={"closure_stable": False, "regime": "heterogeneous_warning"},
        baseline={"closure_stable": True},
        comparison={"delta_ivc_r": 0.2, "repeatability_blocks_positive": {"ivc_r": {"rate": 1.0}}},
    )
    assert none == "sin ganancia"


def test_family_synergy_and_redundancy_helpers() -> None:
    synergy = compute_family_synergy_delta(single_deltas=[0.02, 0.03], combo_delta=0.05)
    assert synergy == pytest.approx(0.02)
    assert family_redundancy_flag(single_delta=0.03, combo_delta=0.031, synergy_delta=0.001, epsilon=0.005)
    assert not family_redundancy_flag(single_delta=0.03, combo_delta=0.05, synergy_delta=0.02, epsilon=0.005)


def test_render_intelligence_verdict_detects_structural_non_cognitive_progress(tmp_path: Path) -> None:
    cognitive_path = tmp_path / "cognitive.json"
    cognitive_path.write_text(
        json.dumps(
            {
                "primary_verdict": "no hay ganancia cognitiva suficiente",
                "strong_regimes": [],
                "conditioned_regimes": [],
                "closure_regressions": [],
            }
        ),
        encoding="utf-8",
    )

    result = render_intelligence_verdict(
        cognitive_verdicts_path=cognitive_path,
        output_root=tmp_path / "reports",
        campaign_id="structural_only",
    )

    assert result["dictamen_principal"] == "hubo mejora estructural pero no cognitiva"
    payload = json.loads(Path(result["intelligence_verdict_path"]).read_text(encoding="utf-8"))
    assert payload["what_really_won"] == ["estructura", "cierre", "disciplina composicional"]
