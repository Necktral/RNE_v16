from __future__ import annotations

import json
from pathlib import Path

from runtime.reasoning.external_models.gating import (
    ExternalReasonerGate,
    ExternalReasonerGateInput,
)
from scripts.benchmark_external_reasoner_gain import (
    guard_external_choice,
    run_campaign,
    run_episode,
)


class FakeExternalClient:
    def __init__(self):
        self.calls = 0

    def generate(self, prompt, **kwargs):
        self.calls += 1
        payload = {
            "claim": "cooling improves the counterfactual transition",
            "reasoning_summary": "activate_cooling lowers global_temp_mean",
            "candidate_hypotheses": ["cooling reduces thermal load"],
            "causal_assumptions": ["lower global_temp_mean is better"],
            "counterfactual_checks": ["activation is cooler than idle"],
            "confidence_proxy": 0.9,
            "recommended_intervention": "activate_cooling",
        }
        return {
            "ok": True,
            "backend": "cuda",
            "output_text": json.dumps(payload),
            "latency_s": 0.001,
            "generation_tps": 50.0,
        }


def test_gate_calls_only_explicit_causal_counterfactual_conflict() -> None:
    gate = ExternalReasonerGate()
    decision = gate.evaluate(
        ExternalReasonerGateInput(
            regime="causal_counterfactual_conflict",
            core_intervention="deactivate_cooling",
            core_metrics={
                "intervention_precision": -0.01,
                "viability_margin": -0.03,
                "closure_stable": False,
            },
        )
    )
    assert decision.considered is True
    assert decision.called is True
    assert decision.reason == "causal_counterfactual_conflict"


def test_gate_skips_stable_non_conflict_core() -> None:
    gate = ExternalReasonerGate()
    decision = gate.evaluate(
        ExternalReasonerGateInput(
            regime="viability_edge",
            core_intervention="activate_cooling",
            core_metrics={
                "intervention_precision": 0.04,
                "viability_margin": 0.02,
                "closure_stable": True,
            },
        )
    )
    assert decision.considered is True
    assert decision.called is False
    assert decision.skip_reason == "core_stable_no_conflict"


def test_guard_rejects_basic_regressions() -> None:
    decision = guard_external_choice(
        allowed_interventions=["activate_cooling", "deactivate_cooling"],
        core_intervention="activate_cooling",
        recommended_intervention="deactivate_cooling",
        core_metrics={
            "viability_margin": 0.10,
            "intervention_precision": 0.08,
            "closure_stable": True,
        },
        candidate_metrics={
            "viability_margin": -0.02,
            "intervention_precision": -0.01,
            "closure_stable": False,
        },
    )
    assert decision.accepted is False
    assert decision.reason == "viability_regression"


def test_guard_rejects_low_confidence_before_influence() -> None:
    decision = guard_external_choice(
        allowed_interventions=["activate_cooling", "deactivate_cooling"],
        core_intervention="deactivate_cooling",
        recommended_intervention="activate_cooling",
        core_metrics={"viability_margin": -0.03, "intervention_precision": -0.01},
        candidate_metrics={"viability_margin": 0.04, "intervention_precision": 0.08},
        external_ok=True,
        schema_validated=True,
        confidence_proxy=0.2,
        confidence_threshold=0.55,
    )
    assert decision.accepted is False
    assert decision.reason == "confidence_below_threshold"


def test_guard_rejects_claim_recommendation_contradiction() -> None:
    decision = guard_external_choice(
        allowed_interventions=["activate_cooling", "deactivate_cooling"],
        core_intervention="deactivate_cooling",
        recommended_intervention="deactivate_cooling",
        core_metrics={"viability_margin": -0.03, "intervention_precision": -0.01},
        candidate_metrics={"viability_margin": -0.03, "intervention_precision": -0.01},
        external_ok=True,
        schema_validated=True,
        confidence_proxy=0.8,
        claim="Activate cooling is better supported",
        reasoning_summary="Activating cooling would lower temperature",
    )
    assert decision.accepted is False
    assert decision.reason == "text_recommendation_contradiction"


def test_lab_episode_can_measure_external_decision_gain() -> None:
    core = run_episode(
        profile="core_only",
        regime="causal_counterfactual_conflict",
        episode_index=0,
        external_input=0.04,
    )
    external = run_episode(
        profile="core_plus_external_reasoner_gated_v1",
        regime="causal_counterfactual_conflict",
        episode_index=0,
        external_input=0.04,
        external_client=FakeExternalClient(),
    )
    assert core["selected_intervention"] == "deactivate_cooling"
    assert external["selected_intervention"] == "activate_cooling"
    assert external["external_reasoner_ok"] is True
    assert external["external_reasoner_schema_validated"] is True
    assert external["external_accepted"] is True
    assert external["intervention_precision"] > core["intervention_precision"]
    assert external["external_reasoner_contribution_proxy"] > 0.0


def test_gated_profile_skips_external_call_for_unvalidated_regime() -> None:
    client = FakeExternalClient()
    row = run_episode(
        profile="core_plus_external_reasoner_gated_v1",
        regime="viability_edge",
        episode_index=0,
        external_input=0.04,
        external_client=client,
    )
    assert client.calls == 0
    assert row["external_reasoner_considered"] is True
    assert row["external_reasoner_called"] is False
    assert row["external_reasoner_skip_reason"] == "external_reasoner_regime_not_validated"
    assert row["selected_intervention"] == row["core_intervention"]


def test_gated_profile_calls_external_for_conflict_regime() -> None:
    client = FakeExternalClient()
    row = run_episode(
        profile="core_plus_external_reasoner_gated_v1",
        regime="causal_counterfactual_conflict",
        episode_index=0,
        external_input=0.04,
        external_client=client,
    )
    assert client.calls == 1
    assert row["external_reasoner_called"] is True
    assert row["external_reasoner_gate_reason"] == "causal_counterfactual_conflict"
    assert row["external_accepted"] is True


def test_legacy_external_profiles_are_rejected() -> None:
    try:
        run_episode(
            profile="core_plus_external_reasoner_guarded",
            regime="causal_counterfactual_conflict",
            episode_index=0,
            external_input=0.04,
            external_client=FakeExternalClient(),
        )
    except ValueError as exc:
        assert "external_reasoner_profile_not_admitted" in str(exc)
    else:
        raise AssertionError("legacy external profile should be rejected")


def test_campaign_writes_required_artifacts(tmp_path: Path) -> None:
    summary = run_campaign(
        campaign_id="unit",
        output_root=tmp_path,
        episodes=1,
        external_input=0.04,
        backend=None,
        allow_cpu_fallback=False,
        external_client=FakeExternalClient(),
    )
    out_dir = tmp_path / "unit"
    assert summary["dictamen"] == "external_reasoner_aporta_ganancia_cognitiva"
    assert (out_dir / "episodes.jsonl").exists()
    assert (out_dir / "summary.json").exists()
    assert (out_dir / "external_reasoner_gain_report.md").exists()
    assert (out_dir / "external_reasoner_gating_report.md").exists()
    assert (out_dir / "external_reasoner_gating_verdict.json").exists()
