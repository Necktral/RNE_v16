"""Tests para EML secondary judge y ablation lab (RTCME-v2 Program 5b).

Valida:
- EML secondary judgment produces structured signals
- Judgment varies with input conditions
- Ablation lab metrics flow correctly
"""

import pytest

from runtime.reality.ablation_lab import (
    EMLSecondaryJudgment,
    compute_eml_secondary_judgment,
)


# ── EML Secondary Judge tests ────────────────────────────────────────────────

class TestEMLSecondaryJudge:
    def test_default_when_disabled(self):
        """No EML payload → neutral judgment."""
        result = compute_eml_secondary_judgment(
            eml_shadow_payload=None,
            transfer_verdict="local_only",
            relation_kind="support",
        )
        assert isinstance(result, EMLSecondaryJudgment)
        assert result.composite_judgment == 0.5

    def test_default_when_not_enabled(self):
        result = compute_eml_secondary_judgment(
            eml_shadow_payload={"enabled": False},
            transfer_verdict="local_only",
            relation_kind="support",
        )
        assert result.composite_judgment == 0.5

    def test_high_quality_eml(self):
        """High-quality EML → high signals."""
        result = compute_eml_secondary_judgment(
            eml_shadow_payload={
                "enabled": True,
                "top_composite": 0.85,
                "top_candidates": [
                    {"composite_score": 0.85},
                    {"composite_score": 0.83},
                    {"composite_score": 0.82},
                ],
            },
            transfer_verdict="compatible_transfer",
            relation_kind="support",
            morphism_score=0.90,
        )
        assert result.regularity_signature >= 0.80
        assert result.composite_judgment > 0.5

    def test_low_quality_eml(self):
        """Low-quality EML → low signals."""
        result = compute_eml_secondary_judgment(
            eml_shadow_payload={
                "enabled": True,
                "top_composite": 0.15,
                "top_candidates": [
                    {"composite_score": 0.15},
                    {"composite_score": 0.05},
                ],
            },
            transfer_verdict="blocked",
            relation_kind="contradiction",
            morphism_score=0.10,
        )
        assert result.regularity_signature <= 0.20
        assert result.composite_judgment < 0.5

    def test_support_boosts_counterfactual_fit(self):
        """Support relation should boost CF fit."""
        support = compute_eml_secondary_judgment(
            eml_shadow_payload={"enabled": True, "top_composite": 0.70, "top_candidates": []},
            transfer_verdict="local_only",
            relation_kind="support",
        )
        contradiction = compute_eml_secondary_judgment(
            eml_shadow_payload={"enabled": True, "top_composite": 0.70, "top_candidates": []},
            transfer_verdict="local_only",
            relation_kind="contradiction",
        )
        assert support.counterfactual_fit_score >= contradiction.counterfactual_fit_score

    def test_morphism_affects_transfer_consistency(self):
        """Higher morphism → higher transfer consistency."""
        high_m = compute_eml_secondary_judgment(
            eml_shadow_payload={"enabled": True, "top_composite": 0.70, "top_candidates": []},
            transfer_verdict="compatible_transfer",
            relation_kind="support",
            morphism_score=0.90,
        )
        low_m = compute_eml_secondary_judgment(
            eml_shadow_payload={"enabled": True, "top_composite": 0.70, "top_candidates": []},
            transfer_verdict="blocked",
            relation_kind="support",
            morphism_score=0.10,
        )
        assert high_m.transfer_consistency_score > low_m.transfer_consistency_score

    def test_stability_from_candidates(self):
        """Low variance among candidates → high stability."""
        tight = compute_eml_secondary_judgment(
            eml_shadow_payload={
                "enabled": True,
                "top_composite": 0.70,
                "top_candidates": [
                    {"composite_score": 0.70},
                    {"composite_score": 0.69},
                    {"composite_score": 0.68},
                ],
            },
            transfer_verdict="local_only",
            relation_kind="support",
        )
        spread = compute_eml_secondary_judgment(
            eml_shadow_payload={
                "enabled": True,
                "top_composite": 0.70,
                "top_candidates": [
                    {"composite_score": 0.90},
                    {"composite_score": 0.10},
                    {"composite_score": 0.50},
                ],
            },
            transfer_verdict="local_only",
            relation_kind="support",
        )
        assert tight.symbolic_stability_score > spread.symbolic_stability_score
