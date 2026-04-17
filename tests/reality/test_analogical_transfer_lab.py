"""Tests para el laboratorio analógico de transferencia + EML shadow."""

from runtime.reality.analogical_lab import (
    eml_concurrence_score,
    run_analogical_transfer_lab,
)


class TestEMLConcurrenceScore:
    def test_both_stable_high_concordance(self):
        """Cuando transfer y EML coinciden en estabilidad → alta concordancia."""
        score = eml_concurrence_score(
            transfer_verdict="certified_transfer_safe",
            eml_shadow_payload={"enabled": True, "top_composite": 0.80},
            relation_kind="support",
        )
        assert score >= 0.80

    def test_both_unstable_moderate(self):
        """Cuando ambos son inestables → concordancia moderada."""
        score = eml_concurrence_score(
            transfer_verdict="rejected_for_transfer",
            eml_shadow_payload={"enabled": True, "top_composite": 0.20},
            relation_kind="contradiction",
        )
        assert 0.50 <= score <= 0.70

    def test_discordance_eml_up_cert_down(self):
        """EML sube pero certificación cae → discordancia crítica."""
        score = eml_concurrence_score(
            transfer_verdict="rejected_for_transfer",
            eml_shadow_payload={"enabled": True, "top_composite": 0.90},
            relation_kind="support",
        )
        assert score < 0.30

    def test_eml_disabled_neutral(self):
        """EML deshabilitado → 0.5 neutral."""
        score = eml_concurrence_score(
            transfer_verdict="certified_local",
            eml_shadow_payload=None,
            relation_kind="support",
        )
        assert score == 0.5

    def test_eml_low_cert_stable(self):
        """EML baja pero cert ok → leve discordancia."""
        score = eml_concurrence_score(
            transfer_verdict="certified_local",
            eml_shadow_payload={"enabled": True, "top_composite": 0.20},
            relation_kind="support",
        )
        assert 0.30 <= score <= 0.50


class TestAnalogicalTransferLab:
    def test_runs_and_produces_deltas(self):
        """Ejecuta lab y produce deltas entre strict y analogical."""
        result = run_analogical_transfer_lab(
            scenarios=["thermal_homeostasis"],
            warmup_episodes=1,
            probe_episodes=1,
            closure_profile="adaptive_min",
            eml_shadow=False,
        )
        assert "delta_continuity" in result
        assert "delta_purity" in result
        assert "delta_transfer_safe" in result
        assert "delta_closure" in result
        assert "eml_concurrence_map" in result

    def test_produces_artifact(self):
        """Produce artifact de tipo analogical_transfer_lab_report."""
        result = run_analogical_transfer_lab(
            scenarios=["thermal_homeostasis"],
            warmup_episodes=1,
            probe_episodes=1,
            closure_profile="adaptive_min",
            eml_shadow=False,
        )
        assert result["artifact"]["kind"] == "analogical_transfer_lab_report"

    def test_both_matrices_present(self):
        """Ambas matrices (strict y analogical) están presentes."""
        result = run_analogical_transfer_lab(
            scenarios=["thermal_homeostasis"],
            warmup_episodes=1,
            probe_episodes=1,
            closure_profile="adaptive_min",
            eml_shadow=False,
        )
        assert "matrix_strict" in result
        assert "matrix_analogical" in result
        assert len(result["matrix_strict"]["cell_reports"]) == 1
        assert len(result["matrix_analogical"]["cell_reports"]) == 1

    def test_deltas_are_numeric(self):
        """Los deltas son numéricos."""
        result = run_analogical_transfer_lab(
            scenarios=["thermal_homeostasis"],
            warmup_episodes=1,
            probe_episodes=1,
            closure_profile="adaptive_min",
            eml_shadow=False,
        )
        for key in ["delta_continuity", "delta_purity", "delta_transfer_safe", "delta_closure"]:
            assert isinstance(result[key], (int, float))

    def test_eml_concurrence_map_present(self):
        """EML concurrence map está presente para cada par."""
        result = run_analogical_transfer_lab(
            scenarios=["thermal_homeostasis"],
            warmup_episodes=1,
            probe_episodes=1,
            closure_profile="adaptive_min",
            eml_shadow=False,
        )
        eml_map = result["eml_concurrence_map"]
        assert isinstance(eml_map, dict)
        for key, val in eml_map.items():
            assert "strict_eml" in val
            assert "analogical_eml" in val
            assert "delta_eml" in val
