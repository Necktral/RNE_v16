"""Tests del estudio de ceguera de la recompensa (mecanismo + sistema real + análisis)."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.reward_blindness_lib import (
    _synthetic_run,
    classify_task,
    effect_size,
    run_mechanism_experiment,
    run_study,
    seed_ci,
)


class TestSyntheticMechanism:
    def test_lambda_zero_suppresses_effective_family(self):
        # Sin término de efectividad, la familia efectiva (cuesta y no mejora coherencia) se suprime.
        out = _synthetic_run(
            lambda_effectiveness=0.0, effectiveness_source="real", episodes=30, seed=1
        )
        assert out["retained"] is False

    def test_high_lambda_retains_effective_family(self):
        out = _synthetic_run(
            lambda_effectiveness=1.0, effectiveness_source="real", episodes=30, seed=1
        )
        assert out["retained"] is True
        assert out["activation_rate_2nd_half"] > 0.5

    def test_noise_does_not_recover_even_with_high_lambda(self):
        # H3: efectividad aleatoria (decorrelacionada) NO recupera la familia.
        out = _synthetic_run(
            lambda_effectiveness=1.0, effectiveness_source="shuffled", episodes=40, seed=3
        )
        assert out["retained"] is False

    def test_deterministic_by_seed(self):
        a = _synthetic_run(lambda_effectiveness=0.5, effectiveness_source="real", episodes=20, seed=7)
        b = _synthetic_run(lambda_effectiveness=0.5, effectiveness_source="real", episodes=20, seed=7)
        assert a == b


class TestMechanismExperiment:
    def test_dose_response_and_noise_control(self):
        out = run_mechanism_experiment(seeds=120, episodes=30)
        v = out["verdicts"]
        real = v["real_retention_by_lambda"]
        # Suprimida en λV=0, recuperada en λV alto (dosis-respuesta).
        assert real[0] < 0.2
        assert real[-1] > 0.7
        assert v["H1_dose_response"] is True
        # El control de ruido se mantiene plano.
        assert v["H3_noise_control_flat"] is True
        json.dumps(out, default=str)


class TestAnalysis:
    def test_seed_ci_structure(self):
        ci = seed_ci([0.1, 0.2, 0.15, 0.18, 0.22])
        assert ci["n"] == 5
        assert ci["ci_lower"] <= ci["mean"] <= ci["ci_upper"]

    def test_effect_size_detects_separation(self):
        es = effect_size([0.0, 0.05, 0.02], [0.9, 0.95, 0.88])
        assert es["delta"] > 0.5
        assert es["ci_excludes_zero"] is True
        assert es["cohen_d"] > 1.0

    def test_effect_size_null_overlap(self):
        es = effect_size([0.4, 0.5, 0.45], [0.46, 0.52, 0.43])
        assert es["ci_excludes_zero"] is False


class TestTaskClassifier:
    def test_conflict_vs_saturated_labeling(self, tmp_path):
        # 0.88 uniform ⇒ greedy elige mal ⇒ conflicto; 0.62 ⇒ saturada.
        conflict = classify_task(
            {"grid_size": 5, "topology": "uniform", "initial_temperature": 0.88,
             "alarm_threshold": 0.85, "cooling_effect": 0.07},
            tmp_path / "c",
        )
        saturated = classify_task(
            {"grid_size": 1, "topology": "uniform", "initial_temperature": 0.62,
             "alarm_threshold": 0.85, "cooling_effect": 0.07},
            tmp_path / "s",
        )
        assert conflict == "conflict"
        assert saturated == "saturated"


class TestStudyEndToEnd:
    def test_smoke_both_modes(self, tmp_path):
        out = run_study(
            mode="both",
            output_root=tmp_path / "study",
            lambda_grid=[0.0, 1.0],
            mechanism_seeds=40,
            mechanism_episodes=20,
            system_lambda_grid=[0.0, 0.5],
            system_seeds=2,
            system_episodes=4,
        )
        root = Path(out["output_root"])
        assert (root / "results.json").exists()
        assert (root / "REPORT.md").exists()
        report = (root / "REPORT.md").read_text(encoding="utf-8")
        assert "Ceguera de la recompensa" in report
        assert "Hipótesis (pre-registradas" in report
        assert "Limitaciones honestas" in report
