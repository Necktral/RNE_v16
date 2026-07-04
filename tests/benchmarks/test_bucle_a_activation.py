"""Tests del estudio A1 — activación del Bucle A (harness puro, sin runtime).

Deterministas por semilla y pequeños (pocas semillas/episodios) para CI rápido.
"""

from __future__ import annotations

from scripts.bucle_a_activation_lib import (
    EFF_MARGIN,
    _effectiveness_from_activations,
    _run_ecology,
    _run_fixed,
    _run_guided,
    run_arms_experiment,
    run_lambda_sweep,
    run_study,
)
from scripts.critical_functional_lib import make_J_reward
from scripts.reward_blindness_lib import EFFECTIVE_FAMILY


class TestArmsMechanics:
    def test_fixed_never_activates_effective_family(self):
        out = _run_fixed(make_J_reward(1.0, "conflict"), episodes=20, seed=0)
        assert out["retained"] is False
        assert all(a == 0.0 for a in out["activations"])

    def test_guided_retains_effective_family_at_o1(self):
        out = _run_guided(make_J_reward(1.0, "conflict"), episodes=30, seed=0)
        assert out["retained"] is True
        # se activa consistentemente en la 2ª mitad
        assert sum(out["activations"][15:]) >= 10

    def test_ecology_converges(self):
        out = _run_ecology(make_J_reward(1.0, "conflict"), episodes=30, seed=0)
        assert out["retained"] is True

    def test_effectiveness_mapping(self):
        # desviar (activa) ⇒ +margen; no desviar ⇒ −margen (régimen con brecha)
        assert _effectiveness_from_activations([1.0, 0.0]) == [+EFF_MARGIN, -EFF_MARGIN]


class TestA1Verdicts:
    def test_guided_beats_fixed_and_threshold_o1(self):
        arms = run_arms_experiment(lam_nu=1.0, seeds=10, episodes=30)
        v = arms["verdicts"]
        # A1a: efectividad guiado > fijo, CI entre-semillas excluye 0.
        assert v["A1a_guided_beats_fixed"] is True
        assert arms["effectiveness_guided_vs_fixed"]["ci_lower"] > 0.0
        # A1b: activación de la familia efectiva sube.
        assert v["A1b_effective_activation_rises"] is True
        # Fijo nunca activa; guiado retiene.
        assert arms["fixed"]["activation"]["mean"] == 0.0
        assert arms["guided"]["retention"]["mean"] == 1.0

    def test_lambda_threshold_is_o1(self):
        sweep = run_lambda_sweep(grid=(0.0, 1.0, 5.0), seeds=10, episodes=30)
        assert sweep["threshold_lambda_nu"] is not None
        assert sweep["threshold_lambda_nu"] <= 2.0
        assert sweep["verdicts"]["A1c_threshold_is_O1"] is True


class TestStudyOrchestration:
    def test_run_study_writes_artifacts(self, tmp_path):
        out = run_study(mode="all", output_root=tmp_path, seeds=8, episodes=24, sweep_grid=(0.0, 1.0, 5.0))
        assert (tmp_path / "results.json").exists()
        assert (tmp_path / "REPORT.md").exists()
        report = (tmp_path / "REPORT.md").read_text(encoding="utf-8")
        assert "Activación del Bucle A" in report
        assert out["arms_verdicts"] is not None
        assert out["sweep_verdicts"] is not None
