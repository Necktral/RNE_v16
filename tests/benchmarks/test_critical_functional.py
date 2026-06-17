"""Tests del experimento del Funcional Crítico J(h|X) (harness puro, sin runtime).

Deterministas por semilla y pequeños (pocas semillas/episodios) para CI rápido.
"""

from __future__ import annotations

import random

import pytest

from scripts.critical_functional_lib import (
    DEFAULT_TAUS,
    EFF_MARGIN,
    Channels,
    _admissible,
    _channels,
    _region_signals,
    make_collapsed_reward,
    make_J_reward,
    run_ablation_experiment,
    run_cure_experiment,
    run_specificity_experiment,
    run_study,
)


class TestChannelsAndAdmissibility:
    def test_region_signals_gap_vs_band(self):
        # Conflicto: desviar ayuda (helps=1,+margen); greedy falla (helps=0,−margen).
        assert _region_signals("conflict", True) == (1.0, +EFF_MARGIN)
        assert _region_signals("conflict", False) == (0.0, -EFF_MARGIN)
        # band_in (sin brecha): ninguna acción ayuda ni tiene margen.
        assert _region_signals("band_in", True) == (0.0, 0.0)
        assert _region_signals("band_in", False) == (0.0, 0.0)

    def test_deviation_drops_identity(self):
        dev = _channels("conflict", True)
        cons = _channels("conflict", False)
        assert dev.identity < cons.identity  # desviar rompe continuidad
        assert dev.nu == 1.0 and cons.nu == 0.0  # ν = helps_goal booleano limpio

    def test_admissibility_gates_below_threshold(self):
        ch = _channels("conflict", True)
        assert _admissible(ch, DEFAULT_TAUS) is True
        bad = Channels(kappa=0.1, sigma=0.8, rho=0.8, alpha=0.8, nu=1.0, identity=0.8, u=0.1, margin=0.0)
        assert _admissible(bad, {"kappa": 0.5}) is False


class TestRewardShapes:
    def test_collapsed_dominated_by_continuity(self):
        # Sin λV, el colapsado prefiere NO desviar (continuidad domina) ⇒ greedy mayor.
        rng = random.Random(0)
        r_dev = sum(make_collapsed_reward(0.0, "conflict")(["opt"], random.Random(s)) for s in range(200))
        r_grv = sum(make_collapsed_reward(0.0, "conflict")(["heur"], random.Random(s)) for s in range(200))
        assert r_grv > r_dev  # la desviación efectiva es penalizada por coherencia

    def test_J_inadmissible_is_excluded(self):
        # τ_κ alto ⇒ toda acción inadmisible ⇒ recompensa muy negativa.
        r = make_J_reward(1.0, "conflict", taus={"kappa": 0.99})(["opt"], random.Random(0))
        assert r < -1e8


class TestG1Cure:
    def test_J_threshold_below_collapsed(self):
        c = run_cure_experiment(grid=(0.0, 1.0, 20.0, 50.0), seeds=60, episodes=24)
        # J recupera con λ pequeño; el colapsado necesita λ mucho mayor.
        assert c["threshold_J"] is not None and c["threshold_J"] <= 2.0
        assert c["threshold_collapsed"] is not None
        assert c["threshold_collapsed"] >= 5.0 * max(c["threshold_J"], 0.5)
        assert c["verdicts"]["G1_J_recovers_at_small_lambda"] is True
        assert c["verdicts"]["G1_collapsed_threshold_much_higher"] is True


class TestG2Specificity:
    def test_active_in_gap_inactive_in_band(self):
        s = run_specificity_experiment(lam_nu=1.0, seeds=80, episodes=24)
        act = s["activation_by_region"]
        assert act["conflict"] > 0.7 and act["band_out"] > 0.7  # con brecha ⇒ activa
        assert act["band_in"] < 0.2  # sin brecha (óptimo interior) ⇒ NO activa
        assert s["verdicts"]["G2_active_in_gap"] is True
        assert s["verdicts"]["G2_inactive_in_band"] is True


class TestG3Ablation:
    def test_nu_necessary_sigma_not(self):
        a = run_ablation_experiment(lam_nu=2.0, seeds=80, episodes=24)
        ret = a["retention_by_variant"]
        assert ret["J_full"] > 0.7
        assert ret["J_sin_nu"] < 0.2          # quitar ν reproduce la supresión
        assert ret["J_sin_sigma"] > 0.7       # quitar σ (no-causal) no suprime
        assert a["verdicts"]["G3_nu_necessary"] is True
        assert a["verdicts"]["G3_sigma_not_necessary"] is True


class TestStudyOrchestration:
    def test_run_study_writes_artifacts(self, tmp_path):
        out = run_study(
            mode="all", output_root=tmp_path,
            cure_grid=(0.0, 1.0, 20.0), seeds=40, episodes=20,
            spec_lam_nu=1.0, ablation_lam_nu=2.0,
        )
        assert (tmp_path / "results.json").exists()
        assert (tmp_path / "REPORT.md").exists()
        report = (tmp_path / "REPORT.md").read_text(encoding="utf-8")
        assert "Funcional Crítico" in report
        assert out["cure_verdicts"] is not None
        assert out["specificity_verdicts"] is not None
        assert out["ablation_verdicts"] is not None
