"""Tests del motor de riesgo 𝔠ₜ⁺ (CVaR + B_safe + regla S-I-E)."""

from __future__ import annotations

import json
from pathlib import Path

from runtime.certification.risk_engine import (
    EpisodeRiskTracker,
    SIE_ACCEPT,
    SIE_BUFFER,
    SIE_REJECT,
    agresti_coull_lcb,
    compute_b_safe,
    compute_cvar,
    phi_bar,
    sie_rule,
)
from runtime.storage import StorageConfig, StorageFactory


def _storage(tmp_path: Path):
    config = StorageConfig(
        mode="sqlite",
        sqlite_db_path=str(tmp_path / "risk.db"),
        postgres_dsn=None,
        artifact_root=tmp_path / "artifacts",
        prefer_postgres_reads=True,
        strict_dual_write=False,
    )
    return StorageFactory.create_facade(config)


class TestComputeCvar:
    def test_empty_returns_zero(self):
        assert compute_cvar([]) == 0.0

    def test_constant_losses(self):
        assert compute_cvar([0.3, 0.3, 0.3], alpha=0.95) == 0.3

    def test_tail_dominated_by_worst_drop(self):
        # Forma del run histórico: 1 caída de 0.09 entre mejoras.
        losses = [0.0, 0.0, 0.0, 0.0, 0.09]
        assert abs(compute_cvar(losses, alpha=0.95) - 0.09) < 1e-9

    def test_cvar_at_least_mean(self):
        losses = [0.01, 0.02, 0.5, 0.03, 0.04]
        mean = sum(losses) / len(losses)
        assert compute_cvar(losses, alpha=0.9) >= mean

    def test_monotonic_in_alpha(self):
        losses = [0.01, 0.05, 0.10, 0.20, 0.50]
        assert compute_cvar(losses, alpha=0.95) >= compute_cvar(losses, alpha=0.50)


class TestAgrestiCoull:
    def test_zero_n(self):
        assert agresti_coull_lcb(0, 0) == 0.0

    def test_known_value_six_of_seven(self):
        # Patrón del histórico calibrado: 6 mejoras de 7 deltas.
        assert round(agresti_coull_lcb(6, 7), 4) == 0.4665

    def test_bounds(self):
        for s, n in [(0, 5), (5, 5), (3, 10), (100, 100)]:
            assert 0.0 <= agresti_coull_lcb(s, n) <= 1.0

    def test_grows_with_evidence(self):
        # Misma proporción, más muestra ⇒ cota inferior más alta.
        assert agresti_coull_lcb(60, 70) > agresti_coull_lcb(6, 7)


class TestBSafe:
    def test_phi_bar_grows_near_limit(self):
        assert phi_bar(0.9) > phi_bar(0.5) > phi_bar(0.1)

    def test_phi_bar_violation_is_none(self):
        assert phi_bar(1.0) is None
        assert phi_bar(1.5) is None

    def test_safe_signals(self):
        block = compute_b_safe({"vram_pressure": 0.5, "temperature": 0.6})
        assert block["violated"] is False
        assert isinstance(block["value"], float)
        assert block["violated_signals"] == []

    def test_violated_signal(self):
        block = compute_b_safe({"vram_pressure": 1.0, "temperature": 0.5})
        assert block["violated"] is True
        assert block["value"] is None
        assert block["violated_signals"] == ["vram_pressure"]

    def test_json_serializable(self):
        for signals in ({"a": 0.3}, {"a": 1.2}):
            json.dumps(compute_b_safe(signals))


class TestSieRule:
    KW = dict(cvar_alpha=0.95, cvar_threshold=0.10, prob_threshold=0.45, min_history=4)

    def test_hard_violation_rejects(self):
        out = sie_rule(deltas=[0.1] * 10, hard_violation_count=1, **self.KW)
        assert out["verdict"] == SIE_REJECT

    def test_b_safe_violation_rejects(self):
        out = sie_rule(deltas=[0.1] * 10, b_safe_violated=True, **self.KW)
        assert out["verdict"] == SIE_REJECT

    def test_short_history_buffers(self):
        out = sie_rule(deltas=[0.1, 0.1], **self.KW)
        assert out["verdict"] == SIE_BUFFER
        assert "insuficiente" in out["reason"]

    def test_healthy_historical_run_accepts(self):
        # Run sano del histórico calibrado: 6 mejoras pequeñas + 1 caída -0.09.
        deltas = [0.01, 0.02, 0.01, -0.09, 0.02, 0.01, 0.01]
        out = sie_rule(deltas=deltas, **self.KW)
        assert out["verdict"] == SIE_ACCEPT

    def test_bad_tail_buffers(self):
        deltas = [0.01, 0.02, 0.01, -0.50, 0.02, 0.01, 0.01]
        out = sie_rule(deltas=deltas, **self.KW)
        assert out["verdict"] == SIE_BUFFER
        assert "CVaR" in out["reason"]


class _CertStub:
    def __init__(self, ioc: float):
        self.ioc_proxy = ioc


class _StorageStub:
    """Devuelve certificados más-reciente-primero, como el storage real."""

    def __init__(self, iocs_chronological):
        self._iocs = list(iocs_chronological)

    def list_episode_certificates(self, *, run_id=None, limit=200):
        return [_CertStub(v) for v in reversed(self._iocs[-limit:])]


class TestEpisodeRiskTracker:
    def test_first_episode_has_no_delta(self):
        tracker = EpisodeRiskTracker()
        block = tracker.assess(run_id="r1", ioc_value=0.8)
        assert block["delta_ioc"] is None
        assert block["n_history"] == 0
        assert block["sie_verdict"] == SIE_BUFFER

    def test_delta_and_history_grow(self):
        tracker = EpisodeRiskTracker()
        tracker.assess(run_id="r1", ioc_value=0.80)
        block = tracker.assess(run_id="r1", ioc_value=0.85)
        assert abs(block["delta_ioc"] - 0.05) < 1e-9
        assert block["n_history"] == 1

    def test_seeds_from_storage_chronologically(self):
        storage = _StorageStub([0.70, 0.72, 0.74, 0.76])
        tracker = EpisodeRiskTracker(storage=storage)
        block = tracker.assess(run_id="r1", ioc_value=0.78)
        # delta contra el ÚLTIMO cronológico (0.76), no contra el primero.
        assert abs(block["delta_ioc"] - 0.02) < 1e-9
        assert block["n_history"] == 4

    def test_max_history_bound(self):
        tracker = EpisodeRiskTracker(max_history=8)
        for i in range(50):
            block = tracker.assess(run_id="r1", ioc_value=0.5 + (i % 3) * 0.01)
        assert block["n_history"] <= 7

    def test_block_is_json_safe(self):
        tracker = EpisodeRiskTracker()
        for i in range(6):
            block = tracker.assess(run_id="r1", ioc_value=0.8 + i * 0.01)
        json.dumps(block)

    def test_runs_are_independent(self):
        tracker = EpisodeRiskTracker()
        tracker.assess(run_id="a", ioc_value=0.9)
        block = tracker.assess(run_id="b", ioc_value=0.5)
        assert block["delta_ioc"] is None


class TestPromotionGateIntegration:
    def test_certificate_carries_risk_plus_in_shadow_mode(self, tmp_path):
        from runtime.world import ScenarioEpisodeRunner

        runner = ScenarioEpisodeRunner(
            storage=_storage(tmp_path), scenario="thermal_homeostasis"
        )
        for _ in range(3):
            result = runner.run_episode()

        certs = runner.storage.list_episode_certificates(
            run_id=runner.run_id, limit=1
        )
        block = certs[0].metadata.get("risk_plus")
        assert block and block["schema"] == "risk_plus.v1"
        assert block["sie_verdict"] in {SIE_ACCEPT, SIE_BUFFER, SIE_REJECT}
        assert block["n_history"] == 2
        # Modo sombra: el veredicto clásico sigue gobernado por la lógica previa.
        assert certs[0].verdict in {"certified", "rejected"}
        # Determinista por defecto: sin telemetría física.
        assert block["b_safe"] is None
        json.dumps(block)
