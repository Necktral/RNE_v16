"""Tests de las palancas de multiplicación de ganancia cognitiva.

Palanca 1: el meta_scheduler ejecuta la secuencia VALIDADA (el contrato de
cierre), no la propuesta sin corregir.
Palanca 2: selector de overlays guiado por la recompensa semi-Markov
(exploración determinista → evidencia → on/off), con núcleo/floors intocables.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from runtime.reasoning.scheduler_meta.policy import select_sequence
from runtime.reasoning.scheduler_meta.reward_guided import (
    DEFAULT_CANDIDATES,
    RewardGuidedOverlaySelector,
    is_reward_guided_enabled,
)
from runtime.storage import StorageConfig, StorageFactory


def _storage(tmp_path: Path):
    config = StorageConfig(
        mode="sqlite",
        sqlite_db_path=str(tmp_path / "rg.db"),
        postgres_dsn=None,
        artifact_root=tmp_path / "artifacts",
        prefer_postgres_reads=False,
        strict_dual_write=False,
    )
    return StorageFactory.create_facade(config)


class TestValidatedSequenceExecution:
    """Palanca 1 — la corrección del validador gobierna la ejecución."""

    def _scheduler_result(self, monkeypatch, *, max_steps: int):
        from runtime.reasoning.scheduler_meta.meta_scheduler import MetaScheduler

        scheduler = MetaScheduler(
            mode="adaptive", family_profile="core_plus_plan", max_steps=max_steps
        )
        return scheduler.run(
            {
                "run_id": "lever1",
                "regime_hint": "viability_edge",
                "observation": {"temperature": 0.95},
            }
        )

    def test_tight_budget_executes_corrected_core(self, monkeypatch):
        # Presupuesto 6: la propuesta legacy mete PLAN tras ABD y expulsa a DED;
        # la ejecutada debe ser la validada (núcleo íntegro).
        result = self._scheduler_result(monkeypatch, max_steps=6)
        executed = result["sequence"]
        assert "DED" in executed
        assert executed[-1] == "PROB"
        assert executed == result["validated_sequence"]

    def test_kill_switch_restores_legacy(self, monkeypatch):
        monkeypatch.setenv("RNFE_EXECUTE_PROPOSED_SEQUENCE", "1")
        result = self._scheduler_result(monkeypatch, max_steps=6)
        assert result["sequence"] == result["proposed_sequence"]

    def test_roomy_budget_keeps_overlay_and_core(self, monkeypatch):
        result = self._scheduler_result(monkeypatch, max_steps=10)
        executed = set(result["sequence"])
        assert {"ABD", "ANA", "CAU", "CTF", "DED", "PROB", "PLAN"} <= executed


class TestOverlayDirectivesInPolicy:
    def test_force_on_bypasses_activation_condition(self):
        # En homogeneous_safe PLAN no se activa por condición; la directiva lo fuerza.
        sequence, _, _ = select_sequence(
            features={},
            budget={"max_steps": 10},
            mode="adaptive",
            profile_name="full_family_exploration",
            regime_hint="homogeneous_safe",
            overlay_directives={"plan": "on", "heur": "off", "dia_adv": "off",
                                "fal_guard": "off", "ind": "off", "opt": "off"},
        )
        assert "plan" in sequence
        assert "heur" not in sequence and "opt" not in sequence

    def test_force_off_vetoes_profile_optional(self):
        sequence, _, _ = select_sequence(
            features={},
            budget={"max_steps": 10},
            mode="fixed",
            profile_name="core_plus_plan",
            regime_hint="viability_edge",
            overlay_directives={"plan": "off"},
        )
        assert "plan" not in sequence
        assert {"abd", "ana", "cau", "ctf", "ded", "prob"} <= set(sequence)

    def test_force_on_outside_profile_contract_is_ignored(self):
        # core_only no admite opcionales: la directiva no puede inyectarlas.
        sequence, _, _ = select_sequence(
            features={},
            budget={"max_steps": 10},
            mode="fixed",
            profile_name="core_only",
            regime_hint="viability_edge",
            overlay_directives={"plan": "on"},
        )
        assert "plan" not in sequence

    def test_no_directives_is_identity(self):
        base, _, _ = select_sequence(
            features={}, budget={"max_steps": 10}, mode="fixed",
            profile_name="core_plus_plan", regime_hint="viability_edge",
        )
        same, _, _ = select_sequence(
            features={}, budget={"max_steps": 10}, mode="fixed",
            profile_name="core_plus_plan", regime_hint="viability_edge",
            overlay_directives=None,
        )
        assert base == same


class TestRewardGuidedSelector:
    def test_disabled_by_default(self, monkeypatch):
        monkeypatch.delenv("RNFE_REWARD_GUIDED_SELECTION", raising=False)
        assert is_reward_guided_enabled() is False

    def test_candidates_exclude_core_and_eml(self):
        assert "eml_sr" not in DEFAULT_CANDIDATES
        assert not set(DEFAULT_CANDIDATES) & {"abd", "ana", "cau", "ctf", "ded", "prob"}

    def test_exploration_is_deterministic_round_robin(self):
        selector = RewardGuidedOverlaySelector(candidates=["heur", "plan"], max_active=1)
        first = selector.directives("r")
        assert sorted(f for f, a in first.items() if a == "on") == ["heur"]
        selector.observe(
            run_id="r", reward_block={"reward": -0.1}, executed_sequence=["ABD", "HEUR"]
        )
        second = selector.directives("r")
        assert sorted(f for f, a in second.items() if a == "on") == ["plan"]

    def test_positive_evidence_turns_family_on(self):
        selector = RewardGuidedOverlaySelector(
            candidates=["plan"], epsilon=0.005, min_obs=2, max_active=2
        )
        for reward, seq in (
            (0.05, ["ABD", "PLAN"]),
            (0.06, ["ABD", "PLAN"]),
            (0.01, ["ABD"]),
            (0.0, ["ABD"]),
        ):
            selector.observe(run_id="r", reward_block={"reward": reward}, executed_sequence=seq)
        assert selector.directives("r")["plan"] == "on"

    def test_negative_evidence_turns_family_off(self):
        selector = RewardGuidedOverlaySelector(
            candidates=["plan"], epsilon=0.005, min_obs=2, max_active=2
        )
        for reward, seq in (
            (-0.20, ["ABD", "PLAN"]),
            (-0.25, ["ABD", "PLAN"]),
            (-0.01, ["ABD"]),
            (0.0, ["ABD"]),
        ):
            selector.observe(run_id="r", reward_block={"reward": reward}, executed_sequence=seq)
        assert selector.directives("r")["plan"] == "off"

    def test_max_active_caps_simultaneous_overlays(self):
        selector = RewardGuidedOverlaySelector(
            candidates=["heur", "plan", "opt"], epsilon=0.001, min_obs=1, max_active=1
        )
        for family in ("heur", "plan", "opt"):
            selector.observe(
                run_id="r",
                reward_block={"reward": 0.10},
                executed_sequence=["ABD", family.upper()],
            )
        selector.observe(run_id="r", reward_block={"reward": -0.5}, executed_sequence=["ABD"])
        directives = selector.directives("r")
        assert sum(1 for action in directives.values() if action == "on") == 1

    def test_summary_is_json_safe_and_auditable(self):
        import json

        selector = RewardGuidedOverlaySelector(candidates=["plan"])
        selector.observe(
            run_id="r", reward_block={"reward": -0.1}, executed_sequence=["ABD", "PLAN"]
        )
        block = selector.summary("r")
        assert block["schema"] == "reward_guided.v2"
        assert block["n_observations"] == 1
        json.dumps(block)


class TestRegimeStratification:
    def _feed(self, selector, run_id, regime, family_reward, base_reward):
        # 2 episodios con familia y 2 sin, en un régimen dado.
        for _ in range(2):
            selector.observe(
                run_id=run_id,
                reward_block={"reward": family_reward},
                executed_sequence=["ABD", "PLAN"],
                regime=regime,
            )
            selector.observe(
                run_id=run_id, reward_block={"reward": base_reward},
                executed_sequence=["ABD"], regime=regime,
            )

    def test_directives_differ_by_regime(self):
        sel = RewardGuidedOverlaySelector(candidates=["plan"], epsilon=0.005, min_obs=2, max_active=2)
        # PLAN paga en viability_edge (+0.1 vs 0.0) y resta en homogeneous_safe (−0.1 vs 0.0).
        self._feed(sel, "r", "viability_edge", family_reward=0.1, base_reward=0.0)
        self._feed(sel, "r", "homogeneous_safe", family_reward=-0.1, base_reward=0.0)
        assert sel.directives("r", regime="viability_edge")["plan"] == "on"
        assert sel.directives("r", regime="homogeneous_safe")["plan"] == "off"

    def test_summary_reports_regimes_seen(self):
        sel = RewardGuidedOverlaySelector(candidates=["plan"])
        sel.observe(run_id="r", reward_block={"reward": 0.0},
                    executed_sequence=["ABD"], regime="viability_edge")
        block = sel.summary("r")
        assert "viability_edge" in block["regimes_seen"]


class TestMergeFrom:
    def test_merge_copies_observations(self):
        donor = RewardGuidedOverlaySelector(candidates=["plan"])
        for _ in range(3):
            donor.observe(run_id="d", reward_block={"reward": 0.1},
                          executed_sequence=["ABD", "PLAN"], regime="viability_edge")
        recipient = RewardGuidedOverlaySelector(candidates=["plan"])
        merged = recipient.merge_from("r", donor.export_evidence("d"), eligible=True)
        assert merged == 3
        assert recipient.summary("r")["n_observations"] == 3

    def test_ineligible_inheritance_blocks_merge(self):
        donor = RewardGuidedOverlaySelector(candidates=["plan"])
        donor.observe(run_id="d", reward_block={"reward": 0.1}, executed_sequence=["ABD", "PLAN"])
        recipient = RewardGuidedOverlaySelector(candidates=["plan"])
        assert recipient.merge_from("r", donor.export_evidence("d"), eligible=False) == 0

    def test_adversarial_morphism_blocks_merge(self):
        class _Adversarial:
            morphism_class = "adversarial"

        donor = RewardGuidedOverlaySelector(candidates=["plan"])
        donor.observe(run_id="d", reward_block={"reward": 0.1}, executed_sequence=["ABD", "PLAN"])
        recipient = RewardGuidedOverlaySelector(candidates=["plan"])
        merged = recipient.merge_from(
            "r", donor.export_evidence("d"), morphism=_Adversarial(), eligible=True
        )
        assert merged == 0


class TestRewardGuidedLiveLoop:
    def test_runner_closes_the_loop_and_reseeds(self, tmp_path, monkeypatch):
        from runtime.world import ScenarioEpisodeRunner

        monkeypatch.setenv("RNFE_REWARD_GUIDED_SELECTION", "1")
        monkeypatch.setenv("RNFE_REASONING_MODE", "adaptive")
        monkeypatch.setenv("RNFE_REASONING_FAMILY_PROFILE", "full_family_exploration")
        monkeypatch.setenv("RNFE_REASONING_MAX_STEPS", "10")
        storage = _storage(tmp_path)
        runner = ScenarioEpisodeRunner(
            storage=storage,
            run_id="rg-live",
            scenario="thermal_homeostasis",
            closure_profile="adaptive_min",
        )
        for _ in range(3):
            result = runner.run_episode()
        block = result["reward_guided"]
        assert block["schema"] == "reward_guided.v2"
        assert block["n_observations"] >= 2
        assert isinstance(block["directives"], dict)
        # El núcleo nunca se toca: cierre certificable en todos los episodios.
        assert result["certification"]["verdict"] in {"certified", "rejected"}
        seq = [step.get("family") for step in result["reasoning"]["trace"]]
        assert {"ABD", "ANA", "CAU", "CTF", "DED", "PROB"} <= set(seq)

        # Runner fresco sobre el mismo run: evidencia re-sembrada de eventos.
        runner2 = ScenarioEpisodeRunner(
            storage=storage,
            run_id="rg-live",
            scenario="thermal_homeostasis",
            closure_profile="adaptive_min",
        )
        result2 = runner2.run_episode()
        assert result2["reward_guided"]["n_observations"] >= 3

    def test_disabled_runner_has_no_reward_guided_block(self, tmp_path, monkeypatch):
        from runtime.world import ScenarioEpisodeRunner

        monkeypatch.delenv("RNFE_REWARD_GUIDED_SELECTION", raising=False)
        storage = _storage(tmp_path)
        runner = ScenarioEpisodeRunner(
            storage=storage, run_id="rg-off", scenario="thermal_homeostasis"
        )
        result = runner.run_episode()
        assert "reward_guided" not in result
