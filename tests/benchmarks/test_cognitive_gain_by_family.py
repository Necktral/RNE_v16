"""Tests del experimento de ganancia cognitiva por tipo de razonamiento.

Cubre: perfiles de aislamiento nuevos, cierre adaptive_min con PLAN/OPT,
captura de señales canon en el benchmark runner, siembra de IoC* del tracker Ω,
y la campaña/análisis/informe end-to-end en miniatura.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from runtime.reality.evaluator import (
    ADAPTIVE_MIN_PROFILE,
    validate_sequence_with_profile,
)
from runtime.reasoning.scheduler_meta.family_profiles import resolve_family_profile
from runtime.reasoning.scheduler_meta.policy import select_sequence
from runtime.storage import StorageConfig, StorageFactory


def _storage_config(tmp_path: Path) -> StorageConfig:
    return StorageConfig(
        mode="sqlite",
        sqlite_db_path=str(tmp_path / "cgf.db"),
        postgres_dsn=None,
        artifact_root=tmp_path / "artifacts",
        prefer_postgres_reads=False,
        strict_dual_write=False,
    )


class TestIsolationProfiles:
    @pytest.mark.parametrize(
        "profile_name,family",
        [("core_plus_ind", "ind"), ("core_plus_plan", "plan"), ("core_plus_opt", "opt")],
    )
    def test_profile_exists_and_is_fixed(self, profile_name, family):
        profile = resolve_family_profile(profile_name)
        assert profile.name == profile_name
        assert profile.optional_families == [family]
        assert profile.adaptive is False

    @pytest.mark.parametrize("profile_name", ["core_plus_plan", "core_plus_opt"])
    def test_fixed_mode_composes_protected_sequence(self, profile_name):
        family = profile_name.rsplit("_", 1)[-1]
        sequence, _, _, meta = select_sequence(
            features={},
            budget={"max_steps": 6},
            mode="fixed",
            profile_name=profile_name,
            regime_hint="viability_edge",
            return_metadata=True,
        )
        assert family in sequence
        assert "ded" in sequence  # la composición protegida nunca expulsa a DED
        assert sequence.index(family) > sequence.index("ctf")  # ancla: tras CTF
        assert sequence[-1] == "prob"
        assert meta["sequence_validation"]["validated_passed"] is True

    def test_ind_fixed_mode_keeps_core_intact(self):
        sequence, _, _, meta = select_sequence(
            features={},
            budget={"max_steps": 6},
            mode="fixed",
            profile_name="core_plus_ind",
            regime_hint="homogeneous_safe",
            return_metadata=True,
        )
        assert "ind" in sequence and "ded" in sequence
        assert meta["sequence_validation"]["validated_passed"] is True


class TestClosureAcceptsDeliberative:
    def test_adaptive_min_whitelists_plan_and_opt(self):
        assert {"PLAN", "OPT"} <= ADAPTIVE_MIN_PROFILE.optional_families

    @pytest.mark.parametrize(
        "sequence",
        [
            ["ABD", "PLAN", "ANA", "CAU", "CTF", "DED", "PROB"],
            ["ABD", "OPT", "ANA", "CAU", "CTF", "DED", "PROB"],
            ["ABD", "PLAN", "OPT", "ANA", "CAU", "CTF", "DED", "PROB"],
        ],
    )
    def test_deliberative_sequences_pass_closure(self, sequence):
        result = validate_sequence_with_profile(sequence, ADAPTIVE_MIN_PROFILE)
        assert result["passed"] is True, result["checks"]

    def test_unknown_family_still_rejected(self):
        result = validate_sequence_with_profile(
            ["ABD", "ALIEN", "ANA", "CAU", "CTF", "DED", "PROB"], ADAPTIVE_MIN_PROFILE
        )
        assert result["passed"] is False


class TestCanonSignalExtraction:
    def test_extracts_reward_and_certificate_blocks(self):
        from tests.benchmarks.benchmark_runner import extract_canon_signal_metrics

        class _Cert:
            ioc_proxy = 0.88
            metadata = {
                "omega": {
                    "omega": 0.05,
                    "ioc_star": 0.865,
                    "pairwise_mean": 0.03,
                    "cycle": {"error": 0.04},
                    "cross_context": True,
                },
                "risk_plus": {
                    "delta_ioc": 0.01,
                    "cvar_neg_delta_ioc": 0.02,
                    "p_delta_nonneg_lcb": 0.55,
                    "sie_verdict": "ACEPTAR",
                },
            }

        payload = {
            "reasoning_reward": {
                "reward": -0.05,
                "delta_ioc": 0.0,
                "delta_ioc_star": -0.002,
                "delta_used": "delta_ioc_star",
                "energy_term": 0.06,
                "bsafe_penalty": 0.0,
                "reasoning_cost": 5.9,
                "cost_budget": 10.0,
            }
        }
        signals = extract_canon_signal_metrics(payload, certificate=_Cert())
        assert signals["reasoning_reward"] == -0.05
        assert signals["reward_delta_used"] == "delta_ioc_star"
        assert signals["ioc_proxy"] == 0.88
        assert signals["omega"] == 0.05
        assert signals["ioc_star"] == 0.865
        assert signals["omega_cycle_error"] == 0.04
        assert signals["omega_cross_context"] == 1.0
        assert signals["risk_sie_verdict"] == "ACEPTAR"
        json.dumps(signals)

    def test_tolerates_legacy_payload_without_blocks(self):
        from tests.benchmarks.benchmark_runner import extract_canon_signal_metrics

        assert extract_canon_signal_metrics({}, certificate=None) == {}


class TestOmegaSeedRestoresIocStar:
    def test_fresh_tracker_recovers_delta_ioc_star(self, tmp_path):
        from runtime.certification.coherence_obstruction import CoherenceObstructionTracker
        from runtime.world import ScenarioEpisodeRunner

        storage = StorageFactory.create_facade(_storage_config(tmp_path))
        runner = ScenarioEpisodeRunner(
            storage=storage, run_id="cgf-seed", scenario="thermal_homeostasis"
        )
        for _ in range(2):
            result = runner.run_episode()
        # Tracker NUEVO (runner fresco por episodio, patrón benchmark): la serie
        # IoC* debe restaurarse del último certificado, no quedar en None.
        tracker = CoherenceObstructionTracker(storage=storage)
        block = tracker.assess(
            run_id="cgf-seed", episode_result=result, ioc_value=0.9
        )
        assert block["delta_ioc_star"] is not None


class TestCampaignEndToEnd:
    @pytest.fixture(scope="class")
    def campaign(self, tmp_path_factory):
        from scripts.cognitive_gain_by_family_lib import run_cognitive_gain_by_family_campaign

        tmp_path = tmp_path_factory.mktemp("cgf_campaign")
        return run_cognitive_gain_by_family_campaign(
            campaign_id="cgf-test",
            output_root=tmp_path,
            blocks=1,
            episodes_per_block=2,
            bootstrap_samples=50,
            profiles=["core_only", "core_plus_plan"],
            regimes=["viability_edge"],
        )

    def test_outputs_exist(self, campaign):
        root = Path(campaign["root_dir"])
        assert (root / "cognitive_gain_by_family_report.md").exists()
        assert (root / "family_verdicts.json").exists()
        assert (root / "block_metrics.jsonl").exists()

    def test_episodes_capture_canon_signals(self, campaign):
        root = Path(campaign["root_dir"])
        episode_files = sorted(root.glob("runs/steps10/core_plus_plan/**/episodes.jsonl"))
        assert episode_files
        rows = [
            json.loads(line)
            for line in episode_files[0].read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        last = rows[-1]
        for key in ("reasoning_reward", "ioc_proxy", "omega", "ioc_star", "risk_sie_verdict"):
            assert key in last, f"falta {key} en episodes.jsonl"
        # ΔIoC* sobrevive al runner-fresco-por-episodio (siembra del tracker Ω).
        assert last.get("reward_delta_used") == "delta_ioc_star"

    def test_plan_certifies_under_primary_protocol(self, campaign):
        verdicts = json.loads(
            (Path(campaign["root_dir"]) / "family_verdicts.json").read_text(encoding="utf-8")
        )
        plan_cell = verdicts["regimes_analysis"]["viability_edge"]["profiles"][
            "core_plus_plan"
        ]["cell"]
        assert plan_cell["success_rate"] == 1.0  # sin el fix de cierre esto era 0.0

    def test_verdict_structure(self, campaign):
        verdicts = json.loads(
            (Path(campaign["root_dir"]) / "family_verdicts.json").read_text(encoding="utf-8")
        )
        assert verdicts["primary_verdict"]
        assert "PLAN" in verdicts["families"]
        plan = verdicts["families"]["PLAN"]["per_regime"]["viability_edge"]
        assert plan["role"] in {"aporta", "aporta condicionado", "neutral", "perjudica"}
        assert "delta_reasoning_reward" in plan
        sensitivity = verdicts["regimes_analysis"]["viability_edge"]["budget_sensitivity"]
        assert "core_plus_plan" in sensitivity

    def test_report_has_all_sections(self, campaign):
        report = Path(campaign["report_path"]).read_text(encoding="utf-8")
        for heading in (
            "## 1. Dictamen primario",
            "## 2. Matriz régimen × perfil",
            "## 3. Aislamiento por familia",
            "## 4. Sinergia deliberativa",
            "## 5. Núcleo ABD/ANA/CAU/CTF/DED/PROB",
            "## 6. Economía del razonamiento",
            "## 7. Coherencia multi-contexto",
            "## 8. Sensibilidad al presupuesto natural",
            "## 9. Comparación con la campaña de abril",
            "## 10. Guardas y riesgos residuales",
        ):
            assert heading in report, f"falta sección: {heading}"
