"""Tests para perfiles de cierre y validación de secuencias adaptativas."""

from pathlib import Path

import pytest

from runtime.reality.evaluator import (
    ADAPTIVE_MIN_PROFILE,
    BASELINE_FIXED_PROFILE,
    CLOSURE_PROFILES,
    ClosureProfile,
    evaluate_episode_closure,
    validate_sequence_with_profile,
    _validate_partial_order,
    _validate_required_families,
    _validate_prob_closes,
)
from runtime.reasoning.context import resolve_reasoning_mode
from runtime.storage import StorageConfig, StorageFactory
from runtime.world.min_cognitive_episode import MinimalCognitiveEpisodeRunner
from runtime.reasoning.scheduler_meta.meta_scheduler import MetaScheduler


def _storage(tmp_path: Path):
    config = StorageConfig(
        mode="sqlite",
        sqlite_db_path=str(tmp_path / "closure.db"),
        postgres_dsn=None,
        artifact_root=tmp_path / "artifacts",
        prefer_postgres_reads=True,
        strict_dual_write=False,
    )
    return StorageFactory.create_facade(config)


class TestClosureProfiles:
    """Tests unitarios para perfiles de cierre."""

    def test_closure_profile_maps_expected_scheduler_mode(self):
        assert resolve_reasoning_mode("baseline_fixed") == "fixed"
        assert resolve_reasoning_mode("adaptive_min") == "adaptive"

    def test_baseline_fixed_profile_exists(self):
        """El perfil baseline_fixed existe en el registro."""
        assert "baseline_fixed" in CLOSURE_PROFILES
        profile = CLOSURE_PROFILES["baseline_fixed"]
        assert profile.name == "baseline_fixed"
        assert profile.required_sequence == ["ABD", "ANA", "CAU", "CTF", "DED", "PROB"]

    def test_adaptive_min_profile_allows_optional_families(self):
        """El perfil adaptive_min permite familias opcionales."""
        assert "adaptive_min" in CLOSURE_PROFILES
        profile = CLOSURE_PROFILES["adaptive_min"]
        assert "DIA_ADV" in profile.optional_families
        assert "HEUR" in profile.optional_families
        assert "FAL_GUARD" in profile.optional_families
        assert "EML_SR" in profile.optional_families


class TestPartialOrderValidation:
    """Tests para validación de orden parcial."""

    def test_valid_partial_order_passes(self):
        """Una secuencia con orden parcial correcto pasa."""
        sequence = ["ABD", "ANA", "CAU", "CTF", "DED", "PROB"]
        partial_order = BASELINE_FIXED_PROFILE.partial_order
        assert _validate_partial_order(sequence, partial_order) is True

    def test_invalid_partial_order_fails(self):
        """Una secuencia con orden parcial incorrecto falla."""
        # PROB antes de ABD es inválido
        sequence = ["PROB", "ABD", "ANA", "CAU", "CTF", "DED"]
        partial_order = BASELINE_FIXED_PROFILE.partial_order
        assert _validate_partial_order(sequence, partial_order) is False

    def test_partial_order_with_optional_families(self):
        """Familias opcionales no rompen el orden parcial."""
        sequence = ["ABD", "HEUR", "ANA", "DIA_ADV", "CAU", "CTF", "DED", "FAL_GUARD", "PROB"]
        partial_order = ADAPTIVE_MIN_PROFILE.partial_order
        assert _validate_partial_order(sequence, partial_order) is True


class TestRequiredFamiliesValidation:
    """Tests para validación de familias requeridas."""

    def test_all_required_present_passes(self):
        """Secuencia con todas las familias requeridas pasa."""
        sequence = ["ABD", "ANA", "CAU", "CTF", "DED", "PROB"]
        required = BASELINE_FIXED_PROFILE.required_sequence
        assert _validate_required_families(sequence, required) is True

    def test_missing_required_fails(self):
        """Secuencia sin una familia requerida falla."""
        sequence = ["ABD", "ANA", "CAU", "CTF", "DED"]  # Falta PROB
        required = BASELINE_FIXED_PROFILE.required_sequence
        assert _validate_required_families(sequence, required) is False

    def test_required_with_extras_passes(self):
        """Secuencia con requeridas + extras pasa."""
        sequence = ["ABD", "HEUR", "ANA", "CAU", "CTF", "DED", "PROB"]
        required = BASELINE_FIXED_PROFILE.required_sequence
        assert _validate_required_families(sequence, required) is True


class TestProbClosesValidation:
    """Tests para validación de PROB como cierre."""

    def test_prob_at_end_passes(self):
        """PROB al final pasa."""
        assert _validate_prob_closes(["ABD", "ANA", "PROB"]) is True

    def test_prob_not_at_end_fails(self):
        """PROB no al final falla."""
        assert _validate_prob_closes(["ABD", "PROB", "ANA"]) is False

    def test_empty_sequence_fails(self):
        """Secuencia vacía falla."""
        assert _validate_prob_closes([]) is False


class TestValidateSequenceWithProfile:
    """Tests para validación completa con perfil."""

    def test_baseline_fixed_exact_sequence_passes(self):
        """La secuencia exacta del baseline pasa."""
        sequence = ["ABD", "ANA", "CAU", "CTF", "DED", "PROB"]
        result = validate_sequence_with_profile(sequence, BASELINE_FIXED_PROFILE)
        assert result["passed"] is True
        assert result["profile"] == "baseline_fixed"

    def test_baseline_fixed_with_extra_families_fails(self):
        """Baseline fixed rechaza familias extra."""
        sequence = ["ABD", "HEUR", "ANA", "CAU", "CTF", "DED", "PROB"]
        result = validate_sequence_with_profile(sequence, BASELINE_FIXED_PROFILE)
        assert result["passed"] is False
        assert result["checks"]["no_unknown_families"] is False

    def test_adaptive_min_with_extra_families_passes(self):
        """Adaptive min acepta familias opcionales."""
        sequence = ["ABD", "HEUR", "ANA", "DIA_ADV", "CAU", "CTF", "DED", "FAL_GUARD", "PROB"]
        result = validate_sequence_with_profile(sequence, ADAPTIVE_MIN_PROFILE)
        assert result["passed"] is True
        assert result["checks"]["no_unknown_families"] is True

    def test_adaptive_min_rejects_unknown_family(self):
        """Adaptive min rechaza familias desconocidas."""
        sequence = ["ABD", "ANA", "CAU", "CTF", "DED", "UNKNOWN_FAM", "PROB"]
        result = validate_sequence_with_profile(sequence, ADAPTIVE_MIN_PROFILE)
        assert result["passed"] is False
        assert result["checks"]["no_unknown_families"] is False

    def test_adaptive_min_without_prob_closing_fails(self):
        """Adaptive min requiere PROB al final."""
        sequence = ["ABD", "ANA", "CAU", "CTF", "DED", "PROB", "HEUR"]
        result = validate_sequence_with_profile(sequence, ADAPTIVE_MIN_PROFILE)
        assert result["passed"] is False
        assert result["checks"]["prob_closes_calibration"] is False

    def test_lowercase_sequence_normalized(self):
        """Secuencias en lowercase se normalizan correctamente."""
        sequence = ["abd", "ana", "cau", "ctf", "ded", "prob"]
        result = validate_sequence_with_profile(sequence, BASELINE_FIXED_PROFILE)
        assert result["passed"] is True
        assert result["sequence_normalized"] == ["ABD", "ANA", "CAU", "CTF", "DED", "PROB"]


class TestEvaluateEpisodeClosureWithProfile:
    """Tests de integración para evaluate_episode_closure con perfiles."""

    def test_baseline_fixed_episode_passes(self, tmp_path: Path):
        """Episodio con secuencia baseline fija pasa."""
        storage = _storage(tmp_path)
        runner = MinimalCognitiveEpisodeRunner(
            storage=storage,
            run_id="run-baseline",
            closure_profile="baseline_fixed",
        )
        result = runner.run_episode(external_heat=0.05)

        closure = evaluate_episode_closure(
            storage=storage,
            run_id="run-baseline",
            result=result,
            closure_profile="baseline_fixed",
        )

        assert closure["closure_passed"] is True
        assert closure["closure_profile"] == "baseline_fixed"
        assert closure["sequence_validation"]["passed"] is True
        assert result["reasoning"]["mode"] == "fixed"
        storage.close()

    def test_adaptive_min_episode_passes(self, tmp_path: Path):
        """Episodio con secuencia adaptativa pasa con perfil adaptive_min."""
        storage = _storage(tmp_path)
        runner = MinimalCognitiveEpisodeRunner(
            storage=storage,
            run_id="run-adaptive",
            closure_profile="adaptive_min",
        )
        result = runner.run_episode(external_heat=0.05)

        closure = evaluate_episode_closure(
            storage=storage,
            run_id="run-adaptive",
            result=result,
            closure_profile="adaptive_min",
        )

        assert closure["closure_passed"] is True
        assert closure["closure_profile"] == "adaptive_min"
        assert result["reasoning"]["mode"] == "adaptive"
        storage.close()

    def test_backward_compatibility_default_profile(self, tmp_path: Path):
        """Sin especificar perfil, usa baseline_fixed por defecto."""
        storage = _storage(tmp_path)
        runner = MinimalCognitiveEpisodeRunner(storage=storage, run_id="run-default")
        result = runner.run_episode(external_heat=0.05)

        closure = evaluate_episode_closure(
            storage=storage,
            run_id="run-default",
            result=result,
        )

        assert closure["closure_passed"] is True
        assert closure["closure_profile"] == "baseline_fixed"
        assert result["reasoning"]["mode"] == "fixed"
        storage.close()


class TestMetaSchedulerAdaptiveWithClosure:
    """Tests de integración MetaScheduler adaptativo con validación de cierre."""

    def test_adaptive_scheduler_trace_validates_with_adaptive_profile(self, tmp_path: Path):
        """Trazas del scheduler adaptativo validan con perfil adaptive_min."""
        storage = _storage(tmp_path)

        scheduler = MetaScheduler(trace_store=storage, mode="adaptive")
        reasoning = scheduler.run(
            {
                "episode_id": "episode-adaptive-test",
                "run_id": "run-adaptive-scheduler",
            }
        )

        sequence = reasoning["sequence"]

        # Verificar que la secuencia pasa con perfil adaptativo
        result = validate_sequence_with_profile(sequence, ADAPTIVE_MIN_PROFILE)
        assert result["passed"] is True

        storage.close()
