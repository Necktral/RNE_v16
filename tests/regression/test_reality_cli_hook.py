"""Tests de integración para CLI y hook de reality validation."""

from pathlib import Path

import pytest

from runtime.reality.cli import main as cli_main, run_reality_validation
from runtime.reality.hook import RealityValidationHook
from runtime.storage import StorageConfig, StorageFactory


def _storage(tmp_path: Path):
    config = StorageConfig(
        mode="sqlite",
        sqlite_db_path=str(tmp_path / "reality_cli.db"),
        postgres_dsn=None,
        artifact_root=tmp_path / "artifacts",
        prefer_postgres_reads=True,
        strict_dual_write=False,
    )
    return StorageFactory.create_facade(config)


class TestRealityValidationCLI:
    """Tests para el entrypoint CLI de validación de realidad."""

    def test_cli_main_returns_zero_on_ci_profile_pass(self, tmp_path: Path, monkeypatch):
        """El CLI retorna 0 cuando el gate CI pasa."""
        storage = _storage(tmp_path)
        monkeypatch.setattr("runtime.reality.cli.get_storage", lambda: storage)

        exit_code = cli_main(["--profile", "ci", "--quiet"])

        assert exit_code == 0
        storage.close()

    def test_cli_main_json_output(self, tmp_path: Path, monkeypatch, capsys):
        """El CLI imprime JSON válido con --json."""
        import json

        storage = _storage(tmp_path)
        monkeypatch.setattr("runtime.reality.cli.get_storage", lambda: storage)

        exit_code = cli_main(["--profile", "ci", "--json"])

        captured = capsys.readouterr()
        output = json.loads(captured.out)

        assert exit_code == 0
        assert "bench_run" in output
        assert "assessments" in output
        assert "passed" in output
        storage.close()

    def test_run_reality_validation_function_returns_result(self, tmp_path: Path, monkeypatch):
        """La función run_reality_validation retorna resultado con estructura esperada."""
        storage = _storage(tmp_path)
        monkeypatch.setattr("runtime.reality.cli.get_storage", lambda: storage)

        result = run_reality_validation(gate_profile="ci")

        assert "bench_run" in result
        assert "assessments" in result
        assert "artifact" in result
        assert "passed" in result
        assert result["bench_run"]["total_episodes"] == 10
        storage.close()


class TestRealityValidationHook:
    """Tests para el hook de validación de realidad."""

    def test_hook_run_validation_persists_result(self, tmp_path: Path):
        """El hook ejecuta validación y persiste resultado."""
        storage = _storage(tmp_path)
        hook = RealityValidationHook(storage=storage, gate_profile="ci", enabled=True)

        result = hook.run_validation(run_id="run-hook-test")

        assert result["passed"] is True
        assert hook.last_result is not None
        assert hook.passed() is True
        storage.close()

    def test_hook_disabled_skips_validation(self, tmp_path: Path):
        """El hook deshabilitado retorna skipped sin ejecutar."""
        storage = _storage(tmp_path)
        hook = RealityValidationHook(storage=storage, gate_profile="ci", enabled=False)

        result = hook.run_validation()

        assert result["skipped"] is True
        assert result["passed"] is True
        storage.close()

    def test_hook_on_shutdown_executes_when_configured(self, tmp_path: Path):
        """El hook ejecuta validación en on_shutdown cuando run_on_shutdown=True."""
        storage = _storage(tmp_path)
        hook = RealityValidationHook(
            storage=storage,
            gate_profile="ci",
            enabled=True,
            run_on_shutdown=True,
        )

        result = hook.on_shutdown(run_id="run-shutdown-test")

        assert result is not None
        assert result["passed"] is True
        storage.close()

    def test_hook_on_shutdown_skips_when_not_configured(self, tmp_path: Path):
        """El hook no ejecuta validación en on_shutdown cuando run_on_shutdown=False."""
        storage = _storage(tmp_path)
        hook = RealityValidationHook(
            storage=storage,
            gate_profile="ci",
            enabled=True,
            run_on_shutdown=False,
        )

        result = hook.on_shutdown(run_id="run-skip-test")

        assert result is None
        storage.close()

    def test_hook_passed_returns_true_before_any_validation(self, tmp_path: Path):
        """El hook retorna True en passed() si no se ha ejecutado validación."""
        storage = _storage(tmp_path)
        hook = RealityValidationHook(storage=storage, gate_profile="ci", enabled=True)

        assert hook.passed() is True
        storage.close()


class TestRealityValidationIntegration:
    """Tests de integración end-to-end runner → service → storage."""

    def test_cli_persists_events_and_artifacts(self, tmp_path: Path, monkeypatch):
        """El CLI persiste eventos y artifacts correctamente."""
        storage = _storage(tmp_path)
        monkeypatch.setattr("runtime.reality.cli.get_storage", lambda: storage)

        result = run_reality_validation(
            gate_profile="ci",
            run_id="run-integration-test",
        )

        # Verificar evento persistido
        events = storage.list_events(run_id="run-integration-test", limit=100)
        reality_events = [e for e in events if e.event_type == "reality.validation.completed"]
        assert len(reality_events) >= 1

        # Verificar artifact materializado
        artifact_path = Path(result["artifact"]["abs_path"])
        assert artifact_path.exists()

        # Verificar assessments persistidos
        bench_run_id = result["bench_run"]["bench_run_id"]
        assessments = storage.list_reality_assessments(
            run_id="run-integration-test",
            bench_run_id=bench_run_id,
            limit=50,
        )
        assert len(assessments) == 10

        storage.close()
