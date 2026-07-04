from pathlib import Path

import pytest

from runtime.reality.collapse import CollapseDetector
from runtime.reality.service import RealityValidationService
from runtime.storage import StorageConfig, StorageFactory
from runtime.storage.records import RealityAssessmentRecord
from runtime.world.min_cognitive_episode import MinimalCognitiveEpisodeRunner


def _storage(tmp_path: Path):
    config = StorageConfig(
        mode="sqlite",
        sqlite_db_path=str(tmp_path / "reality.db"),
        postgres_dsn=None,
        artifact_root=tmp_path / "artifacts",
        prefer_postgres_reads=True,
        strict_dual_write=False,
    )
    return StorageFactory.create_facade(config)


def test_reality_benchmark_ci_persiste_reporte_y_summary(tmp_path: Path):
    storage = _storage(tmp_path)
    service = RealityValidationService(storage=storage)
    result = service.run_benchmark(run_id="run-reality-ci", gate_profile="ci")

    assert result["bench_run"]["total_episodes"] == 10
    assert len(result["assessments"]) == 10
    assert Path(result["artifact"]["abs_path"]).exists()

    bench_rows = storage.list_reality_bench_runs(run_id="run-reality-ci")
    assert bench_rows
    assessments = storage.list_reality_assessments(
        run_id="run-reality-ci", bench_run_id=bench_rows[0].bench_run_id, limit=50
    )
    assert len(assessments) == 10

    events = storage.list_events(run_id="run-reality-ci", limit=100)
    assert any(evt.event_type == "reality.validation.completed" for evt in events)
    storage.close()


def test_reality_continuidad_en_episodios_consecutivos(tmp_path: Path):
    storage = _storage(tmp_path)
    service = RealityValidationService(storage=storage)
    runner = MinimalCognitiveEpisodeRunner(storage=storage, run_id="run-reality-cont")
    bench_run_id = "bench-cont"

    first = runner.run_episode(external_heat=0.04)
    first_assessment = service.evaluate_episode_result(
        run_id="run-reality-cont",
        bench_run_id=bench_run_id,
        result=first,
        previous_result=None,
        recent_assessments=[],
    )
    second = runner.run_episode(external_heat=0.06)
    second_assessment = service.evaluate_episode_result(
        run_id="run-reality-cont",
        bench_run_id=bench_run_id,
        result=second,
        previous_result=first,
        recent_assessments=[first_assessment],
    )

    assert 0.0 <= second_assessment.continuity_score <= 1.0
    assert second_assessment.trace_integrity is True
    storage.close()


def test_reality_detecta_colapso_por_ruptura_de_cierre(tmp_path: Path):
    storage = _storage(tmp_path)
    service = RealityValidationService(storage=storage)
    runner = MinimalCognitiveEpisodeRunner(storage=storage, run_id="run-reality-collapse")
    result = runner.run_episode(external_heat=0.05)
    # Corrupción inducida para verificar detector.
    result["episode"]["context"]["formula"] = ""
    result["episode"]["trace"] = []

    assessment = service.evaluate_episode_result(
        run_id="run-reality-collapse",
        bench_run_id="bench-collapse",
        result=result,
        previous_result=None,
        recent_assessments=[],
    )
    assert assessment.closure_passed is False
    assert assessment.collapse_detected is True
    storage.close()


def test_collapse_detector_por_racha_de_baja_continuidad():
    detector = CollapseDetector(continuity_threshold=0.35, streak=3)
    recent = [
        RealityAssessmentRecord(
            assessment_id="a1",
            episode_id="e1",
            closure_passed=True,
            continuity_score=0.2,
            trace_integrity=True,
            collapse_detected=False,
        ),
        RealityAssessmentRecord(
            assessment_id="a2",
            episode_id="e2",
            closure_passed=True,
            continuity_score=0.3,
            trace_integrity=True,
            collapse_detected=False,
        ),
    ]
    assert (
        detector.detect(
            closure_passed=True,
            trace_integrity=True,
            continuity_score=0.25,
            recent_assessments=recent,
        )
        is True
    )


@pytest.mark.requires_extended_bench
def test_reality_benchmark_extendido_opt_in(tmp_path: Path):
    storage = _storage(tmp_path)
    service = RealityValidationService(storage=storage)
    result = service.run_benchmark(run_id="run-reality-extended", gate_profile="extended")
    assert result["bench_run"]["total_episodes"] == 100
    storage.close()
