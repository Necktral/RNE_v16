from pathlib import Path

import pytest

from runtime.reality.service import RealityValidationService
from runtime.storage import StorageConfig, StorageFactory
from runtime.symbolic.eml import (
    advisory_from_run,
    compare_baseline_vs_shadow,
    evaluate_advisory_promotion,
)
from runtime.symbolic.eml.benchmark import collect_eml_candidates_from_events


def _storage(tmp_path: Path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    config = StorageConfig(
        mode="sqlite",
        sqlite_db_path=str(tmp_path / "eml_benchmark.db"),
        postgres_dsn=None,
        artifact_root=tmp_path / "artifacts",
        prefer_postgres_reads=True,
        strict_dual_write=False,
    )
    return StorageFactory.create_facade(config)


def _pack():
    return [0.02, 0.03, 0.05, 0.07, 0.04, 0.06, 0.01, 0.08, 0.03, 0.05]


def test_eml_advisory_rejects_when_continuity_alert_even_with_support():
    candidates = [
        {"expr": {"op": "var", "name": "x"}, "composite_score": 0.92},
        {"expr": {"op": "var", "name": "x"}, "composite_score": 0.91},
        {"expr": {"op": "var", "name": "x"}, "composite_score": 0.9},
    ]
    decision = evaluate_advisory_promotion(candidates=candidates, continuity_alert=True)
    assert decision.promoted is False
    assert decision.reason == "continuity_alert"


@pytest.mark.requires_extended_bench
def test_eml_shadow_benchmark_no_regression_and_reproducible(tmp_path: Path, monkeypatch):
    storage = _storage(tmp_path)
    service = RealityValidationService(storage=storage)
    pack = _pack()

    monkeypatch.setenv("RNFE_EML_MODE", "disabled")
    baseline = service.run_benchmark(
        run_id="run-eml-baseline",
        gate_profile="ci",
        external_heat_values=pack,
    )

    monkeypatch.setenv("RNFE_EML_MODE", "shadow")
    monkeypatch.setenv("RNFE_EML_SEED", "123")
    monkeypatch.setenv("RNFE_EML_MAX_DEPTH", "3")
    monkeypatch.setenv("RNFE_EML_MAX_EVALS", "256")
    monkeypatch.setenv("RNFE_EML_MAX_CANDIDATES", "48")
    shadow_a = service.run_benchmark(
        run_id="run-eml-shadow-a",
        gate_profile="ci",
        external_heat_values=pack,
    )
    shadow_b = service.run_benchmark(
        run_id="run-eml-shadow-b",
        gate_profile="ci",
        external_heat_values=pack,
    )
    shadow_c = service.run_benchmark(
        run_id="run-eml-shadow-c",
        gate_profile="ci",
        external_heat_values=pack,
    )

    comparison = compare_baseline_vs_shadow(
        baseline_result=baseline,
        shadow_result=shadow_a,
    )
    assert comparison["no_regression"] is True

    cand_a = collect_eml_candidates_from_events(storage, run_id="run-eml-shadow-a")
    cand_b = collect_eml_candidates_from_events(storage, run_id="run-eml-shadow-b")
    cand_c = collect_eml_candidates_from_events(storage, run_id="run-eml-shadow-c")
    assert cand_a and cand_b and cand_c
    sig_a = str(cand_a[0]["expr"])
    sig_b = str(cand_b[0]["expr"])
    sig_c = str(cand_c[0]["expr"])
    assert sig_a == sig_b == sig_c

    advisory = advisory_from_run(storage, run_id="run-eml-shadow-a")
    assert advisory.support >= 3
    assert advisory.mean_score >= 0.0
    storage.close()

