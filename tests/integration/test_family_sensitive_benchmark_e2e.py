"""Integración E2E para benchmark sensible a familias y perfiles."""

from __future__ import annotations

import json
from pathlib import Path

from runtime.storage import StorageConfig
from runtime.world.grid_thermal_scenario import GridThermalScenario
from tests.benchmarks.analysis_report import BenchmarkAnalyzer
from tests.benchmarks.benchmark_runner import BenchmarkConfig, BenchmarkRunner


def _storage_config(tmp_path: Path) -> StorageConfig:
    return StorageConfig(
        mode="sqlite",
        sqlite_db_path=str(tmp_path / "family_sensitive.db"),
        postgres_dsn=None,
        artifact_root=tmp_path / "artifacts",
        prefer_postgres_reads=False,
        strict_dual_write=False,
    )


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _required_family_keys() -> set[str]:
    return {
        "family_activation_counts",
        "family_activation_presence",
        "family_activation_order",
        "family_mix_entropy",
        "family_core_only_flag",
        "family_optional_used_flag",
        "family_optional_count",
        "family_trace_by_family",
        "family_first_activation_step",
        "family_last_activation_step",
        "family_contribution_proxy",
        "family_delta_ivc_r",
        "family_delta_intervention_precision",
        "family_delta_viability_margin",
        "family_delta_reasoning_trace_length",
        "family_delta_success_rate",
        "family_delta_spatial_information_usage",
        "primary_regime_label",
        "cognitive_regime_label",
        "floor_regime_label",
        "mandatory_family_floor",
        "proposed_sequence",
        "validated_sequence",
        "sequence_validation_report",
        "admitted_overlays",
        "default_overlays",
        "correction_steps",
        "fallback_profile_name",
        "backbone_floor_satisfied_flag",
        "sequence_validation_fail_flag",
        "fallback_to_safe_sequence_flag",
        "optional_displacement_flag",
        "closure_break_flag",
        "sequence_autocorrected_flag",
        "budget_overridden_by_floor_flag",
    }


def test_core_only_profile_no_activa_opcionales(tmp_path: Path) -> None:
    runner = BenchmarkRunner(output_root=tmp_path / "out", storage_config=_storage_config(tmp_path))
    output_dir = tmp_path / "core_only_run"

    cfg = BenchmarkConfig(
        scenario_name="grid_thermal_5x5_core_only",
        scenario_class=GridThermalScenario,
        scenario_params={
            "grid_size": 5,
            "topology": "hotspot_center",
            "initial_temperature": 0.90,
            "alarm_threshold": 0.85,
            "cooling_effect": 0.07,
        },
        episodes=2,
        base_seed=100,
        max_steps=50,
        output_dir=output_dir,
        run_id="core-only-family-sensitive",
        reasoning_mode="fixed",
        family_profile="core_only",
        regime_label="heterogeneous_warning",
    )

    runner.run_benchmark(cfg)
    rows = _read_jsonl(output_dir / "episodes.jsonl")
    assert rows

    optional_families = {"HEUR", "DIA_ADV", "FAL_GUARD", "IND", "EML_SR"}
    for row in rows:
        assert _required_family_keys().issubset(set(row.keys()))
        assert row["family_optional_used_flag"] is False
        assert int(row["family_optional_count"]) == 0
        counts = row.get("family_activation_counts", {})
        for family in optional_families:
            assert int(counts.get(family, 0)) == 0


def test_adaptive_family_ecology_activa_opcionales_en_viability_edge(tmp_path: Path) -> None:
    runner = BenchmarkRunner(output_root=tmp_path / "out", storage_config=_storage_config(tmp_path))
    output_dir = tmp_path / "adaptive_run"

    cfg = BenchmarkConfig(
        scenario_name="grid_thermal_5x5_adaptive_ecology",
        scenario_class=GridThermalScenario,
        scenario_params={
            "grid_size": 5,
            "topology": "checkerboard",
            "initial_temperature": 0.95,
            "alarm_threshold": 0.90,
            "cooling_effect": 0.04,
        },
        episodes=2,
        base_seed=200,
        max_steps=50,
        output_dir=output_dir,
        run_id="adaptive-family-sensitive",
        reasoning_mode="adaptive",
        family_profile="adaptive_family_ecology",
        regime_label="viability_edge",
        # Presupuesto holgado: con la palanca 1 (ejecución de la secuencia
        # validada) los opcionales solo se EJECUTAN si caben sin desplazar al
        # núcleo. Con presupuesto ajustado el recorte los quitaría — eso es el
        # comportamiento correcto, no un fallo; aquí medimos la activación real.
        reasoning_max_steps=10,
    )

    runner.run_benchmark(cfg)
    rows = _read_jsonl(output_dir / "episodes.jsonl")
    assert rows

    assert any(bool(row.get("family_optional_used_flag")) for row in rows)
    assert any(int(row.get("family_optional_count", 0)) > 0 for row in rows)

    # Verifica activación de familias esperadas para borde de viabilidad.
    activated_viability_optional = False
    for row in rows:
        counts = row.get("family_activation_counts", {})
        if int(counts.get("DIA_ADV", 0)) > 0 or int(counts.get("FAL_GUARD", 0)) > 0:
            activated_viability_optional = True
    assert activated_viability_optional is True


def test_analyzer_compara_senales_family_sensitive(tmp_path: Path) -> None:
    runner = BenchmarkRunner(output_root=tmp_path / "out", storage_config=_storage_config(tmp_path))

    dir_core = tmp_path / "core_cmp"
    cfg_core = BenchmarkConfig(
        scenario_name="grid_thermal_1x1_core_only",
        scenario_class=GridThermalScenario,
        scenario_params={"grid_size": 1, "initial_temperature": 0.80},
        episodes=1,
        base_seed=300,
        max_steps=50,
        output_dir=dir_core,
        run_id="cmp-core",
        reasoning_mode="fixed",
        family_profile="core_only",
        regime_label="homogeneous_safe",
    )
    runner.run_benchmark(cfg_core)

    dir_adaptive = tmp_path / "adaptive_cmp"
    cfg_adaptive = BenchmarkConfig(
        scenario_name="grid_thermal_5x5_adaptive",
        scenario_class=GridThermalScenario,
        scenario_params={
            "grid_size": 5,
            "topology": "hotspot_center",
            "initial_temperature": 0.93,
            "alarm_threshold": 0.88,
            "cooling_effect": 0.05,
        },
        episodes=1,
        base_seed=400,
        max_steps=50,
        output_dir=dir_adaptive,
        run_id="cmp-adaptive",
        reasoning_mode="adaptive",
        family_profile="adaptive_family_ecology",
        regime_label="heterogeneous_warning",
    )
    runner.run_benchmark(cfg_adaptive)

    analyzer = BenchmarkAnalyzer()
    analyzer.load_results(dir_core, dir_adaptive)
    comparison = analyzer.compute_statistical_comparison()

    assert "family_mix_entropy" in comparison
    assert "optional_family_usage_rate" in comparison
    assert "family_specific_activation_counts" in comparison
    assert "family_contribution_proxy" in comparison
    assert "family_delta_ivc_r" in comparison


def test_adaptive_family_ecology_v2_preserva_floor_y_recupera_regimenes_criticos(tmp_path: Path) -> None:
    runner = BenchmarkRunner(output_root=tmp_path / "out", storage_config=_storage_config(tmp_path))
    regimes = {
        "heterogeneous_elevated": {
            "grid_size": 5,
            "topology": "gradient_ns",
            "initial_temperature": 0.78,
            "alarm_threshold": 0.85,
            "cooling_effect": 0.07,
        },
        "heterogeneous_warning": {
            "grid_size": 5,
            "topology": "checkerboard",
            "initial_temperature": 0.88,
            "alarm_threshold": 0.85,
            "cooling_effect": 0.06,
        },
        "viability_edge": {
            "grid_size": 5,
            "topology": "hotspot_center",
            "initial_temperature": 0.95,
            "alarm_threshold": 0.90,
            "cooling_effect": 0.04,
        },
        "vram_favorable": {
            "grid_size": 5,
            "topology": "uniform",
            "initial_temperature": 0.80,
            "alarm_threshold": 0.85,
            "cooling_effect": 0.07,
        },
    }

    expected_floors = {
        "heterogeneous_elevated": {"ABD", "ANA", "CAU", "CTF"},
        "heterogeneous_warning": {"ABD", "ANA", "CAU", "CTF", "DED"},
        "viability_edge": {"CAU", "CTF", "DED", "PROB"},
        "vram_favorable": {"ABD", "DED", "PROB"},
    }

    for idx, (regime, params) in enumerate(regimes.items()):
        dir_v1 = tmp_path / f"{regime}_v1"
        cfg_v1 = BenchmarkConfig(
            scenario_name=f"{regime}_adaptive_v1",
            scenario_class=GridThermalScenario,
            scenario_params=params,
            episodes=1,
            base_seed=500 + idx * 10,
            max_steps=50,
            output_dir=dir_v1,
            run_id=f"{regime}-v1",
            reasoning_mode="adaptive",
            family_profile="adaptive_family_ecology",
            regime_label=regime,
            reasoning_max_steps=None,
        )
        summary_v1 = runner.run_benchmark(cfg_v1)

        dir_v2 = tmp_path / f"{regime}_v2"
        cfg_v2 = BenchmarkConfig(
            scenario_name=f"{regime}_adaptive_v2",
            scenario_class=GridThermalScenario,
            scenario_params=params,
            episodes=1,
            base_seed=700 + idx * 10,
            max_steps=50,
            output_dir=dir_v2,
            run_id=f"{regime}-v2",
            reasoning_mode="adaptive",
            family_profile="adaptive_family_ecology_v2",
            regime_label=regime,
            reasoning_max_steps=None,
        )
        summary_v2 = runner.run_benchmark(cfg_v2)
        rows_v2 = _read_jsonl(dir_v2 / "episodes.jsonl")

        # Con la palanca 1 (ejecución de la secuencia VALIDADA) el perfil legacy v1
        # ya no rompe el cierre bajo presupuesto ajustado: también cierra. Lo que
        # distingue a v2 es la garantía de floor/no-desplazamiento, no que v1 falle.
        assert summary_v2["success_rate"] > 0.0
        assert summary_v2["optional_family_usage_rate"] > 0.0
        assert summary_v2["backbone_floor_satisfied_rate"] == 1.0
        assert summary_v2["optional_displacement_rate"] == 0.0
        assert rows_v2

        row = rows_v2[0]
        assert expected_floors[regime].issubset(set(row["validated_sequence"]))
        assert {"ABD", "ANA", "CAU", "CTF", "DED", "PROB"}.issubset(set(row["validated_sequence"]))
