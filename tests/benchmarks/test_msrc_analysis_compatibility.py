import json
from pathlib import Path

from tests.benchmarks.analysis_report import BenchmarkAnalyzer


def _write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_analysis_report_accepts_optional_msrc_metrics(tmp_path: Path):
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()

    base = {
        "success_rate": 1.0,
        "viability_margin": 0.2,
        "intervention_precision": 0.1,
        "proposition_diversity": 1.2,
        "spatial_information_usage": 0.25,
        "wall_time_ms": 30.0,
        "artifact_size_bytes": 3000,
        "reasoning_trace_length": 6,
        "ivc_r": 0.35,
        "backbone_floor_satisfied_flag": 1.0,
        "sequence_validation_fail_flag": 0.0,
        "fallback_to_safe_sequence_flag": 0.0,
        "optional_displacement_flag": 0.0,
        "closure_break_flag": 0.0,
    }

    rows_a = [
        {
            **base,
            "scale_selection_accuracy": 0.8,
            "keep_scale_rate": 0.6,
            "upgrade_rate": 0.3,
            "probe_rate": 0.1,
            "probe_commit_rate": 0.5,
            "missed_upgrade_regret": 1.0,
            "oscillation_rate": 0.1,
            "mean_resolution_cost": 1.2,
            "vram_headroom_mean": 0.5,
            "regime_probe_rate": 0.10,
        },
        {
            **base,
            "success_rate": 0.0,
            "scale_selection_accuracy": 0.7,
            "keep_scale_rate": 0.7,
            "upgrade_rate": 0.2,
            "probe_rate": 0.1,
            "probe_commit_rate": 0.4,
            "missed_upgrade_regret": 2.0,
            "oscillation_rate": 0.2,
            "mean_resolution_cost": 1.3,
            "vram_headroom_mean": 0.4,
            "regime_probe_rate": 0.08,
        },
    ]

    rows_b = [
        {
            **base,
            "viability_margin": 0.28,
            "intervention_precision": 0.16,
            "ivc_r": 0.44,
            "scale_selection_accuracy": 0.9,
            "keep_scale_rate": 0.4,
            "upgrade_rate": 0.4,
            "probe_rate": 0.2,
            "probe_commit_rate": 0.7,
            "missed_upgrade_regret": 0.5,
            "oscillation_rate": 0.05,
            "mean_resolution_cost": 2.2,
            "vram_headroom_mean": 0.38,
            "regime_probe_rate": 0.22,
        },
        {
            **base,
            "viability_margin": 0.26,
            "intervention_precision": 0.14,
            "ivc_r": 0.42,
            "scale_selection_accuracy": 0.88,
            "keep_scale_rate": 0.45,
            "upgrade_rate": 0.35,
            "probe_rate": 0.2,
            "probe_commit_rate": 0.65,
            "missed_upgrade_regret": 0.4,
            "oscillation_rate": 0.04,
            "mean_resolution_cost": 2.1,
            "vram_headroom_mean": 0.35,
            "regime_probe_rate": 0.24,
            "backbone_floor_satisfied_flag": 1.0,
            "sequence_validation_fail_flag": 0.0,
            "fallback_to_safe_sequence_flag": 0.0,
            "optional_displacement_flag": 0.0,
            "closure_break_flag": 0.0,
        },
    ]

    _write_jsonl(dir_a / "episodes.jsonl", rows_a)
    _write_jsonl(dir_b / "episodes.jsonl", rows_b)

    analyzer = BenchmarkAnalyzer()
    analyzer.load_results(dir_a, dir_b)
    comparison = analyzer.compute_statistical_comparison()

    assert "success_rate" in comparison
    assert "viability_margin" in comparison
    assert "scale_selection_accuracy" in comparison
    assert "keep_scale_rate" in comparison
    assert "upgrade_rate" in comparison
    assert "probe_rate" in comparison
    assert "probe_commit_rate" in comparison
    assert "missed_upgrade_regret" in comparison
    assert "oscillation_rate" in comparison
    assert "mean_resolution_cost" in comparison
    assert "vram_headroom_mean" in comparison
    assert "regime_probe_rate" in comparison
    assert "backbone_floor_satisfied_flag" in comparison
    assert "sequence_validation_fail_flag" in comparison
    assert "fallback_to_safe_sequence_flag" in comparison
    assert "optional_displacement_flag" in comparison
    assert "closure_break_flag" in comparison
    assert "net_cognitive_gain" in comparison
