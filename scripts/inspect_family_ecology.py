#!/usr/bin/env python3
"""Diagnóstico rápido de ecología de familias por perfil/régimen."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import sys
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime.storage import StorageConfig
from runtime.world.grid_thermal_scenario import GridThermalScenario
from tests.benchmarks.benchmark_runner import BenchmarkConfig, BenchmarkRunner


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True, default=str), encoding="utf-8")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _mode_for_profile(profile: str) -> str:
    return "adaptive" if profile in {"adaptive_family_ecology", "adaptive_family_ecology_v2", "full_family_exploration"} else "fixed"


def _regime_scenario(regime: str) -> Dict[str, Any]:
    mapping = {
        "homogeneous_safe": {
            "grid_size": 1,
            "initial_temperature": 0.62,
            "alarm_threshold": 0.85,
            "cooling_effect": 0.07,
        },
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
    return dict(mapping.get(regime, mapping["viability_edge"]))


def _run_profile(
    *,
    runner: BenchmarkRunner,
    run_id: str,
    profile: str,
    regime: str,
    episodes: int,
    output_dir: Path,
    seed_base: int,
    reasoning_max_steps: int | None,
) -> Dict[str, Any]:
    cfg = BenchmarkConfig(
        scenario_name=f"diag_{regime}_{profile}",
        scenario_class=GridThermalScenario,
        scenario_params=_regime_scenario(regime),
        episodes=episodes,
        base_seed=seed_base,
        max_steps=50,
        output_dir=output_dir,
        run_id=run_id,
        reasoning_mode=_mode_for_profile(profile),
        family_profile=profile,
        regime_label=regime,
        reasoning_max_steps=reasoning_max_steps,
    )
    summary = runner.run_benchmark(cfg)
    rows = _read_jsonl(output_dir / "episodes.jsonl")
    return {
        "summary": summary,
        "episodes": rows,
        "paths": {
            "summary": str(output_dir / "summary.json"),
            "episodes": str(output_dir / "episodes.jsonl"),
        },
    }


def run_diagnostic(args: argparse.Namespace) -> Dict[str, Any]:
    out_root = Path(args.output_root) / args.run_id
    out_root.mkdir(parents=True, exist_ok=True)

    storage_cfg = StorageConfig(
        mode="sqlite",
        sqlite_db_path=str(Path(args.db_path)),
        postgres_dsn=None,
        artifact_root=Path(args.artifact_root),
        prefer_postgres_reads=False,
        strict_dual_write=False,
    )
    runner = BenchmarkRunner(output_root=out_root, storage_config=storage_cfg)

    target = _run_profile(
        runner=runner,
        run_id=f"{args.run_id}-{args.profile}",
        profile=args.profile,
        regime=args.regime,
        episodes=args.episodes,
        output_dir=out_root / args.profile,
        seed_base=args.seed_base,
        reasoning_max_steps=args.reasoning_max_steps,
    )

    core = _run_profile(
        runner=runner,
        run_id=f"{args.run_id}-core_only",
        profile="core_only",
        regime=args.regime,
        episodes=args.episodes,
        output_dir=out_root / "core_only",
        seed_base=args.seed_base + 500,
        reasoning_max_steps=args.reasoning_max_steps,
    )

    target_row = target["episodes"][0] if target["episodes"] else {}
    core_summary = core["summary"]
    target_summary = target["summary"]

    avg_t = target_summary.get("avg_metrics", {})
    avg_c = core_summary.get("avg_metrics", {})

    activated = [
        fam for fam, cnt in (target_row.get("family_activation_counts", {}) or {}).items() if int(cnt) > 0
    ]

    report = {
        "generated_at": datetime.now().isoformat(),
        "run_id": args.run_id,
        "profile": args.profile,
        "regime": args.regime,
        "primary_regime_label": target_row.get("primary_regime_label"),
        "cognitive_regime_label": target_row.get("cognitive_regime_label"),
        "floor_regime_label": target_row.get("floor_regime_label"),
        "mandatory_family_floor": target_row.get("mandatory_family_floor", []),
        "sequence_proposed": target_row.get("proposed_sequence", []),
        "sequence_validated": target_row.get("validated_sequence", []),
        "sequence_executed": target_row.get("family_activation_order", []),
        "families_activated": activated,
        "sequence_validation": target_row.get("sequence_validation_report", {}),
        "sequence_validation_fail_flag": target_row.get("sequence_validation_fail_flag"),
        "backbone_floor_satisfied_flag": target_row.get("backbone_floor_satisfied_flag"),
        "fallback_to_safe_sequence_flag": target_row.get("fallback_to_safe_sequence_flag"),
        "sequence_autocorrected_flag": target_row.get("sequence_autocorrected_flag"),
        "optional_displacement_flag": target_row.get("optional_displacement_flag"),
        "closure_break_flag": target_row.get("closure_break_flag"),
        "admitted_overlays": target_row.get("admitted_overlays", []),
        "default_overlays": target_row.get("default_overlays", []),
        "correction_steps": target_row.get("correction_steps", []),
        "fallback_profile_name": target_row.get("fallback_profile_name"),
        "metrics_by_family": {
            "family_activation_counts": target_row.get("family_activation_counts", {}),
            "family_contribution_proxy": target_row.get("family_contribution_proxy", {}),
            "family_delta_ivc_r": target_row.get("family_delta_ivc_r", {}),
            "family_delta_intervention_precision": target_row.get("family_delta_intervention_precision", {}),
            "family_delta_viability_margin": target_row.get("family_delta_viability_margin", {}),
            "family_delta_reasoning_trace_length": target_row.get("family_delta_reasoning_trace_length", {}),
            "family_delta_success_rate": target_row.get("family_delta_success_rate", {}),
            "family_delta_spatial_information_usage": target_row.get("family_delta_spatial_information_usage", {}),
        },
        "serialized_outputs": {
            "target": target["paths"],
            "core_only": core["paths"],
        },
        "quick_comparison_vs_core_only": {
            "ivc_r_delta": float(avg_t.get("ivc_r", 0.0) or 0.0) - float(avg_c.get("ivc_r", 0.0) or 0.0),
            "intervention_precision_delta": float(avg_t.get("intervention_precision", 0.0) or 0.0)
            - float(avg_c.get("intervention_precision", 0.0) or 0.0),
            "viability_margin_delta": float(avg_t.get("viability_margin", 0.0) or 0.0)
            - float(avg_c.get("viability_margin", 0.0) or 0.0),
            "success_rate_delta": float(target_summary.get("success_rate", 0.0) or 0.0)
            - float(core_summary.get("success_rate", 0.0) or 0.0),
            "family_mix_entropy_delta": float(avg_t.get("family_mix_entropy", 0.0) or 0.0)
            - float(avg_c.get("family_mix_entropy", 0.0) or 0.0),
            "optional_family_usage_rate_target": float(target_summary.get("optional_family_usage_rate", 0.0) or 0.0),
            "optional_family_usage_rate_core_only": float(core_summary.get("optional_family_usage_rate", 0.0) or 0.0),
        },
    }

    out_file = out_root / "family_ecology_report.json"
    _write_json(out_file, report)

    print("FAMILY_ECOLOGY_DIAGNOSTIC_DONE")
    print(f"report={out_file}")
    print(f"profile={args.profile}")
    print(f"regime={args.regime}")
    print(f"primary_regime={report['primary_regime_label']}")
    print(f"cognitive_regime={report['cognitive_regime_label']}")
    print(f"floor_regime={report['floor_regime_label']}")
    print(f"mandatory_floor={report['mandatory_family_floor']}")
    print(f"sequence_proposed={report['sequence_proposed']}")
    print(f"sequence_validated={report['sequence_validated']}")
    print(f"sequence_executed={report['sequence_executed']}")
    print(f"correction_steps={report['correction_steps']}")
    print(f"fallback_to_safe_sequence={report['fallback_to_safe_sequence_flag']}")
    print(f"families_activated={report['families_activated']}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspección de ecología de familias.")
    parser.add_argument("--run-id", default=f"family_ecology_{_now_stamp()}")
    parser.add_argument("--output-root", default="data/diagnostics/family_ecology")
    parser.add_argument("--db-path", default="aeon_event_log.db")
    parser.add_argument("--artifact-root", default="data/artifacts")
    parser.add_argument("--profile", default="adaptive_family_ecology_v2")
    parser.add_argument("--regime", default="viability_edge")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--seed-base", type=int, default=880000)
    parser.add_argument("--reasoning-max-steps", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_diagnostic(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
