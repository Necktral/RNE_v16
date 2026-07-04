#!/usr/bin/env python3
"""Campaña comparativa perfil de familias x régimen.

Salida principal:
- episodes.jsonl y summary.json por celda
- family_profile_regime_table.json
- family_profile_regime_table.md
- family_verdict_report.json
- msrc_cross_summary.json (si está habilitado)
"""

from __future__ import annotations

import argparse
import json
import os
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime.reality.service import RealityValidationService
from runtime.storage import StorageConfig, StorageFactory
from runtime.world.grid_thermal_scenario import GridThermalScenario
from tests.benchmarks.benchmark_runner import BenchmarkConfig, BenchmarkRunner


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


@contextmanager
def _temporary_env(updates: Dict[str, str | None]):
    previous = {k: os.environ.get(k) for k in updates}
    try:
        for key, value in updates.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _mode_for_profile(profile: str) -> str:
    if profile in {"adaptive_family_ecology", "adaptive_family_ecology_v2", "full_family_exploration"}:
        return "adaptive"
    return "fixed"


def _regime_specs() -> Dict[str, Dict[str, Any]]:
    return {
        "homogeneous_safe": {
            "scenario_params": {
                "grid_size": 1,
                "initial_temperature": 0.62,
                "alarm_threshold": 0.85,
                "cooling_effect": 0.07,
            }
        },
        "heterogeneous_elevated": {
            "scenario_params": {
                "grid_size": 5,
                "topology": "gradient_ns",
                "initial_temperature": 0.78,
                "alarm_threshold": 0.85,
                "cooling_effect": 0.07,
            }
        },
        "heterogeneous_warning": {
            "scenario_params": {
                "grid_size": 5,
                "topology": "checkerboard",
                "initial_temperature": 0.88,
                "alarm_threshold": 0.85,
                "cooling_effect": 0.06,
            }
        },
        "viability_edge": {
            "scenario_params": {
                "grid_size": 5,
                "topology": "hotspot_center",
                "initial_temperature": 0.95,
                "alarm_threshold": 0.90,
                "cooling_effect": 0.04,
            }
        },
        "vram_favorable": {
            "scenario_params": {
                "grid_size": 5,
                "topology": "uniform",
                "initial_temperature": 0.80,
                "alarm_threshold": 0.85,
                "cooling_effect": 0.07,
            }
        },
    }


def _profiles(include_full: bool) -> List[str]:
    base = [
        "core_only",
        "core_plus_heur",
        "core_plus_dialectic",
        "core_plus_guard",
        "adaptive_family_ecology",
        "adaptive_family_ecology_v2",
    ]
    if include_full:
        base.append("full_family_exploration")
    return base


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True, default=str), encoding="utf-8")


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _row_from_summary(profile: str, regime: str, summary: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
    avg = summary.get("avg_metrics", {}) or {}
    return {
        "profile": profile,
        "regime": regime,
        "ivc_r": float(avg.get("ivc_r", 0.0) or 0.0),
        "intervention_precision": float(avg.get("intervention_precision", 0.0) or 0.0),
        "success_rate": float(summary.get("success_rate", 0.0) or 0.0),
        "viability_margin": float(avg.get("viability_margin", 0.0) or 0.0),
        "wall_time_ms": float(avg.get("wall_time_ms", 0.0) or 0.0),
        "artifact_size_bytes": float(avg.get("artifact_size_bytes", 0.0) or 0.0),
        "reasoning_trace_length": float(avg.get("reasoning_trace_length", 0.0) or 0.0),
        "family_mix_entropy": float(avg.get("family_mix_entropy", 0.0) or 0.0),
        "optional_family_usage_rate": float(summary.get("optional_family_usage_rate", 0.0) or 0.0),
        "backbone_floor_satisfied_rate": float(summary.get("backbone_floor_satisfied_rate", 0.0) or 0.0),
        "sequence_validation_fail_rate": float(summary.get("sequence_validation_fail_rate", 0.0) or 0.0),
        "fallback_to_safe_sequence_rate": float(summary.get("fallback_to_safe_sequence_rate", 0.0) or 0.0),
        "optional_displacement_rate": float(summary.get("optional_displacement_rate", 0.0) or 0.0),
        "closure_break_rate": float(summary.get("closure_break_rate", 0.0) or 0.0),
        "family_specific_activation_counts": summary.get("family_specific_activation_counts", {}),
        "summary_path": str(output_dir / "summary.json"),
        "episodes_path": str(output_dir / "episodes.jsonl"),
    }


def _dictamen(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_regime: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for row in rows:
        by_regime.setdefault(row["regime"], {})[row["profile"]] = row

    critical_regimes = {
        "heterogeneous_elevated",
        "heterogeneous_warning",
        "viability_edge",
        "vram_favorable",
    }
    recovered_regimes = []
    optional_usage_regimes = []
    evidence = []
    for regime, table in sorted(by_regime.items()):
        core = table.get("core_only")
        adaptive_v1 = table.get("adaptive_family_ecology")
        adaptive_v2 = table.get("adaptive_family_ecology_v2")
        if not core or not adaptive_v1 or not adaptive_v2:
            continue

        optional_used = float(adaptive_v2.get("optional_family_usage_rate", 0.0)) > 0.0
        if optional_used:
            optional_usage_regimes.append(regime)

        deltas_vs_core = {
            "ivc_r": float(adaptive_v2["ivc_r"]) - float(core["ivc_r"]),
            "intervention_precision": float(adaptive_v2["intervention_precision"]) - float(core["intervention_precision"]),
            "viability_margin": float(adaptive_v2["viability_margin"]) - float(core["viability_margin"]),
            "success_rate": float(adaptive_v2["success_rate"]) - float(core["success_rate"]),
        }
        recovered = (
            float(adaptive_v1.get("success_rate", 0.0)) <= 0.0
            and float(adaptive_v2.get("success_rate", 0.0)) > 0.0
            and float(adaptive_v2.get("backbone_floor_satisfied_rate", 0.0)) >= 1.0
            and float(adaptive_v2.get("optional_displacement_rate", 0.0)) <= 0.0
        )
        if recovered:
            recovered_regimes.append(regime)

        evidence.append(
            {
                "regime": regime,
                "optional_used": optional_used,
                "recovered_vs_v1": recovered,
                "v1_success_rate": float(adaptive_v1.get("success_rate", 0.0) or 0.0),
                "v2_success_rate": float(adaptive_v2.get("success_rate", 0.0) or 0.0),
                "v2_backbone_floor_satisfied_rate": float(adaptive_v2.get("backbone_floor_satisfied_rate", 0.0) or 0.0),
                "v2_optional_displacement_rate": float(adaptive_v2.get("optional_displacement_rate", 0.0) or 0.0),
                "v2_closure_break_rate": float(adaptive_v2.get("closure_break_rate", 0.0) or 0.0),
                "deltas_vs_core": deltas_vs_core,
            }
        )

    critical_recovered = sorted(regime for regime in recovered_regimes if regime in critical_regimes)
    if len(critical_recovered) == len(critical_regimes) and optional_usage_regimes:
        positive_deltas = sum(
            1
            for row in evidence
            if row["regime"] in critical_regimes
            and (
                row["deltas_vs_core"]["ivc_r"] > 0.0
                or row["deltas_vs_core"]["intervention_precision"] > 0.0
                or row["deltas_vs_core"]["viability_margin"] > 0.0
            )
        )
        verdict = (
            "ecología adaptativa v2 ya es segura y útil"
            if positive_deltas >= 2
            else "v2 corrige cierre pero aún aporta poco"
        )
    else:
        verdict = "v2 insuficiente"

    return {
        "dictamen": verdict,
        "recovered_regimes": recovered_regimes,
        "critical_recovered_regimes": critical_recovered,
        "optional_usage_regimes": optional_usage_regimes,
        "evidence": evidence,
    }


def _write_markdown_table(path: Path, rows: List[Dict[str, Any]]) -> None:
    lines = [
        "# Family Profile vs Regime",
        "",
        "| Profile | Regime | ivc_r | intervention_precision | success_rate | viability_margin | wall_time_ms | artifact_size_bytes | reasoning_trace_length | family_mix_entropy | optional_family_usage_rate | backbone_floor_satisfied_rate | sequence_validation_fail_rate | fallback_to_safe_sequence_rate | optional_displacement_rate | closure_break_rate | family_specific_activation_counts |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {profile} | {regime} | {ivc_r:.4f} | {intervention_precision:.4f} | "
            "{success_rate:.4f} | {viability_margin:.4f} | {wall_time_ms:.2f} | "
            "{artifact_size_bytes:.1f} | {reasoning_trace_length:.2f} | {family_mix_entropy:.4f} | "
            "{optional_family_usage_rate:.4f} | {backbone_floor_satisfied_rate:.4f} | "
            "{sequence_validation_fail_rate:.4f} | {fallback_to_safe_sequence_rate:.4f} | "
            "{optional_displacement_rate:.4f} | {closure_break_rate:.4f} | "
            "`{family_specific_activation_counts}` |".format(**row)
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _run_msrc_cross(
    *,
    service: RealityValidationService,
    output_root: Path,
    rows: List[Dict[str, Any]],
    episodes: int,
    seed_base: int,
    policies: List[str],
) -> List[Dict[str, Any]]:
    msrc_rows: List[Dict[str, Any]] = []
    regime_specs = _regime_specs()
    for row in rows:
        profile = row["profile"]
        regime = row["regime"]
        mode = _mode_for_profile(profile)
        scenario_params = dict(regime_specs[regime]["scenario_params"])
        for idx, policy in enumerate(policies):
            run_id = f"msrc-{profile}-{regime}-{policy}".replace("_", "-")
            out_dir = output_root / "msrc_cross" / profile / regime / policy
            env_updates = {
                "RNFE_REASONING_MODE": mode,
                "RNFE_REASONING_FAMILY_PROFILE": profile,
                "RNFE_REASONING_REGIME_HINT": regime,
            }
            with _temporary_env(env_updates):
                result = service.run_msrc_policy_benchmark(
                    run_id=run_id,
                    policy_name=policy,
                    episodes=episodes,
                    base_seed=seed_base + idx,
                    output_dir=out_dir,
                    scenario_params=scenario_params,
                    external_input=0.04,
                    level_label=regime,
                    topology_label=str(scenario_params.get("topology", "uniform")),
                )
            summary = result.get("summary", {})
            avg = summary.get("avg_metrics", {})
            msrc_rows.append(
                {
                    "profile": profile,
                    "regime": regime,
                    "policy": policy,
                    "success_rate": float(summary.get("success_rate", 0.0) or 0.0),
                    "ivc_r": float(avg.get("ivc_r", 0.0) or 0.0),
                    "intervention_precision": float(avg.get("intervention_precision", 0.0) or 0.0),
                    "viability_margin": float(avg.get("viability_margin", 0.0) or 0.0),
                    "family_mix_entropy": float(avg.get("family_mix_entropy", 0.0) or 0.0),
                    "optional_family_usage_rate": float(summary.get("optional_family_usage_rate", 0.0) or 0.0),
                    "summary_path": str(out_dir / "summary.json"),
                }
            )
    return msrc_rows


def run_campaign(args: argparse.Namespace) -> int:
    campaign_root = Path(args.output_root) / args.campaign_id
    campaign_root.mkdir(parents=True, exist_ok=True)

    storage_cfg = StorageConfig(
        mode="sqlite",
        sqlite_db_path=str(Path(args.db_path)),
        postgres_dsn=None,
        artifact_root=Path(args.artifact_root),
        prefer_postgres_reads=False,
        strict_dual_write=False,
    )

    runner = BenchmarkRunner(output_root=campaign_root, storage_config=storage_cfg)

    profiles = _profiles(args.include_full_profile)
    regime_specs = _regime_specs()
    table_rows: List[Dict[str, Any]] = []

    for p_idx, profile in enumerate(profiles):
        mode = _mode_for_profile(profile)
        for r_idx, (regime, spec) in enumerate(regime_specs.items()):
            run_id = f"family-{profile}-{regime}".replace("_", "-")
            output_dir = campaign_root / "runs" / profile / regime
            cfg = BenchmarkConfig(
                scenario_name=f"{regime}__{profile}",
                scenario_class=GridThermalScenario,
                scenario_params=dict(spec["scenario_params"]),
                episodes=args.episodes_per_cell,
                base_seed=args.seed_base + p_idx * 1000 + r_idx * 100,
                max_steps=50,
                output_dir=output_dir,
                run_id=run_id,
                reasoning_mode=mode,
                family_profile=profile,
                regime_label=regime,
                reasoning_max_steps=args.reasoning_max_steps,
            )
            summary = runner.run_benchmark(cfg)
            table_rows.append(_row_from_summary(profile, regime, summary, output_dir))

    table_json = campaign_root / "family_profile_regime_table.json"
    table_md = campaign_root / "family_profile_regime_table.md"
    _write_json(table_json, table_rows)
    _write_markdown_table(table_md, table_rows)

    verdict = _dictamen(table_rows)

    out_verdict = {
        "campaign_id": args.campaign_id,
        "generated_at": datetime.now().isoformat(),
        "episodes_per_cell": args.episodes_per_cell,
        "profiles": profiles,
        "regimes": list(regime_specs.keys()),
        "table_path": str(table_json),
        **verdict,
    }

    # Cruce MSRC opcional
    if args.run_msrc_cross:
        storage = StorageFactory.create_facade(storage_cfg)
        service = RealityValidationService(storage=storage)
        msrc_rows = _run_msrc_cross(
            service=service,
            output_root=campaign_root,
            rows=table_rows,
            episodes=args.episodes_per_cell,
            seed_base=args.seed_base + 50_000,
            policies=args.msrc_policies,
        )
        _write_json(campaign_root / "msrc_cross_summary.json", msrc_rows)
        out_verdict["msrc_cross_runs"] = len(msrc_rows)
        out_verdict["msrc_cross_mean_success_rate"] = _mean(float(r.get("success_rate", 0.0)) for r in msrc_rows)
        storage.close()

    _write_json(campaign_root / "family_verdict_report.json", out_verdict)

    print("FAMILY_PROFILE_BENCHMARK_DONE")
    print(f"campaign_id={args.campaign_id}")
    print(f"campaign_root={campaign_root}")
    print(f"table_json={table_json}")
    print(f"table_md={table_md}")
    print(f"verdict={out_verdict['dictamen']}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Campaña perfil de familias x régimen.")
    parser.add_argument("--campaign-id", default=f"family_profiles_{_now_stamp()}")
    parser.add_argument("--output-root", default="data/benchmarks/family_profiles")
    parser.add_argument("--db-path", default="aeon_event_log.db")
    parser.add_argument("--artifact-root", default="data/artifacts")
    parser.add_argument("--episodes-per-cell", type=int, default=8)
    parser.add_argument("--seed-base", type=int, default=710000)
    parser.add_argument("--reasoning-max-steps", type=int, default=None)
    parser.add_argument("--include-full-profile", action="store_true")
    parser.add_argument("--run-msrc-cross", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--msrc-policies",
        nargs="+",
        default=[
            "adaptive_msrc",
            "adaptive_msrc_aggressive",
            "adaptive_msrc_regime_v3",
            "probe_before_switch",
        ],
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return run_campaign(args)


if __name__ == "__main__":
    raise SystemExit(main())
