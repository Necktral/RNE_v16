#!/usr/bin/env python3
"""Campaña comparativa de políticas MSRC.

Genera artifacts:
- manifest.json
- policy_runs_index.jsonl
- scale_decisions.jsonl (agregado)
- policy_summary.json
- final_report.md
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime.reality.service import RealityValidationService
from runtime.storage import StorageConfig, StorageFactory


LEVELS = {
    "SAFE": {"initial_temperature": 0.60, "alarm_threshold": 0.85, "cooling_effect": 0.07},
    "ELEVATED": {"initial_temperature": 0.78, "alarm_threshold": 0.85, "cooling_effect": 0.07},
    "WARNING": {"initial_temperature": 0.87, "alarm_threshold": 0.85, "cooling_effect": 0.07},
    "CRITICAL": {"initial_temperature": 0.95, "alarm_threshold": 0.85, "cooling_effect": 0.07},
}

MATRIX_2_LEVELS = ["ELEVATED", "WARNING", "CRITICAL"]
MATRIX_2_TOPOLOGIES = ["uniform", "hotspot_center", "gradient_ns", "checkerboard"]


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True, default=str) + "\n")


def _load_summary(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def run_campaign(args: argparse.Namespace) -> int:
    campaign_root = Path(args.output_root) / args.campaign_id
    campaign_root.mkdir(parents=True, exist_ok=True)

    storage_config = StorageConfig(
        mode="sqlite",
        sqlite_db_path=str(Path(args.db_path)),
        postgres_dsn=None,
        artifact_root=Path(args.artifact_root),
        prefer_postgres_reads=False,
        strict_dual_write=False,
    )
    storage = StorageFactory.create_facade(storage_config)
    service = RealityValidationService(storage=storage)

    runs_index = campaign_root / "policy_runs_index.jsonl"
    decisions_agg = campaign_root / "scale_decisions.jsonl"

    manifest = {
        "campaign_id": args.campaign_id,
        "created_at": datetime.now().isoformat(),
        "db_path": str(Path(args.db_path)),
        "artifact_root": str(Path(args.artifact_root)),
        "policies": args.policies,
        "matrix_1": {
            "levels": list(LEVELS.keys()),
            "episodes_per_cell": args.matrix1_episodes,
        },
        "matrix_2": {
            "levels": MATRIX_2_LEVELS,
            "topologies": MATRIX_2_TOPOLOGIES,
            "episodes_per_cell": args.matrix2_episodes,
        },
        "proxy_mapping": {
            "closure_rate": "success_rate",
            "continuity_mean": "viability_margin",
        },
    }
    _write_json(campaign_root / "manifest.json", manifest)

    run_records: List[Dict[str, Any]] = []

    # Matriz 1: controlada por niveles
    for policy_name in args.policies:
        for level_name, params in LEVELS.items():
            run_id = f"{args.campaign_id}-M1-{policy_name}-{level_name}"
            out_dir = campaign_root / "matrix_1" / policy_name / level_name.lower()
            result = service.run_msrc_policy_benchmark(
                run_id=run_id,
                policy_name=policy_name,
                episodes=args.matrix1_episodes,
                base_seed=args.seed_base + hash(f"M1-{level_name}") % 10000,
                output_dir=out_dir,
                scenario_params={**params, "grid_size": 1},
                external_input=args.external_input,
                level_label=level_name,
                topology_label="uniform",
            )
            summary = result["summary"]
            record = {
                "matrix": "M1",
                "run_id": run_id,
                "policy": policy_name,
                "level": level_name,
                "topology": "uniform",
                "episodes": args.matrix1_episodes,
                "output_dir": str(out_dir),
                "summary": summary,
            }
            run_records.append(record)
            _append_jsonl(runs_index, record)
            _concat_scale_decisions(out_dir / "scale_decisions.jsonl", decisions_agg)

    # Matriz 2: heterogénea
    for policy_name in args.policies:
        for level_name in MATRIX_2_LEVELS:
            params = LEVELS[level_name]
            for topology in MATRIX_2_TOPOLOGIES:
                run_id = f"{args.campaign_id}-M2-{policy_name}-{level_name}-{topology}"
                out_dir = campaign_root / "matrix_2" / policy_name / level_name.lower() / topology
                result = service.run_msrc_policy_benchmark(
                    run_id=run_id,
                    policy_name=policy_name,
                    episodes=args.matrix2_episodes,
                    base_seed=args.seed_base + hash(f"M2-{level_name}-{topology}") % 10000,
                    output_dir=out_dir,
                    scenario_params={**params, "grid_size": 5, "topology": topology},
                    external_input=args.external_input,
                    level_label=level_name,
                    topology_label=topology,
                )
                summary = result["summary"]
                record = {
                    "matrix": "M2",
                    "run_id": run_id,
                    "policy": policy_name,
                    "level": level_name,
                    "topology": topology,
                    "episodes": args.matrix2_episodes,
                    "output_dir": str(out_dir),
                    "summary": summary,
                }
                run_records.append(record)
                _append_jsonl(runs_index, record)
                _concat_scale_decisions(out_dir / "scale_decisions.jsonl", decisions_agg)

    policy_summary = _aggregate_policy_summary(run_records)
    _write_json(campaign_root / "policy_summary.json", policy_summary)
    _write_final_report(campaign_root / "final_report.md", args.campaign_id, run_records, policy_summary)

    storage.close()

    print("MSRC_POLICY_CAMPAIGN_DONE")
    print(f"campaign_id={args.campaign_id}")
    print(f"output_root={campaign_root}")
    print(f"runs_index={runs_index}")
    print(f"policy_summary={campaign_root / 'policy_summary.json'}")
    print(f"final_report={campaign_root / 'final_report.md'}")
    return 0


def _concat_scale_decisions(source: Path, target: Path) -> None:
    if not source.exists():
        return
    with source.open("r", encoding="utf-8") as src, target.open("a", encoding="utf-8") as dst:
        for line in src:
            if line.strip():
                dst.write(line)


def _aggregate_policy_summary(run_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_policy: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in run_records:
        by_policy[row["policy"]].append(row)

    out: Dict[str, Any] = {}
    for policy, rows in by_policy.items():
        success_rates = [float(r["summary"].get("success_rate", 0.0)) for r in rows]
        ivc_rs = [float((r["summary"].get("avg_metrics") or {}).get("ivc_r", 0.0)) for r in rows]
        mean_cost = [
            float((r["summary"].get("msrc_metrics") or {}).get("mean_resolution_cost", 0.0))
            for r in rows
        ]
        mean_upgrade_rate = [
            float((r["summary"].get("msrc_metrics") or {}).get("upgrade_rate", 0.0))
            for r in rows
        ]
        mean_probe_rate = [
            float((r["summary"].get("msrc_metrics") or {}).get("probe_rate", 0.0))
            for r in rows
        ]
        mean_probe_commit_rate = [
            float((r["summary"].get("msrc_metrics") or {}).get("probe_commit_rate", 0.0))
            for r in rows
        ]
        mean_keep_rate = [
            float((r["summary"].get("msrc_metrics") or {}).get("keep_scale_rate", 0.0))
            for r in rows
        ]
        osc = [float((r["summary"].get("msrc_metrics") or {}).get("oscillation_rate", 0.0)) for r in rows]
        regret_u = [float((r["summary"].get("msrc_metrics") or {}).get("upgrade_regret", 0.0)) for r in rows]
        regret_d = [float((r["summary"].get("msrc_metrics") or {}).get("downgrade_regret", 0.0)) for r in rows]
        regret_missed = [
            float((r["summary"].get("msrc_metrics") or {}).get("missed_upgrade_regret", 0.0))
            for r in rows
        ]
        intervention = [float((r["summary"].get("avg_metrics") or {}).get("intervention_precision", 0.0)) for r in rows]
        wall_time = [float((r["summary"].get("avg_metrics") or {}).get("wall_time_ms", 0.0)) for r in rows]
        artifact_size = [float((r["summary"].get("avg_metrics") or {}).get("artifact_size_bytes", 0.0)) for r in rows]

        out[policy] = {
            "runs": len(rows),
            "mean_success_rate": _mean(success_rates),
            "mean_ivc_r": _mean(ivc_rs),
            "mean_intervention_precision": _mean(intervention),
            "mean_wall_time_ms": _mean(wall_time),
            "mean_artifact_size_bytes": _mean(artifact_size),
            "mean_resolution_cost": _mean(mean_cost),
            "mean_keep_scale_rate": _mean(mean_keep_rate),
            "mean_upgrade_rate": _mean(mean_upgrade_rate),
            "mean_probe_rate": _mean(mean_probe_rate),
            "mean_probe_commit_rate": _mean(mean_probe_commit_rate),
            "mean_oscillation_rate": _mean(osc),
            "mean_upgrade_regret": _mean(regret_u),
            "mean_downgrade_regret": _mean(regret_d),
            "mean_missed_upgrade_regret": _mean(regret_missed),
        }
    return out


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _write_final_report(
    path: Path,
    campaign_id: str,
    run_records: List[Dict[str, Any]],
    policy_summary: Dict[str, Any],
) -> None:
    lines = [
        "# MSRC Policy Benchmark Report",
        "",
        f"- campaign_id: `{campaign_id}`",
        f"- total_runs: `{len(run_records)}`",
        "",
        "## Policy Summary",
        "| Policy | Runs | Mean Success | Mean IVC-R | Mean IntPrecision | Mean Wall(ms) | Mean Artifact(B) | Mean ResCost | Keep Rate | Upgrade Rate | Probe Rate | Probe Commit | Missed Upgrade Regret | Mean Oscillation |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for policy, row in sorted(policy_summary.items()):
        lines.append(
            f"| {policy} | {row['runs']} | {row['mean_success_rate']:.4f} | {row['mean_ivc_r']:.4f} | "
            f"{row['mean_intervention_precision']:.4f} | {row['mean_wall_time_ms']:.2f} | {row['mean_artifact_size_bytes']:.1f} | "
            f"{row['mean_resolution_cost']:.4f} | {row['mean_keep_scale_rate']:.4f} | {row['mean_upgrade_rate']:.4f} | "
            f"{row['mean_probe_rate']:.4f} | {row['mean_probe_commit_rate']:.4f} | {row['mean_missed_upgrade_regret']:.4f} | "
            f"{row['mean_oscillation_rate']:.4f} |"
        )

    lines.extend([
        "",
        "## Notes",
        "- Los proxies se mantienen explícitos: success_rate y viability_margin.",
        "- MSRC prioriza suficiencia cognitiva y viabilidad antes de costo meta.",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark comparativo de políticas MSRC.")
    parser.add_argument("--campaign-id", default=f"msrc_policy_{_now_stamp()}")
    parser.add_argument("--output-root", default="data/benchmarks/msrc")
    parser.add_argument("--db-path", default="aeon_event_log.db")
    parser.add_argument("--artifact-root", default="rnfe_artifacts")
    parser.add_argument("--external-input", type=float, default=0.04)
    parser.add_argument("--seed-base", type=int, default=500000)
    parser.add_argument("--matrix1-episodes", type=int, default=20)
    parser.add_argument("--matrix2-episodes", type=int, default=12)
    parser.add_argument(
        "--policies",
        nargs="+",
        default=[
            "always_1x1",
            "always_5x5",
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
