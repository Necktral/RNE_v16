#!/usr/bin/env python3
"""CLI para correr una ecología multi-organismo (una sola configuración).

Para el A/B entre modos de transferencia usa
``scripts/ecology_multiplication_experiment.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime.organism.ecology import OrganismEcology, TransferMode, build_member
from runtime.storage import StorageConfig, StorageFactory
from scripts.intelligence_campaign_lib import _regime_specs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Corre una ecología multi-organismo.")
    p.add_argument("--output-root", default="data/benchmarks/ecology")
    p.add_argument("--ecology-id", default="ecology-run")
    p.add_argument("--members", type=int, default=4)
    p.add_argument("--generations", type=int, default=3)
    p.add_argument("--episodes-per-member", type=int, default=6)
    p.add_argument(
        "--transfer-mode",
        default="reasoning_policy",
        choices=[m.value for m in TransferMode],
    )
    p.add_argument("--regime", default="viability_edge")
    p.add_argument("--seed", type=int, default=990000)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.output_root) / args.ecology_id
    root.mkdir(parents=True, exist_ok=True)
    config = StorageConfig(
        mode="sqlite",
        sqlite_db_path=str(root / "ecology.db"),
        postgres_dsn=None,
        artifact_root=root / "artifacts",
        prefer_postgres_reads=False,
        strict_dual_write=False,
    )
    storage = StorageFactory.create_facade(config)
    params = dict(_regime_specs()[args.regime]["scenario_params"])
    members = [
        build_member(
            member_id=f"m{i}",
            scenario="grid_thermal_5x5",
            scenario_kwargs=params,
            storage=storage,
        )
        for i in range(args.members)
    ]
    eco = OrganismEcology(
        members,
        storage=storage,
        transfer_mode=TransferMode(args.transfer_mode),
        seed=args.seed,
    )
    summaries = []
    for _ in range(args.generations):
        summary = eco.run_generation(episodes_per_member=args.episodes_per_member)
        summaries.append(summary)
        best = summary["ranking"][0]
        print(
            f"[ecology] gen {summary['generation']}: best={best['member_id']} "
            f"fitness={best['fitness']} certified={best['certified_episodes']}",
            flush=True,
        )
    payload = {
        "ecology_id": args.ecology_id,
        "transfer_mode": args.transfer_mode,
        "population_summary": eco.population_summary(),
        "generations": summaries,
    }
    (root / "ecology_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=True, default=str), encoding="utf-8"
    )
    print(json.dumps(eco.population_summary(), indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
