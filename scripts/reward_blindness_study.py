#!/usr/bin/env python3
"""CLI del estudio controlado de ceguera de la recompensa (process vs outcome reward)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.reward_blindness_lib import LAMBDA_GRID_DEFAULT, run_study


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Estudio de ceguera de la recompensa: ¿un reward de coherencia suprime la efectividad?"
    )
    p.add_argument("--mode", default="both", choices=["mechanism", "system", "both"])
    p.add_argument("--output-root", default="data/reports/reward_blindness")
    p.add_argument("--mechanism-seeds", type=int, default=1000)
    p.add_argument("--mechanism-episodes", type=int, default=30)
    p.add_argument("--system-seeds", type=int, default=8)
    p.add_argument("--system-episodes", type=int, default=36)
    p.add_argument(
        "--lambda-grid",
        default=",".join(str(x) for x in LAMBDA_GRID_DEFAULT),
        help="Coma-separados para el experimento de mecanismo.",
    )
    p.add_argument(
        "--system-lambda-grid",
        default="0.0,5.0,20.0,50.0",
        help="Coma-separados para el sistema real (cruza el umbral real ~20).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out = run_study(
        mode=args.mode,
        output_root=args.output_root,
        lambda_grid=[float(x) for x in args.lambda_grid.split(",")],
        mechanism_seeds=args.mechanism_seeds,
        mechanism_episodes=args.mechanism_episodes,
        system_lambda_grid=[float(x) for x in args.system_lambda_grid.split(",")],
        system_seeds=args.system_seeds,
        system_episodes=args.system_episodes,
    )
    print(json.dumps(out, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
