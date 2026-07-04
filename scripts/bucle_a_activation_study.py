#!/usr/bin/env python3
"""CLI del estudio A1: activación del Bucle A (guiado-por-recompensa vs perfil fijo + ecología)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.bucle_a_activation_lib import LAMBDA_GRID, LAMBDA_NU_DEFAULT, run_study


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="A1 Bucle A: ¿la selección guiada-por-recompensa (ν descompuesto) supera al perfil fijo?"
    )
    p.add_argument("--mode", default="all", choices=["arms", "sweep", "all"])
    p.add_argument("--output-root", default="data/reports/bucle_a_activation")
    p.add_argument("--lam-nu", type=float, default=LAMBDA_NU_DEFAULT)
    p.add_argument("--seeds", type=int, default=12)
    p.add_argument("--episodes", type=int, default=36)
    p.add_argument(
        "--sweep-grid",
        default=",".join(str(x) for x in LAMBDA_GRID),
        help="Coma-separados; barrido de λ_ν para el umbral de recuperación O(1).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out = run_study(
        mode=args.mode,
        output_root=args.output_root,
        lam_nu=args.lam_nu,
        seeds=args.seeds,
        episodes=args.episodes,
        sweep_grid=[float(x) for x in args.sweep_grid.split(",")],
    )
    print(json.dumps(out, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
