#!/usr/bin/env python3
"""CLI del experimento del Funcional Crítico J(h|X): ¿la descomposición cura la ceguera?"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.critical_functional_lib import CURE_GRID_DEFAULT, run_study


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Funcional Crítico J(h|X): ¿descomponer IoC (ν de 1ª clase) cura la ceguera?"
    )
    p.add_argument("--mode", default="all", choices=["cure", "specificity", "ablation", "all"])
    p.add_argument("--output-root", default="data/reports/critical_functional")
    p.add_argument("--seeds", type=int, default=400)
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument(
        "--cure-grid",
        default=",".join(str(x) for x in CURE_GRID_DEFAULT),
        help="Coma-separados; debe cruzar el umbral real ~20 para exhibir el contraste.",
    )
    p.add_argument("--spec-lam-nu", type=float, default=1.0)
    p.add_argument("--ablation-lam-nu", type=float, default=2.0)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out = run_study(
        mode=args.mode,
        output_root=args.output_root,
        cure_grid=[float(x) for x in args.cure_grid.split(",")],
        seeds=args.seeds,
        episodes=args.episodes,
        spec_lam_nu=args.spec_lam_nu,
        ablation_lam_nu=args.ablation_lam_nu,
    )
    print(json.dumps(out, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
