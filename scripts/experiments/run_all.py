from __future__ import annotations

import argparse
from pathlib import Path

from . import exp1_regret, exp2_learning, exp3_brier, exp4_regime, exp5_n4_closed_loop


def run(*, root: Path, quick: bool = False):
    return {
        "exp1": exp1_regret.run(root=root, quick=quick),
        "exp2": exp2_learning.run(root=root, quick=quick),
        "exp3": exp3_brier.run(root=root, quick=quick),
        "exp4": exp4_regime.run(root=root, quick=quick),
        "exp5": exp5_n4_closed_loop.run(root=root, quick=quick),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    run(root=args.root, quick=args.quick)
