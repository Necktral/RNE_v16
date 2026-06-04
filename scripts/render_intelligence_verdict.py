#!/usr/bin/env python3
"""CLI para dictamen duro de progreso en inteligencia útil con cierre."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.intelligence_campaign_lib import render_intelligence_verdict


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Renderiza el dictamen final de progreso en inteligencia."
    )
    parser.add_argument("--cognitive-verdicts-path", required=True)
    parser.add_argument("--family-causal-path", default=None)
    parser.add_argument("--output-root", default="data/reports/intelligence_verdict")
    parser.add_argument("--campaign-id", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = render_intelligence_verdict(
        cognitive_verdicts_path=args.cognitive_verdicts_path,
        family_causal_path=args.family_causal_path,
        output_root=args.output_root,
        campaign_id=args.campaign_id,
    )
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
