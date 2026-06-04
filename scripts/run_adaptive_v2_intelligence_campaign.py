#!/usr/bin/env python3
"""CLI secuencial para la campaña fuerte de inteligencia de adaptive_family_ecology_v2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.intelligence_campaign_lib import run_adaptive_v2_intelligence_campaign


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ejecuta prompt 1, prompt 2 condicional y prompt 3 para adaptive_family_ecology_v2."
    )
    parser.add_argument("--campaign-id", default=None)
    parser.add_argument("--cognitive-output-root", default="data/benchmarks/cognitive_gain")
    parser.add_argument("--family-output-root", default="data/benchmarks/family_causal_gain")
    parser.add_argument("--verdict-output-root", default="data/reports/intelligence_verdict")
    parser.add_argument("--db-path", default="aeon_event_log.db")
    parser.add_argument("--artifact-root", default="data/artifacts")
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--episodes-per-block", type=int, default=8)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed-base", type=int, default=910000)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_adaptive_v2_intelligence_campaign(
        campaign_id=args.campaign_id,
        cognitive_output_root=args.cognitive_output_root,
        family_output_root=args.family_output_root,
        verdict_output_root=args.verdict_output_root,
        db_path=args.db_path,
        artifact_root=args.artifact_root,
        blocks=args.blocks,
        episodes_per_block=args.episodes_per_block,
        bootstrap_samples=args.bootstrap_samples,
        seed_base=args.seed_base,
    )
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
