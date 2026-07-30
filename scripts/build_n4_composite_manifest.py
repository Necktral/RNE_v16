"""Construye y verifica un manifiesto compuesto N4 desde una definición."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime.neural.training.n4_campaign import (
    COMPOSITE_SCHEMA,
    load_composite_manifest,
)
from runtime.neural.training.n4_ranking import FEATURE_NAMES


def build_composite_manifest(
    *, definition_path: Path, output_path: Path
) -> dict:
    if output_path.exists():
        raise FileExistsError("n4_composite_output_must_be_new")
    definition = json.loads(definition_path.read_bytes())
    payload = {
        "schema": COMPOSITE_SCHEMA,
        "campaign_protocol_version": definition[
            "campaign_protocol_version"
        ],
        "code_checkpoint": definition["code_checkpoint"],
        "feature_order": list(FEATURE_NAMES),
        "components": definition["components"],
        "sampling_policies": ["natural", "stratified_50_50"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(
        json.dumps(
            payload,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        + b"\n"
    )
    loaded = load_composite_manifest(output_path)
    return {
        "manifest_sha256": loaded.manifest_sha256,
        "train_candidate_set_count": len(loaded.train),
        "validation_candidate_set_count": len(loaded.validation),
        "component_stats": list(loaded.component_stats),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--definition", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_composite_manifest(
        definition_path=args.definition,
        output_path=args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
