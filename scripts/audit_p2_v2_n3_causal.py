#!/usr/bin/env python3
"""Independent receipt-first audit for P2-v2."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from runtime.neural.integration.contracts import canonical_sha256
from runtime.neural.integration.p2_v2_n3_decision import ARMS, contrast_statistics


def _dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False,
                               allow_nan=False) + "\n")


def audit_receipts(rows: Sequence[Mapping[str, Any]], *, expected_units: int) -> dict[str, Any]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        for key in ("regret", "chosen_utility", "optimal_utility"):
            value = row.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("p2_v2_nonfinite_receipt")
        groups[(row.get("scenario"), row.get("seed"), row.get("episode_index"))].append(row)
    if len(groups) != expected_units or any({x.get("arm_id") for x in unit} != set(ARMS)
                                             or len(unit) != 3 for unit in groups.values()):
        raise ValueError("p2_v2_three_arm_pairing_invalid")
    nonconstant = 0
    treatment = {"n3-reference": 0, "n3-trained": 0}
    for unit in groups.values():
        canonical = next(x for x in unit if x["arm_id"] == "canonical")
        raw_ids = canonical["raw_candidate_ids"]
        raw_structures = canonical["raw_candidate_structures"]
        for row in unit:
            if (row["snapshot_sha256"] != row["actual_pre_action_state_sha256"]
                    or row.get("snapshot_matches_actual") is not True):
                raise ValueError("p2_v2_actual_state_parity_invalid")
            if row["raw_candidate_ids"] != raw_ids or row["raw_candidate_structures"] != raw_structures:
                raise ValueError("p2_v2_raw_pool_parity_invalid")
            if set(row["arm_order_ids"]) != set(raw_ids):
                raise ValueError("p2_v2_raw_pool_mutation")
            if not row.get("decision_sealed") or not row.get("oracle_opened_after_seal"):
                raise ValueError("p2_v2_oracle_order_invalid")
            if (row.get("external_reasoner_used") or row.get("training_executed")
                    or row.get("live_authority") or not row.get("chosen_intervention_is_allowed")):
                raise ValueError("p2_v2_authority_or_action_violation")
            if abs((row["optimal_utility"] - row["chosen_utility"]) - row["regret"]) > 1e-12:
                raise ValueError("p2_v2_regret_mismatch")
        if len(set(canonical.get("raw_candidate_scores", []))) > 1:
            nonconstant += 1
        for arm in treatment:
            treatment[arm] += int(next(x for x in unit if x["arm_id"] == arm)["exposed_set_changed"])
    return {
        "receipt_count": len(rows), "unit_count": len(groups), "three_arm_pairing": 1.0,
        "actual_state_hash_parity": 1.0, "raw_pool_set_parity": 1.0,
        "oracle_order_integrity": 1.0, "unauthorized_interventions": 0,
        "external_reasoner_calls": 0, "training_calls": 0,
        "units_with_nonconstant_scores_rate": nonconstant / expected_units,
        "treatment_delivery": {arm: changed / expected_units for arm, changed in treatment.items()},
    }


def recompute_matrix(rows: Sequence[Mapping[str, Any]], prereg: Mapping[str, Any]) -> dict[str, Any]:
    expected_units = len(prereg["seeds"]) * len(prereg["scenarios"]) * prereg["episodes_per_scenario"]
    integrity = audit_receipts(rows, expected_units=expected_units)
    grouped = defaultdict(dict)
    for row in rows:
        grouped[(row["scenario"], row["seed"], row["episode_index"])][row["arm_id"]] = row
    definitions = {
        "reference - canonical": ("canonical", "n3-reference"),
        "trained - canonical": ("canonical", "n3-trained"),
        "trained - reference": ("n3-reference", "n3-trained"),
    }
    contrasts = {}
    for name, (baseline, treatment) in definitions.items():
        seed_means = []
        for seed in prereg["seeds"]:
            scenario_means = []
            for scenario in prereg["scenarios"]:
                values = [arms[baseline]["regret"] - arms[treatment]["regret"]
                          for (s, sd, _), arms in grouped.items() if s == scenario and sd == seed]
                scenario_means.append(sum(values) / len(values))
            seed_means.append(sum(scenario_means) / len(scenario_means))
        contrasts[name] = contrast_statistics(seed_means, name=name)
    delivered = {arm: rate >= .10 for arm, rate in integrity["treatment_delivery"].items()}
    passes = {
        "reference": delivered["n3-reference"] and contrasts["reference - canonical"]["gate_passed"],
        "trained": delivered["n3-trained"] and contrasts["trained - canonical"]["gate_passed"],
    }
    valid = integrity["units_with_nonconstant_scores_rate"] >= .80
    if not valid:
        verdict = "p2_v2_invalid"
    elif not any(delivered.values()):
        verdict = "p2_v2_treatment_not_delivered"
    elif all(passes.values()):
        verdict = "p2_v2_n3_decision_gain_supported_both"
    elif passes["reference"]:
        verdict = "p2_v2_n3_decision_gain_supported_reference"
    elif passes["trained"]:
        verdict = "p2_v2_n3_decision_gain_supported_trained"
    else:
        verdict = "p2_v2_n3_decision_gain_not_demonstrated"
    return {"schema_version": "p2-v2-audit-v1", "integrity": integrity,
            "contrasts": contrasts, "verdict": verdict,
            "authority": {"P3_DESIGN_AUTHORIZED": False, "LIVE_AUTHORITY": False,
                          "STAGING_AUTHORIZED": False, "PROMOTION_AUTHORIZED": False,
                          "MAIN_MERGE_AUTHORIZED": False}}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--receipts", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--verdict-output", type=Path, required=True)
    args = parser.parse_args()
    prereg = json.loads(args.preregistration.read_text())
    rows = [json.loads(line) for line in args.receipts.read_text().splitlines() if line]
    audit = recompute_matrix(rows, prereg)
    matrix = json.loads(args.matrix.read_text())
    if matrix.get("contrasts") != audit["contrasts"] or matrix.get("integrity") != audit["integrity"]:
        audit["verdict"] = "p2_v2_invalid"
        audit["error"] = "P2_V2_MATRIX_AUDIT_MISMATCH"
    _dump(args.audit_output, audit)
    _dump(args.verdict_output, {"verdict": audit["verdict"], "authority": audit["authority"],
                                "audit_sha256": hashlib.sha256(args.audit_output.read_bytes()).hexdigest()})
    return 0 if "error" not in audit else 2


if __name__ == "__main__":
    raise SystemExit(main())
