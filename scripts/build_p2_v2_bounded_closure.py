#!/usr/bin/env python3
"""Build the bounded P2-v2 closure from frozen receipts only."""
from __future__ import annotations

import argparse, hashlib, json, math, subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

ARMS = ("canonical", "n3-reference", "n3-trained")
N3_ARMS = ARMS[1:]
CATEGORIES = ("NO_INFLUENCE", "ORDER_CHANGED_OUTSIDE_TOP4", "TOP4_SEQUENCE_CHANGED",
              "TOP4_MEMBERSHIP_CHANGED", "ACTION_CHANGED_WITHOUT_MEMBERSHIP_CHANGE",
              "ACTION_CHANGED_WITH_MEMBERSHIP_CHANGE")
AUTHORITY = {key: False for key in ("P3_DESIGN_AUTHORIZED", "LIVE_AUTHORITY",
                                     "STAGING_AUTHORIZED", "PROMOTION_AUTHORIZED",
                                     "MAIN_MERGE_AUTHORIZED")}
EXACT_GEOMETRY = {
    "status": "NOT_RECONSTRUCTIBLE_FROM_FROZEN_EVIDENCE",
    "missing_fields": ["observed_micro_signal", "observed_meso_signal", "observed_macro_signal",
                       "candidate_multiplier", "candidate_adjusted_score"],
    "forbidden_reconstruction_methods": ["rerun_reference_backend", "rerun_trained_backend",
                                           "infer_multipliers_from_final_order",
                                           "derive_unique_scores_from_partial_order"],
}
OBSERVABILITY_FIELDS = [
    "directive_status", "directive_reason", "directive_candidate_hash",
    "observed_micro_signal", "observed_meso_signal", "observed_macro_signal",
    "scale_signal_vector_sha256", "memory_id", "canonical_score", "scale",
    "observed_scale_signal", "applied_multiplier", "adjusted_score", "rank_before",
    "rank_after", "inside_top_k_before", "inside_top_k_after", "k_exposed",
    "canonical_boundary_score", "adjusted_boundary_score", "boundary_crossed",
    "rerank_formula_id", "rerank_formula_version", "scorer_version", "retrieval_version",
    "raw_pool_sha256", "adjusted_pool_sha256", "exposed_sequence_sha256",
    "exposed_membership_sha256",
]

def dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False,
                               allow_nan=False) + "\n")

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def group_rows(rows: Sequence[Mapping[str, Any]], expected_units: int):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row.get("scenario"), row.get("seed"), row.get("episode_index"))].append(row)
    if len(grouped) != expected_units:
        raise ValueError("p2_v2_unit_count_invalid")
    out = {}
    for key, unit in grouped.items():
        names = [row.get("arm_id") for row in unit]
        if len(unit) != 3 or Counter(names) != Counter(ARMS):
            raise ValueError("p2_v2_three_arm_pairing_invalid")
        out[key] = {row["arm_id"]: row for row in unit}
    return out

def classify_influence(canonical, arm):
    c, a = canonical["arm_order_ids"], arm["arm_order_ids"]
    membership = set(c[:4]) != set(a[:4])
    action = canonical["chosen_intervention"] != arm["chosen_intervention"]
    if action and membership: return "ACTION_CHANGED_WITH_MEMBERSHIP_CHANGE"
    if action: return "ACTION_CHANGED_WITHOUT_MEMBERSHIP_CHANGE"
    if membership: return "TOP4_MEMBERSHIP_CHANGED"
    if c[:4] != a[:4]: return "TOP4_SEQUENCE_CHANGED"
    if c != a: return "ORDER_CHANGED_OUTSIDE_TOP4"
    return "NO_INFLUENCE"

def audit_frozen_receipts(rows, *, expected_units, campaign_id, original_contrasts):
    groups = group_rows(rows, expected_units)
    counts = {arm: Counter() for arm in N3_ARMS}
    taxonomy = {arm: Counter() for arm in N3_ARMS}
    gains = {arm: defaultdict(list) for arm in N3_ARMS}
    for unit in groups.values():
        c = unit["canonical"]
        if c["arm_order_ids"] != c["canonical_order_ids"]:
            raise ValueError("p2_v2_canonical_order_invalid")
        actual = {r["actual_pre_action_state_sha256"] for r in unit.values()}
        snapshots = {r["snapshot_sha256"] for r in unit.values()}
        if len(actual) != 1 or len(snapshots) != 1 or actual != snapshots:
            raise ValueError("p2_v2_actual_state_parity_invalid")
        raw = ("raw_candidate_ids", "raw_candidate_scores", "raw_candidate_scales",
               "raw_candidate_structures", "raw_candidate_pool_sha256")
        if any(r.get(field) != c.get(field) for r in unit.values() for field in raw):
            raise ValueError("p2_v2_raw_pool_parity_invalid")
        c_order = c["arm_order_ids"]
        for r in unit.values():
            for field in ("chosen_utility", "optimal_utility", "regret"):
                value = r.get(field)
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError("p2_v2_nonfinite_receipt")
            if abs((r["optimal_utility"] - r["chosen_utility"]) - r["regret"]) > 1e-12:
                raise ValueError("p2_v2_regret_mismatch")
            if not r.get("decision_sealed") or not r.get("oracle_opened_after_seal"):
                raise ValueError("p2_v2_oracle_order_invalid")
            if not r.get("chosen_intervention_is_allowed"):
                raise ValueError("p2_v2_unauthorized_intervention")
            if r.get("external_reasoner_used") or r.get("training_executed") or r.get("live_authority"):
                raise ValueError("p2_v2_authority_violation")
            if (r.get("closure_passed") is not None or r.get("certified") is not None
                or r.get("full_safety_evaluation") is not None
                or r.get("measurement_status") != "NOT_MEASURED_IN_P2_V2"):
                raise ValueError("p2_v2_unmeasured_success_asserted")
        for arm_name in N3_ARMS:
            r, a = unit[arm_name], unit[arm_name]["arm_order_ids"]
            if len(a) != len(c_order) or len(a) != len(set(a)) or set(a) != set(c_order):
                raise ValueError("p2_v2_raw_pool_mutation")
            derived = {"full_order_changed": a != c_order,
                       "top1_changed": a[0] != c_order[0],
                       "exposed_set_changed": set(a[:4]) != set(c_order[:4])}
            if any(r.get(k) is not v for k, v in derived.items()):
                raise ValueError("P2_V2_RECEIPT_DERIVED_FIELD_MISMATCH")
            flags = {"full_order_change": derived["full_order_changed"],
                     "top1_change": derived["top1_changed"],
                     "top4_sequence_change": a[:4] != c_order[:4],
                     "top4_membership_change": derived["exposed_set_changed"],
                     "action_change": r["chosen_intervention"] != c["chosen_intervention"],
                     "regret_change": abs(r["regret"] - c["regret"]) > 1e-15}
            counts[arm_name].update({k: int(v) for k, v in flags.items()})
            category = classify_influence(c, r); taxonomy[arm_name][category] += 1
            gain = c["regret"] - r["regret"]; gains[arm_name][category].append(gain)
            if flags["action_change"]: gains[arm_name]["ACTION_CHANGE_ANY"].append(gain)
    geometry = {}
    for arm in N3_ARMS:
        geometry[arm] = {}
        for metric in ("full_order_change", "top1_change", "top4_sequence_change",
                       "top4_membership_change", "action_change", "regret_change"):
            geometry[arm][metric + "_count"] = counts[arm][metric]
            geometry[arm][metric + "_rate"] = counts[arm][metric] / expected_units
        geometry[arm]["membership_treatment_delivered"] = geometry[arm]["top4_membership_change_rate"] >= .10
        geometry[arm]["order_treatment_delivered"] = geometry[arm]["top4_sequence_change_rate"] >= .10
    tax = {arm: {cat: {"count": taxonomy[arm][cat], "rate": taxonomy[arm][cat] / expected_units}
                 for cat in CATEGORIES} for arm in N3_ARMS}
    diagnostic = {"status": "EXPLORATORY_DIAGNOSTIC_ONLY", "preregistered": False,
                  "confirmatory_authority": False, "arms": {}}
    for arm in N3_ARMS:
        diagnostic["arms"][arm] = {cat: {"unit_count": len(vals),
                                                  "mean_regret_gain": sum(vals)/len(vals) if vals else None,
                                                  "regret_change_count": sum(abs(v) > 1e-15 for v in vals)}
                                            for cat, vals in sorted(gains[arm].items())}
    delivered = any(geometry[a]["membership_treatment_delivered"] for a in N3_ARMS)
    return {"schema_version": "p2-v2-receipt-audit-v2", "campaign_id": campaign_id,
            "receipt_count": len(rows), "unit_count": len(groups),
            "integrity": {"arms_per_unit": 3, "three_arm_pairing": 1.0,
                          "actual_state_hash_parity": 1.0, "snapshot_actual_match_rate": 1.0,
                          "raw_pool_parity": 1.0, "canonical_order_valid": True,
                          "pool_conservation_valid": True, "oracle_and_regret_valid": True},
            "treatment_geometry_observed": geometry, "treatment_taxonomy": tax,
            "action_geometry": {a: {"change_count": counts[a]["action_change"],
                                    "change_rate": counts[a]["action_change"]/expected_units} for a in N3_ARMS},
            "regret_diagnostics": diagnostic, "exact_treatment_geometry": EXACT_GEOMETRY,
            "original_confirmatory_contrasts": original_contrasts, "authority": AUTHORITY,
            "valid": True, "verdict": "p2_v2_n3_decision_gain_not_demonstrated" if delivered
                                      else "p2_v2_treatment_not_delivered"}

def observability_contract():
    return {"schema_version": "p2-next-observability-contract-v1", "authorization_effect": "none",
            "required_per_arm_unit_fields": OBSERVABILITY_FIELDS,
            "invariants": {"ADJUSTED_SCORES_PERSISTED": True, "MULTIPLIERS_PERSISTED": True,
                           "DIRECTIVE_SIGNALS_PERSISTED": True, "RANK_TRANSITION_PERSISTED": True,
                           "BOUNDARY_DIAGNOSTICS_RECONSTRUCTIBLE": True,
                           "NO_BACKEND_RERUN_REQUIRED_FOR_AUDIT": True}, "authority": AUTHORITY}

def build(prereg_path, receipts_path, matrix_path, audit_v1_path, verdict_v1_path,
          suite_path, output_dir, evidence_commit="edb79dbcfcddd99ac793d86608295d13a8038904"):
    prereg=json.loads(prereg_path.read_text()); matrix=json.loads(matrix_path.read_text())
    rows=[json.loads(line) for line in receipts_path.read_text().splitlines() if line]
    audit=audit_frozen_receipts(rows, expected_units=len(prereg["seeds"])*len(prereg["scenarios"])*prereg["episodes_per_scenario"],
                               campaign_id=prereg["campaign_id"], original_contrasts=matrix["contrasts"])
    audit["source_receipt_sha256"]=sha(receipts_path); dump(output_dir/"matrix.audit-v2.json", audit)
    membership=all(not audit["treatment_geometry_observed"][a]["membership_treatment_delivered"] for a in N3_ARMS)
    verdict={"schema_version":"p2-v2-verdict-v2","P2_V2_STATUS":"CLOSED",
             "P2_V2_RESULT":"VALID_TREATMENT_FEASIBILITY_FAILURE_WITH_INSTRUMENTATION_LIMITATION",
             "P2_V2_ORDER_TREATMENT":"MEASURED","P2_V2_MEMBERSHIP_TREATMENT":"NOT_DELIVERED" if membership else "DELIVERED",
             "P2_V2_EXACT_TREATMENT_GEOMETRY":"NOT_RECONSTRUCTIBLE",
             "N3_DECISIONAL_GAIN":"NOT_EVALUABLE_TREATMENT_NOT_DELIVERED" if membership else "RETAIN_ORIGINAL_VERDICT",
             "N3_BACKEND_PREFERENCE":"NONE", **AUTHORITY}; dump(output_dir/"verdict.v2.json",verdict)
    dump(output_dir/"P2_NEXT_OBSERVABILITY_CONTRACT.json", observability_contract())
    suite=json.loads(suite_path.read_text())
    closure={"closure_schema_version":"p2-v2-bounded-closure-v1","campaign_id":prereg["campaign_id"],
             "branch":"codex/p2-n3-causal-decision-v2","base_commit":prereg["p1_closure_commit"],
             "preregistration_commit":prereg["preregistration_commit"],"execution_commit":"55f716e24804ac4ae58f656af8f8d0805b8aa5df",
             "evidence_commit":evidence_commit,"receipt_sha256":sha(receipts_path),"matrix_sha256":sha(matrix_path),
             "audit_v1_sha256":sha(audit_v1_path),"audit_v2_sha256":sha(output_dir/"matrix.audit-v2.json"),
             "verdict_v1_sha256":sha(verdict_v1_path),"verdict_v2_sha256":sha(output_dir/"verdict.v2.json"),
             "focused_test_result":suite.get("focused"),"full_suite_result":suite.get("summary"),
             "git_diff_check":"pending_final_verification","scope_check":"pending_final_verification",
             "final_status":"CLOSED","final_result":verdict["P2_V2_RESULT"],"authority_ceiling":"none"}
    dump(output_dir/"closure.json",closure)
    return audit, verdict, closure

def main():
    p=argparse.ArgumentParser(); p.add_argument("--package",type=Path,required=True); p.add_argument("--suite",type=Path,required=True)
    a=p.parse_args(); d=a.package
    build(d/"preregistration.json",d/"decision-receipts.jsonl",d/"matrix.json",d/"matrix.audit-v1.json",
          d/"verdict.json",a.suite,d)
    return 0
if __name__=="__main__": raise SystemExit(main())
