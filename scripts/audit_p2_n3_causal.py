#!/usr/bin/env python3
"""Deterministically audit P2 receipts and derive its bounded verdict."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

def dump(path, obj): path.write_text(json.dumps(obj, sort_keys=True, indent=2, allow_nan=False) + "\n")

def main():
    p=argparse.ArgumentParser()
    for name in ("manifest","matrix","receipts","output","verdict"): p.add_argument(f"--{name}", type=Path, required=True)
    a=p.parse_args(); manifest=json.loads(a.manifest.read_text()); matrix=json.loads(a.matrix.read_text())
    rows=[json.loads(line) for line in a.receipts.read_text().splitlines() if line]
    integrity=dict(matrix["integrity"])
    expected=len(manifest["arms"])*len(manifest["seeds"])*len(manifest["scenarios_included"])*manifest["steps_per_lane"]
    checks={"receipt_count":len(rows)==expected,"receipt_hash":hashlib.sha256(a.receipts.read_bytes()).hexdigest()==matrix["receipt_sha256"],
            "all_authority_false":all(not r["live_authority"] for r in rows),"all_finite":all(isinstance(r[k],(int,float)) for r in rows for k in ("chosen_utility","optimal_utility","regret")),
            "integrity_zero":all(integrity[k]==0 for k in ("oracle_leakage","shared_state_writes","external_reasoner_calls","training_calls","candidate_pool_mutations","unauthorized_actions"))}
    valid=all(checks.values()) and all(integrity[k]==1.0 for k in ("pre_action_hash_parity","candidate_pool_set_parity","candidate_pool_count_parity","seed_pairing","scenario_pairing","decision_receipt_completeness"))
    ref=matrix["contrasts"]["reference - canonical"]["gate_passed"]; trained=matrix["contrasts"]["trained - canonical"]["gate_passed"]
    if not valid: value="p2_invalid"; preferred="none"
    elif ref and trained:
        value="p2_n3_decision_gain_supported_both"; direct=matrix["contrasts"]["trained - reference"]
        preferred=("trained" if direct["gate_passed"] and direct["mean"]>0 else "reference" if direct["gate_passed"] else "inconclusive")
    elif trained: value="p2_n3_decision_gain_supported_trained"; preferred="trained"
    elif ref: value="p2_n3_decision_gain_supported_reference"; preferred="reference"
    else: value="p2_n3_signal_not_transferred_to_decision"; preferred="none"
    audit={"schema_version":"p2-n3-causal-audit-v1","campaign_id":manifest["campaign_id"],"checks":checks,"valid":valid,"verdict":value,"preferred_backend":preferred}
    verdict={"schema_version":"p2-n3-causal-verdict-v1","verdict":value,"preferred_backend":preferred,"P3_DESIGN_AUTHORIZED":False,"N3_DECISIONAL_INFLUENCE":"DEMONSTRATED_IN_PAIRED_SANDBOX" if (ref or trained) and valid else "NOT_DEMONSTRATED","live_authority":False,"staging_authorized":False,"promotion_authorized":False,"main_merge_authorized":False}
    dump(a.output,audit); dump(a.verdict,verdict)
if __name__=="__main__": main()
