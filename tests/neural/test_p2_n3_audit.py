import json, subprocess, sys
from pathlib import Path

def test_audit_negative_verdict_is_deterministic(tmp_path):
    manifest={"campaign_id":"x","arms":["canonical","n3-reference","n3-trained"],"seeds":[1],"scenarios_included":["s"],"steps_per_lane":1}
    rows=[]
    for arm in manifest["arms"]:
        rows.append({"arm_id":arm,"live_authority":False,"chosen_utility":0.0,"optimal_utility":0.0,"regret":0.0})
    receipts=tmp_path/"r.jsonl"; receipts.write_text("".join(json.dumps(x)+"\n" for x in rows))
    import hashlib
    matrix={"receipt_sha256":hashlib.sha256(receipts.read_bytes()).hexdigest(),"integrity":{"oracle_leakage":0,"shared_state_writes":0,"external_reasoner_calls":0,"training_calls":0,"candidate_pool_mutations":0,"unauthorized_actions":0,"pre_action_hash_parity":1.0,"candidate_pool_set_parity":1.0,"candidate_pool_count_parity":1.0,"seed_pairing":1.0,"scenario_pairing":1.0,"decision_receipt_completeness":1.0},"contrasts":{"reference - canonical":{"gate_passed":False},"trained - canonical":{"gate_passed":False},"trained - reference":{"gate_passed":False,"mean":0.0}}}
    mp=tmp_path/"m.json"; xp=tmp_path/"x.json"; mp.write_text(json.dumps(manifest)); xp.write_text(json.dumps(matrix))
    audit=tmp_path/"a.json"; verdict=tmp_path/"v.json"; script=Path(__file__).parents[2]/"scripts/audit_p2_n3_causal.py"
    cmd=[sys.executable,str(script),"--manifest",str(mp),"--matrix",str(xp),"--receipts",str(receipts),"--output",str(audit),"--verdict",str(verdict)]
    subprocess.check_call(cmd); first=audit.read_bytes(); subprocess.check_call(cmd)
    assert audit.read_bytes()==first
    assert json.loads(verdict.read_text())["verdict"]=="p2_n3_signal_not_transferred_to_decision"
