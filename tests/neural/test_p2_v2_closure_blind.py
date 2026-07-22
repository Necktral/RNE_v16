import copy
import pytest
from scripts.review_p2_v2_closure_blind import AUTH, dimensions, disposition, review, stats, validate_primary_evidence

ARMS=["canonical","n3-reference","n3-trained"]
def inputs(k=2,gate=.1):
 p={"campaign_id":"x","seeds":[1],"scenarios":["s"],"episodes_per_scenario":1,"k_exposed":k,"treatment_delivery_gate":gate,"bootstrap_samples":20};m={"arms":ARMS};ids=[f"m{i}" for i in range(6)]
 def row(a):return {"scenario":"s","seed":1,"episode_index":0,"arm_id":a,"snapshot_sha256":"x","actual_pre_action_state_sha256":"x","raw_candidate_pool_sha256":"p","raw_candidate_ids":ids,"raw_candidate_scores":[1,1,.5,.5,0,0],"raw_candidate_scales":["micro","meso","macro"]*2,"raw_candidate_structures":[{} for _ in ids],"canonical_order_ids":ids,"arm_order_ids":ids.copy(),"chosen_intervention":"a","chosen_utility":1.,"optimal_utility":1.,"regret":0.,"decision_sealed":True,"oracle_opened_after_seal":True,"chosen_intervention_is_allowed":True,"full_order_changed":False,"top1_changed":False,"exposed_set_changed":False}
 return p,m,[row(a) for a in ARMS]
def test_dimensions_derive_k_gate_and_counts():
 for k,g in ((2,.05),(5,.4)):
  p,m,r=inputs(k,g);d=dimensions(p,m);assert (d["k"],d["gate"],d["units"],d["receipts"])==(k,g,1,3)
def test_zero_and_nonzero_membership_follow_gate():
 p,m,r=inputs();assert review(p,m,r)["review_result"]=="TREATMENT_NOT_DELIVERED"
 r[1]["arm_order_ids"]=["m2","m1","m0","m3","m4","m5"];r[1]["full_order_changed"]=True;r[1]["top1_changed"]=True;r[1]["exposed_set_changed"]=True
 out=review(p,m,r);assert out["observable_treatment"]["n3-reference"]["membership_delivered"]
def test_pairing_pool_regret_and_derived_fraud_invalidate_or_detect():
 p,m,r=inputs();assert not review(p,m,r[:-1])["review_valid"]
 for field,value in (("raw_candidate_scores",[9]*6),("regret",1.)):
  b=copy.deepcopy(r);b[1][field]=value;assert not review(p,m,b)["review_valid"]
 b=copy.deepcopy(r);b[1]["top1_changed"]=True;assert review(p,m,b)["primary_evidence_integrity"]["derived_field_mismatches"]==1
def test_neutral_machine_all_routes_and_authority():
 c={"reference - canonical":{"gate_passed":False},"trained - canonical":{"gate_passed":False}}
 t={a:{"membership_delivered":False} for a in ARMS[1:]};assert disposition(True,t,c)[1]=="TREATMENT_NOT_DELIVERED"
 t["n3-reference"]["membership_delivered"]=True;assert disposition(True,t,c)[1]=="DECISION_GAIN_NOT_DEMONSTRATED"
 c["reference - canonical"]["gate_passed"]=True;assert disposition(True,t,c)[1]=="DECISION_GAIN_SUPPORTED_REFERENCE"
 t["n3-trained"]["membership_delivered"]=True;c["trained - canonical"]["gate_passed"]=True;assert disposition(True,t,c)[1]=="DECISION_GAIN_SUPPORTED_BOTH"
 assert disposition(False,t,c)[1]=="INVALID_PRIMARY_EVIDENCE" and all(v is False for v in AUTH.values())
def test_deterministic_and_no_backend_dependency():
 p,m,r=inputs();assert review(p,m,r)==review(p,m,r)
 import scripts.review_p2_v2_closure_blind as x
 assert not any(n in x.__dict__ for n in ("N3Adapter","MemoryRetrieval","Mamba2TemporalTorchBackend"))
 assert review(p,m,r)["exact_treatment_geometry"]["status"]=="NOT_RECONSTRUCTIBLE_FROM_FROZEN_EVIDENCE"
def test_stats_empty_singleton_and_measured_are_explicit():
 with pytest.raises(ValueError,match="empty_sample"): stats([],"empty",10)
 one=stats([.25],"one",20); assert one["sample_size"]==1 and one["standard_deviation_ddof1"] is None
 assert one["standard_deviation_status"]=="NOT_DEFINED_N_LT_2" and one["assignments_enumerated"]==2
 assert one["bootstrap_ci95"]==[.25,.25] and one==stats([.25],"one",20)
 two=stats([.1,.2],"two",20); assert isinstance(two["standard_deviation_ddof1"],float)
 assert two["standard_deviation_status"]=="MEASURED" and two["assignments_enumerated"]==4
def test_invalid_pairing_skips_all_contrasts(monkeypatch):
 import scripts.review_p2_v2_closure_blind as module
 p,m,r=inputs(); called=False
 def forbidden(*args,**kwargs):
  nonlocal called; called=True; raise AssertionError("contrasts must not run")
 monkeypatch.setattr(module,"compute_confirmatory_contrasts",forbidden)
 out=module.review(p,m,r[:-1]); assert not called
 assert out["review_status"]=="BLOCKED" and out["review_result"]=="INVALID_PRIMARY_EVIDENCE"
 assert out["confirmatory_contrasts_recomputed"] is None
def test_validated_units_are_preserved_into_contrasts(monkeypatch):
 import scripts.review_p2_v2_closure_blind as module
 p,m,r=inputs(); validation=validate_primary_evidence(p,m,r); assert validation["valid"]
 assert len(validation["units"])==validation["expected_dimensions"]["unit_count"]==1
 original=module.validate_primary_evidence; received=[]
 monkeypatch.setattr(module,"validate_primary_evidence",lambda *args: validation)
 monkeypatch.setattr(module,"compute_confirmatory_contrasts",lambda prereg,units: received.append(units) or {
  "reference - canonical":{"gate_passed":False},"trained - canonical":{"gate_passed":False},"trained - reference":{"gate_passed":False}})
 module.review(p,m,r); assert received[0] is validation["units"]
 monkeypatch.setattr(module,"validate_primary_evidence",original)
def test_validated_unit_count_mismatch_is_guarded(monkeypatch):
 import scripts.review_p2_v2_closure_blind as module
 p,m,r=inputs(); validation=validate_primary_evidence(p,m,r); validation["units"]={}
 monkeypatch.setattr(module,"validate_primary_evidence",lambda *args: validation)
 with pytest.raises(ValueError,match="validated_unit_count_mismatch"): module.review(p,m,r)
