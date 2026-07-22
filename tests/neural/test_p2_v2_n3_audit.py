import copy
import pytest
from scripts.build_p2_v2_bounded_closure import audit_frozen_receipts, classify_influence

ARMS=("canonical","n3-reference","n3-trained")
IDS=[f"m{i}" for i in range(12)]
SCALES=["micro","micro","meso","meso","macro","macro"]*2
STRUCT=[{"intervention":iv,"relation_kind":rel} for iv in ("a","b")
        for scale in ("micro","meso","macro") for rel in ("support","contradiction")]

def receipt(arm):
    return {"campaign_id":"c","scenario":"s","seed":1,"episode_index":0,"arm_id":arm,
      "snapshot_sha256":"state","actual_pre_action_state_sha256":"state","snapshot_matches_actual":True,
      "raw_candidate_pool_sha256":"pool","raw_candidate_ids":IDS.copy(),
      "raw_candidate_scores":[float(i) for i in range(12)],"raw_candidate_scales":SCALES.copy(),
      "raw_candidate_structures":copy.deepcopy(STRUCT),"canonical_order_ids":IDS.copy(),
      "arm_order_ids":IDS.copy(),"exposed_memory_ids":IDS[:4],"full_order_changed":False,
      "top1_changed":False,"exposed_set_changed":False,"decision_sealed":True,
      "oracle_opened_after_seal":True,"chosen_intervention":"a","chosen_intervention_is_allowed":True,
      "optimal_intervention":"a","chosen_utility":1.0,"optimal_utility":1.0,"regret":0.0,
      "external_reasoner_used":False,"training_executed":False,"live_authority":False,
      "closure_passed":None,"certified":None,"full_safety_evaluation":None,
      "measurement_status":"NOT_MEASURED_IN_P2_V2"}

def rows(): return [receipt(a) for a in ARMS]
def audit(value): return audit_frozen_receipts(value,expected_units=1,campaign_id="c",original_contrasts={})

def test_pairing_complete_missing_and_duplicate():
    assert audit(rows())["integrity"]["three_arm_pairing"]==1.0
    with pytest.raises(ValueError,match="pairing"): audit(rows()[:-1])
    bad=rows(); bad[2]["arm_id"]="n3-reference"
    with pytest.raises(ValueError,match="pairing"): audit(bad)

@pytest.mark.parametrize("field,value",[
 ("actual_pre_action_state_sha256","different"),("raw_candidate_ids",["x"]*12),
 ("raw_candidate_scores",[99.0]*12),("raw_candidate_scales",["micro"]*12),
 ("raw_candidate_structures",[{}]*12),("raw_candidate_pool_sha256","different")])
def test_state_and_raw_pool_differences_fail(field,value):
    bad=rows(); bad[1][field]=value
    with pytest.raises(ValueError): audit(bad)

def test_pool_mutation_fails():
    bad=rows(); bad[1]["arm_order_ids"]=["m0"]*12
    with pytest.raises(ValueError,match="mutation"): audit(bad)

@pytest.mark.parametrize("stored",["full_order_changed","top1_changed","exposed_set_changed"])
def test_fraudulent_derived_boolean_is_detected(stored):
    bad=rows(); bad[1][stored]=True
    with pytest.raises(ValueError,match="DERIVED_FIELD"): audit(bad)

def classified(order, action="a"):
    c,r=receipt("canonical"),receipt("n3-reference"); r["arm_order_ids"]=order; r["chosen_intervention"]=action
    return classify_influence(c,r)

def test_taxonomy_all_geometries():
    assert classified(IDS)=="NO_INFLUENCE"
    assert classified(IDS[:4]+IDS[5:]+[IDS[4]])=="ORDER_CHANGED_OUTSIDE_TOP4"
    assert classified([IDS[1],IDS[0],*IDS[2:]])=="TOP4_SEQUENCE_CHANGED"
    assert classified([IDS[4],*IDS[1:4],IDS[0],*IDS[5:]])=="TOP4_MEMBERSHIP_CHANGED"
    assert classified(IDS,"b")=="ACTION_CHANGED_WITHOUT_MEMBERSHIP_CHANGE"
    assert classified([IDS[4],*IDS[1:4],IDS[0],*IDS[5:]],"b")=="ACTION_CHANGED_WITH_MEMBERSHIP_CHANGE"

def test_regret_nonfinite_oracle_action_and_unmeasured_guards():
    mutations=[("regret",.1),("regret",float("nan")),("regret",float("inf")),
               ("decision_sealed",False),("chosen_intervention_is_allowed",False),
               ("closure_passed",True)]
    for field,value in mutations:
        bad=rows(); bad[1][field]=value
        with pytest.raises(ValueError): audit(bad)

def test_audit_deterministic_geometry_unknown_and_authority_false():
    first=audit(rows()); assert first==audit(rows())
    assert first["exact_treatment_geometry"]["status"]=="NOT_RECONSTRUCTIBLE_FROM_FROZEN_EVIDENCE"
    assert all(value is False for value in first["authority"].values())
