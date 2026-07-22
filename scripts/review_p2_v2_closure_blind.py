#!/usr/bin/env python3
"""Independent outcome-agnostic reviewer; never imports the v2 builder."""
import argparse, hashlib, itertools, json, math, random, statistics
from collections import Counter, defaultdict
from pathlib import Path

AUTH={k:False for k in ("P3_DESIGN_AUTHORIZED","LIVE_AUTHORITY","STAGING_AUTHORIZED","PROMOTION_AUTHORIZED","MAIN_MERGE_AUTHORIZED")}
def dump(p,v): p.write_text(json.dumps(v,sort_keys=True,indent=2,allow_nan=False)+"\n")
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def dimensions(p,m):
 a=m["arms"]; return {"seeds":len(p["seeds"]),"scenarios":len(p["scenarios"]),"episodes":p["episodes_per_scenario"],"k":p["k_exposed"],"gate":p["treatment_delivery_gate"],"arms":a,"units":len(p["seeds"])*len(p["scenarios"])*p["episodes_per_scenario"],"receipts":len(p["seeds"])*len(p["scenarios"])*p["episodes_per_scenario"]*len(a)}
def stats(vals,name,samples=10000):
 vals=list(map(float,vals))
 if not vals: raise ValueError("p2_v2_contrast_empty_sample")
 if any(not math.isfinite(v) for v in vals): raise ValueError("p2_v2_contrast_nonfinite_sample")
 rng=random.Random(int.from_bytes(hashlib.sha256(f"p2-v2:{name}".encode()).digest()[:8],"big")); boot=sorted(statistics.mean(rng.choice(vals) for _ in vals) for _ in range(samples)); lo,hi=boot[int(.025*samples)],boot[int(.975*samples)]; obs=abs(statistics.mean(vals)); signs=list(itertools.product((-1.,1.),repeat=len(vals))); p=sum(abs(statistics.mean(v*s for v,s in zip(vals,x)))>=obs-1e-15 for x in signs)/len(signs)
 measured=len(vals)>=2
 out={"sample_size":len(vals),"mean":statistics.mean(vals),"median":statistics.median(vals),"standard_deviation_ddof1":statistics.stdev(vals) if measured else None,"standard_deviation_status":"MEASURED" if measured else "NOT_DEFINED_N_LT_2","minimum":min(vals),"maximum":max(vals),"positive_count":sum(v>0 for v in vals),"zero_count":sum(v==0 for v in vals),"negative_count":sum(v<0 for v in vals),"bootstrap_ci95":[lo,hi],"bootstrap_samples":samples,"exact_sign_flip_p_value":p,"assignments_enumerated":2**len(vals),"seed_values":vals}; out["gate_passed"]=out["mean"]>0 and lo>0 and p<.05 and out["positive_count"]>=math.ceil(.75*len(vals)) and out["negative_count"]<=math.floor(.25*len(vals)); return out
def disposition(valid,treatment,contrasts):
 if not valid:return "BLOCKED","INVALID_PRIMARY_EVIDENCE","NOT_EVALUABLE","NONE"
 delivered={a:x["membership_delivered"] for a,x in treatment.items()}
 if not any(delivered.values()):return "READY_TO_CLOSE","TREATMENT_NOT_DELIVERED","NOT_EVALUABLE","NONE"
 passed={"n3-reference":delivered.get("n3-reference",False) and contrasts["reference - canonical"]["gate_passed"],"n3-trained":delivered.get("n3-trained",False) and contrasts["trained - canonical"]["gate_passed"]}
 if all(passed.values()):return "READY_TO_CLOSE","DECISION_GAIN_SUPPORTED_BOTH","SUPPORTED_BOTH","BOTH"
 if passed["n3-reference"]:return "READY_TO_CLOSE","DECISION_GAIN_SUPPORTED_REFERENCE","SUPPORTED_REFERENCE","REFERENCE"
 if passed["n3-trained"]:return "READY_TO_CLOSE","DECISION_GAIN_SUPPORTED_TRAINED","SUPPORTED_TRAINED","TRAINED"
 return "READY_TO_CLOSE","DECISION_GAIN_NOT_DEMONSTRATED","NOT_DEMONSTRATED","NONE"
def validate_primary_evidence(prereg,manifest,rows):
 d=dimensions(prereg,manifest); groups=defaultdict(list); errors=[]
 for r in rows: groups[(r.get("scenario"),r.get("seed"),r.get("episode_index"))].append(r)
 if len(rows)!=d["receipts"]: errors.append({"code":"P2_V2_RECEIPT_COUNT_INVALID","expected":d["receipts"],"observed":len(rows)})
 if len(groups)!=d["units"]: errors.append({"code":"P2_V2_UNIT_COUNT_INVALID","expected":d["units"],"observed":len(groups)})
 units={}
 for key,g in groups.items():
  observed=[r.get("arm_id") for r in g]
  if Counter(observed)!=Counter(d["arms"]):
   errors.append({"code":"P2_V2_THREE_ARM_PAIRING_INVALID","unit":"|".join(map(str,key)),"expected_arms":d["arms"],"observed_arms":observed}); continue
  u={r["arm_id"]:r for r in g}; units[key]=u; c=u["canonical"]
  if len({r["actual_pre_action_state_sha256"] for r in g})!=1 or any(r["snapshot_sha256"]!=r["actual_pre_action_state_sha256"] for r in g): errors.append({"code":"P2_V2_STATE_INVALID","unit":"|".join(map(str,key))})
  fields=("raw_candidate_ids","raw_candidate_scores","raw_candidate_scales","raw_candidate_structures","raw_candidate_pool_sha256")
  if any(r.get(f)!=c.get(f) for r in g for f in fields): errors.append({"code":"P2_V2_POOL_INVALID","unit":"|".join(map(str,key))})
  for r in g:
   nums=(r.get("chosen_utility"),r.get("optimal_utility"),r.get("regret"))
   if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in nums): errors.append({"code":"P2_V2_NONFINITE","unit":"|".join(map(str,key))}); continue
   if abs((nums[1]-nums[0])-nums[2])>1e-12: errors.append({"code":"P2_V2_REGRET_INVALID","unit":"|".join(map(str,key))})
   if not r.get("decision_sealed") or not r.get("oracle_opened_after_seal") or not r.get("chosen_intervention_is_allowed"): errors.append({"code":"P2_V2_DECISION_OR_ORACLE_INVALID","unit":"|".join(map(str,key))})
 valid=not errors
 return {"valid":valid,"errors":errors,"units":units if valid else {},
         "expected_dimensions":{"unit_count":d["units"],"receipt_count":d["receipts"],"arms":d["arms"]},
         "observed_dimensions":{"unit_count":len(groups),"receipt_count":len(rows)},"dimensions":d}
def compute_confirmatory_contrasts(prereg,units):
 defs={"reference - canonical":("canonical","n3-reference"),"trained - canonical":("canonical","n3-trained"),"trained - reference":("n3-reference","n3-trained")};cs={}
 for name,(b,a) in defs.items():
  sv=[]
  for seed in prereg["seeds"]:
   per=[]
   for scenario in prereg["scenarios"]:
    values=[u[b]["regret"]-u[a]["regret"] for (s,sd,e),u in units.items() if s==scenario and sd==seed]; per.append(sum(values)/len(values))
   sv.append(sum(per)/len(per))
  cs[name]=stats(sv,name,prereg.get("bootstrap_samples",10000))
 return cs
def review(prereg,manifest,rows,matrix=None,audit2=None,verdict2=None):
 validation=validate_primary_evidence(prereg,manifest,rows)
 if not validation["valid"]:
  return {"schema_version":"p2-v2-blind-review-v3","campaign_id":prereg["campaign_id"],"derived_dimensions":validation["dimensions"],"observed_dimensions":validation["observed_dimensions"],"primary_evidence_integrity":{"valid":False,"errors":validation["errors"]},"observable_treatment":None,"confirmatory_contrasts_recomputed":None,"exact_treatment_geometry":{"status":"NOT_RECONSTRUCTIBLE_FROM_FROZEN_EVIDENCE"},"authority":AUTH,"review_valid":False,"review_status":"BLOCKED","review_result":"INVALID_PRIMARY_EVIDENCE","n3_decisional_gain":"NOT_EVALUABLE","backend_preference":"NONE"}
 d=validation["dimensions"]; arms=d["arms"]; k=d["k"]; errors=[]
 expected_units=validation["expected_dimensions"]["unit_count"]
 if expected_units<=0: raise ValueError("p2_v2_expected_unit_count_invalid")
 validated_units=validation["units"]
 if len(validated_units)!=expected_units: raise ValueError("p2_v2_validated_unit_count_mismatch")
 counts={a:Counter() for a in arms if a!="canonical"}; mismatches=0
 for key,u in validated_units.items():
  c=u["canonical"];co=c["arm_order_ids"]
  for a in counts:
   r=u[a];ao=r["arm_order_ids"]
   if len(ao)!=len(co) or len(set(ao))!=len(ao) or set(ao)!=set(co):errors.append("conservation")
   f={"full_order":ao!=co,"top1":ao[0]!=co[0],"sequence":ao[:k]!=co[:k],"membership":set(ao[:k])!=set(co[:k]),"action":r["chosen_intervention"]!=c["chosen_intervention"],"regret":abs(r["regret"]-c["regret"])>1e-15};counts[a].update({x:int(y) for x,y in f.items()})
   if (r.get("full_order_changed"),r.get("top1_changed"),r.get("exposed_set_changed"))!=(f["full_order"],f["top1"],f["membership"]):mismatches+=1
 t={}
 for a,c in counts.items():
  t[a]={x+"_count":c[x] for x in ("full_order","top1","sequence","membership","action","regret")};t[a].update({x+"_rate":c[x]/d["units"] for x in ("full_order","top1","sequence","membership","action","regret")});t[a]["membership_delivered"]=t[a]["membership_rate"]>=d["gate"]
 cs=compute_confirmatory_contrasts(prereg,validated_units)
 valid=not errors;status,result,gain,pref=disposition(valid,t,cs)
 return {"schema_version":"p2-v2-blind-review-v3","campaign_id":prereg["campaign_id"],"derived_dimensions":d,"observed_dimensions":validation["observed_dimensions"],"primary_evidence_integrity":{"valid":valid,"errors":sorted(set(errors)),"derived_field_mismatches":mismatches},"observable_treatment":t,"confirmatory_contrasts_recomputed":cs,"exact_treatment_geometry":{"status":"NOT_RECONSTRUCTIBLE_FROM_FROZEN_EVIDENCE"},"comparison_with_v1":{"contrasts_match":matrix is not None and matrix.get("contrasts")==cs},"comparison_with_v2":{"numeric_treatment_match":audit2 is not None and all(abs(t[a]["membership_rate"]-audit2["treatment_geometry_observed"][a]["top4_membership_change_rate"])<1e-15 for a in t)},"authority":AUTH,"review_valid":valid,"review_status":status,"review_result":result,"n3_decisional_gain":gain,"backend_preference":pref}
def bias_findings():
 specs=[
  ("HARDCODED_K_EXPOSED",True,"high","scripts/build_p2_v2_bounded_closure.py","classify_influence slices [:4]","synthetic k=2 or k=5 is classified with four","metrics",True,False),
  ("HARDCODED_TREATMENT_GATE",True,"high","scripts/build_p2_v2_bounded_closure.py","delivery compares against .10","synthetic gate=.40 still uses .10","disposition",True,False),
  ("HARDCODED_VALID_TRUE",True,"high","scripts/build_p2_v2_bounded_closure.py","audit return sets valid true","inconsistent fixture cannot select invalid output","validity",True,False),
  ("HARDCODED_CLOSED_STATUS",True,"medium","scripts/build_p2_v2_bounded_closure.py","verdict sets CLOSED","invalid fixture still receives closure template","disposition",True,False),
  ("HARDCODED_RESULT",True,"high","scripts/build_p2_v2_bounded_closure.py","failure result is literal","delivered synthetic campaign cannot choose supported result","disposition",True,False),
  ("HARDCODED_BACKEND_PREFERENCE",True,"medium","scripts/build_p2_v2_bounded_closure.py","backend preference is NONE","supported synthetic arm cannot become preferred","disposition",True,False),
  ("NON_NEUTRAL_DELIVERED_BRANCH",True,"high","scripts/build_p2_v2_bounded_closure.py","delivered branch ignores contrast gates","synthetic delivered+passing contrast selects not-demonstrated","disposition",True,False),
  ("EXPECTED_DIMENSIONS_NOT_FULLY_DERIVED",True,"medium","scripts/build_p2_v2_bounded_closure.py","N3 arms and top-k are module literals","manifest arms and prereg k variants are not authoritative","metrics",True,False),
  ("SELF_CONFIRMING_INPUT_DEPENDENCY",False,"none","scripts/build_p2_v2_bounded_closure.py","v2 does not consume its own prior audit as calculation input","independent recomputation path exists","none",False,False)]
 return {"schema_version":"p2-v2-bias-findings-v1","findings":[{"finding_id":i,"present":p,"severity":s,"file":f,"evidence":e,"behavioral_test":b,"behavioral_impact":impact,"can_change_metrics":cm,"can_change_disposition":cd,"did_change_observed_v2_result":changed} for i,p,s,f,e,b,impact,cm,changed in specs for cd in [impact=="disposition"]]}
def knowledge_gain(result):
 t=result["observable_treatment"]
 return {"resolved_questions":[{"question":"Did N3 observably alter ranking and exposure sequence?","evidence":"observable_treatment full_order and sequence rates"},{"question":"Was preregistered membership treatment delivered?","evidence":"membership_rate and preregistered gate"}],"partially_resolved_questions":[{"question":"Can order alone alter IND action?","evidence":"action changes observed without membership change; causal quality not isolated"}],"unresolved_questions":[{"question":"Which recorded N3 signal or multiplier caused each rank transition?","evidence":"exact_treatment_geometry NOT_RECONSTRUCTIBLE"}],"unexpected_findings":[{"finding":"Action changed despite unchanged exposure membership","evidence":{"reference_action_rate":t["n3-reference"]["action_rate"],"trained_action_rate":t["n3-trained"]["action_rate"]}}],"causal_claims_authorized":["N3 arms changed observable ordering under the frozen harness"],"causal_claims_not_authorized":["N3 improved decisional gain","Either backend is superior","Operational readiness"],"instrumentation_gaps":["observed scale signals","applied multipliers","candidate adjusted scores"],"highest_information_remaining_question":"Does a preregistered N3 intervention that changes top-k membership reduce seed-level decisional regret relative to canonical retrieval?"}
def main():
 p=argparse.ArgumentParser();p.add_argument("--source",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();s=a.source;o=a.output;o.mkdir(parents=True,exist_ok=True);load=lambda n:json.loads((s/n).read_text());rows=[json.loads(x) for x in (s/"decision-receipts.jsonl").read_text().splitlines() if x];r=review(load("preregistration.json"),load("manifest.json"),rows,load("matrix.json"),load("matrix.audit-v2.json"),load("verdict.v2.json"));r["source_hashes"]={n:sha(s/n) for n in ("preregistration.json","manifest.json","decision-receipts.jsonl")};dump(o/"matrix.audit-v3.json",r);t=r["observable_treatment"];classification="CONFIRMED_WITH_IMPLEMENTATION_BIAS" if r["comparison_with_v2"]["numeric_treatment_match"] and r["review_result"]=="TREATMENT_NOT_DELIVERED" else "SUPERSEDED";v={"P2_V2_REVIEW_STATUS":r["review_status"],"P2_V2_REVIEW_RESULT":r["review_result"],"P2_V2_CLOSURE_V2_CLASSIFICATION":classification,"P2_V2_MEMBERSHIP_TREATMENT_REFERENCE":"DELIVERED" if t["n3-reference"]["membership_delivered"] else "NOT_DELIVERED","P2_V2_MEMBERSHIP_TREATMENT_TRAINED":"DELIVERED" if t["n3-trained"]["membership_delivered"] else "NOT_DELIVERED","P2_V2_ORDER_TREATMENT_REFERENCE":"OBSERVED" if t["n3-reference"]["sequence_rate"]>0 else "NOT_OBSERVED","P2_V2_ORDER_TREATMENT_TRAINED":"OBSERVED" if t["n3-trained"]["sequence_rate"]>0 else "NOT_OBSERVED","N3_DECISIONAL_GAIN":r["n3_decisional_gain"],"N3_BACKEND_PREFERENCE":r["backend_preference"],**AUTH};dump(o/"verdict.v3.json",v);dump(o/"BIAS_FINDINGS.json",bias_findings());dump(o/"closure.review.json",{"schema_version":"p2-v2-closure-review-v1","classification":classification,"review_result":r["review_result"],"knowledge_gain":knowledge_gain(r),"authority":AUTH})
if __name__=="__main__":main()
