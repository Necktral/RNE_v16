from runtime.neural.integration.p2_n3_decision import P2CandidatePool, rank_pool, seed_values, contrast_statistics

def pool():
    return P2CandidatePool.freeze((
        {"memory_id":"a","scale":"micro","score":1.0,"structure":{}},
        {"memory_id":"b","scale":"macro","score":1.0,"structure":{}},
    ))

def test_canonical_and_permutation_only():
    frozen=pool(); canonical=rank_pool(frozen,arm_id="canonical")
    trained=rank_pool(frozen,arm_id="n3-trained",scale_signals={"micro":0.0,"macro":1.0})
    assert [x["memory_id"] for x in canonical]==["a","b"]
    assert [x["memory_id"] for x in trained]==["b","a"]
    assert {x["memory_id"] for x in canonical}=={x["memory_id"] for x in trained}

def test_seeds_exact_and_statistics_deterministic():
    assert seed_values()==[1752726358,2028904492,451178818,401174225,734129406,965802336,237833712,1695529792,254447928,310802106,1260027411,1959994480]
    values=[0.1]*12
    assert contrast_statistics(values,name="x")==contrast_statistics(values,name="x")
    assert contrast_statistics(values,name="x")["assignments_enumerated"]==4096
