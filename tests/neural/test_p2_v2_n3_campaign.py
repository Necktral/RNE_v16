from runtime.neural.integration.p1_n3 import derive_n3_shadow_directive


def test_reference_and_trained_share_official_p1_bridge():
    reference = derive_n3_shadow_directive(
        {"uncertainty": .2, "trend": .3}, candidate_hash="r",
        optimization_direction="minimize", alarm_threshold=.7,
    )
    trained = derive_n3_shadow_directive(
        {"uncertainty": .2, "retrieval_priority": .4, "importance": .3,
         "risk": .2, "continuity": .7}, candidate_hash="t",
        optimization_direction="minimize", alarm_threshold=.7,
    )
    assert reference.eligible and trained.eligible
    assert trained.scale_signals == {"micro": .2, "meso": .3, "macro": .7}
    assert reference.scale_signals.keys() == trained.scale_signals.keys()


def test_authority_ceiling_is_structural():
    source = __import__(
        "runtime.neural.integration.p2_v2_n3_decision", fromlist=["x"]
    )
    assert not hasattr(source, "promote")
    assert not hasattr(source, "train")

