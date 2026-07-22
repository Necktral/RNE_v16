from dataclasses import replace

import pytest

from runtime.neural.integration.p1_n3 import N3ShadowDirective
from runtime.neural.integration.p2_v2_n3_decision import (
    DecisionSeal,
    FrozenRetrieval,
    contrast_statistics,
    open_oracle,
    rerank,
)


def directive(*, micro=0.0, meso=0.0, macro=1.0):
    return N3ShadowDirective(
        status="eligible", reason="test", candidate_hash="abc",
        optimization_direction="minimize", uncertainty=0.2,
        retrieval_priority=0.2, risk=micro, importance=meso,
        continuity=macro,
    )


def pool():
    return FrozenRetrieval.freeze((
        {"memory_id": "a", "scale": "micro", "score": 1.0, "structure": {"x": 1}},
        {"memory_id": "b", "scale": "macro", "score": 1.0, "structure": {"x": 2}},
        {"memory_id": "c", "scale": "meso", "score": 0.5, "structure": {"x": 3}},
    ))


def test_canonical_preserves_real_retrieval_order():
    frozen = pool()
    assert rerank(frozen, arm_id="canonical") == frozen.hits


def test_n3_only_permutes_and_uses_p1_semantics():
    frozen = pool()
    ranked = rerank(frozen, arm_id="n3-trained", directive=directive())
    assert [x["memory_id"] for x in ranked][:2] == ["b", "a"]
    assert {x["memory_id"] for x in ranked} == set(frozen.ids)
    assert {x["memory_id"]: x["structure"] for x in ranked} == {
        x["memory_id"]: x["structure"] for x in frozen.hits
    }


def test_ineligible_or_arbitrary_mapping_fails_closed():
    with pytest.raises(ValueError, match="directive_not_eligible"):
        rerank(pool(), arm_id="n3-trained", directive=None)
    with pytest.raises(ValueError, match="directive_not_eligible"):
        rerank(pool(), arm_id="n3-trained", directive=replace(directive(), status="unavailable"))


def test_oracle_requires_decision_seal():
    with pytest.raises(ValueError, match="P2_ORACLE_BEFORE_DECISION_SEAL"):
        open_oracle(scenario=object(), external_input=0.0,
                    allowed_interventions=("a", "b"), seal=None)


def test_statistics_are_seed_level_and_deterministic():
    first = contrast_statistics([0.1] * 12, name="reference-canonical")
    assert first == contrast_statistics([0.1] * 12, name="reference-canonical")
    assert first["assignments_enumerated"] == 4096
    assert first["gate_passed"] is True

