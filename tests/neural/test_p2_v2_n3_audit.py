import copy

import pytest

from scripts.audit_p2_v2_n3_causal import audit_receipts


def receipt(arm):
    return {
        "scenario": "s", "seed": 1, "episode_index": 0, "arm_id": arm,
        "snapshot_sha256": "x", "actual_pre_action_state_sha256": "x",
        "snapshot_matches_actual": True, "raw_candidate_pool_sha256": "p",
        "raw_candidate_ids": ["a", "b"], "raw_candidate_scores": [1.0, .5],
        "canonical_order_ids": ["a", "b"],
        "arm_order_ids": ["a", "b"], "raw_candidate_structures": [{}, {}],
        "exposed_memory_ids": ["a"], "exposed_memory_sha256": "dummy",
        "decision_sealed": True, "oracle_opened_after_seal": True,
        "chosen_intervention": "a", "optimal_intervention": "a",
        "chosen_utility": 1.0, "optimal_utility": 1.0, "regret": 0.0,
        "exposed_set_changed": arm != "canonical",
        "chosen_intervention_is_allowed": True, "external_reasoner_used": False,
        "training_executed": False, "live_authority": False,
    }


def test_auditor_recomputes_pairing_and_detects_missing_arm():
    rows = [receipt(x) for x in ("canonical", "n3-reference", "n3-trained")]
    assert audit_receipts(rows, expected_units=1)["three_arm_pairing"] == 1.0
    with pytest.raises(ValueError, match="three_arm_pairing"):
        audit_receipts(rows[:-1], expected_units=1)


def test_auditor_detects_pool_mutation_and_nonfinite():
    rows = [receipt(x) for x in ("canonical", "n3-reference", "n3-trained")]
    broken = copy.deepcopy(rows)
    broken[1]["raw_candidate_ids"] = ["a", "c"]
    with pytest.raises(ValueError, match="raw_pool"):
        audit_receipts(broken, expected_units=1)
    broken = copy.deepcopy(rows)
    broken[0]["regret"] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        audit_receipts(broken, expected_units=1)
