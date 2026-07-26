"""W0-R3 seven-arm factorial treatment-kernel tests."""

from __future__ import annotations

import json
import random
import sys
from dataclasses import replace

import pytest

from runtime.neural.integration.p2_arm_isolation import capture_unit_state
from runtime.neural.integration.p2_factorial_kernel import (
    FACTORIAL_ARMS,
    FactorialArmReceipt,
    GoldenUnitArtifact,
    execute_factorial_unit,
    verify_factorial_receipt,
    verify_golden_unit_artifact,
    write_golden_unit_artifact,
)
from runtime.world.thermal_scenario import ThermalScenario


def _candidate(
    memory_id: str,
    *,
    scale: str,
    score: float,
    rank: int,
    intervention: str,
):
    return {
        "memory_id": memory_id,
        "run_id": "run",
        "episode_id": f"episode-{memory_id}",
        "scale": scale,
        "canonical_score": score,
        "canonical_rank": rank,
        "source_order": rank,
        "structure": {
            "relation_kind": "support",
            "intervention": intervention,
            "propositions": ["TEMP_HIGH"],
        },
    }


def _snapshot(*, signals=None, pool=None, reference_deriver=None, trained_deriver=None):
    signals = signals or {"micro": 0.0, "meso": 0.2, "macro": 1.0}
    pool = pool or [
        _candidate(
            "m0",
            scale="micro",
            score=0.90,
            rank=0,
            intervention="activate_cooling",
        ),
        _candidate(
            "m1",
            scale="meso",
            score=0.80,
            rank=1,
            intervention="deactivate_cooling",
        ),
        _candidate(
            "m2",
            scale="macro",
            score=0.70,
            rank=2,
            intervention="activate_cooling",
        ),
        _candidate(
            "m3",
            scale="macro",
            score=0.69,
            rank=3,
            intervention="deactivate_cooling",
        ),
        _candidate(
            "m4",
            scale="micro",
            score=0.68,
            rank=4,
            intervention="activate_cooling",
        ),
    ]
    python_state = random.getstate()
    numpy = sys.modules.get("numpy")
    numpy_state = numpy.random.get_state() if numpy is not None else None
    torch = sys.modules.get("torch")
    torch_cpu_state = torch.random.get_rng_state() if torch is not None else None
    torch_cuda_states = (
        torch.cuda.get_rng_state_all()
        if torch is not None and torch.cuda.is_available()
        else None
    )
    random.seed(1704)
    if numpy is not None:
        numpy.random.seed(1704)
    if torch is not None:
        torch.manual_seed(1704)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(1704)
    try:
        return capture_unit_state(
            unit_id="thermal_homeostasis:17:4",
            scenario=ThermalScenario(initial_temperature=0.91),
            seed=17,
            episode_index=4,
            external_input=0.01,
            storage_records=[],
            retrieval_configuration={"limit": 3, "embedding_mode": "off"},
            canonical_scored_pool={
                "scoring_version": "mfm-canonical-structural-v1",
                "retrieval_configuration_fingerprint": "config",
                "storage_input_fingerprint": "storage",
                "candidates": pool,
            },
            reference_state={"schema_version": "n3-temporal-checkpoint-v1", "entries": []},
            trained_state={"schema_version": "mamba2-temporal-history-v1", "entries": []},
            reference_deriver=reference_deriver
            or (
                lambda: {
                    "status": "ok",
                    "backend": "reference",
                    "scale_signals": dict(signals),
                }
            ),
            trained_deriver=trained_deriver
            or (
                lambda: {
                    "status": "ok",
                    "backend": "trained",
                    "scale_signals": dict(signals),
                }
            ),
            environment={},
        )
    finally:
        random.setstate(python_state)
        if numpy is not None:
            numpy.random.set_state(numpy_state)
        if torch is not None:
            torch.random.set_rng_state(torch_cpu_state)
            if torch_cuda_states is not None:
                torch.cuda.set_rng_state_all(torch_cuda_states)


@pytest.fixture
def golden():
    return execute_factorial_unit(
        snapshot=_snapshot(),
        campaign_id="w0-r3-golden",
        top_k=3,
    )


def _by_arm(golden):
    return {receipt.arm_plan.arm_id: receipt for receipt in golden.receipts}


def test_complete_seven_arm_factorial_unit(golden):
    assert tuple(receipt.arm_plan.arm_id for receipt in golden.receipts) == FACTORIAL_ARMS
    assert len(golden.receipts) == 7
    assert verify_golden_unit_artifact(golden) is True
    assert all(receipt.authority_effect == "none" for receipt in golden.receipts)
    assert all(receipt.training_executed is False for receipt in golden.receipts)
    assert all(receipt.shared_state_writes == 0 for receipt in golden.receipts)


def test_membership_and_sequence_factors_are_delivered_independently(golden):
    arms = _by_arm(golden)

    assert arms["C"].membership.delivered is False
    assert arms["C"].sequence.delivered is False
    for arm in ("R-S", "T-S"):
        assert arms[arm].membership.membership_after_ids == (
            arms[arm].membership.membership_before_ids
        )
        assert arms[arm].sequence.delivered is True
    for arm in ("R-M", "T-M"):
        assert arms[arm].membership.delivered is True
        assert arms[arm].sequence.delivered is False
        expected = tuple(
            sorted(
                arms[arm].membership.membership_after_ids,
                key={"m0": 0, "m1": 1, "m2": 2, "m3": 3, "m4": 4}.get,
            )
        )
        assert arms[arm].sequence.sequence_after_ids == expected
    for arm in ("R-SM", "T-SM"):
        assert arms[arm].membership.delivered is True
        assert arms[arm].sequence.delivered is True


def test_membership_is_exactly_one_real_top_k_swap_without_invented_candidates(golden):
    raw_ids = ("m0", "m1", "m2", "m3", "m4")
    for receipt in golden.receipts:
        assert verify_factorial_receipt(receipt, raw_pool_ids=raw_ids)
        before = set(receipt.membership.membership_before_ids)
        after = set(receipt.membership.membership_after_ids)
        if receipt.membership.delivered:
            assert len(before - after) == len(after - before) == 1
            assert receipt.membership.swap_in_id == "m3"
            assert receipt.membership.swap_out_id == "m1"
            assert receipt.membership.swap_margin > 0
        else:
            assert before == after


def test_sequence_treatment_preserves_membership_and_reports_metrics(golden):
    for receipt in golden.receipts:
        before = receipt.sequence.sequence_before_ids
        after = receipt.sequence.sequence_after_ids
        assert set(before) == set(after)
        assert -1.0 <= receipt.sequence.kendall_tau <= 1.0
        assert receipt.sequence.spearman_distance >= 0
        assert receipt.sequence.inversion_count >= 0
        assert receipt.sequence.delivered is (before != after)


def test_real_ind_decision_is_sealed_before_oracle_and_regret_is_reconstructible(golden):
    for receipt in golden.receipts:
        seal = receipt.decision_seal
        oracle = receipt.oracle
        assert seal.allowed_action_validated is True
        assert seal.oracle_unopened is True
        assert seal.decision_order == 1
        assert oracle.oracle_order == 2
        assert oracle.decision_hash == seal.decision_hash
        assert oracle.decision_already_sealed is True
        assert oracle.opened_after_seal is True
        assert oracle.regret == pytest.approx(
            oracle.optimal_utility - oracle.chosen_utility
        )


def test_receipts_round_trip_and_golden_artifact_are_reconstructible(golden, tmp_path):
    path = tmp_path / "golden-unit.json"
    persisted_hash = write_golden_unit_artifact(path, golden)
    raw = json.loads(path.read_text())
    reconstructed = GoldenUnitArtifact.from_dict(raw)

    assert reconstructed == golden
    assert persisted_hash
    assert reconstructed.artifact_hash == golden.artifact_hash
    for original, rebuilt in zip(golden.receipts, reconstructed.receipts):
        assert FactorialArmReceipt.from_dict(original.to_dict()) == rebuilt
        assert original.receipt_hash == rebuilt.receipt_hash


def _scientific_view(receipt):
    value = receipt.to_dict()
    value["arm_plan"].pop("execution_position")
    return value


def test_arm_order_invariance_and_repeated_unit_hashes():
    snapshot = _snapshot()
    orders = (
        FACTORIAL_ARMS,
        tuple(reversed(FACTORIAL_ARMS)),
        ("R-M", "T-S", "C", "T-SM", "R-S", "T-M", "R-SM"),
    )
    artifacts = [
        execute_factorial_unit(
            snapshot=snapshot,
            campaign_id="w0-r3-order",
            top_k=3,
            arm_order=order,
        )
        for order in orders
    ]
    views = [{r.arm_plan.arm_id: _scientific_view(r) for r in a.receipts} for a in artifacts]
    assert views[0] == views[1] == views[2]

    repeated = execute_factorial_unit(
        snapshot=snapshot,
        campaign_id="w0-r3-order",
        top_k=3,
    )
    assert repeated.artifact_hash == artifacts[0].artifact_hash


def test_raw_pool_and_prestate_hash_are_identical_across_all_arms(golden):
    assert len({receipt.raw_pool_hash for receipt in golden.receipts}) == 1
    assert len({receipt.unit_snapshot_hash for receipt in golden.receipts}) == 1
    assert all(
        tuple(memory_id for memory_id, _ in receipt.adjusted_scores)
        == ("m0", "m1", "m2", "m3", "m4")
        for receipt in golden.receipts
    )
    arms = _by_arm(golden)
    assert arms["C"].backend_output_hash is None
    assert len({arms[arm].backend_output_hash for arm in ("R-S", "R-M", "R-SM")}) == 1
    assert len({arms[arm].backend_output_hash for arm in ("T-S", "T-M", "T-SM")}) == 1
    assert arms["R-S"].backend_output_hash != arms["T-S"].backend_output_hash


def test_backends_are_not_rederived_inside_any_factorial_arm():
    calls = {"reference": 0, "trained": 0}
    signals = {"micro": 0.0, "meso": 0.2, "macro": 1.0}

    def reference():
        calls["reference"] += 1
        return {"status": "ok", "scale_signals": signals}

    def trained():
        calls["trained"] += 1
        return {"status": "ok", "scale_signals": signals}

    snapshot = _snapshot(
        reference_deriver=reference,
        trained_deriver=trained,
    )
    assert calls == {"reference": 1, "trained": 1}
    execute_factorial_unit(
        snapshot=snapshot,
        campaign_id="w0-r3-single-derivation",
        top_k=3,
    )
    assert calls == {"reference": 1, "trained": 1}


def test_membership_ineligible_when_swap_margin_is_not_positive():
    flat = {"micro": 0.0, "meso": 0.0, "macro": 0.0}
    artifact = execute_factorial_unit(
        snapshot=_snapshot(signals=flat),
        campaign_id="w0-r3-ineligible",
        top_k=3,
    )
    for arm in ("R-M", "R-SM", "T-M", "T-SM"):
        receipt = _by_arm(artifact)[arm]
        assert receipt.membership.eligible is False
        assert receipt.membership.delivered is False
        assert receipt.membership.ineligibility_reason == "nonpositive_swap_margin"


def test_nonfinite_signals_and_incomplete_arm_orders_fail_closed():
    with pytest.raises(ValueError, match="finite"):
        _snapshot(
            signals={"micro": 0.0, "meso": 0.0, "macro": float("nan")}
        )
    with pytest.raises(ValueError, match="complete"):
        execute_factorial_unit(
            snapshot=_snapshot(),
            campaign_id="w0-r3-invalid",
            top_k=3,
            arm_order=("C",),
        )


def test_tampered_receipt_is_rejected(golden):
    receipt = _by_arm(golden)["R-M"]
    tampered_membership = replace(
        receipt.membership,
        membership_after_ids=("invented", *receipt.membership.membership_after_ids[1:]),
    )
    tampered = replace(receipt, membership=tampered_membership)
    with pytest.raises(ValueError, match="invented"):
        verify_factorial_receipt(
            tampered,
            raw_pool_ids=("m0", "m1", "m2", "m3", "m4"),
        )
