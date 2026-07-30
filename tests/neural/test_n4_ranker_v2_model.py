from __future__ import annotations

import pytest
import torch

from runtime.neural.training.n4_ranking import (
    CandidateSetSample,
    GAIN_EPSILON,
    N4CandidateRecord,
    N4ModelOutput,
    RISK_EPSILON,
    collate_candidate_sets,
    compare_candidates,
    create_n4_ranker_v2,
    informative_pairs,
    n4_multitask_loss,
    score_n4_artifact_v2,
    train_n4_ranking_v2,
)
from runtime.neural.contracts import NeuralInferenceRequest
from runtime.neural.organs import N4CausalRankingBackend


def _candidate(
    name,
    *,
    risk=0.0,
    valid=False,
    gain=0.0,
):
    return N4CandidateRecord(
        hypothesis_id=name,
        candidate_source="test",
        features=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
        valid_label=valid,
        mae_gain_label=gain,
        invariant_risk_label=risk,
    )


def _historical(name, *, valid=False, gain=0.0):
    return N4CandidateRecord(
        hypothesis_id=name,
        candidate_source="historical",
        features=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
        valid_label=valid,
        mae_gain_label=gain,
        invariant_risk_label=None,
        risk_label_available=False,
        risk_label_version=None,
    )


def _sample(name, candidates):
    return CandidateSetSample(
        candidate_set_id=name,
        scenario_id="scenario",
        seed=1,
        split="train",
        candidates=tuple(sorted(candidates, key=lambda item: item.hypothesis_id)),
    )


def test_preference_is_lexicographic_safety_validity_gain():
    safe_invalid = _candidate("a", risk=0.0, valid=False)
    risky_valid = _candidate("b", risk=0.1, valid=True, gain=1.0)
    assert compare_candidates(safe_invalid, risky_valid) == 1
    assert compare_candidates(risky_valid, safe_invalid) == -1

    valid = _candidate("a", valid=True, gain=-1.0)
    invalid = _candidate("b", valid=False, gain=1.0)
    assert compare_candidates(valid, invalid) == 1

    higher_gain = _candidate("a", valid=True, gain=0.5)
    lower_gain = _candidate("b", valid=True, gain=0.1)
    assert compare_candidates(higher_gain, lower_gain) == 1


def test_preference_tolerances_create_real_ties():
    left = _candidate(
        "a", risk=0.2, valid=True, gain=0.3
    )
    right = _candidate(
        "b",
        risk=0.2 + RISK_EPSILON / 2,
        valid=True,
        gain=0.3 + GAIN_EPSILON / 2,
    )
    assert compare_candidates(left, right) == 0


def test_all_and_only_informative_pairs_are_generated_once():
    sample = _sample(
        "set",
        [
            _candidate("a", valid=True, gain=0.8),
            _candidate("b", valid=True, gain=0.2),
            _candidate("c", valid=False),
            _candidate("d", valid=False),
        ],
    )
    pairs = informative_pairs(sample)
    assert len(pairs) == 5
    assert len(set(frozenset(pair) for pair in pairs)) == len(pairs)
    assert (2, 3) not in pairs and (3, 2) not in pairs


def test_model_produces_four_mask_compatible_heads():
    model = create_n4_ranker_v2(torch)
    output = model(torch.zeros((2, 3, 6)))
    assert output.rank_score.shape == (2, 3)
    assert output.validity_logit.shape == (2, 3)
    assert output.expected_mae_gain.shape == (2, 3)
    assert output.risk_logit.shape == (2, 3)
    assert torch.all(output.expected_mae_gain <= 1.0)
    assert torch.all(output.expected_mae_gain >= -1.0)


def test_loss_is_finite_with_padding_and_without_positive_candidates():
    samples = (
        _sample(
            "set-a",
            [_candidate("a"), _candidate("b", risk=0.2)],
        ),
        _sample("set-b", [_candidate("c")]),
    )
    batch = collate_candidate_sets(samples)
    model = create_n4_ranker_v2(torch)
    output = model(batch.features)
    total, components = n4_multitask_loss(torch, output, batch, samples)
    total.backward()
    assert torch.isfinite(total)
    assert components["gain"].item() == 0.0
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_pair_loss_is_normalized_equally_per_candidate_set():
    small = _sample(
        "small",
        [_candidate("a", valid=True), _candidate("b", valid=False)],
    )
    large = _sample(
        "large",
        [
            _candidate("c", valid=True),
            _candidate("d", valid=False),
            _candidate("e", valid=False),
        ],
    )
    samples = (small, large)
    batch = collate_candidate_sets(samples)
    ranks = torch.tensor(
        [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], requires_grad=True
    )
    zeros = torch.zeros_like(ranks, requires_grad=True)
    output = N4ModelOutput(ranks, zeros, torch.tanh(zeros), zeros)
    _, components = n4_multitask_loss(torch, output, batch, samples)
    expected = torch.nn.functional.softplus(torch.tensor(-1.0))
    assert components["rank"].item() == pytest.approx(expected.item())


def test_sets_without_informative_pairs_have_zero_pair_loss():
    samples = (
        _sample("set", [_candidate("a"), _candidate("b")]),
    )
    batch = collate_candidate_sets(samples)
    model = create_n4_ranker_v2(torch)
    output = model(batch.features)
    total, components = n4_multitask_loss(torch, output, batch, samples)
    assert torch.isfinite(total)
    assert components["rank"].item() == 0.0


def test_loss_is_reproducible_for_same_seed():
    samples = (
        _sample(
            "set",
            [_candidate("a", valid=True), _candidate("b", valid=False)],
        ),
    )
    batch = collate_candidate_sets(samples)
    torch.manual_seed(17)
    first = create_n4_ranker_v2(torch)
    first_loss, _ = n4_multitask_loss(
        torch, first(batch.features), batch, samples
    )
    torch.manual_seed(17)
    second = create_n4_ranker_v2(torch)
    second_loss, _ = n4_multitask_loss(
        torch, second(batch.features), batch, samples
    )
    assert first_loss.item() == second_loss.item()


def test_historical_batch_has_finite_zero_risk_loss():
    samples = (
        _sample(
            "historical",
            [
                _historical("a", valid=True, gain=0.5),
                _historical("b", valid=False),
            ],
        ),
    )
    batch = collate_candidate_sets(samples)
    model = create_n4_ranker_v2(torch)
    total, components = n4_multitask_loss(
        torch, model(batch.features), batch, samples
    )
    total.backward()
    assert torch.isfinite(total)
    assert components["risk"].item() == 0.0
    assert not batch.risk_label_mask.any()


def test_historical_preference_ignores_risk_and_uses_validity_gain():
    valid = _historical("a", valid=True, gain=-0.5)
    invalid = _historical("b", valid=False, gain=1.0)
    assert compare_candidates(valid, invalid) == 1
    higher = _historical("a", valid=True, gain=0.5)
    lower = _historical("b", valid=True, gain=0.1)
    assert compare_candidates(higher, lower) == 1


def test_continuous_risk_uses_sigmoid_huber_not_binary_target():
    samples = (_sample("set", [_candidate("a", risk=0.25)]),)
    batch = collate_candidate_sets(samples)
    zeros = torch.zeros((1, 1), requires_grad=True)
    output = N4ModelOutput(zeros, zeros, zeros, zeros)
    _, components = n4_multitask_loss(torch, output, batch, samples)
    expected = torch.nn.functional.smooth_l1_loss(
        torch.tensor([0.5]), torch.tensor([0.25])
    )
    assert components["risk"].item() == pytest.approx(expected.item())


def test_configurable_risk_weight_changes_total_loss():
    samples = (_sample("set", [_candidate("a", risk=0.25)]),)
    batch = collate_candidate_sets(samples)
    zeros = torch.zeros((1, 1), requires_grad=True)
    output = N4ModelOutput(zeros, zeros, zeros, zeros)
    low, _ = n4_multitask_loss(
        torch,
        output,
        batch,
        samples,
        loss_weights={"rank": 1.0, "valid": 0.5, "gain": 0.3, "risk": 0.2},
    )
    high, _ = n4_multitask_loss(
        torch,
        output,
        batch,
        samples,
        loss_weights={"rank": 1.0, "valid": 0.5, "gain": 0.3, "risk": 0.5},
    )
    assert high.item() > low.item()


def test_v2_training_exports_full_model_and_diagnostic_gates(tmp_path):
    samples = []
    for split_index, split in enumerate(("train", "validation")):
        for set_index in range(3):
            samples.append(
                _sample(
                    f"{split}-{set_index}",
                    [
                        _candidate("a-valid", valid=True, gain=0.5),
                        _candidate("b-invalid", valid=False, gain=-0.5),
                    ],
                )
            )
            original = samples[-1]
            samples[-1] = CandidateSetSample(
                candidate_set_id=original.candidate_set_id,
                scenario_id=original.scenario_id,
                seed=split_index * 10 + set_index,
                split=split,
                candidates=original.candidates,
            )
    path = tmp_path / "n4-v2.json"
    artifact = train_n4_ranking_v2(
        samples,
        artifact_path=path,
        epochs=3,
        patience=2,
        loss_weights={
            "rank": 1.0,
            "valid": 0.5,
            "gain": 0.3,
            "risk": 0.5,
        },
    )
    assert artifact["schema"] == "n4-ranking-artifact.v2"
    assert artifact["model"]["trunk"]["layers"][0]["weights"]
    assert set(artifact["model"]["heads"]) == {"rank", "validity", "gain", "risk"}
    assert artifact["calibration"]["a"] > 0
    assert artifact["calibration"]["target"] == "validity_logit"
    assert artifact["gates"]["risk_target_informative"] is False
    assert artifact["gates"]["promotable"] is False
    assert "holdout_metrics" not in artifact
    assert artifact["training_provenance"]["loss_weights"]["risk"] == 0.5
    assert path.read_bytes().endswith(b"\n")
    features = samples[0].candidates[0].features
    offline = score_n4_artifact_v2(features, artifact)
    runtime = N4CausalRankingBackend(artifact).infer(
        NeuralInferenceRequest(
            inference_id="parity",
            run_id="parity",
            logical_time=1,
            payload={
                "candidates": [
                    {
                        "hypothesis_id": "h",
                        **dict(zip(
                            (
                                "empirical_gain",
                                "holdout_support",
                                "coverage",
                                "simplicity",
                                "stability",
                                "invariant_safety",
                            ),
                            features,
                        )),
                    }
                ]
            },
        )
    ).candidate_output["rankings"][0]
    for name in (
        "rank_score",
        "validity_logit",
        "validity_probability",
        "expected_mae_gain",
        "invariant_risk",
    ):
        assert runtime[name] == pytest.approx(offline[name], abs=1e-6)


def test_development_trainer_rejects_holdout_and_invalid_weights(tmp_path):
    train = CandidateSetSample(
        candidate_set_id="train",
        scenario_id="scenario",
        seed=1,
        split="train",
        candidates=(
            _candidate("a", valid=True, risk=0.1),
            _candidate("b", valid=False, risk=0.4),
        ),
    )
    validation = CandidateSetSample(
        candidate_set_id="validation",
        scenario_id="scenario",
        seed=2,
        split="validation",
        candidates=(
            _candidate("a", valid=True, risk=0.1),
            _candidate("b", valid=False, risk=0.4),
        ),
    )
    holdout = CandidateSetSample(
        candidate_set_id="holdout",
        scenario_id="scenario",
        seed=3,
        split="holdout",
        candidates=(
            _candidate("a", valid=True, risk=0.1),
            _candidate("b", valid=False, risk=0.4),
        ),
    )
    with pytest.raises(ValueError, match="rejects_holdout"):
        train_n4_ranking_v2(
            (train, validation, holdout),
            artifact_path=tmp_path / "holdout.json",
            epochs=1,
        )
    with pytest.raises(ValueError, match="loss_weights_invalid"):
        train_n4_ranking_v2(
            (train, validation),
            artifact_path=tmp_path / "bad.json",
            epochs=1,
            loss_weights={
                "rank": 1.0,
                "valid": 0.5,
                "gain": 0.3,
                "risk": -0.1,
            },
        )
