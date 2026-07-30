from __future__ import annotations

import random

from scripts.generate_n4_dataset import (
    _episode_external_input,
    _resolve_seeds,
    _scenario_kwargs,
    summarize_risk_informativeness,
)


def test_seeded_trajectory_policy_is_deterministic():
    first = random.Random(17)
    second = random.Random(17)
    assert _scenario_kwargs("thermal_with_battery", first) == (
        _scenario_kwargs("thermal_with_battery", second)
    )
    assert [_episode_external_input(first) for _ in range(8)] == [
        _episode_external_input(second) for _ in range(8)
    ]


def test_seeded_trajectory_policy_varies_across_seeds():
    trajectories = set()
    for seed in range(6):
        rng = random.Random(seed)
        initial = tuple(
            sorted(_scenario_kwargs("thermal_with_battery", rng).items())
        )
        inputs = tuple(_episode_external_input(rng) for _ in range(8))
        trajectories.add((initial, inputs))
    assert len(trajectories) == 6


def test_seeded_thermal_ranges_cover_safe_preflight_domain():
    for seed in range(100):
        rng = random.Random(seed)
        kwargs = _scenario_kwargs("thermal_with_battery", rng)
        assert 0.86 <= kwargs["initial_temperature"] <= 0.94
        assert 0.45 <= kwargs["initial_battery"] <= 0.95
        assert all(
            0.025 <= _episode_external_input(rng) <= 0.055
            for _ in range(30)
        )


def test_seed_count_can_start_at_reserved_offset():
    assert _resolve_seeds(["4"], seed_start=100) == (100, 101, 102, 103)


def _risk_row(set_id, candidate, risk, events):
    return {
        "candidate_set_id": set_id,
        "hypothesis_id": candidate,
        "risk_label_available": True,
        "risk_label": risk,
        "risk_report": {
            "oracle_events": events,
            "candidate_events": [],
        },
    }


def test_informativeness_gate_requires_intra_set_variance():
    events = [
        {
            "predicate_id": "temp_above_threshold",
            "step": 2,
            "severity": 0.2,
        },
        {
            "predicate_id": "battery_low",
            "step": 3,
            "severity": 0.5,
        },
    ]
    homogeneous = [
        _risk_row("set-a", "a", 0.0, events),
        _risk_row("set-a", "b", 0.0, events),
        _risk_row("set-b", "a", 0.5, events),
        _risk_row("set-b", "b", 0.5, events),
    ]
    summary = summarize_risk_informativeness(homogeneous)
    assert summary["checks"]["global_variance"]
    assert not summary["checks"]["intra_candidate_set_variance"]
    assert not summary["passed"]

    informative = list(homogeneous)
    informative[1] = _risk_row("set-a", "b", 0.25, events)
    summary = summarize_risk_informativeness(informative)
    assert summary["checks"]["intra_candidate_set_variance"]
    assert summary["passed"]
