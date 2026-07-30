from __future__ import annotations

import random

from scripts.generate_n4_dataset import (
    _episode_external_input,
    _scenario_kwargs,
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
