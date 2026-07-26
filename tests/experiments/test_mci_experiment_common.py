from __future__ import annotations

import json

import pytest

from runtime.symbolic.mci import ExternalHypothesis
from runtime.symbolic.mci import MCIPlanningConfig
from scripts.experiments.common import (
    bootstrap_paired_diff,
    generate_external_inputs,
    run_paired_campaign,
    write_canonical_json,
)
from scripts.experiments.synthetic_provider import ScheduledHypothesisProvider


class _FakeRunner:
    def __init__(self, profile: str, seed: int) -> None:
        self.profile = profile
        self.seed = seed
        self.calls = 0

    def run_episode(self, *, external_input: float, replay_unit_id: str):
        self.calls += 1
        return {
            "value": external_input
            + (1.0 if self.profile == "mci_integrated_v1" else 0.0),
            "replay_unit_id": replay_unit_id,
        }


def test_inputs_and_campaign_are_reproducible():
    assert generate_external_inputs(42, 5) == generate_external_inputs(42, 5)

    def factory(*, profile: str, seed: int):
        return _FakeRunner(profile, seed)

    kwargs = {
        "runner_factory": factory,
        "seeds": (42, 123),
        "n_episodes_per_seed": 3,
        "external_input_gen": generate_external_inputs,
        "metric": lambda result: result["value"],
    }
    assert run_paired_campaign(**kwargs) == run_paired_campaign(**kwargs)


def test_bootstrap_preserves_known_paired_difference():
    lower, upper = bootstrap_paired_diff(
        (2.0, 3.0, 4.0),
        (1.0, 2.0, 3.0),
        n_resamples=500,
    )
    assert lower == pytest.approx(1.0)
    assert upper == pytest.approx(1.0)


def test_provider_emits_only_on_scheduled_episode_and_receives_feedback():
    proposal = ExternalHypothesis(
        kind="parameter",
        target_id="cooling_effect",
        proposed_value=0.14,
    )
    provider = ScheduledHypothesisProvider({2: (proposal,)})
    assert provider.infer_hypotheses({}) == ()
    assert provider.infer_hypotheses({}) == (proposal,)
    assert provider.infer_hypotheses({}) == ()


def test_canonical_json_has_stable_order_and_newline(tmp_path):
    path = write_canonical_json(tmp_path / "result.json", {"z": 1, "a": 2})
    assert path.read_text(encoding="utf-8") == '{"a":2,"z":1}\n'
    assert json.loads(path.read_text(encoding="utf-8")) == {"a": 2, "z": 1}


def test_runner_factory_keeps_planning_configuration_explicit(tmp_path):
    from scripts.experiments.common import build_isolated_runner

    config = MCIPlanningConfig(
        horizon=3,
        exact_horizon=True,
        objective_mode="trajectory_loss",
    )
    mci = build_isolated_runner(
        work_root=tmp_path,
        experiment="isolation",
        profile="mci_integrated_v1",
        seed=1,
        scenario="deferred_load_trap",
        mci_planning_config=config,
    )
    baseline = build_isolated_runner(
        work_root=tmp_path,
        experiment="isolation",
        profile="core_plus_opt",
        seed=2,
        scenario="deferred_load_trap",
    )

    assert mci._mci_runtime is not None
    assert baseline._mci_runtime is not None
    assert mci._mci_runtime.planning_config == config
    assert baseline._mci_runtime.planning_config == MCIPlanningConfig()
