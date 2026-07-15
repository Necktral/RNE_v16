from copy import deepcopy
from dataclasses import replace
import json

import pytest

from runtime.neural.training import (
    CounterfactualDatasetBuilder,
    CounterfactualSample,
    DatasetQualityReport,
    counterfactual_initial_state_hash,
    train_n1_router,
)


def test_counterfactual_hash_excludes_identity_and_family_treatment() -> None:
    state = {
        "organism_id": "org-off",
        "run_id": "run-off",
        "episode_id": "episode-off",
        "world": {
            "scenario_id": "thermal_homeostasis",
            "scenario_version": "1.0",
            "scenario_config_hash": "config",
            "observation_hash": "observation",
            "world_state_hash": "world",
            "main_variable": "temperature",
            "observable_alarm": False,
        },
        "regime": {
            "regime_id": "homeostatic_cooling",
            "regime_model_version": "v1",
            "equilibrium_class": "stable",
            "recovery_profile": "fast",
            "measurement_status": "measured",
        },
        "organism": {
            "organism_state_id": "state-off",
            "organism_state_hash": "organism-off",
            "viability": {"value": 1.0, "measurement_status": "measured"},
            "continuity": {"value": 0.8, "measurement_status": "measured"},
            "risk": {"value": None, "measurement_status": "unmeasured"},
            "closure": {"value": None, "measurement_status": "unmeasured"},
        },
        "resources": {},
        "homeostasis": {
            "alarm": False,
            "viability_margin": {"value": 1.0, "measurement_status": "measured"},
            "distance_to_edge": {"value": 1.0, "measurement_status": "measured"},
            "rollback_required": False,
        },
        "policy": {"active_overlays": []},
        "state_hash": "full-off",
    }
    treated = deepcopy(state)
    treated.update(
        organism_id="org-on",
        run_id="run-on",
        episode_id="episode-on",
        state_hash="full-on",
    )
    treated["organism"]["organism_state_id"] = "state-on"
    treated["organism"]["organism_state_hash"] = "organism-on"
    treated["policy"]["active_overlays"] = ["IND"]

    assert counterfactual_initial_state_hash(state) == counterfactual_initial_state_hash(treated)

    changed_world = deepcopy(treated)
    changed_world["world"]["observation_hash"] = "different-observation"
    assert counterfactual_initial_state_hash(state) != counterfactual_initial_state_hash(changed_world)


def _record(enabled: bool, *, seed=1, generator="grid", context="ctx", family="IND"):
    return {
        "context_key": context,
        "scenario_generator": generator,
        "seed": seed,
        "family": family,
        "family_enabled": enabled,
        "initial_state_hash": "same-state",
        "features": {"pressure": 0.2, "uncertainty": 0.7},
        "reward": 0.8 if enabled else 0.5,
        "effectiveness": 0.9 if enabled else 0.4,
        "closure": 1.0,
        "certified": 1.0 if enabled else 0.0,
        "continuity": 0.9 if enabled else 0.8,
        "viability": 0.95 if enabled else 0.8,
    }


def test_n1_builder_requires_real_paired_ablation_and_grouped_split() -> None:
    builder = CounterfactualDatasetBuilder()
    samples, report = builder.build([_record(False), _record(True)])
    assert report.valid_pairs == 1
    assert samples[0].positive_utility is True
    split = builder.split(samples)
    assert sum(len(values) for values in split.values()) == 1
    assert report.training_ready() is False


def test_n1_builder_rejects_historical_proxy_as_causal_label() -> None:
    bad = _record(False)
    bad["family_delta_reward"] = 0.3
    _, report = CounterfactualDatasetBuilder().build([bad, _record(True)])
    assert report.valid_pairs == 0
    assert report.rejected_records == 2


def test_n1_builder_rejects_ambiguous_boolean_and_unpaired_features() -> None:
    ambiguous = _record(False)
    ambiguous["family_enabled"] = "false"
    _, ambiguous_report = CounterfactualDatasetBuilder().build([ambiguous, _record(True)])
    assert ambiguous_report.valid_pairs == 0

    off = _record(False)
    on = _record(True)
    on["features"] = {"pressure": 0.9, "uncertainty": 0.7}
    _, mismatch_report = CounterfactualDatasetBuilder().build([off, on])
    assert mismatch_report.valid_pairs == 0


def _sample(index: int, *, family: str = "IND") -> CounterfactualSample:
    utility = 0.2 if index % 2 else -0.1
    return CounterfactualSample(
        pair_id=f"pair-{index}",
        context_key=f"context-{index}",
        scenario_generator=f"generator-{index % 3}",
        seed=index,
        family=family,
        features={"pressure": (index % 10) / 10.0, "uncertainty": 0.5},
        utility_delta=utility,
        positive_utility=utility > 0.0,
        effectiveness_delta=0.0,
        closure_delta=0.0,
        certification_delta=0.0,
        continuity_delta=0.0,
        viability_delta=0.0,
    )


def test_n1_trainer_uses_validation_for_calibration_and_reports_test(tmp_path) -> None:
    train = [_sample(index, family=("IND", "PLAN", "OPT")[index % 3]) for index in range(30)]
    validation = [
        replace(_sample(index, family=("IND", "PLAN", "OPT")[index % 3]), pair_id=f"val-{index}")
        for index in range(30, 36)
    ]
    test = [
        replace(_sample(index, family=("IND", "PLAN", "OPT")[index % 3]), pair_id=f"test-{index}")
        for index in range(36, 42)
    ]
    report = DatasetQualityReport(
        total_records=600,
        valid_pairs=300,
        unique_contexts=60,
        generators=3,
        families=("IND", "PLAN", "OPT"),
        rejected_records=0,
        positive_pairs=150,
        negative_pairs=150,
        utility_min=-0.1,
        utility_max=0.2,
    )

    _manifest, evidence = train_n1_router(
        train,
        report,
        artifact_root=tmp_path,
        epochs=1,
        validation_samples=validation,
        test_samples=test,
    )

    assert evidence["calibration_split"] == "validation"
    assert evidence["calibration_method"] == "validation_scalar_temperature_log_grid_v1"
    assert 0.05 <= evidence["temperature"] <= 10.0
    assert evidence["validation_ece_before_calibration"] is not None
    assert evidence["heldout_evaluated"] is True
    assert evidence["split_metrics"]["validation"]["records"] == 6
    assert evidence["split_metrics"]["test"]["evaluated"] is True
    assert evidence["promotion_eligible"] is False
    artifact = json.loads((tmp_path / "n1/router-lab-v1.json").read_text())
    assert artifact["temperature"] == evidence["temperature"]


def test_n1_trainer_rejects_split_overlap(tmp_path) -> None:
    rows = [_sample(index, family=("IND", "PLAN", "OPT")[index % 3]) for index in range(12)]
    report = DatasetQualityReport(
        600, 300, 60, 3, ("IND", "PLAN", "OPT"), 0, 150, 150, -0.1, 0.2
    )

    with pytest.raises(ValueError, match="train_validation_test_overlap"):
        train_n1_router(
            rows,
            report,
            artifact_root=tmp_path,
            epochs=1,
            validation_samples=(rows[0],),
        )


def test_n1_quality_gate_rejects_single_class_dataset() -> None:
    report = DatasetQualityReport(
        720,
        360,
        60,
        3,
        ("HEUR", "DIA_ADV", "FAL_GUARD", "IND", "PLAN", "OPT"),
        0,
        positive_pairs=0,
        negative_pairs=360,
        utility_min=-0.46,
        utility_max=-0.45,
    )

    assert report.training_ready() is False
