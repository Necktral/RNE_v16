from pathlib import Path

from runtime.storage import StorageConfig, StorageFactory
from runtime.world.causal_attestation import build_causal_attestation
from runtime.world.registry import get_scenario
from runtime.world.scenario_runner import ScenarioEpisodeRunner


def _storage(tmp_path: Path):
    return StorageFactory.create_facade(
        StorageConfig(
            mode="sqlite",
            sqlite_db_path=str(tmp_path / "causal_attestation.db"),
            postgres_dsn=None,
            artifact_root=tmp_path / "artifacts",
            prefer_postgres_reads=False,
            strict_dual_write=False,
        )
    )


def test_causal_attestation_passes_when_counterfactual_supports_relation():
    scenario = get_scenario("thermal_homeostasis")
    attestation = build_causal_attestation(
        scenario_name="thermal_homeostasis",
        main_variable="temperature",
        intervention="activate_cooling",
        observation={"temperature": 0.88},
        factual={"temperature": 0.82},
        counterfactual={"temperature": 0.92},
        relation_kind="support",
        signature=scenario.causal_signature,
    )

    assert attestation["schema"] == "causal_attestation.v1"
    assert attestation["validation_status"] == "pass"
    assert attestation["supports_choice"] is True
    assert attestation["agreement_with_relation_kind"] is True
    assert attestation["signature"]["signature_present"] is True


def test_causal_attestation_fails_when_relation_contradicts_counterfactual():
    scenario = get_scenario("thermal_homeostasis")
    attestation = build_causal_attestation(
        scenario_name="thermal_homeostasis",
        main_variable="temperature",
        intervention="activate_cooling",
        observation={"temperature": 0.88},
        factual={"temperature": 0.94},
        counterfactual={"temperature": 0.82},
        relation_kind="support",
        signature=scenario.causal_signature,
    )

    assert attestation["validation_status"] == "fail"
    assert attestation["degradation_level"] == "relation_mismatch"
    assert attestation["supports_choice"] is False
    assert attestation["agreement_with_relation_kind"] is False


def test_scenario_runner_persists_causal_attestation_in_episode_trace_and_certificate(tmp_path: Path):
    storage = _storage(tmp_path)
    runner = ScenarioEpisodeRunner(
        storage=storage,
        run_id="run-causal-attestation",
        scenario="thermal_homeostasis",
        closure_profile="adaptive_min",
    )

    result = runner.run_episode(external_input=0.05)

    attestation = result["episode"]["context"]["causal_attestation"]
    assert attestation["schema"] == "causal_attestation.v1"
    assert attestation["validation_status"] in {"pass", "warn"}
    governance = result["reasoning"]["governance"]
    assert governance["causality"]["attestation"]["schema"] == "causal_attestation.v1"
    assert result["reasoning"]["trace"][0]["detail"]["governance"] == governance

    certs = storage.list_episode_certificates(run_id="run-causal-attestation", limit=1)
    assert certs
    transfer = certs[0].metadata["transfer_assessment"]
    assert transfer["causal_attestation"]["schema"] == "causal_attestation.v1"
    storage.close()
