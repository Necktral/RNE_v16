from __future__ import annotations

import inspect
from pathlib import Path

from runtime.neural.integration import (
    SymbiosisIdentity,
    SymbioticNeuralCoordinator,
    canonical_adapter_registry,
)
from runtime.neural.organs.n4_causal import CausalMessagePassingBackend
from runtime.storage import StorageConfig, StorageFactory


def _storage(tmp_path: Path):
    return StorageFactory.create_facade(
        StorageConfig(
            mode="sqlite",
            sqlite_db_path=str(tmp_path / "canonical-adapters.db"),
            postgres_dsn=None,
            artifact_root=tmp_path / "artifacts",
            prefer_postgres_reads=False,
            strict_dual_write=False,
        )
    )


def test_registry_has_one_canonical_adapter_per_organ_and_no_inline_producers() -> None:
    registry = canonical_adapter_registry()
    assert set(registry) == {"N1", "N2", "N3", "N4", "N5", "N6"}
    assert {adapter.organ for adapter in registry.values()} == set(registry)
    source = inspect.getsource(SymbioticNeuralCoordinator)
    for organ in range(1, 7):
        assert f"def _n{organ}" not in source


def test_live_n4_path_calls_canonical_typed_backend(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("RNFE_NEURAL_MODE", "shadow")
    calls: list[str] = []
    original = CausalMessagePassingBackend.infer

    def observed(self, request):
        calls.append(str(request.payload["graph"]["schema_version"]))
        return original(self, request)

    monkeypatch.setattr(CausalMessagePassingBackend, "infer", observed)
    storage = _storage(tmp_path)
    coordinator = SymbioticNeuralCoordinator(storage=storage)
    coordinator.begin_episode(
        identity=SymbiosisIdentity(
            trace_group_id="trace-adapter",
            organism_id="organism-adapter",
            lineage_id="lineage-adapter",
            run_id="run-adapter",
            episode_id="episode-adapter",
            scenario_id="scenario@1",
            decision_id="decision-adapter",
        ),
        observation={"temperature": 0.8},
        formula="temperature > 0.5",
        proposition="temperature high",
        memory_hits=[],
        scenario_metadata={"main_variable": "temperature"},
        causal_attestation={
            "factual_delta": -0.2,
            "counterfactual_delta": 0.1,
            "supports_choice": True,
        },
        resources={},
    )
    assert calls == ["n4-causal-graph-v1"]
    storage.close()
