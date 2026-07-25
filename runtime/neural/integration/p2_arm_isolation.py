"""Deterministic per-arm state isolation for the W0-R2 causal harness.

This module deliberately contains no treatment logic.  It captures one
pre-action unit, freezes the two N3 outputs once, and creates independent
scenario/storage/RNG contexts from that immutable snapshot.
"""

from __future__ import annotations

import json
import os
import random
import sys
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from runtime.neural.integration.contracts import (
    canonical_json_bytes,
    canonical_sha256,
    canonicalize,
)
from runtime.world.deferred_load_scenario import DeferredLoadScenario, DeferredLoadState
from runtime.world.grid_thermal_scenario import CellState, GridState, GridThermalScenario
from runtime.world.resource_scenario import ResourceScenario, ResourceWorldState
from runtime.world.thermal_scenario import ThermalScenario, ThermalWorldState


SNAPSHOT_SCHEMA_VERSION = "p2-unit-state-snapshot-v1"
BACKEND_OUTPUT_SCHEMA_VERSION = "p2-frozen-backend-output-v1"
RNG_SCHEMA_VERSION = "p2-rng-state-v1"
ENVIRONMENT_SCHEMA_VERSION = "p2-environment-state-v1"

_ORACLE_FORBIDDEN_KEYS = frozenset(
    {
        "outcomes",
        "optimal_intervention",
        "optimal_utility",
        "utility",
        "utilities",
        "regret",
        "oracle_action",
        "best_actions",
    }
)
_ENVIRONMENT_KEYS = (
    "RNFE_MEMORY_EMBEDDINGS",
    "RNFE_MEMORY_EMBEDDINGS_WEIGHT",
    "RNFE_MEMORY_CANDIDATE_POOL",
    "RNFE_EMBEDDINGS_GGUF",
    "RNFE_LLAMA_EMBEDDING_CLI",
    "RNFE_EXTERNAL_REASONER_NGL",
    "RNFE_EMBEDDINGS_TIMEOUT_S",
    "RNFE_ARTIFACT_ROOT",
    "RNFE_NEURAL_N3_MANIFEST",
    "RNFE_REASONING_ACTUATES",
)


def _strict_json(value: Any) -> str:
    return canonical_json_bytes(value).decode("utf-8")


def _strict_load(payload: str) -> Any:
    value = json.loads(payload)
    canonicalize(value)
    return value


def _tuple_tree(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_tuple_tree(item) for item in value)
    return value


def _assert_no_oracle_fields(value: Any, *, path: str = "$") -> None:
    if isinstance(value, Mapping):
        forbidden = _ORACLE_FORBIDDEN_KEYS.intersection(value)
        if forbidden:
            raise ValueError(
                f"p2_oracle_field_in_prestate:{path}:{sorted(forbidden)[0]}"
            )
        for key, item in value.items():
            _assert_no_oracle_fields(item, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_no_oracle_fields(item, path=f"{path}[{index}]")


@dataclass(frozen=True, slots=True)
class EnvironmentSnapshot:
    schema_version: str
    values_json: str
    sha256: str

    @classmethod
    def capture(
        cls, values: Mapping[str, str | None] | None = None
    ) -> "EnvironmentSnapshot":
        selected = (
            {key: os.environ.get(key) for key in _ENVIRONMENT_KEYS}
            if values is None
            else {str(key): value for key, value in values.items()}
        )
        payload = {"schema_version": ENVIRONMENT_SCHEMA_VERSION, "values": selected}
        return cls(
            schema_version=ENVIRONMENT_SCHEMA_VERSION,
            values_json=_strict_json(selected),
            sha256=canonical_sha256(payload),
        )

    @property
    def values(self) -> dict[str, str | None]:
        return dict(_strict_load(self.values_json))

    def verify_current(self) -> None:
        current = {key: os.environ.get(key) for key in self.values}
        if current != self.values:
            raise ValueError("p2_environment_drift")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "values": self.values,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EnvironmentSnapshot":
        captured = cls.capture(value.get("values") or {})
        if value.get("schema_version") != captured.schema_version:
            raise ValueError("p2_environment_schema_mismatch")
        if value.get("sha256") != captured.sha256:
            raise ValueError("p2_environment_hash_mismatch")
        return captured


@dataclass(frozen=True, slots=True)
class RNGStateSnapshot:
    schema_version: str
    python_state_json: str
    numpy_state_json: str | None
    torch_cpu_state_json: str | None
    torch_cuda_states_json: str | None
    sha256: str

    @classmethod
    def capture(cls) -> "RNGStateSnapshot":
        python_state = canonicalize(random.getstate())
        numpy_state: Any | None = None
        numpy = sys.modules.get("numpy")
        if numpy is not None:
            raw = numpy.random.get_state()
            numpy_state = {
                "bit_generator": str(raw[0]),
                "keys": raw[1].tolist(),
                "position": int(raw[2]),
                "has_gauss": int(raw[3]),
                "cached_gaussian": float(raw[4]),
            }
        torch_cpu: Any | None = None
        torch_cuda: Any | None = None
        torch = sys.modules.get("torch")
        if torch is not None:
            torch_cpu = torch.random.get_rng_state().tolist()
            if torch.cuda.is_available():
                torch_cuda = [state.tolist() for state in torch.cuda.get_rng_state_all()]
        payload = {
            "schema_version": RNG_SCHEMA_VERSION,
            "python": python_state,
            "numpy": numpy_state,
            "torch_cpu": torch_cpu,
            "torch_cuda": torch_cuda,
        }
        return cls(
            schema_version=RNG_SCHEMA_VERSION,
            python_state_json=_strict_json(python_state),
            numpy_state_json=_strict_json(numpy_state) if numpy_state is not None else None,
            torch_cpu_state_json=_strict_json(torch_cpu) if torch_cpu is not None else None,
            torch_cuda_states_json=(
                _strict_json(torch_cuda) if torch_cuda is not None else None
            ),
            sha256=canonical_sha256(payload),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "python": _strict_load(self.python_state_json),
            "numpy": (
                _strict_load(self.numpy_state_json)
                if self.numpy_state_json is not None
                else None
            ),
            "torch_cpu": (
                _strict_load(self.torch_cpu_state_json)
                if self.torch_cpu_state_json is not None
                else None
            ),
            "torch_cuda": (
                _strict_load(self.torch_cuda_states_json)
                if self.torch_cuda_states_json is not None
                else None
            ),
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RNGStateSnapshot":
        payload = {
            "schema_version": value.get("schema_version"),
            "python": value.get("python"),
            "numpy": value.get("numpy"),
            "torch_cpu": value.get("torch_cpu"),
            "torch_cuda": value.get("torch_cuda"),
        }
        if payload["schema_version"] != RNG_SCHEMA_VERSION:
            raise ValueError("p2_rng_schema_mismatch")
        digest = canonical_sha256(payload)
        if digest != value.get("sha256"):
            raise ValueError("p2_rng_hash_mismatch")
        return cls(
            schema_version=RNG_SCHEMA_VERSION,
            python_state_json=_strict_json(payload["python"]),
            numpy_state_json=(
                _strict_json(payload["numpy"])
                if payload["numpy"] is not None
                else None
            ),
            torch_cpu_state_json=(
                _strict_json(payload["torch_cpu"])
                if payload["torch_cpu"] is not None
                else None
            ),
            torch_cuda_states_json=(
                _strict_json(payload["torch_cuda"])
                if payload["torch_cuda"] is not None
                else None
            ),
            sha256=digest,
        )

    def instantiate(self) -> tuple[random.Random, Any | None, Any | None, tuple[Any, ...]]:
        python_rng = random.Random()
        python_rng.setstate(_tuple_tree(_strict_load(self.python_state_json)))

        numpy_rng: Any | None = None
        if self.numpy_state_json is not None:
            try:
                import numpy
            except ImportError as exc:
                raise RuntimeError("p2_numpy_rng_unavailable") from exc
            data = _strict_load(self.numpy_state_json)
            numpy_rng = numpy.random.RandomState()
            numpy_rng.set_state(
                (
                    data["bit_generator"],
                    numpy.array(data["keys"], dtype=numpy.uint32),
                    data["position"],
                    data["has_gauss"],
                    data["cached_gaussian"],
                )
            )

        torch_cpu_rng: Any | None = None
        torch_cuda_rngs: tuple[Any, ...] = ()
        if self.torch_cpu_state_json is not None:
            try:
                import torch
            except ImportError as exc:
                raise RuntimeError("p2_torch_rng_unavailable") from exc
            torch_cpu_rng = torch.Generator(device="cpu")
            torch_cpu_rng.set_state(
                torch.tensor(
                    _strict_load(self.torch_cpu_state_json), dtype=torch.uint8
                )
            )
            if self.torch_cuda_states_json is not None:
                generators = []
                for index, state in enumerate(
                    _strict_load(self.torch_cuda_states_json)
                ):
                    generator = torch.Generator(device=f"cuda:{index}")
                    generator.set_state(torch.tensor(state, dtype=torch.uint8))
                    generators.append(generator)
                torch_cuda_rngs = tuple(generators)
        return python_rng, numpy_rng, torch_cpu_rng, torch_cuda_rngs


@dataclass(frozen=True, slots=True)
class BackendOutputSnapshot:
    schema_version: str
    reference_output_json: str
    trained_output_json: str
    sha256: str

    @classmethod
    def freeze(
        cls,
        *,
        reference_output: Mapping[str, Any],
        trained_output: Mapping[str, Any],
    ) -> "BackendOutputSnapshot":
        _assert_no_oracle_fields(reference_output)
        _assert_no_oracle_fields(trained_output)
        payload = {
            "schema_version": BACKEND_OUTPUT_SCHEMA_VERSION,
            "reference": dict(reference_output),
            "trained": dict(trained_output),
        }
        return cls(
            schema_version=BACKEND_OUTPUT_SCHEMA_VERSION,
            reference_output_json=_strict_json(payload["reference"]),
            trained_output_json=_strict_json(payload["trained"]),
            sha256=canonical_sha256(payload),
        )

    @property
    def reference_output(self) -> dict[str, Any]:
        return dict(_strict_load(self.reference_output_json))

    @property
    def trained_output(self) -> dict[str, Any]:
        return dict(_strict_load(self.trained_output_json))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "reference": self.reference_output,
            "trained": self.trained_output,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BackendOutputSnapshot":
        frozen = cls.freeze(
            reference_output=value.get("reference") or {},
            trained_output=value.get("trained") or {},
        )
        if value.get("schema_version") != frozen.schema_version:
            raise ValueError("p2_backend_output_schema_mismatch")
        if value.get("sha256") != frozen.sha256:
            raise ValueError("p2_backend_output_hash_mismatch")
        return frozen


def derive_frozen_backend_outputs(
    *,
    reference_deriver: Callable[[], Mapping[str, Any]],
    trained_deriver: Callable[[], Mapping[str, Any]],
) -> BackendOutputSnapshot:
    """Evaluate each stateful producer exactly once and freeze detached outputs."""

    reference = reference_deriver()
    trained = trained_deriver()
    if not isinstance(reference, Mapping) or not isinstance(trained, Mapping):
        raise TypeError("p2_backend_output_must_be_mapping")
    return BackendOutputSnapshot.freeze(
        reference_output=dict(reference), trained_output=dict(trained)
    )


def _capture_scenario(scenario: Any) -> dict[str, Any]:
    if isinstance(scenario, ThermalScenario):
        return {
            "type": "thermal_homeostasis",
            "parameters": {
                "alarm_threshold": scenario._alarm_threshold,
                "cooling_effect": scenario._cooling_effect,
            },
            "state": {
                "temperature": scenario._state.temperature,
                "cooling_active": scenario._state.cooling_active,
                "alarm": scenario._state.alarm,
            },
        }
    if isinstance(scenario, ResourceScenario):
        return {
            "type": "resource_management",
            "parameters": {
                "scarcity_threshold": scenario._scarcity_threshold,
                "production_rate": scenario._production_rate,
            },
            "state": {
                "stock_level": scenario._state.stock_level,
                "production_active": scenario._state.production_active,
                "scarcity_alert": scenario._state.scarcity_alert,
            },
        }
    if isinstance(scenario, DeferredLoadScenario):
        return {
            "type": "deferred_load_trap",
            "parameters": {
                "alarm_threshold": scenario._alarm_threshold,
                "boost_effect": scenario._boost_effect,
                "shed_effect": scenario._shed_effect,
                "boost_debt": scenario._boost_debt,
                "shed_debt": scenario._shed_debt,
            },
            "state": {
                "load": scenario._state.load,
                "debt": scenario._state.debt,
                "boosting": scenario._state.boosting,
                "alarm": scenario._state.alarm,
            },
        }
    if isinstance(scenario, GridThermalScenario):
        return {
            "type": "grid_thermal_5x5",
            "parameters": {
                "alarm_threshold": scenario._alarm_threshold,
                "cooling_effect": scenario._cooling_effect,
                "grid_size": scenario._grid_size,
            },
            "state": {
                "cells": [
                    {
                        "row": cell.row,
                        "col": cell.col,
                        "temperature": cell.temperature,
                        "cooling_active": cell.cooling_active,
                    }
                    for cell in scenario._grid.cells
                ],
                "global_temp_mean": scenario._grid.global_temp_mean,
                "global_temp_max": scenario._grid.global_temp_max,
                "global_alarm": scenario._grid.global_alarm,
                "cooling_cells_count": scenario._grid.cooling_cells_count,
            },
        }
    raise TypeError(f"p2_scenario_type_unsupported:{type(scenario).__name__}")


def _restore_scenario(payload: Mapping[str, Any]) -> Any:
    kind = payload.get("type")
    parameters = dict(payload.get("parameters") or {})
    state = dict(payload.get("state") or {})
    if kind == "thermal_homeostasis":
        scenario = ThermalScenario(
            initial_temperature=state["temperature"],
            alarm_threshold=parameters["alarm_threshold"],
            cooling_effect=parameters["cooling_effect"],
        )
        scenario._state = ThermalWorldState(**state)
        return scenario
    if kind == "resource_management":
        scenario = ResourceScenario(
            initial_stock=state["stock_level"],
            scarcity_threshold=parameters["scarcity_threshold"],
            production_rate=parameters["production_rate"],
        )
        scenario._state = ResourceWorldState(**state)
        return scenario
    if kind == "deferred_load_trap":
        scenario = DeferredLoadScenario(
            initial_load=state["load"],
            **parameters,
        )
        scenario._state = DeferredLoadState(**state)
        return scenario
    if kind == "grid_thermal_5x5":
        scenario = GridThermalScenario(
            initial_temperature=state["global_temp_mean"],
            alarm_threshold=parameters["alarm_threshold"],
            cooling_effect=parameters["cooling_effect"],
            grid_size=parameters["grid_size"],
            topology="uniform",
        )
        scenario._grid = GridState(
            cells=[CellState(**dict(item)) for item in state["cells"]],
            global_temp_mean=state["global_temp_mean"],
            global_temp_max=state["global_temp_max"],
            global_alarm=state["global_alarm"],
            cooling_cells_count=state["cooling_cells_count"],
        )
        return scenario
    raise ValueError(f"p2_scenario_snapshot_unsupported:{kind}")


def _capture_storage_record(record: Any) -> dict[str, Any]:
    """Capture only fields retrieval actually consumes (B24 excluded)."""

    return {
        "memory_id": str(record.memory_id),
        "run_id": str(record.run_id),
        "episode_id": str(record.episode_id),
        "scale": str(record.scale),
        "structure_json": dict(record.structure_json or {}),
        "ttl_seconds": (
            int(record.ttl_seconds) if record.ttl_seconds is not None else None
        ),
        "certificate_id": (
            str(record.certificate_id)
            if record.certificate_id is not None
            else None
        ),
        "ioc_proxy": (
            float(record.ioc_proxy) if record.ioc_proxy is not None else None
        ),
        "support_count": int(record.support_count),
        "created_at": str(record.created_at),
        "metadata": dict(record.metadata or {}),
    }


@dataclass(slots=True)
class _IsolatedMemoryRecord:
    memory_id: str
    run_id: str
    episode_id: str
    scale: str
    structure_json: dict[str, Any]
    ttl_seconds: int | None
    certificate_id: str | None
    ioc_proxy: float | None
    support_count: int
    created_at: str
    metadata: dict[str, Any]


class IsolatedMemoryStorage:
    """Read-only detached storage surface sufficient for canonical retrieval."""

    def __init__(self, records: Sequence[Mapping[str, Any]]) -> None:
        self._records_json = [_strict_json(dict(record)) for record in records]
        self._disposed = False

    def retrieve_memory_records(
        self,
        *,
        run_id: str,
        scales: Sequence[str] | None = None,
        limit: int = 100,
    ) -> list[_IsolatedMemoryRecord]:
        if self._disposed:
            raise RuntimeError("p2_arm_context_disposed")
        allowed = set(scales or ())
        out: list[_IsolatedMemoryRecord] = []
        for payload in self._records_json:
            item = dict(_strict_load(payload))
            if item["run_id"] != run_id:
                continue
            if allowed and item["scale"] not in allowed:
                continue
            out.append(_IsolatedMemoryRecord(**item))
            if len(out) >= limit:
                break
        return out

    def dispose(self) -> None:
        self._records_json.clear()
        self._disposed = True

    def __getattr__(self, name: str) -> Any:
        if name.startswith(("write", "append", "upsert", "delete", "purge")):
            raise RuntimeError("p2_isolated_storage_read_only")
        raise AttributeError(name)


@dataclass(frozen=True, slots=True)
class UnitStateSnapshot:
    schema_version: str
    unit_id: str
    seed: int
    episode_index: int
    payload_json: str
    snapshot_hash: str

    @property
    def payload(self) -> dict[str, Any]:
        return dict(_strict_load(self.payload_json))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "unit_id": self.unit_id,
            "seed": self.seed,
            "episode_index": self.episode_index,
            "payload": self.payload,
            "snapshot_hash": self.snapshot_hash,
        }


def capture_unit_state(
    *,
    unit_id: str,
    scenario: Any,
    seed: int,
    episode_index: int,
    external_input: float,
    storage_records: Sequence[Any],
    retrieval_configuration: Mapping[str, Any],
    canonical_scored_pool: Mapping[str, Any],
    reference_state: Mapping[str, Any],
    trained_state: Mapping[str, Any],
    reference_deriver: Callable[[], Mapping[str, Any]],
    trained_deriver: Callable[[], Mapping[str, Any]],
    environment: Mapping[str, str | None] | None = None,
) -> UnitStateSnapshot:
    if not str(unit_id).strip():
        raise ValueError("p2_unit_id_required")

    # Freeze X_unit before either stateful producer is allowed to advance history
    # or consume process RNG.  Strict JSON snapshots also detach caller-owned
    # mappings that a producer might mutate while deriving its one unit output.
    rng = RNGStateSnapshot.capture()
    env = EnvironmentSnapshot.capture(environment)
    scenario_payload = _strict_load(_strict_json(_capture_scenario(scenario)))
    storage_payload = _strict_load(
        _strict_json([_capture_storage_record(record) for record in storage_records])
    )
    retrieval_payload = _strict_load(_strict_json(dict(retrieval_configuration)))
    pool_payload = _strict_load(_strict_json(dict(canonical_scored_pool)))
    reference_state_payload = _strict_load(_strict_json(dict(reference_state)))
    trained_state_payload = _strict_load(_strict_json(dict(trained_state)))
    allowed_actions = _strict_load(_strict_json(list(scenario.config.interventions)))
    scenario_identity = str(scenario.config.name)
    frozen_external_input = float(external_input)

    backend_outputs = derive_frozen_backend_outputs(
        reference_deriver=reference_deriver,
        trained_deriver=trained_deriver,
    )
    payload = {
        "scenario_identity": scenario_identity,
        "scenario": scenario_payload,
        "external_input": frozen_external_input,
        "allowed_actions": allowed_actions,
        "storage_records": storage_payload,
        "storage_order": [record["memory_id"] for record in storage_payload],
        "retrieval_configuration": retrieval_payload,
        "canonical_scored_pool": pool_payload,
        "reference_state": reference_state_payload,
        "trained_state": trained_state_payload,
        "backend_outputs": backend_outputs.to_dict(),
        "rng": rng.to_dict(),
        "environment": env.to_dict(),
        "producer_versions": {
            "snapshot": SNAPSHOT_SCHEMA_VERSION,
            "backend_output": BACKEND_OUTPUT_SCHEMA_VERSION,
            "rng": RNG_SCHEMA_VERSION,
            "environment": ENVIRONMENT_SCHEMA_VERSION,
        },
    }
    _assert_no_oracle_fields(payload)
    envelope = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "unit_id": str(unit_id),
        "seed": int(seed),
        "episode_index": int(episode_index),
        "payload": payload,
    }
    return UnitStateSnapshot(
        schema_version=SNAPSHOT_SCHEMA_VERSION,
        unit_id=str(unit_id),
        seed=int(seed),
        episode_index=int(episode_index),
        payload_json=_strict_json(payload),
        snapshot_hash=canonical_sha256(envelope),
    )


def serialize_snapshot(snapshot: UnitStateSnapshot) -> bytes:
    return canonical_json_bytes(snapshot.to_dict())


def deserialize_snapshot(raw: bytes | str) -> UnitStateSnapshot:
    data = json.loads(raw)
    canonicalize(data)
    if data.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
        raise ValueError("p2_snapshot_schema_mismatch")
    payload = data.get("payload")
    if not isinstance(payload, Mapping):
        raise ValueError("p2_snapshot_payload_required")
    _assert_no_oracle_fields(payload)
    BackendOutputSnapshot.from_dict(payload["backend_outputs"])
    RNGStateSnapshot.from_dict(payload["rng"])
    EnvironmentSnapshot.from_dict(payload["environment"])
    envelope = {
        "schema_version": data["schema_version"],
        "unit_id": data["unit_id"],
        "seed": data["seed"],
        "episode_index": data["episode_index"],
        "payload": payload,
    }
    digest = canonical_sha256(envelope)
    if digest != data.get("snapshot_hash"):
        raise ValueError("p2_snapshot_hash_mismatch")
    return UnitStateSnapshot(
        schema_version=SNAPSHOT_SCHEMA_VERSION,
        unit_id=str(data["unit_id"]),
        seed=int(data["seed"]),
        episode_index=int(data["episode_index"]),
        payload_json=_strict_json(payload),
        snapshot_hash=digest,
    )


class _PreActionScenarioView:
    """Scenario proxy that keeps the oracle closed until a decision is sealed."""

    def __init__(self, scenario: Any) -> None:
        self.__scenario = scenario
        self.__decision_sha256: str | None = None
        self.__disposed = False

    @property
    def config(self) -> Any:
        self._require_live()
        return self.__scenario.config

    @property
    def causal_signature(self) -> Any:
        self._require_live()
        return self.__scenario.causal_signature

    def observe(self) -> Any:
        self._require_live()
        return self.__scenario.observe()

    def select_intervention(self, observation: Any) -> str:
        self._require_live()
        return self.__scenario.select_intervention(observation)

    def factual_transition(self, *, intervention: str, external_input: float) -> Any:
        self._require_live()
        return self.__scenario.factual_transition(
            intervention=intervention, external_input=external_input
        )

    def seal_preaction_decision(self, decision_sha256: str) -> None:
        self._require_live()
        digest = str(decision_sha256)
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError("p2_decision_seal_invalid")
        if self.__decision_sha256 is not None and self.__decision_sha256 != digest:
            raise RuntimeError("p2_decision_already_sealed")
        self.__decision_sha256 = digest

    def simulate_counterfactual(
        self, *, intervention: str, external_input: float
    ) -> Any:
        self._require_live()
        if self.__decision_sha256 is None:
            raise RuntimeError("p2_oracle_access_before_decision_seal")
        return self.__scenario.simulate_counterfactual(
            intervention=intervention, external_input=external_input
        )

    def semantic_state(self) -> dict[str, Any]:
        self._require_live()
        return _capture_scenario(self.__scenario)

    def dispose(self) -> None:
        self.__scenario = None
        self.__decision_sha256 = None
        self.__disposed = True

    def _require_live(self) -> None:
        if self.__disposed:
            raise RuntimeError("p2_arm_context_disposed")


@dataclass(slots=True)
class ArmExecutionContext:
    snapshot_hash: str
    arm_id: str
    scenario: _PreActionScenarioView
    storage: IsolatedMemoryStorage
    python_rng: random.Random
    numpy_rng: Any | None
    torch_cpu_rng: Any | None
    torch_cuda_rngs: tuple[Any, ...]
    backend_outputs: BackendOutputSnapshot
    canonical_pool_json: str
    retrieval_configuration_json: str
    environment: EnvironmentSnapshot
    disposed: bool = False

    @property
    def canonical_pool(self) -> dict[str, Any]:
        if self.disposed:
            raise RuntimeError("p2_arm_context_disposed")
        return dict(_strict_load(self.canonical_pool_json))

    @property
    def retrieval_configuration(self) -> dict[str, Any]:
        if self.disposed:
            raise RuntimeError("p2_arm_context_disposed")
        return dict(_strict_load(self.retrieval_configuration_json))


def instantiate_arm_context(
    snapshot: UnitStateSnapshot, arm_id: str
) -> ArmExecutionContext:
    if not str(arm_id).strip():
        raise ValueError("p2_arm_id_required")
    payload = snapshot.payload
    environment = EnvironmentSnapshot.from_dict(payload["environment"])
    environment.verify_current()
    rng = RNGStateSnapshot.from_dict(payload["rng"])
    python_rng, numpy_rng, torch_cpu_rng, torch_cuda_rngs = rng.instantiate()
    context = ArmExecutionContext(
        snapshot_hash=snapshot.snapshot_hash,
        arm_id=str(arm_id),
        scenario=_PreActionScenarioView(_restore_scenario(payload["scenario"])),
        storage=IsolatedMemoryStorage(payload["storage_records"]),
        python_rng=python_rng,
        numpy_rng=numpy_rng,
        torch_cpu_rng=torch_cpu_rng,
        torch_cuda_rngs=torch_cuda_rngs,
        backend_outputs=BackendOutputSnapshot.from_dict(payload["backend_outputs"]),
        canonical_pool_json=_strict_json(payload["canonical_scored_pool"]),
        retrieval_configuration_json=_strict_json(
            payload["retrieval_configuration"]
        ),
        environment=environment,
    )
    verify_arm_prestate(snapshot, context)
    return context


def verify_arm_prestate(
    snapshot: UnitStateSnapshot, context: ArmExecutionContext
) -> bool:
    if context.disposed:
        raise RuntimeError("p2_arm_context_disposed")
    if context.snapshot_hash != snapshot.snapshot_hash:
        raise ValueError("p2_arm_snapshot_hash_mismatch")
    payload = snapshot.payload
    _assert_no_oracle_fields(payload)
    if context.scenario.semantic_state() != payload["scenario"]:
        raise ValueError("p2_arm_scenario_prestate_mismatch")
    if list(context.scenario.config.interventions) != payload["allowed_actions"]:
        raise ValueError("p2_arm_allowed_actions_mismatch")
    if context.canonical_pool != payload["canonical_scored_pool"]:
        raise ValueError("p2_arm_pool_prestate_mismatch")
    if context.retrieval_configuration != payload["retrieval_configuration"]:
        raise ValueError("p2_arm_retrieval_configuration_mismatch")
    if context.backend_outputs.to_dict() != payload["backend_outputs"]:
        raise ValueError("p2_arm_backend_output_mismatch")
    restored_records = context.storage.retrieve_memory_records(
        run_id=payload["storage_records"][0]["run_id"]
        if payload["storage_records"]
        else "",
        scales=[],
        limit=max(1, len(payload["storage_records"])),
    )
    observed_records = [_capture_storage_record(record) for record in restored_records]
    if observed_records != payload["storage_records"]:
        raise ValueError("p2_arm_storage_prestate_mismatch")
    if [record["memory_id"] for record in observed_records] != payload["storage_order"]:
        raise ValueError("p2_arm_storage_order_mismatch")
    return True


def dispose_arm_context(context: ArmExecutionContext) -> None:
    if context.disposed:
        return
    context.scenario.dispose()
    context.storage.dispose()
    context.python_rng = random.Random(0)
    context.numpy_rng = None
    context.torch_cpu_rng = None
    context.torch_cuda_rngs = ()
    context.disposed = True
