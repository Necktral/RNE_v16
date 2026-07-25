"""W0-R1 characterization of the canonical scored-pool seam."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError

import pytest

import runtime.memory.mfm_lite.retrieval as retrieval_module
from runtime.memory.mfm_lite.retrieval import (
    MemoryRetrieval,
    ScoredMemoryCandidate,
    ScoredMemoryPool,
    select_top_k,
)
from runtime.storage.records import MemoryRecord


QUERY = {"proposition": "TEMP_HIGH", "alarm": True}


def _record(
    memory_id: str,
    *,
    structure: dict | None = None,
    scale: str = "micro",
    scenario: str = "thermal_homeostasis",
    version: str = "1.0",
    created_at: str = "2026-01-01T00:00:00+00:00",
) -> MemoryRecord:
    return MemoryRecord(
        memory_id=memory_id,
        run_id="run",
        episode_id=f"episode-{memory_id}",
        scale=scale,
        structure_json=structure
        or {
            "proposition": "TEMP_HIGH",
            "alarm": True,
            "propositions": ["TEMP_HIGH"],
        },
        metadata={
            "scenario_metadata": {
                "scenario_name": scenario,
                "scenario_version": version,
            }
        },
        support_count=1,
        created_at=created_at,
    )


class FixedStorage:
    def __init__(self, records: list[MemoryRecord]):
        self.records = records
        self.calls: list[dict] = []

    def retrieve_memory_records(self, **kwargs):
        self.calls.append(dict(kwargs))
        return list(self.records[: kwargs["limit"]])


def _fixture_records() -> list[MemoryRecord]:
    return [
        _record("tie-first", scale="micro"),
        _record("tie-second", scale="macro"),
        _record(
            "low",
            scale="meso",
            structure={"proposition": "OTHER", "alarm": False},
        ),
        _record("cross", scenario="resource_management"),
    ]


def _build(
    records: list[MemoryRecord] | None = None,
    **kwargs,
) -> tuple[MemoryRetrieval, ScoredMemoryPool]:
    retrieval = MemoryRetrieval(
        storage=FixedStorage(_fixture_records() if records is None else records)
    )
    pool = retrieval.build_scored_pool(
        run_id="run",
        query=QUERY,
        candidate_pool_size=20,
        **kwargs,
    )
    return retrieval, pool


def test_pre_cut_pool_and_public_retrieve_are_exactly_equivalent():
    retrieval, pool = _build()
    expected = select_top_k(pool, 3)
    observed = retrieval.retrieve(
        run_id="run",
        query=QUERY,
        limit=3,
        candidate_pool_size=20,
    )

    assert isinstance(pool, ScoredMemoryPool)
    assert [item.memory_id for item in pool.candidates] == [
        "tie-first",
        "tie-second",
        "cross",
        "low",
    ]
    assert [item.canonical_score for item in pool.candidates] == [
        1.0,
        1.0,
        1.0,
        0.0,
    ]
    assert observed == expected
    assert [item["memory_id"] for item in observed] == [
        "tie-first",
        "tie-second",
        "cross",
    ]


@pytest.mark.parametrize(
    ("records", "limit", "expected"),
    [
        ([], 5, []),
        ([_record("one")], 5, ["one"]),
        (_fixture_records(), 2, ["tie-first", "tie-second"]),
        (_fixture_records(), 10, ["tie-first", "tie-second", "cross", "low"]),
    ],
)
def test_empty_short_long_and_overfull_pools_preserve_public_behavior(
    records, limit, expected
):
    storage = FixedStorage(records)
    retrieval = MemoryRetrieval(storage=storage)
    kwargs = {
        "run_id": "run",
        "query": QUERY,
        "limit": limit,
        "candidate_pool_size": 20,
    }
    pool = retrieval.build_scored_pool(**kwargs)
    assert retrieval.retrieve(**kwargs) == select_top_k(pool, limit)
    assert [item["memory_id"] for item in select_top_k(pool, limit)] == expected


def test_ties_preserve_historical_source_order_and_expose_metadata():
    _, pool = _build()
    first, second, cross, low = pool.candidates

    assert (first.source_order, second.source_order, cross.source_order) == (0, 1, 3)
    assert (first.canonical_rank, second.canonical_rank, cross.canonical_rank) == (
        0,
        1,
        2,
    )
    assert first.tie_group == second.tie_group == cross.tie_group == 0
    assert first.tie_size == second.tie_size == cross.tie_size == 3
    assert low.tie_group == 1 and low.tie_size == 1
    assert [item["memory_id"] for item in select_top_k(pool, 2)] == [
        "tie-first",
        "tie-second",
    ]


def test_strict_filter_and_analogical_penalty_match_historical_contract():
    retrieval, strict = _build(
        scenario_name="thermal_homeostasis",
        scenario_filter_mode="strict_same_scenario",
    )
    strict_kwargs = {
        "run_id": "run",
        "query": QUERY,
        "limit": 10,
        "candidate_pool_size": 20,
        "scenario_name": "thermal_homeostasis",
        "scenario_filter_mode": "strict_same_scenario",
    }
    assert retrieval.retrieve(**strict_kwargs) == select_top_k(strict, 10)
    assert [item.memory_id for item in strict.candidates] == [
        "tie-first",
        "tie-second",
        "low",
    ]
    assert strict.filtered_cross_scenario_count == 1

    retrieval, analog = _build(
        scenario_name="thermal_homeostasis",
        scenario_filter_mode="analogical",
    )
    analog_kwargs = {
        **strict_kwargs,
        "scenario_filter_mode": "analogical",
    }
    observed = retrieval.retrieve(**analog_kwargs)
    assert observed == select_top_k(analog, 10)
    by_id = {item.memory_id: item for item in analog.candidates}
    assert by_id["cross"].canonical_score == 0.5
    assert by_id["cross"].cross_scenario_source is True
    assert observed[0]["retrieval_metrics"]["cross_scenario_penalty_applied"] is True


def test_cross_version_penalty_is_preserved_and_auditable():
    records = [_record("same"), _record("older-version", version="0.9")]
    retrieval, pool = _build(
        records,
        scenario_name="thermal_homeostasis",
        scenario_version="1.0",
    )
    kwargs = {
        "run_id": "run",
        "query": QUERY,
        "limit": 5,
        "candidate_pool_size": 20,
        "scenario_name": "thermal_homeostasis",
        "scenario_version": "1.0",
    }
    observed = retrieval.retrieve(**kwargs)
    assert observed == select_top_k(pool, 5)
    assert [item.canonical_score for item in pool.candidates] == [1.0, 0.8]
    assert pool.candidates[1].cross_version_source is True
    assert observed[1]["cross_version_source"] is True


def test_scale_weights_use_the_same_canonical_score_for_pool_and_retrieve():
    records = [
        _record("micro", scale="micro"),
        _record("macro", scale="macro"),
    ]
    retrieval, pool = _build(records, scale_weights={"micro": 0.5, "macro": 1.0})
    kwargs = {
        "run_id": "run",
        "query": QUERY,
        "limit": 2,
        "candidate_pool_size": 20,
        "scale_weights": {"micro": 0.5, "macro": 1.0},
    }
    assert retrieval.retrieve(**kwargs) == select_top_k(pool, 2)
    assert [(item.memory_id, item.canonical_score) for item in pool.candidates] == [
        ("macro", 1.0),
        ("micro", 0.5),
    ]
    assert dict(pool.candidates[1].scoring_trace)["scale_weight"] == 0.5


class DeterministicEmbedder:
    def embed(self, text: str) -> list[float]:
        return [float(text.count("TEMP_HIGH")), float(text.count("OTHER"))]


@pytest.mark.parametrize("enabled", [False, True])
def test_embedding_modes_are_deterministic_and_equivalent(monkeypatch, enabled):
    embedder = DeterministicEmbedder() if enabled else None
    monkeypatch.setattr(retrieval_module, "get_embedder", lambda: embedder)
    monkeypatch.setenv("RNFE_MEMORY_EMBEDDINGS", "hashed" if enabled else "off")
    monkeypatch.setenv("RNFE_MEMORY_EMBEDDINGS_WEIGHT", "0.4")

    retrieval, first = _build()
    second = retrieval.build_scored_pool(
        run_id="run",
        query=QUERY,
        limit=5,
        candidate_pool_size=20,
    )
    kwargs = {
        "run_id": "run",
        "query": QUERY,
        "limit": 5,
        "candidate_pool_size": 20,
    }
    assert retrieval.retrieve(**kwargs) == select_top_k(first, 5)
    assert first.to_dict() == second.to_dict()
    assert first.retrieval_configuration_fingerprint == (
        second.retrieval_configuration_fingerprint
    )
    assert all(
        dict(item.scoring_trace)["embedding_used"] is enabled
        for item in first.candidates
        if item.memory_id != "low"
    )


def test_pool_is_immutable_and_consumer_results_are_detached():
    _, pool = _build()
    original = pool.to_dict()
    two = select_top_k(pool, 2)
    four = select_top_k(pool, 4)

    with pytest.raises(FrozenInstanceError):
        pool.requested_pool_size = 99  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        pool.candidates[0].canonical_rank = 99  # type: ignore[misc]

    two[0]["structure"]["propositions"].append("MUTATED")
    two[0]["retrieval_metrics"]["candidate_pool_size"] = -1
    four[0]["rag_attestation"]["trace_memory_ids"].append("MUTATED")

    assert pool.to_dict() == original
    assert "MUTATED" not in pool.candidates[0].structure["propositions"]
    assert select_top_k(pool, 2)[0]["retrieval_metrics"]["candidate_pool_size"] == 20
    assert len(pool.candidates) == 4


def test_pool_representation_is_deterministic_json_and_has_fingerprints():
    _, first = _build()
    _, second = _build()

    first_json = json.dumps(first.to_dict(), sort_keys=True, separators=(",", ":"))
    second_json = json.dumps(second.to_dict(), sort_keys=True, separators=(",", ":"))
    assert first_json == second_json
    assert first.storage_input_fingerprint == second.storage_input_fingerprint
    assert first.retrieval_configuration_fingerprint == (
        second.retrieval_configuration_fingerprint
    )
    assert first.scoring_version == "mfm-canonical-structural-v1"
    assert isinstance(first.candidates[0], ScoredMemoryCandidate)
