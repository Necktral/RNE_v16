from __future__ import annotations

from pathlib import Path

import pytest

from runtime.memory.mfm_lite.retrieval import MemoryRetrieval
from runtime.neural.integration.p1_n3 import (
    compare_n3_shadow_retrieval,
    derive_n3_shadow_directive,
    retrieval_scale_weights,
    shadow_retrieval_limit,
)
from runtime.storage import StorageConfig, StorageFactory


def _storage(tmp_path: Path):
    return StorageFactory.create_facade(
        StorageConfig(
            mode="sqlite",
            sqlite_db_path=str(tmp_path / "p1-n3.db"),
            postgres_dsn=None,
            artifact_root=tmp_path / "artifacts",
            prefer_postgres_reads=True,
            strict_dual_write=False,
        )
    )


def _reference(*, trend=0.2, uncertainty=0.4):
    return {"status": "ok", "trend": trend, "uncertainty": uncertainty}


@pytest.mark.parametrize(
    ("direction", "trend", "expected_risk"),
    [
        ("minimize", 0.2, 0.2),
        ("minimize", -0.2, 0.0),
        ("maximize", -0.2, 0.2),
        ("maximize", 0.2, 0.0),
    ],
)
def test_reference_directive_respects_optimization_direction(
    direction: str, trend: float, expected_risk: float
) -> None:
    directive = derive_n3_shadow_directive(
        _reference(trend=trend),
        candidate_hash="candidate-hash",
        optimization_direction=direction,
        alarm_threshold=1.0,
    )

    assert directive.eligible is True
    assert directive.risk == pytest.approx(expected_risk)
    assert directive.importance == pytest.approx(0.2)
    assert directive.continuity == pytest.approx(0.8)
    assert directive.authority_effect == "none"
    assert directive.retrieval_limit_delta == 1


def test_target_band_fails_closed_without_explicit_band() -> None:
    directive = derive_n3_shadow_directive(
        _reference(),
        candidate_hash="candidate-hash",
        optimization_direction="target_band",
        alarm_threshold=0.5,
    )

    assert directive.status == "unavailable"
    assert directive.eligible is False
    assert directive.scale_signals == {}
    assert directive.reason == "target_band_requires_explicit_lower_and_upper_bounds"


@pytest.mark.parametrize(
    ("candidate", "candidate_hash", "status", "reason"),
    [
        ({"uncertainty": 0.4, "trend": None}, "hash", "warmup", "reference_trend_not_measured"),
        ({"uncertainty": float("nan"), "trend": 0.2}, "hash", "unavailable", "uncertainty_missing_nonfinite_or_out_of_range"),
        ({"uncertainty": 0.4, "trend": 0.2}, None, "unavailable", "admitted_candidate_hash_required"),
        (None, "hash", "unavailable", "candidate_missing_or_not_mapping"),
    ],
)
def test_missing_or_invalid_evidence_never_becomes_zero(
    candidate, candidate_hash, status: str, reason: str
) -> None:
    directive = derive_n3_shadow_directive(
        candidate,
        candidate_hash=candidate_hash,
        optimization_direction="minimize",
        alarm_threshold=1.0,
    )

    assert directive.status == status
    assert directive.reason == reason
    assert directive.risk is None
    assert directive.retrieval_priority is None


def test_trained_temporal_heads_are_bounded_and_do_not_require_trend() -> None:
    directive = derive_n3_shadow_directive(
        {
            "uncertainty": 0.2,
            "retrieval_priority": 0.9,
            "importance": 0.7,
            "risk": 0.8,
            "continuity": 0.3,
        },
        candidate_hash="trained-hash",
        optimization_direction="minimize",
        alarm_threshold=1.0,
    )

    assert directive.eligible
    assert directive.reason == "bounded_trained_temporal_heads"
    assert directive.retrieval_limit_delta == 2
    assert retrieval_scale_weights(directive) == pytest.approx(
        {"micro": 0.95, "meso": 0.925, "macro": 0.825}
    )
    assert shadow_retrieval_limit(directive, canonical_limit=7) == 8


def test_partial_trained_heads_fail_closed_instead_of_falling_back() -> None:
    directive = derive_n3_shadow_directive(
        {"uncertainty": 0.2, "trend": 0.2, "risk": 0.7},
        candidate_hash="partial-hash",
        optimization_direction="minimize",
        alarm_threshold=1.0,
    )

    assert directive.status == "unavailable"
    assert directive.reason == "trained_temporal_heads_incomplete_or_invalid"


def test_retrieve_shadow_is_default_neutral(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("RNFE_MEMORY_EMBEDDINGS", "off")
    storage = _storage(tmp_path)
    for index, scale in enumerate(("micro", "meso", "macro")):
        storage.write_memory_record(
            run_id="run",
            episode_id=f"episode-{index}",
            scale=scale,
            structure_json={"proposition": "TEMP_HIGH", "alarm": True},
            memory_id=f"memory-{scale}",
        )
    retrieval = MemoryRetrieval(storage=storage)
    kwargs = {
        "run_id": "run",
        "query": {"proposition": "TEMP_HIGH", "alarm": True},
        "limit": 3,
    }

    canonical = retrieval.retrieve(**kwargs)
    neutral_shadow = retrieval.retrieve_shadow(**kwargs)

    assert neutral_shadow == canonical
    storage.close()


def test_retrieve_shadow_weights_scales_without_writes(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("RNFE_MEMORY_EMBEDDINGS", "off")
    storage = _storage(tmp_path)
    storage.write_memory_record(
        run_id="run",
        episode_id="micro-episode",
        scale="micro",
        structure_json={"proposition": "TEMP_HIGH"},
        memory_id="memory-micro",
    )
    storage.write_memory_record(
        run_id="run",
        episode_id="macro-episode",
        scale="macro",
        structure_json={"proposition": "TEMP_HIGH"},
        memory_id="memory-macro",
    )
    retrieval = MemoryRetrieval(storage=storage)
    query = {"proposition": "TEMP_HIGH"}
    canonical = retrieval.retrieve(run_id="run", query=query, limit=2)
    shadow = retrieval.retrieve_shadow(
        run_id="run",
        query=query,
        limit=1,
        scale_signals={"micro": 1.0, "meso": 0.5, "macro": 0.0},
    )

    assert [row["memory_id"] for row in canonical] == ["memory-macro", "memory-micro"]
    assert [row["memory_id"] for row in shadow] == ["memory-micro"]
    assert len(storage.retrieve_memory_records(run_id="run", limit=10)) == 2
    storage.close()


def test_report_is_non_authoritative_and_hashes_both_branches() -> None:
    directive = derive_n3_shadow_directive(
        _reference(),
        candidate_hash="candidate-hash",
        optimization_direction="minimize",
        alarm_threshold=1.0,
    )
    canonical = [{"memory_id": "a", "scale": "macro", "score": 1.0}]
    shadow = [
        {"memory_id": "a", "scale": "macro", "score": 0.9},
        {"memory_id": "b", "scale": "micro", "score": 0.8},
    ]

    report = compare_n3_shadow_retrieval(
        directive=directive,
        canonical_hits=canonical,
        shadow_hits=shadow,
        canonical_scheduler_sequence=("ABD", "PROB"),
        shadow_scheduler_sequence=("ABD", "CAU", "PROB"),
        snapshot_match=True,
    )

    assert report.status == "compared"
    assert report.overlap_count == 1
    assert report.canonical_retrieval_hash != report.shadow_retrieval_hash
    assert report.writes_performed is False
    assert report.authority_effect == "none"
    assert report.snapshot_match is True


@pytest.mark.parametrize(
    "signals",
    [
        {"micro": float("nan")},
        {"micro": -0.1},
        {"micro": 1.1},
        {"unknown": 0.5},
    ],
)
def test_retrieve_shadow_rejects_invalid_scale_signals(
    tmp_path: Path, signals: dict[str, float]
) -> None:
    storage = _storage(tmp_path)
    retrieval = MemoryRetrieval(storage=storage)
    with pytest.raises(ValueError):
        retrieval.retrieve_shadow(run_id="run", query={}, scale_signals=signals)
    storage.close()

