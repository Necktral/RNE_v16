from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts import audit_p1_n3_attribution as audit


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = (
    ROOT
    / "data/reports/p1_experimental/neural-p1-final-20260721-06e95c8"
)
MATRIX_PATH = EVIDENCE / "matrix.json"
AUDIT_V2_PATH = EVIDENCE / "matrix.audit-v2.json"
SHA256S_PATH = EVIDENCE / "SHA256SUMS"


@pytest.fixture(scope="module")
def source_payloads():
    return (
        json.loads(MATRIX_PATH.read_text()),
        json.loads(AUDIT_V2_PATH.read_text()),
    )


def _profile(matrix: dict, profile_id: str) -> dict:
    return next(item for item in matrix["profiles"] if item["profile_id"] == profile_id)


def _build(matrix: dict, audit_v2: dict) -> dict:
    return audit.build_audit(
        matrix,
        audit_v2,
        matrix_sha256=audit.sha256_file(MATRIX_PATH),
        audit_v2_sha256=audit.sha256_file(AUDIT_V2_PATH),
    )


def test_rejects_incorrect_sha256(tmp_path: Path) -> None:
    matrix = tmp_path / "matrix.json"
    audit_v2 = tmp_path / "matrix.audit-v2.json"
    matrix.write_bytes(MATRIX_PATH.read_bytes())
    audit_v2.write_bytes(AUDIT_V2_PATH.read_bytes())
    sha256s = tmp_path / "SHA256SUMS"
    sha256s.write_text(
        f"{'0' * 64}  matrix.json\n{audit.sha256_file(audit_v2)}  matrix.audit-v2.json\n"
    )

    with pytest.raises(audit.AuditError, match="sha256_mismatch:matrix.json"):
        audit.validate_published_hashes(
            matrix_path=matrix, audit_v2_path=audit_v2, sha256s_path=sha256s
        )


@pytest.mark.parametrize(
    ("target", "field", "value", "message"),
    [
        ("matrix", "campaign_id", "other", "matrix_campaign_id_invalid"),
        ("matrix", "commit", "bad", "experiment_commit_invalid"),
        ("audit", "audit_commit", "bad", "audit_v2_commit_invalid"),
    ],
)
def test_rejects_campaign_or_commit_inconsistency(
    source_payloads, target: str, field: str, value: str, message: str
) -> None:
    matrix, audit_v2 = copy.deepcopy(source_payloads)
    (matrix if target == "matrix" else audit_v2)[field] = value
    with pytest.raises(audit.AuditError, match=message):
        audit.validate_evidence(matrix, audit_v2)


@pytest.mark.parametrize("mutation", ["missing", "duplicate"])
def test_rejects_missing_or_duplicate_seeds(source_payloads, mutation: str) -> None:
    matrix, audit_v2 = copy.deepcopy(source_payloads)
    lanes = _profile(matrix, "only-n3-trained")["lanes"]
    if mutation == "missing":
        lanes.pop()
    else:
        lanes[-1]["seed"] = lanes[0]["seed"]
    with pytest.raises(audit.AuditError, match="seed"):
        audit.validate_evidence(matrix, audit_v2)


def test_rejects_missing_expected_profile(source_payloads) -> None:
    matrix, audit_v2 = copy.deepcopy(source_payloads)
    matrix["profiles"] = [
        profile for profile in matrix["profiles"] if profile["profile_id"] != "p1-all"
    ]
    with pytest.raises(audit.AuditError, match="profile_set_invalid"):
        audit.validate_evidence(matrix, audit_v2)


def test_detects_zero_filling_in_profile_without_n3(source_payloads) -> None:
    matrix, audit_v2 = copy.deepcopy(source_payloads)
    profiles = audit.validate_evidence(matrix, audit_v2)
    profiles["off"]["lanes"][0]["summary"]["n3"]["mean_ndcg_delta"] = 0.0
    with pytest.raises(audit.AuditError, match="invalid_metric_missingness_semantics"):
        audit.audit_missingness(profiles)


def test_pairing_is_by_seed_not_input_order() -> None:
    left = {seed: float(index) for index, seed in enumerate(audit.EXPECTED_SEEDS)}
    right = dict(reversed(list({seed: float(index - 1) for index, seed in enumerate(audit.EXPECTED_SEEDS)}.items())))
    differences = audit.paired_differences(left, right)
    assert list(differences) == list(audit.EXPECTED_SEEDS)
    assert set(differences.values()) == {1.0}


def test_bootstrap_is_deterministic() -> None:
    values = [float(index) / 10.0 for index in range(12)]
    assert audit.bootstrap_mean_ci95(values, label="same") == audit.bootstrap_mean_ci95(
        values, label="same"
    )


def test_exact_randomization_enumerates_all_4096_assignments() -> None:
    result = audit.exact_sign_flip_test([1.0] * 12)
    assert result["assignments_enumerated"] == 4096
    assert result["extreme_assignments"] == 2
    assert result["p_value"] == pytest.approx(2 / 4096)


def test_trained_reference_contrast_is_seed_paired(source_payloads) -> None:
    matrix, audit_v2 = source_payloads
    report = _build(matrix, audit_v2)
    contrast = report["contrasts"]["trained_vs_reference"]
    assert contrast["paired_seed_count"] == 12
    assert contrast["mean"] > 0.0
    assert contrast["metrics"]["brier_improvement"]["mean"] > 0.0
    assert contrast["metrics"]["mrr_delta"]["mean"] == 0.0


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        (
            dict(
                integrity_valid=True,
                isolated_supported=True,
                p1_all_signal=True,
                cross_context_consistent=True,
                trained_beats_reference=True,
                limitation_present=True,
            ),
            "n3_attribution_supported_limited",
        ),
        (
            dict(
                integrity_valid=True,
                isolated_supported=False,
                p1_all_signal=True,
                cross_context_consistent=True,
                trained_beats_reference=True,
                limitation_present=True,
            ),
            "n3_signal_detected_but_attribution_inconclusive",
        ),
        (
            dict(
                integrity_valid=True,
                isolated_supported=False,
                p1_all_signal=False,
                cross_context_consistent=False,
                trained_beats_reference=False,
                limitation_present=True,
            ),
            "n3_attribution_not_supported",
        ),
        (
            dict(
                integrity_valid=False,
                isolated_supported=True,
                p1_all_signal=True,
                cross_context_consistent=True,
                trained_beats_reference=True,
                limitation_present=True,
            ),
            "audit_invalid",
        ),
    ],
)
def test_classifies_all_four_verdicts(kwargs: dict, expected: str) -> None:
    assert audit.classify_verdict(**kwargs) == expected


def test_marks_standard_ndcg_not_recomputable(source_payloads) -> None:
    report = _build(*source_payloads)
    status = report["metric_semantics"]["standard_ndcg_recomputation"]
    assert status["status"] == "not_recomputable_from_published_aggregate"
    assert "complete_eligible_pool" in status["missing"]


def test_marks_brier_decomposition_not_recomputable(source_payloads) -> None:
    report = _build(*source_payloads)
    decomposition = report["brier_analysis"]["decomposition"]
    assert decomposition["status"] == "not_recomputable_from_published_aggregate"
    assert decomposition["missing"] == ["individual_predictions", "individual_labels"]


def test_build_preserves_source_files_and_never_authorizes_p2(source_payloads) -> None:
    before = {
        MATRIX_PATH: hashlib.sha256(MATRIX_PATH.read_bytes()).hexdigest(),
        AUDIT_V2_PATH: hashlib.sha256(AUDIT_V2_PATH.read_bytes()).hexdigest(),
    }
    report = _build(*source_payloads)
    after = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in before}
    assert after == before
    assert report["source"]["source_files_preserved"] is True
    assert report["runtime_modified"] is False
    assert report["p2_authorized"] is False
    assert report["gates"]["p2_authorized"] is False
    assert report["invariants"]["raw_runtime_tree_used"] is False

