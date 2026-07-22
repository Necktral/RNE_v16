from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts import build_p1_closure as closure


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "data/reports/p1_experimental/neural-p1-final-20260721-06e95c8"
MATRIX = EVIDENCE / "matrix.json"
AUDIT_V2 = EVIDENCE / "matrix.audit-v2.json"
N3 = EVIDENCE / "n3-attribution.audit-v1.json"
SHA256S = EVIDENCE / "SHA256SUMS"


@pytest.fixture(scope="module")
def sources():
    return json.loads(MATRIX.read_text()), json.loads(AUDIT_V2.read_text()), json.loads(
        N3.read_text()
    )


@pytest.fixture(scope="module")
def model(sources):
    return closure.build_closure_model(
        *sources,
        matrix_sha256=closure.sha256_file(MATRIX),
        audit_v2_sha256=closure.sha256_file(AUDIT_V2),
        n3_sha256=closure.sha256_file(N3),
    )


def _build(sources):
    return closure.build_closure_model(
        *sources,
        matrix_sha256=closure.sha256_file(MATRIX),
        audit_v2_sha256=closure.sha256_file(AUDIT_V2),
        n3_sha256=closure.sha256_file(N3),
    )


def test_rejects_incorrect_sha(tmp_path: Path) -> None:
    matrix = tmp_path / "matrix.json"
    audit = tmp_path / "matrix.audit-v2.json"
    n3 = tmp_path / "n3-attribution.audit-v1.json"
    matrix.write_bytes(MATRIX.read_bytes())
    audit.write_bytes(AUDIT_V2.read_bytes())
    n3.write_bytes(N3.read_bytes())
    sha = tmp_path / "SHA256SUMS"
    sha.write_text(
        f"{'0' * 64}  matrix.json\n"
        f"{closure.sha256_file(audit)}  matrix.audit-v2.json\n"
        f"{closure.sha256_file(n3)}  n3-attribution.audit-v1.json\n"
    )
    with pytest.raises(closure.ClosureError, match="sha256_mismatch:matrix.json"):
        closure.validate_hashes(
            matrix_path=matrix,
            audit_v2_path=audit,
            n3_attribution_path=n3,
            sha256s_path=sha,
        )


def test_rejects_incorrect_campaign(sources) -> None:
    matrix, audit, n3 = copy.deepcopy(sources)
    matrix["campaign_id"] = "wrong"
    with pytest.raises(closure.ClosureError, match="matrix_campaign_invalid"):
        _build((matrix, audit, n3))


def test_rejects_incorrect_commit(sources) -> None:
    matrix, audit, n3 = copy.deepcopy(sources)
    audit["audit_commit"] = "wrong"
    with pytest.raises(closure.ClosureError, match="closure_fix_commit_invalid"):
        _build((matrix, audit, n3))


def test_rejects_incomplete_closure_integrity(sources) -> None:
    matrix, audit, n3 = copy.deepcopy(sources)
    audit["integrity"]["matched_closures"] = 3249
    with pytest.raises(closure.ClosureError, match="campaign_integrity_incomplete"):
        _build((matrix, audit, n3))


def test_rejects_n2_as_approved(model) -> None:
    changed = copy.deepcopy(model)
    changed["organ_results"]["N2"]["status"] = "PASSED"
    with pytest.raises(closure.ClosureError, match="n2_must_fail"):
        closure.validate_closure_model(changed)


def test_rejects_n4_as_approved(model) -> None:
    changed = copy.deepcopy(model)
    changed["organ_results"]["N4"]["status"] = "PASSED"
    with pytest.raises(closure.ClosureError, match="n4_must_fail"):
        closure.validate_closure_model(changed)


def test_rejects_n3_global_superiority(model) -> None:
    changed = copy.deepcopy(model)
    changed["organ_results"]["N3"]["global_superiority"] = "DEMONSTRATED"
    with pytest.raises(closure.ClosureError, match="n3_global_superiority_forbidden"):
        closure.validate_closure_model(changed)


def test_ranking_trained_reference_is_inconclusive(model) -> None:
    ranking = model["organ_results"]["N3"]["trained_vs_reference"]["ranking"]
    assert ranking["status"] == "INCONCLUSIVE"
    assert ranking["exact_sign_flip_p"] == 0.125


def test_brier_trained_reference_is_supported(model) -> None:
    brier = model["organ_results"]["N3"]["trained_vs_reference"]["brier"]
    assert brier["status"] == "SUPPORTED"
    assert brier["positive_seeds"] == 12


def test_mrr_trained_reference_is_not_supported(model) -> None:
    mrr = model["organ_results"]["N3"]["trained_vs_reference"]["mrr"]
    assert mrr == {"status": "NOT_SUPPORTED", "mean_delta": 0.0, "zero_seeds": 12}


def test_balanced_accuracy_trained_reference_is_refuted(model) -> None:
    balanced = model["organ_results"]["N3"]["trained_vs_reference"][
        "balanced_accuracy"
    ]
    assert balanced["status"] == "REFUTED"
    assert balanced["negative_seeds"] == 12


@pytest.mark.parametrize(
    "key", ["p2_authorized", "main_merge_authorized", "live_authority"]
)
def test_authority_decisions_are_false(model, key: str) -> None:
    assert model["decisions"][key] is False
    assert model["closure_gates"][key] is False


def test_preserves_three_source_files(tmp_path: Path) -> None:
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in (MATRIX, AUDIT_V2, N3)}
    closure.run(
        matrix_path=MATRIX,
        audit_v2_path=AUDIT_V2,
        n3_attribution_path=N3,
        sha256s_path=SHA256S,
        json_output=tmp_path / "closure.json",
        markdown_output=tmp_path / "closure.md",
    )
    after = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in before}
    assert after == before


def test_json_generation_is_deterministic(model) -> None:
    assert closure._encoded_json(model) == closure._encoded_json(copy.deepcopy(model))


def test_markdown_generation_is_deterministic(model) -> None:
    assert closure.render_markdown(model) == closure.render_markdown(copy.deepcopy(model))


def test_atomic_pair_writes_complete_files(tmp_path: Path) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    closure.atomic_write_pair({first: b"alpha", second: b"beta"})
    assert first.read_bytes() == b"alpha"
    assert second.read_bytes() == b"beta"
    assert not list(tmp_path.glob(".*"))


def test_rejects_nonfinite_fields(model) -> None:
    changed = copy.deepcopy(model)
    changed["integrity"]["certification_rate"] = float("nan")
    with pytest.raises(closure.ClosureError, match="nonfinite_value"):
        closure.validate_closure_model(changed)


def test_rejects_ambiguous_gate_as_final(model) -> None:
    changed = copy.deepcopy(model)
    changed["closure_gates"]["trained_vs_reference"] = True
    with pytest.raises(
        closure.ClosureError, match="ambiguous_trained_vs_reference_gate_forbidden"
    ):
        closure.validate_closure_model(changed)


def test_final_declaration_is_mandatory_and_last(model) -> None:
    markdown = closure.render_markdown(model)
    required = """P1 queda CLOSED como experimento SHADOW de atribución cognitiva.

N2: FAILED.
N3: SUPPORTED_LIMITED.
N4: FAILED.

La contribución aislada de N3 queda demostrada dentro de las métricas y
condiciones de P1. La superioridad global del backend trained frente al
reference no queda demostrada.

Ningún resultado concede autoridad operativa, staging, promoción, merge a
main ni autorización de P2.

La selección de un nuevo objetivo requiere una decisión humana explícita.
"""
    assert markdown.endswith(required)


def test_builder_has_no_cognitive_runtime_imports() -> None:
    tree = ast.parse((ROOT / "scripts/build_p1_closure.py").read_text())
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    assert not [name for name in imported if name.startswith(("runtime", "contracts"))]


def test_external_xpass_is_recorded_as_non_blocking(model) -> None:
    observation = model["qa_observations"]["external_xpass"]
    assert observation["node_id"].endswith("::test_no_undesired_memory_effects")
    assert observation["belongs_to_p1_scope"] is False
    assert observation["closure_limitation"] == "recorded_non_blocking_external_xpass"
    assert observation["closure_impact"] == "none"
