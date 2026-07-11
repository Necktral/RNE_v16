from __future__ import annotations

from runtime.neural.organs import (
    DeterministicChunker,
    KANSpline,
    LTCCell,
    StructuralEvolutionGate,
    StructuralMutationProposal,
    UnstructuredIngestionService,
)


def test_n5_chunker_is_unicode_safe_and_calls_real_ingestion_ports() -> None:
    chunker = DeterministicChunker(max_bytes=32)
    text = "Órgano uno.\n\n```python\nprint('simbiosis')\n```\nFin."
    chunks = chunker.chunk(text)
    assert "".join(item.text for item in chunks) == text
    assert all(len(item.text.encode("utf-8")) <= 32 for item in chunks)

    signs = []
    memories = []
    service = UnstructuredIngestionService(
        sign_sink=lambda item: signs.append(dict(item)) or f"sign-{len(signs)}",
        memory_candidate_sink=lambda item: memories.append(dict(item)) or f"memory-{len(memories)}",
        fallback_chunker=chunker,
    )
    result = service.ingest(text, run_id="run-5", source_id="source-5")
    assert len(result["chunks"]) == len(signs) == len(memories)
    assert all(item["status"] == "candidate" for item in signs)
    assert all(item["promotion"] == "requires_existing_mfm_gate" for item in memories)


def test_n6_kan_exports_equivalent_sympy_expression() -> None:
    spline = KANSpline(knots=(0.0, 1.0, 2.0), coefficients=(0.0, 2.0, 2.0))
    expression = spline.to_sympy("x")
    assert spline.evaluate(0.5) == 1.0
    assert float(expression.subs({"x": 0.5})) == 1.0
    assert spline.evaluate(3.0) == 2.0


def test_n6_ltc_is_bounded_and_structural_gate_requires_real_apply_fn() -> None:
    cell = LTCCell(
        input_weights=((1.0,), (0.5,)),
        recurrent_weights=((0.1, 0.0), (0.0, 0.1)),
        bias=(0.0, 0.0),
        tau=(1.0, 2.0),
    )
    state = cell.step((0.0, 0.0), (0.8,), dt=0.1)
    assert all(-1.0 <= value <= 1.0 for value in state)

    proposal = StructuralMutationProposal(
        mutation_type="parameter_bound",
        target="n1.max_optional_families",
        value=2,
        expected_gain=0.1,
        rollback_token="rollback-1",
    )
    gate = StructuralEvolutionGate()
    blocked = gate.evaluate_and_apply(
        proposal,
        sandbox=lambda item: {"safe": True},
        certify=lambda report: bool(report["safe"]),
        apply_fn=None,
        rollback=None,
    )
    assert blocked == {"applied": False, "reason": "apply_fn_and_rollback_required_p29"}

    applied = []
    result = gate.evaluate_and_apply(
        proposal,
        sandbox=lambda item: {"safe": True, "closure_delta": 0.01},
        certify=lambda report: bool(report["safe"]),
        apply_fn=lambda item: applied.append(item) or "applied",
        rollback=lambda token: None,
    )
    assert result["applied"] is True
    assert applied == [proposal]
