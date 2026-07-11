"""B41 — CausalContext.v1: sobre de correlación de auditoría (P-CADENA-CAUSAL).

Contrato frozen versionado, aditivo y gated. Tests del round-trip y de la acuñación
por step (trace_group_id como hilo de la cadena decisión→episodio→traza).
"""

from __future__ import annotations

from runtime.organism.identity import (
    CausalContext,
    causal_context_enabled,
    mint_trace_group_id,
)


def test_causal_context_round_trip():
    """to_dict/from_dict preserva todos los campos del sobre (contrato v1)."""
    ctx = CausalContext(
        organism_id="org-abc",
        lineage_id="lin-xyz",
        run_id="life-123",
        trace_group_id="tg-org-abc-4-deadbeef",
        parent_trace_group_id="tg-org-abc-3-cafef00d",
        decision_id="dec-000111222333",
        step_index=4,
    )
    payload = ctx.to_dict()
    assert payload["schema_version"] == "causal_context.v1"

    restored = CausalContext.from_dict(payload)
    assert restored == ctx
    # Doble round-trip estable.
    assert CausalContext.from_dict(restored.to_dict()) == ctx


def test_causal_context_from_empty_payload_defaults_v1():
    """Un payload viejo/ausente se lee con defaults (campo opcional, compat)."""
    ctx = CausalContext.from_dict(None)
    assert ctx.schema_version == "causal_context.v1"
    assert ctx.organism_id == ""
    assert ctx.step_index == -1
    assert ctx.parent_trace_group_id is None


def test_for_step_mints_trace_group_id_bound_to_organism_and_step():
    """for_step acuña un trace_group_id fresco por episodio, atado al genoma y al step."""
    ctx = CausalContext.for_step(
        organism_id="org-abc",
        lineage_id="lin-xyz",
        run_id="life-123",
        step_index=7,
        decision_id="dec-abc",
    )
    assert ctx.organism_id == "org-abc"
    assert ctx.lineage_id == "lin-xyz"
    assert ctx.run_id == "life-123"
    assert ctx.decision_id == "dec-abc"
    assert ctx.step_index == 7
    assert ctx.trace_group_id.startswith("tg-org-abc-7-")

    # Dos steps del mismo organismo NO comparten trace_group_id (hilos distintos).
    other = CausalContext.for_step(
        organism_id="org-abc", lineage_id="lin-xyz", run_id="life-123", step_index=8,
    )
    assert other.trace_group_id != ctx.trace_group_id


def test_with_decision_sets_decision_id_immutably():
    """with_decision copia el sobre fijando decision_id (frozen: no muta el original)."""
    base = CausalContext.for_step(
        organism_id="org-abc", lineage_id="lin-xyz", run_id="life-123", step_index=1,
        trace_group_id="tg-fixed",
    )
    stamped = base.with_decision("dec-999")
    assert stamped.decision_id == "dec-999"
    assert stamped.trace_group_id == base.trace_group_id
    assert base.decision_id is None  # inmutable


def test_mint_trace_group_id_shape():
    tg = mint_trace_group_id(organism_id="org-abc", step_index=3)
    assert tg.startswith("tg-org-abc-3-")


def test_flag_off_by_default(monkeypatch):
    monkeypatch.delenv("RNFE_CAUSAL_CONTEXT", raising=False)
    assert causal_context_enabled() is False
    monkeypatch.setenv("RNFE_CAUSAL_CONTEXT", "1")
    assert causal_context_enabled() is True
