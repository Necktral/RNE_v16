"""Tests del router GPU-aware y del tier ejecutable (Bloque B)."""

from __future__ import annotations

from runtime.conjunction.router import ComputeRouter
from runtime.conjunction.contracts import OperationContext, OperationalConstraints
from runtime.conjunction.execution import tier_execution_directives, routing_enforced


def _ctx(**kw) -> OperationContext:
    # task_type no-causal + incertidumbre alta + externo permitido => rama tier_3.
    base = dict(
        run_id="r",
        user_intent="test",
        task_type="life_cycle",
        requested_action="act",
        constraints=OperationalConstraints(max_compute_tier="tier_3_external", allow_external=True),
        uncertainty_score=0.9,
        complexity_score=0.9,
    )
    base.update(kw)
    return OperationContext.create(**base)


def test_router_byte_identical_without_gpu():
    """Sin señal de GPU, la ruta no marca gpu_backed y no degrada por VRAM."""
    route = ComputeRouter().route(_ctx())
    assert route.gpu_backed is False
    # tier_3 alcanzable (uncertainty alta + allow_external + techo tier_3)
    assert route.selected_compute_tier == "tier_3_external"


def test_router_marks_gpu_backed_when_gpu_present_and_idle():
    route = ComputeRouter().route(_ctx(gpu_available=True, vram_pressure=0.1, vram_headroom=0.9))
    assert route.selected_compute_tier == "tier_3_external"
    assert route.gpu_backed is True


def test_router_degrades_under_vram_saturation():
    """GPU presente pero VRAM saturada -> los tiers GPU (2/3) quedan fuera."""
    from runtime.conjunction.contracts import CausalAssumption

    # Contexto que normalmente rutea a tier_2 (causal soportado).
    causal = OperationContext.create(
        run_id="r",
        user_intent="test",
        task_type="causal_decision",
        requested_action="act",
        constraints=OperationalConstraints(max_compute_tier="tier_3_external", allow_external=True),
        causal_assumptions=[CausalAssumption(name="x->y", statement="x causes y", supported=True)],
        gpu_available=True,
        vram_pressure=0.95,
        vram_headroom=0.05,
    )
    route = ComputeRouter().route(causal)
    # Cae al techo efectivo tier_1 (no hay VRAM para el workload GPU).
    assert route.selected_compute_tier == "tier_1_local_light"
    assert route.gpu_backed is False
    assert "vram_saturated" in route.reason

    # Sin saturación, el mismo contexto sí alcanza tier_2 (con GPU).
    healthy = OperationContext.create(
        run_id="r",
        user_intent="test",
        task_type="causal_decision",
        requested_action="act",
        constraints=OperationalConstraints(max_compute_tier="tier_3_external", allow_external=True),
        causal_assumptions=[CausalAssumption(name="x->y", statement="x causes y", supported=True)],
        gpu_available=True,
        vram_pressure=0.1,
    )
    hroute = ComputeRouter().route(healthy)
    assert hroute.selected_compute_tier == "tier_2_specialized"
    assert hroute.gpu_backed is True


def test_idle_gpu_does_not_change_tier_selection():
    """Una GPU ociosa no altera la selección respecto a no tener GPU."""
    no_gpu = ComputeRouter().route(_ctx())
    idle_gpu = ComputeRouter().route(_ctx(gpu_available=True, vram_pressure=0.05))
    assert no_gpu.selected_compute_tier == idle_gpu.selected_compute_tier


def test_routing_enforced_off_by_default(monkeypatch):
    monkeypatch.delenv("RNFE_CONJUNCTION_ROUTING_ENFORCED", raising=False)
    assert routing_enforced() is False
    monkeypatch.setenv("RNFE_CONJUNCTION_ROUTING_ENFORCED", "1")
    assert routing_enforced() is True


def test_tier_execution_directives_mapping():
    t0 = tier_execution_directives("tier_0_deterministic")
    assert t0.closure_profile == "baseline_fixed"
    assert t0.memory_retrieval_limit == 1
    assert t0.external_reasoner_enabled is False

    t3 = tier_execution_directives("tier_3_external", gpu_backed=True)
    assert t3.closure_profile == "adaptive_min"
    assert t3.memory_retrieval_limit == 5
    assert t3.external_reasoner_enabled is True
    assert t3.gpu_backed is True

    # Tier desconocido cae a tier_1.
    unknown = tier_execution_directives("tier_99")
    assert unknown.memory_retrieval_limit == 3


def test_resource_pressure_gate_precedes_gpu():
    """La presión de recursos del host sigue teniendo prioridad (tier_0)."""
    route = ComputeRouter().route(
        _ctx(gpu_available=True, vram_pressure=0.1, resource_pressure=0.95)
    )
    assert route.selected_compute_tier == "tier_0_deterministic"
