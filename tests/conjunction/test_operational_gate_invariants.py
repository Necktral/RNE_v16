"""Invariantes del gate operacional del LifeKernel (P-SEG: B48 + B39).

B48 — invariante total de bloqueo: ninguna acción bajo veredicto "block" llega
al runner por fall-through de nombre de acción (hueco H5). Toda acción
ejecutante bloqueada se transforma o se detiene.

B39 — identidad causal: las ramas transformadoras del gate preservan
``decision_id``/``created_at`` de la decisión original (misma decisión, otra
forma; no una decisión nueva sin linaje).

Incluye la separación validation_tier/execution_tier: una acción crítica puede
VALIDAR en un tier y solo EJECUTA (su reemplazo autorizado) en otro — o en
ninguno, si el ciclo se detiene.
"""

from pathlib import Path

from runtime.conjunction.contracts import OperationalConjunctionResult
from runtime.life import LifeKernel, LifeKernelConfig
from runtime.life.contracts import AutonomyDecision
from runtime.life.kernel import NON_EPISODE_ACTIONS
from runtime.storage import StorageConfig, StorageFactory


def _storage(tmp_path: Path):
    return StorageFactory.create_facade(
        StorageConfig(
            mode="sqlite",
            sqlite_db_path=str(tmp_path / "gate-invariants.db"),
            postgres_dsn=None,
            artifact_root=tmp_path / "artifacts",
            prefer_postgres_reads=False,
            strict_dual_write=False,
        )
    )


def _kernel(tmp_path: Path, run_id: str) -> LifeKernel:
    return LifeKernel(
        config=LifeKernelConfig(
            run_id=run_id,
            scenarios=("thermal_homeostasis",),
            restore=False,
            enable_msrc=False,
        ),
        storage=_storage(tmp_path),
    )


def _operational(
    final_decision: str = "block",
    *,
    tier: str = "tier_2_specialized",
    context_summary: dict | None = None,
) -> OperationalConjunctionResult:
    blocked = final_decision == "block"
    return OperationalConjunctionResult(
        operation_id="op-gate-invariants",
        selected_compute_tier=tier,  # type: ignore[arg-type]
        selected_reasoning_path="test_path",
        validation_status="fail" if blocked else "pass",
        compensation_status="none",
        confidence_state="conflicted" if blocked else "high",
        execution_permissions={"automatic_execution_allowed": not blocked},
        final_decision=final_decision,  # type: ignore[arg-type]
        validation_findings=(),
        compensations=(),
        trace=(),
        context_summary=dict(context_summary or {}),
    )


def _decision(action: str, **overrides) -> AutonomyDecision:
    payload = {
        "action": action,
        "mode": "normal",
        "reason": "test_decision",
        "priority": 0.7,
        "scenario": "thermal_homeostasis",
        "external_input": 0.05,
        "directives": {"origin": "test"},
    }
    payload.update(overrides)
    return AutonomyDecision(**payload)


# ---------------------------------------------------------------------------
# B48 — invariante total de bloqueo (hueco H5)
# ---------------------------------------------------------------------------


def test_block_halts_non_critical_action_no_fall_through(tmp_path: Path):
    """Acción NO crítica bajo "block" -> el ciclo se DETIENE (no fall-through H5)."""
    kernel = _kernel(tmp_path, "gate-b48-act")
    for action in ("act", "observe", "explore"):
        original = _decision(action)
        gated = kernel._apply_operational_gate(
            decision=original,
            operational=_operational("block"),
        )
        # La decisión resultante NO llega al runner: es una acción no-ejecutante.
        assert gated.action in NON_EPISODE_ACTIONS, action
        assert gated.action == "sleep"
        assert gated.mode == "conservative"
        assert gated.directives["blocked_action"] == action
        # Detenido => no hay tier de ejecución; la validación queda registrada.
        assert gated.directives["execution_tier"] is None
        assert gated.directives["validation_tier"] == "tier_2_specialized"
        # B39: la transformación preserva la identidad causal.
        assert gated.decision_id == original.decision_id
        assert gated.created_at == original.created_at


def test_block_non_episode_action_does_not_execute_episode(tmp_path: Path):
    """Acción no-ejecutante bajo "block" queda anotada y sigue sin ejecutar episodio."""
    kernel = _kernel(tmp_path, "gate-b48-nonepisode")
    original = _decision("quarantine", mode="quarantine")
    gated = kernel._apply_operational_gate(
        decision=original,
        operational=_operational("block"),
    )
    assert gated.action == "quarantine"
    assert gated.action in NON_EPISODE_ACTIONS
    assert gated.directives["execution_tier"] is None
    assert gated.decision_id == original.decision_id
    assert gated.created_at == original.created_at


def test_block_rollback_with_evidence_stays_non_episode(tmp_path: Path):
    """Rollback CON checkpoint sano bajo "block": sigue siendo rollback (no runner)."""
    kernel = _kernel(tmp_path, "gate-b48-rollback-evidence")
    original = _decision("rollback", mode="rollback")
    gated = kernel._apply_operational_gate(
        decision=original,
        operational=_operational(
            "block",
            context_summary={"available_evidence_kinds": ["healthy_checkpoint"]},
        ),
    )
    assert gated.action == "rollback"
    assert gated.action in NON_EPISODE_ACTIONS
    assert gated.directives["execution_tier"] is None
    assert gated.decision_id == original.decision_id
    assert gated.created_at == original.created_at


def test_life_kernel_step_halts_episode_under_generic_block(tmp_path: Path):
    """Integración H5: un "block" genérico sobre "act" detiene el episodio en step()."""

    class _ForcedBlockConjunction:
        def __init__(self, result: OperationalConjunctionResult):
            self.result = result
            self.seen_decisions: list[AutonomyDecision] = []

        def evaluate_life_cycle(self, **kwargs) -> OperationalConjunctionResult:
            self.seen_decisions.append(kwargs["decision"])
            return self.result

    kernel = _kernel(tmp_path, "gate-b48-step-halt")
    stub = _ForcedBlockConjunction(_operational("block"))
    kernel.conjunction = stub  # type: ignore[assignment]

    result = kernel.step(external_input=0.05)

    # El supervisor pidió una acción ejecutante NO crítica...
    assert stub.seen_decisions, "la conjunción debió evaluarse"
    supervisor_decision = stub.seen_decisions[0]
    assert supervisor_decision.action == "act"
    # ...y bajo "block" el ciclo se DETIENE: no hay episodio (antes: fall-through).
    assert result.episode_result is None
    assert result.decision.action == "sleep"
    assert result.decision.directives["blocked_action"] == "act"
    assert result.operational["final_decision"] == "block"
    # B39 end-to-end: la decisión detenida ES la decisión del supervisor.
    assert result.decision.decision_id == supervisor_decision.decision_id
    assert result.decision.created_at == supervisor_decision.created_at
    events = kernel.storage.list_events(
        run_id="gate-b48-step-halt",
        event_types=["life.sleep"],
        limit=5,
    )
    assert events, "el alto por bloqueo debe quedar persistido como ciclo no-actuante"


# ---------------------------------------------------------------------------
# B39 — identidad causal en las tres ramas transformadoras
# ---------------------------------------------------------------------------


def test_blocked_self_modify_preserves_causal_identity(tmp_path: Path):
    kernel = _kernel(tmp_path, "gate-b39-selfmod")
    original = _decision("self_modify")
    gated = kernel._apply_operational_gate(
        decision=original,
        operational=_operational("block"),
    )
    assert gated.action == "act"
    assert gated.mode == "recovery"
    assert gated.directives["blocked_action"] == "self_modify"
    assert gated.decision_id == original.decision_id
    assert gated.created_at == original.created_at
    # Separación de tiers: validó en tier_2, su reemplazo seguro ejecuta en tier_0.
    assert gated.directives["validation_tier"] == "tier_2_specialized"
    assert gated.directives["execution_tier"] == "tier_0_deterministic"


def test_blocked_consult_external_preserves_causal_identity(tmp_path: Path):
    kernel = _kernel(tmp_path, "gate-b39-consult")
    original = _decision("consult_external")
    gated = kernel._apply_operational_gate(
        decision=original,
        operational=_operational("block", tier="tier_3_external"),
    )
    assert gated.action == "act"
    assert gated.mode == "conservative"
    assert gated.directives["blocked_action"] == "consult_external"
    assert gated.decision_id == original.decision_id
    assert gated.created_at == original.created_at
    assert gated.directives["validation_tier"] == "tier_3_external"
    assert gated.directives["execution_tier"] == "tier_0_deterministic"


def test_blocked_rollback_without_evidence_preserves_causal_identity(tmp_path: Path):
    kernel = _kernel(tmp_path, "gate-b39-rollback")
    original = _decision("rollback", mode="rollback")
    gated = kernel._apply_operational_gate(
        decision=original,
        operational=_operational("block", context_summary={"available_evidence_kinds": []}),
    )
    assert gated.action == "quarantine"
    assert gated.directives["blocked_action"] == "rollback"
    assert gated.decision_id == original.decision_id
    assert gated.created_at == original.created_at
    assert gated.directives["execution_tier"] is None


def test_degrade_preserves_causal_identity(tmp_path: Path):
    kernel = _kernel(tmp_path, "gate-b39-degrade")
    original = _decision("act")
    gated = kernel._apply_operational_gate(
        decision=original,
        operational=_operational("degrade"),
    )
    assert gated.action == "act"
    assert gated.mode == "conservative"
    assert gated.decision_id == original.decision_id
    assert gated.created_at == original.created_at
    # Degradado pero autorizado: ejecuta en el mismo tier en el que validó.
    assert gated.directives["validation_tier"] == "tier_2_specialized"
    assert gated.directives["execution_tier"] == "tier_2_specialized"


def test_allow_pass_through_keeps_identity_and_annotates_tiers(tmp_path: Path):
    kernel = _kernel(tmp_path, "gate-b39-allow")
    original = _decision("act")
    gated = kernel._apply_operational_gate(
        decision=original,
        operational=_operational("allow"),
    )
    assert gated.action == "act"
    assert gated.decision_id == original.decision_id
    assert gated.created_at == original.created_at
    assert gated.directives["validation_tier"] == "tier_2_specialized"
    assert gated.directives["execution_tier"] == "tier_2_specialized"
