"""Coordinador persistente por runner y sesión efímera por episodio."""

from __future__ import annotations

from typing import Any, Mapping

from .causal_learning import CausalLearningEngine, TransitionEvidence
from .compiler import TransitionCompiler
from .contracts import CausalOverlay, Justification, MCICommitReport, TransitionSpec
from .jtms import TemporalAssumptionLedger
from .planner import SMTPlanner
from .regime_detector import RegimeDetector
from .self_model import IncrementalSelfModel


class MCIRuntime:
    def __init__(self, spec: TransitionSpec):
        self.base_spec = spec
        self.learner = CausalLearningEngine(spec)
        self.ledger = TemporalAssumptionLedger()
        self.self_model = IncrementalSelfModel()
        self.regime_detector = RegimeDetector()
        self._pending: dict[str, Any] | None = None

    def restore_overlay(self, payload: Mapping[str, Any]) -> None:
        self.learner.restore_overlay(
            CausalOverlay(
                overlay_id=str(payload["overlay_id"]),
                version=int(payload["version"]),
                base_spec_sha256=str(payload["base_spec_sha256"]),
                parameter_updates=dict(payload.get("parameter_updates") or {}),
                precondition_updates=dict(payload.get("precondition_updates") or {}),
                edge_updates=tuple(
                    tuple(item) for item in payload.get("edge_updates") or ()
                ),
                evidence=dict(payload.get("evidence") or {}),
                train_mae_before=float(payload["train_mae_before"]),
                train_mae_after=float(payload["train_mae_after"]),
                holdout_mae_before=float(payload["holdout_mae_before"]),
                holdout_mae_after=float(payload["holdout_mae_after"]),
                parent_overlay_id=payload.get("parent_overlay_id"),
            )
        )

    def propose(
        self,
        *,
        state: Mapping[str, Any],
        external_input: float,
        logical_time: int,
        regime: str,
    ) -> dict[str, Any]:
        compiler = TransitionCompiler(self.learner.active_spec)
        plan = SMTPlanner(compiler).plan(state, external_input=external_input)
        planned_action = plan.actions[0] if plan.actions else self.base_spec.safe_action
        self_report = self.self_model.predict(
            spec=self.learner.active_spec,
            regime=regime,
            action=planned_action,
            plan_length=len(plan.actions),
            risk=float(plan.risk if plan.risk is not None else 1.0),
        )
        action = self_report.safe_action if self_report.decision == "abstain" else planned_action
        model_belief = self.ledger.add_belief(
            belief_id=f"model/t/{logical_time}",
            proposition=f"model={self.learner.active_spec.sha256}",
            confidence=1.0,
            logical_time=logical_time,
        )
        prediction_belief = self.ledger.add_belief(
            belief_id=f"prediction/t/{logical_time}",
            proposition=f"action={action}",
            confidence=self_report.success_probability,
            logical_time=logical_time,
        )
        justification = Justification(
            justification_id=f"justification/t/{logical_time}",
            antecedent_ids=(model_belief.belief_id,),
            consequence_id=prediction_belief.belief_id,
            kind="model_prediction",
        )
        self.ledger.justify(justification)
        report = MCICommitReport(
            schema="mci_commit.v1",
            spec_id=self.base_spec.spec_id,
            spec_sha256=self.base_spec.sha256,
            overlay_id=(
                self.learner.active_overlay.overlay_id
                if self.learner.active_overlay is not None
                else None
            ),
            plan=plan,
            causal_graph=compiler.causal_graph(),
            beliefs=self.ledger.beliefs,
            belief_revision={"out": [], "revisable": []},
            self_model=self_report,
            constraint_ids=plan.constraint_ids,
        )
        predicted = compiler.execute(state, action=action, external_input=external_input)
        self._pending = {
            "state": dict(state),
            "action": action,
            "external_input": float(external_input),
            "predicted": predicted,
            "belief_id": prediction_belief.belief_id,
            "signature": self_report.signature,
            "plan_length": len(plan.actions),
            "report": report,
        }
        return {
            "mci_active": True,
            "mci_first_action": action,
            "mci_plan_report": plan.to_dict(),
            "mci_self_model": self_report.to_dict(),
            "mci_commit_report": report.to_dict(),
        }

    def observe_outcome(
        self,
        observed: Mapping[str, Any],
        *,
        committed_action: str,
        logical_time: int,
        reasoning_cost: float,
    ) -> dict[str, Any]:
        if self._pending is None:
            return {"status": "no_pending_decision"}
        pending = self._pending
        variable = self.base_spec.main_variable
        committed = str(committed_action)
        compiler = TransitionCompiler(self.learner.active_spec)
        predicted = compiler.execute(
            pending["state"],
            action=committed,
            external_input=pending["external_input"],
        )
        error = float(observed[variable]) - float(predicted[variable])
        success = not bool(observed[self.base_spec.alarm_variable])
        recommendation_committed = committed == pending["action"]
        if recommendation_committed:
            self.self_model.observe(
                pending["signature"], success=success, cost=float(reasoning_cost)
            )
        revision = {"out": (), "revisable": ()}
        if abs(error) > 1e-6:
            revision = self.ledger.contradict(pending["belief_id"])
        regime_change = None
        if self.regime_detector.check(error, logical_time=logical_time):
            stale = self.ledger.mark_empirically_stale(
                before_logical_time=logical_time
            )
            self.learner.segment_regime(logical_time=logical_time)
            regime_change = {
                **self.regime_detector.last_event.to_dict(),
                "stale_belief_ids": stale,
            }
        overlay = self.learner.observe(
            TransitionEvidence(
                state=pending["state"],
                action=committed,
                external_input=pending["external_input"],
                observed=dict(observed),
                predicted=predicted,
            )
        )
        self._pending = None
        return {
            "status": "observed",
            "prediction_error": round(error, 9),
            "success": success,
            "recommendation_committed": recommendation_committed,
            "belief_revision": revision,
            "promoted_overlay": overlay.to_dict() if overlay is not None else None,
            "causal_event": self.learner.last_event,
            "regime_change": regime_change,
            "logical_time": logical_time,
        }
