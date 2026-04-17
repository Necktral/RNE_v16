"""Gate formal de propuesta -> certificación -> promoción/rechazo."""

from __future__ import annotations

from typing import Any, Dict
from uuid import uuid4

from runtime.memory.mfm_lite import EpisodeMemoryStore, MFMCondenser, MacroPromotion
from runtime.reality.evaluator import evaluate_episode_closure
from runtime.storage import StorageFacade

from .certificate_builder import CertificateBuilder
from .continuity_guard import ContinuityGuard
from .ioc_proxy import IoCProxy


class PromotionGate:
    def __init__(self, *, storage: StorageFacade):
        self.storage = storage
        self.builder = CertificateBuilder(storage=storage)
        self.continuity_guard = ContinuityGuard()
        self.ioc_proxy = IoCProxy()
        self.memory_store = EpisodeMemoryStore(storage=storage)
        self.condenser = MFMCondenser()
        self.macro_promotion = MacroPromotion(storage=storage)

    def process_episode(
        self,
        *,
        run_id: str,
        episode_result: Dict[str, Any],
        reality_assessment=None,
    ) -> Dict[str, Any]:
        episode = episode_result.get("episode", {})

        # Extract scenario_metadata with fallback
        scenario_metadata = episode.get("scenario_metadata")
        if scenario_metadata is None:
            scenario_name = episode.get("scenario", "unknown")
            scenario_metadata = {"scenario_name": scenario_name}

        closure = evaluate_episode_closure(storage=self.storage, run_id=run_id, result=episode_result)
        previous = self.storage.list_episode_certificates(run_id=run_id, limit=1)
        previous_certificate = previous[0] if previous else None
        fallback_continuity = (
            float(reality_assessment.continuity_score) if reality_assessment is not None else None
        )
        continuity = self.continuity_guard.score(
            previous_certificate=previous_certificate,
            current_episode=episode,
            fallback_continuity=fallback_continuity,
        )
        continuity_alert = self.continuity_guard.has_alert(continuity)
        collapse_detected = bool(
            getattr(reality_assessment, "collapse_detected", False)
        )
        uncertainty = float(episode.get("context", {}).get("uncertainty", 0.2))
        ioc_value = self.ioc_proxy.compute(
            continuity_score=continuity,
            closure_passed=closure["closure_passed"],
            trace_integrity=closure["trace_integrity"],
            collapse_detected=collapse_detected,
            uncertainty=uncertainty,
        )

        proposal = {
            "proposal_id": f"proposal-{uuid4()}",
            "origin": "promotion_gate",
            "change": {"episode_id": episode.get("episode_id"), "run_id": run_id},
            "risk": "low" if ioc_value >= 0.72 else "medium",
            "metadata": {
                "ioc_proxy": ioc_value,
                "continuity_score": continuity,
                "scenario_metadata": scenario_metadata,
            },
        }
        self.storage.append_event(
            event_type="proposal.evaluated",
            run_id=run_id,
            source="promotion_gate",
            payload=proposal,
        )

        certificate = self.builder.build_and_persist(
            run_id=run_id,
            episode_result=episode_result,
            continuity_score=continuity,
            ioc_proxy=ioc_value,
            continuity_alert=continuity_alert,
            closure_passed=closure["closure_passed"],
            trace_integrity=closure["trace_integrity"],
            collapse_detected=collapse_detected,
        )
        decision_verdict = "promote" if certificate.promotion_candidate else "reject"
        decision_reason = (
            "certificate_gate_passed"
            if certificate.promotion_candidate
            else "certificate_gate_failed"
        )
        decision = self.storage.write_promotion_decision(
            episode_id=certificate.episode_id,
            run_id=run_id,
            certificate_id=certificate.certificate_id,
            verdict=decision_verdict,
            reason=decision_reason,
            rollback_ready=certificate.rollback_ready,
            metadata={
                "ioc_proxy": certificate.ioc_proxy,
                "risk_score": certificate.risk_score,
                "scenario_metadata": scenario_metadata,
            },
        )

        memory_updates: Dict[str, Any] = {"micro": None, "meso": None, "macro": None}
        memory_extra_metadata = {"scenario_metadata": scenario_metadata}
        if certificate.verdict == "certified":
            micro = self.memory_store.write_micro(
                run_id=run_id,
                episode_id=certificate.episode_id,
                certificate_id=certificate.certificate_id,
                ioc_proxy=certificate.ioc_proxy,
                structure=self.condenser.micro(
                    episode_result=episode_result,
                    certificate=certificate,
                ),
                extra_metadata=memory_extra_metadata,
            )
            meso_structure = self.condenser.meso(
                episode_result=episode_result,
                certificate=certificate,
            )
            meso = self.memory_store.write_meso(
                run_id=run_id,
                episode_id=certificate.episode_id,
                certificate_id=certificate.certificate_id,
                ioc_proxy=certificate.ioc_proxy,
                structure=meso_structure,
                extra_metadata=memory_extra_metadata,
            )
            memory_updates["micro"] = micro
            memory_updates["meso"] = meso
            macro_eval = self.macro_promotion.should_promote(
                run_id=run_id,
                pattern_key=meso_structure["pattern_key"],
                continuity_alert=continuity_alert,
            )
            if decision_verdict == "promote" and macro_eval.get("promote"):
                macro = self.memory_store.write_macro(
                    run_id=run_id,
                    episode_id=certificate.episode_id,
                    certificate_id=certificate.certificate_id,
                    ioc_proxy=certificate.ioc_proxy,
                    support_count=int(macro_eval["support"]),
                    structure={
                        "pattern_key": meso_structure["pattern_key"],
                        "support_count": macro_eval["support"],
                        "mean_ioc": macro_eval["mean_ioc"],
                        "std_ioc": macro_eval["std_ioc"],
                    },
                    extra_metadata=memory_extra_metadata,
                )
                memory_updates["macro"] = macro

        self.storage.append_event(
            event_type="promotion.decision",
            run_id=run_id,
            source="promotion_gate",
            payload={
                "episode_id": certificate.episode_id,
                "certificate_id": certificate.certificate_id,
                "decision_id": decision.decision_id,
                "verdict": decision.verdict,
                "rollback_ready": decision.rollback_ready,
            },
        )
        return {
            "proposal": proposal,
            "certificate": certificate,
            "decision": decision,
            "memory_updates": memory_updates,
        }
