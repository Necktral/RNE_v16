"""Corte constitucional T5: trayectoria + flujo + renormalizacion + riesgo."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict
from uuid import uuid4

from runtime.storage import StorageFacade

from .constitution_flow import ConstitutionalFlowEngine
from .regime_model import get_regime_for_scenario
from .regime_renormalization import RegimeRenormalizationEngine, RenormalizationResult
from .risk_process import ConstitutionalRiskProcess, RiskState
from .t5_mode import get_t5_mode
from .trajectory import OrganismTrajectory
from .trajectory_state_machine import TrajectoryStateMachine
from .viability_kernel import TrajectoryViabilityKernel


@dataclass(frozen=True)
class CourtEpisodeResult:
    trajectory_id: str
    canonical_scope: str
    legacy_scope: str
    transfer_advice: str
    flow_validity: bool
    erosion: float
    phase_drift: float
    rollback_obligation: bool
    viability_score: float
    organism_risk: float
    edge_risk: float
    modification_risk: float
    inheritance_risk: float
    failure_mode_count: int
    renormalization_residual: float
    renormalization_uncertainty: float
    expected_recovery_cost: float


class ConstitutionalCourtRuntime:
    """Orquesta el dictamen constitucional primario del organismo."""

    def __init__(self, *, storage: StorageFacade):
        self.storage = storage
        self.flow_engine = ConstitutionalFlowEngine()
        self.renorm_engine = RegimeRenormalizationEngine()
        self.risk_process = ConstitutionalRiskProcess()
        self.viability_kernel = TrajectoryViabilityKernel()
        self.state_machine = TrajectoryStateMachine()
        self._trajectory_by_run: Dict[str, OrganismTrajectory] = {}
        self._last_regime_by_run: Dict[str, str] = {}

    def _legacy_scope_from_canonical(self, canonical_scope: str) -> str:
        mapping = {
            "local_safe": "local_only",
            "transfer_safe": "compatible_transfer",
            "modification_safe": "compatible_transfer",
            "inheritance_safe": "compatible_transfer",
            "quarantine_only": "analogical_hint_only",
            "blocked": "blocked",
        }
        return mapping.get(canonical_scope, "blocked")

    def _scope_from_risk(
        self,
        *,
        cross_regime: bool,
        modification_pending: bool,
        flow_valid: bool,
        rollback: bool,
        viability_score: float,
        organism_risk: float,
        edge_risk: float,
        modification_risk: float,
        inheritance_risk: float,
    ) -> tuple[str, str]:
        aggregate = max(organism_risk, edge_risk, modification_risk, inheritance_risk)
        if rollback or viability_score < 0.20 or aggregate >= 0.85:
            return "blocked", ""
        if not flow_valid or viability_score < 0.35 or aggregate >= 0.65:
            return "quarantine_only", "analogical_hint"

        if modification_pending:
            if modification_risk < 0.45 and organism_risk < 0.55:
                return "modification_safe", ""
            return "quarantine_only", "analogical_hint"

        if cross_regime:
            if inheritance_risk < 0.30 and edge_risk < 0.35 and organism_risk < 0.40:
                return "inheritance_safe", ""
            if edge_risk < 0.50 and organism_risk < 0.55:
                return "transfer_safe", ""
            return "quarantine_only", "analogical_hint"

        if organism_risk < 0.55:
            return "local_safe", ""
        return "quarantine_only", "analogical_hint"

    def _seed_risk_from_storage(
        self,
        *,
        run_id: str,
        trajectory_id: str,
        scope_type: str,
        scope_key: str,
    ) -> None:
        if self.risk_process.get(scope_type=scope_type, scope_key=scope_key) is not None:
            return
        previous = self.storage.list_constitutional_risk_states(
            run_id=run_id,
            trajectory_id=trajectory_id,
            scope_type=scope_type,
            scope_key=scope_key,
            limit=1,
        )
        if previous:
            self.risk_process.seed_score(
                scope_type=scope_type,  # type: ignore[arg-type]
                scope_key=scope_key,
                risk_score=previous[0].risk_score,
            )

    def _persist_risk_scope(
        self,
        *,
        run_id: str,
        trajectory_id: str,
        window_end_episode: int,
        scope_type: str,
        scope_key: str,
        update_payload: Dict[str, Any],
        state: RiskState,
        mode: str,
    ) -> None:
        previous = self.storage.list_constitutional_risk_states(
            run_id=run_id,
            trajectory_id=trajectory_id,
            scope_type=scope_type,
            scope_key=scope_key,
            limit=1,
        )
        prev_state_id = previous[0].state_id if previous else None
        step_index = (previous[0].step_index + 1) if previous else 0
        state_id = f"risk-{trajectory_id}-{scope_type}-{window_end_episode}-{step_index}"

        failure_events = [
            {
                "name": f.name,
                "severity": f.severity,
                "reversible": f.reversible,
                "recovery_protocol": f.recovery_protocol,
                "trigger": f.signature.trigger,
                "metrics": f.signature.metrics,
            }
            for f in state.failure_atlas.events
        ]

        self.storage.write_constitutional_risk_state(
            state_id=state_id,
            run_id=run_id,
            trajectory_id=trajectory_id,
            scope_type=scope_type,
            scope_key=scope_key,
            risk_score=state.risk_score,
            prev_state_id=prev_state_id,
            step_index=step_index,
            risk_json={
                "update": update_payload,
                "state": {
                    "scope_type": state.scope_type,
                    "scope_key": state.scope_key,
                    "risk_score": state.risk_score,
                    "drift_identity": state.drift_identity,
                    "drift_policy": state.drift_policy,
                    "delta_viability": state.delta_viability,
                    "delta_purity": state.delta_purity,
                    "delta_modification": state.delta_modification,
                    "failure_modes": [f["name"] for f in failure_events],
                    "failure_events": failure_events,
                },
                "chain": {
                    "prev_state_id": prev_state_id,
                    "step_index": step_index,
                },
            },
            metadata={"mode": mode},
        )
        for index, failure in enumerate(failure_events):
            self.storage.write_failure_atlas_event(
                event_id=f"failure-{trajectory_id}-{scope_type}-{window_end_episode}-{step_index}-{index}",
                run_id=run_id,
                trajectory_id=trajectory_id,
                scope_type=scope_type,
                scope_key=scope_key,
                failure_class=str(failure["name"]),
                severity=str(failure["severity"]),
                reversible=bool(failure["reversible"]),
                recovery_protocol=str(failure["recovery_protocol"]),
                signature_json={
                    "trigger": failure["trigger"],
                    "metrics": failure["metrics"],
                },
                metadata={"mode": mode, "step_index": step_index},
            )

    def ingest_episode(self, *, run_id: str, episode_result: Dict[str, Any]) -> CourtEpisodeResult | None:
        mode = get_t5_mode()
        if mode == "off":
            return None

        episode = episode_result.get("episode", {})
        episode_id = str(episode.get("episode_id", f"episode-{uuid4().hex[:8]}"))
        scenario_metadata = episode.get("scenario_metadata", {})
        scenario_name = str(scenario_metadata.get("scenario_name", episode.get("scenario", "unknown")))
        timestamp = str(episode.get("timestamp", ""))

        trajectory = self._trajectory_by_run.get(run_id)
        if trajectory is None:
            trajectory = OrganismTrajectory(trajectory_id=f"traj-{run_id}")
            self._trajectory_by_run[run_id] = trajectory

        snapshot = self.state_machine.advance(
            trajectory=trajectory,
            regime=scenario_name,
            episode_result=episode_result,
            new_snapshot_id=f"snap-{uuid4().hex[:12]}",
            timestamp=timestamp,
        )

        window = trajectory.window(8)
        digest = trajectory.digest(window_size=8)
        window_id = f"window-{trajectory.trajectory_id}-{window.end_episode}"
        self.storage.write_organism_snapshot(
            snapshot_id=snapshot.snapshot_id,
            run_id=run_id,
            episode_id=episode_id,
            trajectory_id=trajectory.trajectory_id,
            regime=scenario_name,
            snapshot_json=snapshot.to_dict(),
            metadata={"mode": mode},
        )
        self.storage.write_trajectory_window(
            window_id=window_id,
            run_id=run_id,
            trajectory_id=trajectory.trajectory_id,
            start_episode=window.start_episode,
            end_episode=window.end_episode,
            snapshots_json={"snapshots": [point.snapshot.to_dict() for point in window.points]},
            digest_json=asdict(digest),
            metadata={"window_size": len(window.points)},
        )

        flow_result = self.flow_engine.evaluate(trajectory, window_size=8)
        flow_report = self.flow_engine.to_report(trajectory=trajectory, result=flow_result, window_size=8)
        self.storage.write_trajectory_flow_report(
            report_id=f"flow-{trajectory.trajectory_id}-{window.end_episode}",
            run_id=run_id,
            trajectory_id=trajectory.trajectory_id,
            window_id=window_id,
            flow_validity=flow_result.flow_validity,
            erosion=flow_result.erosion,
            phase_drift=flow_result.phase_drift,
            rollback_obligation=flow_result.rollback_obligation,
            report_json=asdict(flow_report),
            metadata={
                "mode": mode,
                "point_violations": list(flow_result.point_violations),
                "flow_violations": list(flow_result.flow_violations),
                "thresholds": flow_result.thresholds,
            },
        )

        viability = self.viability_kernel.assess(
            trajectory=trajectory,
            flow_result=flow_result,
            window_size=8,
        )

        prev_regime_name = self._last_regime_by_run.get(run_id)
        renorm_result: RenormalizationResult | None = None
        renorm_residual = 0.0
        renorm_uncertainty = 0.0
        renorm_recovery_cost = 0.0
        if prev_regime_name and prev_regime_name != scenario_name:
            source_regime = get_regime_for_scenario(prev_regime_name)
            target_regime = get_regime_for_scenario(scenario_name)
            if source_regime is not None and target_regime is not None:
                renorm_result = self.renorm_engine.renormalize(
                    source_regime=source_regime,
                    target_regime=target_regime,
                    snapshot=snapshot,
                )
                renorm_residual = renorm_result.regime_residual.residual_error
                renorm_uncertainty = renorm_result.uncertainty.transport_uncertainty
                renorm_recovery_cost = renorm_result.regime_residual.expected_recovery_cost
                self.storage.write_renormalization_event(
                    event_id=f"renorm-{trajectory.trajectory_id}-{window.end_episode}",
                    run_id=run_id,
                    trajectory_id=trajectory.trajectory_id,
                    source_regime=renorm_result.renormalization_map.source_regime,
                    target_regime=renorm_result.renormalization_map.target_regime,
                    residual_error=renorm_residual,
                    transport_uncertainty=renorm_uncertainty,
                    expected_recovery_cost=renorm_recovery_cost,
                    map_json=asdict(renorm_result),
                    metadata={"mode": mode},
                )

        previous_snapshot = window.points[-2].snapshot if len(window.points) > 1 else snapshot
        d_viability = snapshot.viability.viability_margin - previous_snapshot.viability.viability_margin
        d_purity = max(0.0, previous_snapshot.belief.memory_purity_estimate - snapshot.belief.memory_purity_estimate)
        d_modification = 1.0 if snapshot.modification.lineage_delta_pending else 0.0

        scope_defs = [
            ("organism", run_id),
            ("modification", snapshot.modification.active_proposals[0].proposal_id if snapshot.modification.active_proposals else f"{run_id}:mod"),
            ("inheritance", snapshot.identity.lineage_id or run_id),
        ]
        if renorm_result is not None:
            scope_defs.append(
                (
                    "edge",
                    f"{renorm_result.renormalization_map.source_regime}->{renorm_result.renormalization_map.target_regime}",
                )
            )

        updates: Dict[str, RiskState] = {}
        for scope_type, scope_key in scope_defs:
            self._seed_risk_from_storage(
                run_id=run_id,
                trajectory_id=trajectory.trajectory_id,
                scope_type=scope_type,
                scope_key=scope_key,
            )
            update = self.risk_process.update(
                scope_type=scope_type,  # type: ignore[arg-type]
                scope_key=scope_key,
                drift_identity=digest.identity_curvature,
                drift_policy=digest.policy_phase_drift,
                delta_viability=d_viability,
                delta_purity=d_purity,
                delta_modification=d_modification,
                erosion=flow_result.erosion,
                renorm_residual=renorm_residual,
            )
            state = self.risk_process.get(scope_type=scope_type, scope_key=scope_key)
            assert state is not None
            updates[scope_type] = state
            self._persist_risk_scope(
                run_id=run_id,
                trajectory_id=trajectory.trajectory_id,
                window_end_episode=window.end_episode,
                scope_type=scope_type,
                scope_key=scope_key,
                update_payload=asdict(update),
                state=state,
                mode=mode,
            )

        organism_state = updates["organism"]
        edge_state = updates.get("edge")
        mod_state = updates["modification"]
        inheritance_state = updates["inheritance"]
        edge_risk = edge_state.risk_score if edge_state is not None else 0.0

        canonical_scope, transfer_advice = self._scope_from_risk(
            cross_regime=renorm_result is not None,
            modification_pending=snapshot.modification.lineage_delta_pending,
            flow_valid=flow_result.flow_validity,
            rollback=flow_result.rollback_obligation or viability.rollback_required,
            viability_score=viability.viability_score,
            organism_risk=organism_state.risk_score,
            edge_risk=edge_risk,
            modification_risk=mod_state.risk_score,
            inheritance_risk=inheritance_state.risk_score,
        )
        legacy_scope = self._legacy_scope_from_canonical(canonical_scope)

        self._last_regime_by_run[run_id] = scenario_name
        return CourtEpisodeResult(
            trajectory_id=trajectory.trajectory_id,
            canonical_scope=canonical_scope,
            legacy_scope=legacy_scope,
            transfer_advice=transfer_advice,
            flow_validity=flow_result.flow_validity,
            erosion=flow_result.erosion,
            phase_drift=flow_result.phase_drift,
            rollback_obligation=flow_result.rollback_obligation or viability.rollback_required,
            viability_score=viability.viability_score,
            organism_risk=organism_state.risk_score,
            edge_risk=round(edge_risk, 4),
            modification_risk=mod_state.risk_score,
            inheritance_risk=inheritance_state.risk_score,
            failure_mode_count=len(organism_state.failure_atlas.events),
            renormalization_residual=renorm_residual,
            renormalization_uncertainty=renorm_uncertainty,
            expected_recovery_cost=renorm_recovery_cost,
        )

