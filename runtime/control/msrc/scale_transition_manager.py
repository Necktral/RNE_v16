"""Gestor de transición de escala para MSRC."""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, Optional

from runtime.storage.records import utc_now_iso

from .contracts import ProbeResult, ScaleAction, ScaleEstimate, ScaleTransitionRecord
from .scale_catalog import ScaleCatalog


ProbeExecutor = Callable[[str], ProbeResult]


class ScaleTransitionManager:
    """Coordina cambios de escala, probes y rollback."""

    def __init__(self, *, catalog: ScaleCatalog):
        self.catalog = catalog

    def execute_action(
        self,
        *,
        run_id: str,
        episode_id: str,
        current_scale_id: str,
        action: ScaleAction,
        estimate: ScaleEstimate,
        probe_executor: Optional[ProbeExecutor] = None,
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        source_spec = self.catalog.get(current_scale_id)
        target_scale_id = action.target_scale_id or current_scale_id
        rollback_applied = False
        probe_result: Optional[ProbeResult] = None

        try:
            if action.action_type == "fork_probe":
                if probe_executor is None:
                    raise RuntimeError("probe_executor requerido para fork_probe")
                probe_result = probe_executor(target_scale_id)
                real_time_cost = (time.perf_counter() - t0) * 1000.0
                return {
                    "selected_scale_id": current_scale_id,
                    "probe_result": probe_result,
                    "transition_record": self._build_record(
                        run_id=run_id,
                        episode_id=episode_id,
                        action=action,
                        source_scale_id=current_scale_id,
                        target_scale_id=current_scale_id,
                        estimate=estimate,
                        real_time_cost=real_time_cost,
                        real_artifact_cost=0.0,
                        ioc_delta=probe_result.cognitive_gain_delta,
                        viability_delta=0.0 if probe_result.viability_preserved else -1.0,
                        rollback_applied=False,
                    ),
                }

            if action.action_type in {"keep_scale", "lock_scale_for_n_steps", "discard_probe_result"}:
                real_time_cost = (time.perf_counter() - t0) * 1000.0
                return {
                    "selected_scale_id": current_scale_id,
                    "probe_result": None,
                    "transition_record": self._build_record(
                        run_id=run_id,
                        episode_id=episode_id,
                        action=action,
                        source_scale_id=current_scale_id,
                        target_scale_id=current_scale_id,
                        estimate=estimate,
                        real_time_cost=real_time_cost,
                        real_artifact_cost=0.0,
                        ioc_delta=0.0,
                        viability_delta=0.0,
                        rollback_applied=False,
                    ),
                }

            if action.action_type in {"upgrade_scale", "downgrade_scale", "commit_probe_result"}:
                if not self.catalog.has_scale(target_scale_id):
                    raise KeyError(f"Escala destino inválida: {target_scale_id}")

                target_spec = self.catalog.get(target_scale_id)
                if not target_spec.is_executable:
                    target_spec = self.catalog.nearest_executable(target_scale_id)
                    target_scale_id = target_spec.scale_id

                real_time_cost = (time.perf_counter() - t0) * 1000.0
                real_artifact_cost = max(
                    target_spec.expected_artifact_cost - source_spec.expected_artifact_cost,
                    0.0,
                )

                return {
                    "selected_scale_id": target_scale_id,
                    "probe_result": None,
                    "transition_record": self._build_record(
                        run_id=run_id,
                        episode_id=episode_id,
                        action=action,
                        source_scale_id=current_scale_id,
                        target_scale_id=target_scale_id,
                        estimate=estimate,
                        real_time_cost=real_time_cost,
                        real_artifact_cost=real_artifact_cost,
                        ioc_delta=action.expected_gain,
                        viability_delta=0.0,
                        rollback_applied=False,
                    ),
                }

            raise ValueError(f"Acción de escala no soportada: {action.action_type}")

        except Exception as exc:
            rollback_applied = True
            real_time_cost = (time.perf_counter() - t0) * 1000.0
            record = self._build_record(
                run_id=run_id,
                episode_id=episode_id,
                action=action,
                source_scale_id=current_scale_id,
                target_scale_id=current_scale_id,
                estimate=estimate,
                real_time_cost=real_time_cost,
                real_artifact_cost=0.0,
                ioc_delta=0.0,
                viability_delta=-1.0,
                rollback_applied=True,
                extra_metadata={"error": f"{type(exc).__name__}: {exc}"},
            )
            return {
                "selected_scale_id": current_scale_id,
                "probe_result": probe_result,
                "transition_record": record,
            }

    def _build_record(
        self,
        *,
        run_id: str,
        episode_id: str,
        action: ScaleAction,
        source_scale_id: str,
        target_scale_id: str,
        estimate: ScaleEstimate,
        real_time_cost: float,
        real_artifact_cost: float,
        ioc_delta: float,
        viability_delta: float,
        rollback_applied: bool,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> ScaleTransitionRecord:
        source_spec = self.catalog.get(source_scale_id)
        target_spec = self.catalog.get(target_scale_id)
        metadata = {
            "required_resolution_score": estimate.required_resolution_score,
            "risk_score": estimate.risk_score,
            "vram_pressure": estimate.vram_pressure,
            "vram_opportunity_score": estimate.vram_opportunity_score,
            **action.metadata,
        }
        if extra_metadata:
            metadata.update(extra_metadata)

        return ScaleTransitionRecord(
            run_id=run_id,
            episode_id=episode_id,
            action_type=action.action_type,
            source_scale_id=source_scale_id,
            target_scale_id=target_scale_id,
            reason=action.reason,
            estimated_time_cost=target_spec.expected_time_cost,
            estimated_artifact_cost=target_spec.expected_artifact_cost,
            real_time_cost=real_time_cost,
            real_artifact_cost=real_artifact_cost,
            ioc_delta=ioc_delta,
            viability_delta=viability_delta,
            rollback_applied=rollback_applied,
            timestamp=utc_now_iso(),
            metadata=metadata,
        )
