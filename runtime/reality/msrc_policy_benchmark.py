"""Benchmark de políticas de escala para MSRC."""

from __future__ import annotations

import json
import math
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from runtime.control.msrc import MSRCController, ProbeResult, ScalePolicyEngine, ScalePolicyState
from runtime.reasoning.scheduler_meta.family_metrics import (
    aggregate_family_activation_counts,
    aggregate_family_dict_metric,
    build_family_sensitive_bundle,
)
from runtime.storage.records import utc_now_iso
from runtime.world.grid_thermal_scenario import GridThermalScenario
from runtime.world.scenario_runner import ScenarioEpisodeRunner


class MSRCPolicyBenchmarkRunner:
    """Ejecutor de benchmark comparativo de políticas de escala."""

    VALID_POLICIES = {
        "always_1x1",
        "always_5x5",
        "adaptive_msrc",
        "adaptive_msrc_aggressive",
        "adaptive_msrc_regime_v3",
        "always_highest_available",
        "probe_before_switch",
    }

    def __init__(self, *, storage, output_root: Path):
        self.storage = storage
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)

    def run_policy(
        self,
        *,
        run_id: str,
        policy_name: str,
        episodes: int,
        base_seed: int,
        output_dir: Path,
        scenario_params: Dict[str, Any],
        external_input: float = 0.04,
        level_label: str = "",
        topology_label: str = "uniform",
    ) -> Dict[str, Any]:
        if policy_name not in self.VALID_POLICIES:
            raise ValueError(f"Política no soportada: {policy_name}")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        controller: Optional[MSRCController] = None
        state: Optional[ScalePolicyState] = None
        pending_probe_result: Optional[ProbeResult] = None

        if policy_name in {
            "adaptive_msrc",
            "adaptive_msrc_aggressive",
            "adaptive_msrc_regime_v3",
            "probe_before_switch",
        }:
            if policy_name == "adaptive_msrc_aggressive":
                policy_engine = ScalePolicyEngine.aggressive()
            elif policy_name == "adaptive_msrc_regime_v3":
                policy_engine = ScalePolicyEngine.regime_v3()
            else:
                policy_engine = ScalePolicyEngine.baseline()
            controller = MSRCController(
                storage=self.storage,
                output_dir=output_dir,
                policy_engine=policy_engine,
            )
            state = controller.ensure_state(None, default_scale_id="1x1")

        results: List[Dict[str, Any]] = []
        decision_rows: List[Dict[str, Any]] = []
        action_rows: List[Dict[str, Any]] = []
        transition_rows: List[Dict[str, Any]] = []
        contamination_rows: List[Dict[str, Any]] = []
        probe_rows: List[Dict[str, Any]] = []
        selected_scales: List[str] = []
        vram_snapshots: List[Dict[str, Any]] = []
        recompute_avoided = 0

        expected_spatial_complexity = self._expected_spatial_complexity(topology_label=topology_label)

        for i in range(episodes):
            seed = base_seed + i
            episode_id = f"{policy_name}-ep-{seed}"

            if policy_name == "always_1x1":
                scale_id = "1x1"
            elif policy_name in {"always_5x5", "always_highest_available"}:
                scale_id = "5x5"
            else:
                assert state is not None
                scale_id = state.current_scale_id

            episode_result, metrics = self._run_single_episode(
                run_id=run_id,
                episode_id=episode_id,
                scale_id=scale_id,
                scenario_params=scenario_params,
                external_input=external_input,
                topology_label=topology_label,
            )
            selected_scales.append(scale_id)
            results.append(metrics)

            if controller is None or state is None:
                action_rows.append(
                    {
                        "action_type": "keep_scale",
                        "target_scale_id": scale_id,
                        "reason": "política estática",
                        "metadata": {
                            "policy": policy_name,
                            "current_scale_id": scale_id,
                            "candidate_scale_id": scale_id,
                            "triggered_upgrade_signal": False,
                            "triggered_probe_signal": False,
                        },
                    }
                )
                continue

            memory_payload = {
                "world_level": metrics.get("world_level", 0.0),
                "viability_margin": metrics.get("viability_margin", 0.0),
                "regime_category": level_label,
                "heterogeneity_score": metrics.get("heterogeneity_score_proxy", 0.0),
                "scale_success_history": selected_scales[-6:],
                "cost_history": [x.get("wall_time_ms", 0.0) for x in results[-6:]],
                "episode_signature": metrics.get("episode_id"),
                "prior_scale_verdict": metrics.get("certification_verdict"),
                "cell_states": episode_result.get("episode", {}).get("context", {}).get("observation", {}).get("cell_states"),
            }

            probe_enabled = policy_name in {
                "adaptive_msrc",
                "adaptive_msrc_aggressive",
                "adaptive_msrc_regime_v3",
                "probe_before_switch",
            }

            def _probe_executor(target_scale_id: str) -> ProbeResult:
                probe_episode_id = f"{episode_id}-probe-{target_scale_id}"
                _, probe_metrics = self._run_single_episode(
                    run_id=run_id,
                    episode_id=probe_episode_id,
                    scale_id=target_scale_id,
                    scenario_params=scenario_params,
                    external_input=external_input,
                    topology_label=topology_label,
                )
                base_ivc = float(metrics.get("ivc_r", 0.0))
                probe_ivc = float(probe_metrics.get("ivc_r", 0.0))
                delta = probe_ivc - base_ivc
                viable = bool(probe_metrics.get("is_viable", True))
                evidence = min(max(0.5 + delta, 0.0), 1.0)
                outcome = "positive" if viable and delta > 0.01 else "negative"
                probe_result = ProbeResult(
                    target_scale_id=target_scale_id,
                    cognitive_gain_delta=delta,
                    viability_preserved=viable,
                    evidence_score=evidence,
                    outcome=outcome,
                    details={
                        "base_ivc_r": base_ivc,
                        "probe_ivc_r": probe_ivc,
                        "delta_ivc_r": delta,
                        "base_intervention_precision": float(metrics.get("intervention_precision", 0.0) or 0.0),
                        "probe_intervention_precision": float(probe_metrics.get("intervention_precision", 0.0) or 0.0),
                        "delta_intervention_precision": (
                            float(probe_metrics.get("intervention_precision", 0.0) or 0.0)
                            - float(metrics.get("intervention_precision", 0.0) or 0.0)
                        ),
                        "base_spatial_information_usage": float(metrics.get("spatial_information_usage", 0.0) or 0.0),
                        "probe_spatial_information_usage": float(probe_metrics.get("spatial_information_usage", 0.0) or 0.0),
                        "delta_spatial_information_usage": (
                            float(probe_metrics.get("spatial_information_usage", 0.0) or 0.0)
                            - float(metrics.get("spatial_information_usage", 0.0) or 0.0)
                        ),
                        "probe_episode_id": probe_episode_id,
                    },
                )
                probe_rows.append(probe_result.to_dict())
                return probe_result

            step_out = controller.step(
                run_id=run_id,
                episode_id=episode_id,
                state=state,
                observation=episode_result.get("episode", {}).get("context", {}).get("observation", {}),
                viability_margin=episode_result.get("viability_assessment", {}).get("viability_margin"),
                certification_verdict=episode_result.get("certification", {}).get("verdict"),
                metrics={
                    **metrics,
                    "factual_delta": episode_result.get("episode", {}).get("result", {}).get("factual_delta", 0.0),
                    "counterfactual_delta": episode_result.get("episode", {}).get("result", {}).get("counterfactual_delta", 0.0),
                    "operational_budget_ratio": min(metrics.get("wall_time_ms", 0.0) / 1200.0, 1.0),
                    "cumulative_cost_ratio": min(sum(x.get("wall_time_ms", 0.0) for x in results) / (max(len(results), 1) * 1200.0), 1.0),
                    "expected_spatial_complexity": expected_spatial_complexity,
                    "topology_label": topology_label,
                    "previous_probe_failure": 1.0 if (pending_probe_result is not None and pending_probe_result.outcome == "negative") else 0.0,
                    "recent_oscillation": float(state.oscillation_events) / max(state.step_index, 1),
                },
                probe_result=pending_probe_result,
                probe_executor=_probe_executor if probe_enabled else None,
            )

            pending_probe_result = step_out.get("probe_result")
            action_rows.append(step_out["action"].to_dict())
            decision_record = step_out.get("decision_record")
            if decision_record is not None:
                decision_rows.append(decision_record.to_dict())
            transition_rows.append(step_out["transition_record"].to_dict())
            vram_snapshots.append(step_out.get("vram_snapshot", {}))
            if step_out.get("vram_snapshot", {}).get("vram_opportunity_score", 0.0) > 0.60 and step_out["selected_scale_id"] != "1x1":
                recompute_avoided += 1

            report = controller.sanitize_cross_scale_memory(
                source_scale_id=scale_id,
                target_scale_id=step_out["selected_scale_id"],
                payload=memory_payload,
            )
            contamination_rows.append(report.to_dict())

        self._persist_jsonl(output_dir / "episodes.jsonl", results)
        if decision_rows:
            self._persist_jsonl(output_dir / "scale_decisions.jsonl", decision_rows)
        self._persist_jsonl(output_dir / "scale_actions.jsonl", action_rows)

        summary = self._build_summary(
            policy_name=policy_name,
            run_id=run_id,
            results=results,
            selected_scales=selected_scales,
            actions=action_rows,
            transitions=transition_rows,
            probes=probe_rows,
            contamination_rows=contamination_rows,
            vram_snapshots=vram_snapshots,
            recompute_avoided=recompute_avoided,
            level_label=level_label,
            topology_label=topology_label,
        )

        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=True, indent=2, default=str),
            encoding="utf-8",
        )

        self.storage.write_reality_bench_run(
            bench_run_id=run_id,
            run_id=run_id,
            total_episodes=summary["total_episodes"],
            closure_rate=summary["success_rate"],
            continuity_mean=summary["avg_metrics"].get("viability_margin", 0.0),
            collapse_count=summary["collapse_count"],
            gate_profile=f"msrc_policy::{policy_name}",
            passed=bool(summary["errors"] == 0 and summary["successful"] > 0),
            summary=summary,
        )

        self.storage.append_event(
            event_type="reality.msrc_policy_benchmark.completed",
            run_id=run_id,
            source="msrc_policy_benchmark",
            payload={
                "run_id": run_id,
                "policy_name": policy_name,
                "summary": summary,
                "timestamp": utc_now_iso(),
            },
        )

        return summary

    def _run_single_episode(
        self,
        *,
        run_id: str,
        episode_id: str,
        scale_id: str,
        scenario_params: Dict[str, Any],
        external_input: float,
        topology_label: str,
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        params = dict(scenario_params)
        if scale_id == "1x1":
            params["grid_size"] = 1
            params.pop("topology", None)
            scenario_name = "grid_thermal_1x1"
        else:
            params["grid_size"] = 5
            params.setdefault("topology", topology_label or "uniform")
            scenario_name = f"grid_thermal_5x5_{params['topology']}"

        scenario = GridThermalScenario(**params)
        profile_name = os.environ.get("RNFE_REASONING_FAMILY_PROFILE", "").strip().lower()
        closure_profile = "baseline_fixed" if profile_name in {"", "core_only"} else "adaptive_min"

        runner = ScenarioEpisodeRunner(
            storage=self.storage,
            run_id=run_id,
            scenario=scenario,
            closure_profile=closure_profile,
        )

        t0 = time.perf_counter()
        result = runner.run_episode(external_input=external_input)
        wall_time_ms = (time.perf_counter() - t0) * 1000.0

        artifact_path = result.get("artifact", {}).get("abs_path")
        artifact_size = 0
        if artifact_path and Path(artifact_path).exists():
            artifact_size = Path(artifact_path).stat().st_size

        metrics = self._episode_to_metrics(
            episode_id=episode_id,
            scenario_name=scenario_name,
            scale_id=scale_id,
            result=result,
            wall_time_ms=wall_time_ms,
            artifact_size_bytes=artifact_size,
        )
        return result, metrics

    def _episode_to_metrics(
        self,
        *,
        episode_id: str,
        scenario_name: str,
        scale_id: str,
        result: Dict[str, Any],
        wall_time_ms: float,
        artifact_size_bytes: int,
    ) -> Dict[str, Any]:
        episode = result.get("episode", {})
        observation = episode.get("context", {}).get("observation", {})
        updated_world = episode.get("result", {}).get("updated_world", {})
        propositions = observation.get("propositions") or []

        cert_verdict = result.get("certification", {}).get("verdict")
        outcome = "success" if cert_verdict in {"passed", "certified", "PASSED", "CONDITIONALLY_PASSED"} else "failure"
        is_viable = bool(result.get("viability_assessment", {}).get("is_viable", True))
        viability_margin = float(result.get("viability_assessment", {}).get("viability_margin", 0.0) or 0.0)

        intervention_precision = self._compute_intervention_precision(observation, updated_world)
        proposition_diversity = self._compute_shannon_entropy(propositions)
        spatial_information_usage = self._compute_spatial_usage(propositions)
        reasoning_trace_length = len(result.get("reasoning", {}).get("sequence", []))
        ivc_r = self._compute_ivc_r(
            cierre_rate=1.0 if outcome == "success" else 0.0,
            continuity_score=max(viability_margin, 0.0),
            intervention_precision=max(intervention_precision, 0.0),
            proposition_diversity=proposition_diversity,
            wall_time_ms=wall_time_ms,
        )

        failure_primary = None
        if outcome != "success":
            if not is_viable:
                failure_primary = "viability_failed"
            else:
                failure_primary = "certification_failed"

        world_level = float(observation.get("world_level", observation.get("temperature", 0.0) or 0.0))
        heterogeneity_proxy = float(observation.get("local_std_temp", 0.0) or 0.0)
        reasoning = result.get("reasoning", {}) or {}
        family_bundle = build_family_sensitive_bundle(
            reasoning_sequence=reasoning.get("sequence", []),
            reasoning_trace=reasoning.get("trace", []),
            profile_name=reasoning.get("family_profile"),
            mode=reasoning.get("mode", "fixed"),
            proposed_sequence=reasoning.get("proposed_sequence", []),
            validated_sequence=reasoning.get("validated_sequence", []),
            sequence_validation=reasoning.get("sequence_validation", {}),
            final_metrics={
                "ivc_r": ivc_r,
                "intervention_precision": intervention_precision,
                "viability_margin": viability_margin,
                "reasoning_trace_length": reasoning_trace_length,
                "success_rate": 1.0 if outcome == "success" else 0.0,
                "spatial_information_usage": spatial_information_usage,
            },
        )

        return {
            "episode_id": episode_id,
            "scenario": scenario_name,
            "scale_id": scale_id,
            "outcome": outcome,
            "certification_verdict": cert_verdict,
            "is_viable": is_viable,
            "viability_margin": viability_margin,
            "success_rate": 1.0 if outcome == "success" else 0.0,
            "intervention_precision": intervention_precision,
            "proposition_diversity": proposition_diversity,
            "spatial_information_usage": spatial_information_usage,
            "wall_time_ms": wall_time_ms,
            "artifact_size_bytes": artifact_size_bytes,
            "reasoning_trace_length": reasoning_trace_length,
            "ivc_r": ivc_r,
            "failure_primary": failure_primary,
            "world_level": world_level,
            "heterogeneity_score_proxy": heterogeneity_proxy,
            "reasoning_mode": reasoning.get("mode"),
            "family_profile": reasoning.get("family_profile"),
            "regime_label": reasoning.get("regime_label"),
            "timestamp": utc_now_iso(),
            **family_bundle,
        }

    def _build_summary(
        self,
        *,
        policy_name: str,
        run_id: str,
        results: List[Dict[str, Any]],
        selected_scales: List[str],
        actions: List[Dict[str, Any]],
        transitions: List[Dict[str, Any]],
        probes: List[Dict[str, Any]],
        contamination_rows: List[Dict[str, Any]],
        vram_snapshots: List[Dict[str, Any]],
        recompute_avoided: int,
        level_label: str,
        topology_label: str,
    ) -> Dict[str, Any]:
        total = len(results)
        successful = sum(1 for row in results if row.get("outcome") == "success")
        failed = sum(1 for row in results if row.get("outcome") == "failure")
        errors = sum(1 for row in results if row.get("outcome") == "error")

        avg_metrics = {}
        for key in [
            "intervention_precision",
            "proposition_diversity",
            "spatial_information_usage",
            "wall_time_ms",
            "artifact_size_bytes",
            "reasoning_trace_length",
            "ivc_r",
            "viability_margin",
            "family_mix_entropy",
        ]:
            values = [float(row.get(key, 0.0) or 0.0) for row in results]
            avg_metrics[key] = sum(values) / len(values) if values else 0.0

        optional_family_usage_rate = (
            sum(1.0 for row in results if bool(row.get("family_optional_used_flag"))) / len(results)
            if results
            else 0.0
        )
        backbone_floor_satisfied_rate = (
            sum(1.0 for row in results if bool(row.get("backbone_floor_satisfied_flag"))) / len(results)
            if results
            else 0.0
        )
        sequence_validation_fail_rate = (
            sum(1.0 for row in results if bool(row.get("sequence_validation_fail_flag"))) / len(results)
            if results
            else 0.0
        )
        fallback_to_safe_sequence_rate = (
            sum(1.0 for row in results if bool(row.get("fallback_to_safe_sequence_flag"))) / len(results)
            if results
            else 0.0
        )
        optional_displacement_rate = (
            sum(1.0 for row in results if bool(row.get("optional_displacement_flag"))) / len(results)
            if results
            else 0.0
        )
        closure_break_rate = (
            sum(1.0 for row in results if bool(row.get("closure_break_flag"))) / len(results)
            if results
            else 0.0
        )
        family_specific_activation_counts = aggregate_family_activation_counts(results)
        family_impact_aggregates = {
            "family_contribution_proxy": aggregate_family_dict_metric(results, "family_contribution_proxy"),
            "family_delta_ivc_r": aggregate_family_dict_metric(results, "family_delta_ivc_r"),
            "family_delta_intervention_precision": aggregate_family_dict_metric(
                results,
                "family_delta_intervention_precision",
            ),
            "family_delta_viability_margin": aggregate_family_dict_metric(results, "family_delta_viability_margin"),
            "family_delta_reasoning_trace_length": aggregate_family_dict_metric(
                results,
                "family_delta_reasoning_trace_length",
            ),
            "family_delta_success_rate": aggregate_family_dict_metric(results, "family_delta_success_rate"),
            "family_delta_spatial_information_usage": aggregate_family_dict_metric(
                results,
                "family_delta_spatial_information_usage",
            ),
        }

        failures = Counter(row.get("failure_primary") for row in results if row.get("failure_primary"))
        collapse_count = sum(1 for row in results if row.get("is_viable") is False)

        action_counts = Counter(row.get("action_type") for row in actions if row.get("action_type"))
        total_actions = sum(action_counts.values())
        keep_rate = action_counts.get("keep_scale", 0) / total_actions if total_actions else 0.0
        upgrade_rate = action_counts.get("upgrade_scale", 0) / total_actions if total_actions else 0.0
        downgrade_rate = action_counts.get("downgrade_scale", 0) / total_actions if total_actions else 0.0
        probe_rate = action_counts.get("fork_probe", 0) / total_actions if total_actions else 0.0
        probe_commit_rate = (
            action_counts.get("commit_probe_result", 0) / action_counts.get("fork_probe", 1)
            if action_counts.get("fork_probe", 0) > 0
            else 0.0
        )
        regime_counts: Counter[str] = Counter()
        regime_probe_counts: Counter[str] = Counter()
        for row in actions:
            metadata = row.get("metadata") or {}
            regime = metadata.get("regime_label")
            if not regime:
                continue
            regime_counts[str(regime)] += 1
            if row.get("action_type") == "fork_probe":
                regime_probe_counts[str(regime)] += 1
        regime_probe_rate = {
            regime: (regime_probe_counts.get(regime, 0) / count if count else 0.0)
            for regime, count in regime_counts.items()
        }

        missed_upgrade_regret = sum(
            1
            for row in actions
            if row.get("action_type") == "keep_scale"
            and bool((row.get("metadata") or {}).get("triggered_upgrade_signal", False))
        )

        scale_selection_accuracy = self._compute_scale_selection_accuracy(selected_scales, transitions)
        probe_positive = sum(1 for probe in probes if probe.get("outcome") == "positive")
        probe_total = len(probes)
        probe_value_rate = probe_positive / probe_total if probe_total else 0.0

        contamination_rate = 0.0
        if contamination_rows:
            contamination_rate = sum(
                float(row.get("cross_scale_memory_contamination_rate", 0.0))
                for row in contamination_rows
            ) / len(contamination_rows)

        vram_headroom_values = [float(v.get("vram_headroom", 0.0) or 0.0) for v in vram_snapshots]
        vram_pressure_values = [float(v.get("vram_pressure", 0.0) or 0.0) for v in vram_snapshots]
        vram_headroom_mean = sum(vram_headroom_values) / len(vram_headroom_values) if vram_headroom_values else 0.0
        vram_peak_ratio = max(vram_pressure_values) if vram_pressure_values else 0.0

        fork_actions = [row for row in actions if row.get("action_type") == "fork_probe"]
        vram_probe_candidates = [
            idx
            for idx, action in enumerate(fork_actions)
            if float((action.get("metadata") or {}).get("vram_opportunity_score", 0.0) or 0.0) > 0.5
        ]
        vram_probe_positive = sum(
            1
            for idx in vram_probe_candidates
            if idx < len(probes) and probes[idx].get("outcome") == "positive"
        )
        vram_enabled_probe_success_rate = (
            vram_probe_positive / len(vram_probe_candidates)
            if vram_probe_candidates
            else 0.0
        )

        mean_resolution_cost = self._mean_resolution_cost(selected_scales)
        resolution_efficiency = avg_metrics["ivc_r"] / max(mean_resolution_cost, 1e-6)
        vram_efficiency_after_intelligence = (
            resolution_efficiency / max(vram_peak_ratio, 1e-6)
            if vram_peak_ratio > 0
            else resolution_efficiency
        )

        summary = {
            "run_id": run_id,
            "policy_name": policy_name,
            "level": level_label,
            "topology": topology_label,
            "total_episodes": total,
            "successful": successful,
            "failed": failed,
            "errors": errors,
            "success_rate": successful / total if total else 0.0,
            "avg_metrics": avg_metrics,
            "failure_distribution": dict(failures),
            "collapse_count": collapse_count,
            "optional_family_usage_rate": optional_family_usage_rate,
            "backbone_floor_satisfied_rate": backbone_floor_satisfied_rate,
            "sequence_validation_fail_rate": sequence_validation_fail_rate,
            "fallback_to_safe_sequence_rate": fallback_to_safe_sequence_rate,
            "optional_displacement_rate": optional_displacement_rate,
            "closure_break_rate": closure_break_rate,
            "family_specific_activation_counts": family_specific_activation_counts,
            "family_impact_aggregates": family_impact_aggregates,
            "proxy_mapping": {
                "closure_rate": "success_rate",
                "continuity_mean": "viability_margin",
            },
            "msrc_metrics": {
                "scale_selection_accuracy": scale_selection_accuracy,
                "upgrade_regret": sum(1 for a in actions if a.get("action_type") == "discard_probe_result"),
                "downgrade_regret": sum(1 for t in transitions if t.get("action_type") == "rollback"),
                "missed_upgrade_regret": missed_upgrade_regret,
                "keep_scale_rate": keep_rate,
                "upgrade_rate": upgrade_rate,
                "downgrade_rate": downgrade_rate,
                "probe_rate": probe_rate,
                "probe_commit_rate": probe_commit_rate,
                "oscillation_rate": self._compute_oscillation_rate(transitions),
                "probe_value_rate": probe_value_rate,
                "mean_resolution_cost": mean_resolution_cost,
                "resolution_efficiency": resolution_efficiency,
                "cross_scale_memory_contamination_rate": contamination_rate,
                "vram_headroom_mean": vram_headroom_mean,
                "vram_peak_ratio": vram_peak_ratio,
                "vram_recompute_avoided": recompute_avoided,
                "vram_enabled_probe_success_rate": vram_enabled_probe_success_rate,
                "vram_efficiency_after_intelligence": vram_efficiency_after_intelligence,
                "selected_scale_distribution": dict(Counter(selected_scales)),
                "regime_distribution": dict(regime_counts),
                "regime_probe_rate": regime_probe_rate,
            },
            "generated_at": utc_now_iso(),
        }
        return summary

    def _compute_scale_selection_accuracy(self, selected_scales: List[str], transitions: List[Dict[str, Any]]) -> float:
        if not selected_scales:
            return 0.0
        bad = sum(1 for row in transitions if row.get("rollback_applied"))
        return max(0.0, 1.0 - (bad / len(selected_scales)))

    def _compute_oscillation_rate(self, transitions: List[Dict[str, Any]]) -> float:
        if len(transitions) < 2:
            return 0.0
        oscillations = 0
        for idx in range(1, len(transitions)):
            prev_action = transitions[idx - 1].get("action_type")
            curr_action = transitions[idx].get("action_type")
            if {prev_action, curr_action} <= {"upgrade_scale", "downgrade_scale"} and prev_action != curr_action:
                oscillations += 1
        return oscillations / len(transitions)

    def _mean_resolution_cost(self, selected_scales: List[str]) -> float:
        if not selected_scales:
            return 0.0
        mapping = {
            "1x1": 1.0,
            "2x2": 1.2,
            "3x3": 1.5,
            "5x5": 2.2,
            "10x10": 4.5,
            "30x30": 12.0,
        }
        vals = [mapping.get(scale, 1.0) for scale in selected_scales]
        return sum(vals) / len(vals)

    def _compute_intervention_precision(self, observation: Dict[str, Any], updated_world: Dict[str, Any]) -> float:
        initial = observation.get("world_level") or observation.get("temperature")
        final = updated_world.get("world_level") or updated_world.get("temperature")
        if initial is None or final is None:
            return 0.0
        try:
            initial_f = float(initial)
            final_f = float(final)
        except (TypeError, ValueError):
            return 0.0
        if abs(initial_f) < 1e-6:
            return 0.0
        return (initial_f - final_f) / initial_f

    def _compute_shannon_entropy(self, propositions: List[str]) -> float:
        if not propositions:
            return 0.0
        counts = Counter(propositions)
        total = len(propositions)
        entropy = 0.0
        for count in counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)
        return entropy

    def _compute_spatial_usage(self, propositions: List[str]) -> float:
        if not propositions:
            return 0.0
        indicators = {
            "HOTSPOT_DETECTED",
            "MULTI_HOTSPOT",
            "HOTSPOT_CENTRAL",
            "HOTSPOT_PERIPHERAL",
            "HOTSPOT_CRITICAL",
            "THERMAL_GRADIENT",
            "THERMAL_GRADIENT_NS",
            "THERMAL_GRADIENT_EW",
            "CRITICAL_ZONE_PRESENT",
            "QUADRANT_IMBALANCE",
            "UNIFORM_TEMPERATURE",
        }
        return sum(1 for item in propositions if item in indicators) / len(propositions)

    def _compute_ivc_r(
        self,
        *,
        cierre_rate: float,
        continuity_score: float,
        intervention_precision: float,
        proposition_diversity: float,
        wall_time_ms: float,
    ) -> float:
        eps = 1e-6
        cierre = min(max(cierre_rate, 0.0), 1.0)
        continuity = min(max(continuity_score, 0.0), 1.0)
        precision = max(intervention_precision, 0.0)
        diversity = min(max(proposition_diversity / 4.0, 0.0), 1.0)
        time_norm = max(wall_time_ms / 50.0, eps)

        ivc_log = (
            0.35 * math.log(cierre + eps)
            + 0.25 * math.log(continuity + eps)
            + 0.20 * math.log(precision + eps)
            + 0.10 * math.log(diversity + eps)
            - 0.10 * math.log(time_norm + eps)
        )
        return math.exp(ivc_log)

    def _persist_jsonl(self, path: Path, rows: List[Dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=True, default=str) + "\n")

    def _expected_spatial_complexity(self, *, topology_label: str) -> float:
        mapping = {
            "uniform": 0.12,
            "hotspot_center": 0.78,
            "gradient_ns": 0.62,
            "gradient_ew": 0.62,
            "checkerboard": 0.85,
        }
        return float(mapping.get((topology_label or "uniform").lower(), 0.22))
