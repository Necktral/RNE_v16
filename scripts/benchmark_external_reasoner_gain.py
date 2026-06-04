#!/usr/bin/env python3
"""Benchmark lab-only para medir ganancia del razonador externo.

No usa ScenarioEpisodeRunner: la decision nominal del runner ocurre antes del
scheduler. Este benchmark compara directamente la decision core del escenario
contra una decision asesorada por EXT_OPEN_THINKER.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime.reasoning.external_models.gating import (
    ExternalReasonerGate,
    ExternalReasonerGateDecision,
    ExternalReasonerGateInput,
)
from runtime.reasoning.families import ext_open_thinker
from runtime.reasoning.scheduler_meta.family_profiles import (
    EXT_OPEN_THINKER_ADMISSION,
    validate_external_reasoner_admission,
)
from runtime.world.grid_thermal_scenario import GridThermalScenario
from tests.benchmarks.metrics_cognitive_quality import compute_all_cognitive_metrics
from tests.benchmarks.metrics_ivc_r import compute_ivc_r_from_episode


GATED_PROFILE = "core_plus_external_reasoner_gated_v1"
CORE_PROFILE = "core_only"
PROFILES = [
    CORE_PROFILE,
    GATED_PROFILE,
]
BLOCKED_EXTERNAL_PROFILES = {
    "core_plus_external_reasoner",
    "core_plus_external_reasoner_guarded",
    "core_plus_external_reasoner_guarded_always_in_target_regimes",
}
EXTERNAL_PROFILES = {GATED_PROFILE}
GUARDED_PROFILES = {GATED_PROFILE}
CRITICAL_REGIMES = {
    "heterogeneous_warning",
    "viability_edge",
    "causal_counterfactual_conflict",
}


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else 0.0


def _std(values: Iterable[float]) -> float:
    values = list(values)
    if len(values) < 2:
        return 0.0
    return float(statistics.pstdev(values))


def _p95(values: Iterable[float]) -> float:
    values = sorted(values)
    if not values:
        return 0.0
    index = int(round((len(values) - 1) * 0.95))
    return float(values[min(max(index, 0), len(values) - 1)])


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def regime_specs() -> Dict[str, Dict[str, Any]]:
    return {
        "heterogeneous_warning": {
            "scenario_params": {
                "grid_size": 5,
                "topology": "checkerboard",
                "topology_params": {"delta": 0.12},
                "initial_temperature": 0.88,
                "alarm_threshold": 0.85,
                "cooling_effect": 0.06,
            }
        },
        "viability_edge": {
            "scenario_params": {
                "grid_size": 5,
                "topology": "hotspot_center",
                "topology_params": {"hotspot_temp": 0.99, "hotspot_radius": 1},
                "initial_temperature": 0.90,
                "alarm_threshold": 0.90,
                "cooling_effect": 0.04,
            }
        },
        "causal_counterfactual_conflict": {
            "scenario_params": {
                "grid_size": 5,
                "topology": "uniform",
                "initial_temperature": 0.88,
                "alarm_threshold": 0.85,
                "cooling_effect": 0.07,
            }
        },
        "homogeneous_safe": {
            "scenario_params": {
                "grid_size": 5,
                "topology": "uniform",
                "initial_temperature": 0.62,
                "alarm_threshold": 0.85,
                "cooling_effect": 0.07,
            }
        },
    }


def build_scenario(regime: str) -> GridThermalScenario:
    specs = regime_specs()
    if regime not in specs:
        raise ValueError(f"unknown_regime:{regime}")
    return GridThermalScenario(**specs[regime]["scenario_params"])


def transition_viability_margin(scenario: GridThermalScenario, transition_dict: Mapping[str, Any]) -> float:
    temp = _safe_float(
        transition_dict.get("global_temp_mean", transition_dict.get("world_level")),
        default=0.0,
    )
    return float(scenario.config.alarm_threshold - temp)


def transition_temperature(transition_dict: Mapping[str, Any]) -> float:
    return _safe_float(transition_dict.get("global_temp_mean", transition_dict.get("world_level")), 0.0)


def opposite_intervention(intervention: str) -> str:
    if intervention == "activate_cooling":
        return "deactivate_cooling"
    return "activate_cooling"


def compute_ioc_proxy(
    *,
    ivc_r: float,
    intervention_precision: float,
    viability_margin: float,
    spatial_information_usage: float | None,
    spatial_applicable: bool,
) -> float:
    components = {
        "ivc_r": (0.45, max(_safe_float(ivc_r), 0.0)),
        "intervention_precision": (0.25, max(_safe_float(intervention_precision), 0.0)),
        "viability_margin": (0.20, max(_safe_float(viability_margin), 0.0)),
    }
    if spatial_applicable and spatial_information_usage is not None:
        components["spatial_information_usage"] = (
            0.10,
            max(_safe_float(spatial_information_usage), 0.0),
        )
    total = sum(weight for weight, _ in components.values())
    return sum((weight / total) * value for weight, value in components.values()) if total > 0 else 0.0


def metric_bundle(
    *,
    scenario: GridThermalScenario,
    observation_dict: Mapping[str, Any],
    transition_dict: Mapping[str, Any],
    wall_time_ms: float,
    regime: str,
) -> Dict[str, Any]:
    cognitive_metrics = compute_all_cognitive_metrics(
        {
            "observation": dict(observation_dict),
            "updated_world": dict(transition_dict),
        }
    )
    precision = _safe_float(cognitive_metrics.get("intervention_precision"), 0.0)
    viability_margin = transition_viability_margin(scenario, transition_dict)
    initial_temp = _safe_float(
        observation_dict.get("global_temp_mean", observation_dict.get("world_level")),
        0.0,
    )
    final_temp = transition_temperature(transition_dict)
    success_rate = 1.0 if viability_margin >= 0.0 else 0.0
    closure_stable = bool(success_rate >= 1.0 or final_temp < initial_temp)
    continuity_score = max(0.0, min(1.0, 1.0 - max(final_temp - initial_temp, 0.0)))
    ivc_input = {
        "cierre_rate": success_rate,
        "continuity_score": continuity_score,
        "intervention_precision": max(precision, 0.0),
        "proposition_diversity": _safe_float(cognitive_metrics.get("proposition_diversity"), 0.0),
        "wall_time_ms": max(float(wall_time_ms), 1.0),
    }
    ivc = compute_ivc_r_from_episode(ivc_input)
    spatial_usage = cognitive_metrics.get("spatial_information_usage")
    ioc_proxy = compute_ioc_proxy(
        ivc_r=_safe_float(ivc.get("ivc_r"), 0.0),
        intervention_precision=precision,
        viability_margin=viability_margin,
        spatial_information_usage=None if spatial_usage is None else _safe_float(spatial_usage),
        spatial_applicable=regime != "homogeneous_safe",
    )
    return {
        "ivc_r": _safe_float(ivc.get("ivc_r"), 0.0),
        "ivc_r_log": _safe_float(ivc.get("ivc_r_log"), 0.0),
        "intervention_precision": precision,
        "proposition_diversity": _safe_float(cognitive_metrics.get("proposition_diversity"), 0.0),
        "spatial_information_usage": None if spatial_usage is None else _safe_float(spatial_usage),
        "viability_margin": viability_margin,
        "success_rate": success_rate,
        "closure_stable": closure_stable,
        "ioc_proxy": ioc_proxy,
    }


@dataclass(frozen=True)
class GuardDecision:
    accepted: bool
    reason: str


def guard_external_choice(
    *,
    allowed_interventions: List[str],
    core_intervention: str,
    recommended_intervention: str | None,
    core_metrics: Mapping[str, Any],
    candidate_metrics: Mapping[str, Any] | None,
    external_ok: bool = True,
    schema_validated: bool = True,
    confidence_proxy: float = 1.0,
    confidence_threshold: float = 0.55,
    claim: str = "",
    reasoning_summary: str = "",
) -> GuardDecision:
    if not external_ok:
        return GuardDecision(False, "external_not_ok")
    if not schema_validated:
        return GuardDecision(False, "schema_not_validated")
    if _safe_float(confidence_proxy, 0.0) < confidence_threshold:
        return GuardDecision(False, "confidence_below_threshold")
    if not recommended_intervention:
        return GuardDecision(False, "no_recommendation")
    if recommended_intervention not in allowed_interventions:
        return GuardDecision(False, "invalid_intervention")
    evidence_text = f"{claim} {reasoning_summary}".strip().lower()
    if recommended_intervention == "deactivate_cooling" and (
        "activate cooling is better" in evidence_text
        or "activating cooling is better" in evidence_text
        or "activating cooling would lower" in evidence_text
    ):
        return GuardDecision(False, "text_recommendation_contradiction")
    if recommended_intervention == "activate_cooling" and (
        "deactivate cooling is better" in evidence_text
        or "deactivating cooling is better" in evidence_text
    ):
        return GuardDecision(False, "text_recommendation_contradiction")
    if candidate_metrics is None:
        return GuardDecision(False, "missing_candidate_metrics")
    if recommended_intervention == core_intervention:
        return GuardDecision(True, "same_as_core")

    core_margin = _safe_float(core_metrics.get("viability_margin"), 0.0)
    candidate_margin = _safe_float(candidate_metrics.get("viability_margin"), 0.0)
    if candidate_margin + 1e-9 < core_margin:
        return GuardDecision(False, "viability_regression")

    core_precision = _safe_float(core_metrics.get("intervention_precision"), 0.0)
    candidate_precision = _safe_float(candidate_metrics.get("intervention_precision"), 0.0)
    if candidate_precision + 1e-9 < core_precision:
        return GuardDecision(False, "precision_regression")

    if candidate_metrics.get("closure_stable") is False and core_metrics.get("closure_stable") is True:
        return GuardDecision(False, "closure_regression")

    return GuardDecision(True, "guard_passed")


def build_external_state(
    *,
    scenario: GridThermalScenario,
    regime: str,
    observation: Any,
    observation_dict: Mapping[str, Any],
    core_intervention: str,
    core_transition_dict: Mapping[str, Any],
    counterfactual_transition_dict: Mapping[str, Any],
    backend: str | None,
    allow_cpu_fallback: bool,
    external_client: Any | None,
) -> Dict[str, Any]:
    state = {
        "observation": dict(observation_dict),
        "intervention": core_intervention,
        "updated_world": dict(core_transition_dict),
        "counterfactual": dict(counterfactual_transition_dict),
        "formula": scenario.get_formula(observation),
        "regime_hint": regime,
        "scenario_metadata": {
            "scenario": scenario.config.name,
            "interventions": list(scenario.config.interventions),
            "main_variable": scenario.config.main_variable,
            "alarm_threshold": scenario.config.alarm_threshold,
        },
        "_meta": {"selected_family": ext_open_thinker.FAMILY_ID},
        "external_reasoner_allow_cpu_fallback": allow_cpu_fallback,
    }
    if backend:
        state["external_reasoner_backend"] = backend
    if external_client is not None:
        state["_external_reasoner_client"] = external_client
    return state


def run_episode(
    *,
    profile: str,
    regime: str,
    episode_index: int,
    external_input: float,
    backend: str | None = None,
    allow_cpu_fallback: bool = False,
    confidence_threshold: float = 0.55,
    external_client: Any | None = None,
    external_gate: ExternalReasonerGate | None = None,
    gate_history: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    if profile in BLOCKED_EXTERNAL_PROFILES:
        raise ValueError(
            "external_reasoner_profile_not_admitted:"
            f"{profile}:requires_{GATED_PROFILE}"
        )
    if profile not in set(PROFILES):
        raise ValueError(f"unknown_profile:{profile}")
    scenario = build_scenario(regime)
    observation = scenario.observe()
    observation_dict = scenario.to_observation_dict(observation)
    allowed_interventions = list(scenario.config.interventions)

    core_intervention = scenario.select_intervention(observation)
    core_transition = scenario.simulate_counterfactual(
        intervention=core_intervention,
        external_input=external_input,
    )
    core_transition_dict = scenario.to_transition_dict(core_transition)
    alternate_intervention = opposite_intervention(core_intervention)
    alternate_transition = scenario.simulate_counterfactual(
        intervention=alternate_intervention,
        external_input=external_input,
    )
    alternate_transition_dict = scenario.to_transition_dict(alternate_transition)
    core_metrics = metric_bundle(
        scenario=scenario,
        observation_dict=observation_dict,
        transition_dict=core_transition_dict,
        wall_time_ms=1.0,
        regime=regime,
    )

    start = time.perf_counter()
    external_result: Dict[str, Any] = {}
    recommendation: str | None = None
    external_used = False
    external_ok = False
    external_latency_s = 0.0
    external_generation_tps = 0.0
    external_schema_validated = False
    external_confidence_proxy = 0.0
    external_claim = ""
    external_reasoning_summary = ""
    guard_decision = GuardDecision(True, "not_guarded")
    gate_decision = ExternalReasonerGateDecision(
        considered=False,
        called=False,
        reason="not_external_profile",
    )

    if profile in EXTERNAL_PROFILES:
        admission = validate_external_reasoner_admission(
            profile_name=profile,
            regime=regime,
            gate_present=True,
            guard_present=profile in GUARDED_PROFILES,
            schema_present=True,
            fallback_present=True,
        )
        if not admission.allowed:
            gate_decision = ExternalReasonerGateDecision(
                considered=True,
                called=False,
                reason="skip",
                skip_reason=admission.reason,
                expected_failure_mode="external_reasoner_not_admitted_for_context",
                trigger_signals={
                    "admission_allowed": False,
                    "validated_regime": regime in set(EXT_OPEN_THINKER_ADMISSION.validated_regimes),
                    "admitted_profile": profile in set(EXT_OPEN_THINKER_ADMISSION.admitted_lab_profiles),
                },
            )
        else:
            gate = external_gate or ExternalReasonerGate()
            gate_decision = gate.evaluate(
                ExternalReasonerGateInput(
                    regime=regime,
                    core_intervention=core_intervention,
                    causal_recommended_intervention=core_intervention,
                    counterfactual_recommended_intervention=None,
                    core_metrics=core_metrics,
                    history=gate_history or {},
                )
            )

    if profile in EXTERNAL_PROFILES and gate_decision.called:
        external_state = build_external_state(
            scenario=scenario,
            regime=regime,
            observation=observation,
            observation_dict=observation_dict,
            core_intervention=core_intervention,
            core_transition_dict=core_transition_dict,
            counterfactual_transition_dict=alternate_transition_dict,
            backend=backend,
            allow_cpu_fallback=allow_cpu_fallback,
            external_client=external_client,
        )
        external_result = ext_open_thinker.execute(external_state)
        state_delta = external_result.get("state_delta") or {}
        external_used = bool(state_delta.get("external_reasoner_used"))
        external_ok = bool(state_delta.get("external_reasoner_ok"))
        external_latency_s = _safe_float(state_delta.get("external_reasoner_latency_s"), 0.0)
        external_generation_tps = _safe_float(state_delta.get("external_reasoner_generation_tps"), 0.0)
        external_schema_validated = bool(state_delta.get("external_reasoner_schema_validated"))
        external_confidence_proxy = _safe_float(state_delta.get("external_reasoner_confidence_proxy"), 0.0)
        external_claim = str(state_delta.get("external_reasoner_claim") or "")
        external_reasoning_summary = str(state_delta.get("external_reasoner_reasoning_summary") or "")
        recommendation = str(state_delta.get("external_reasoner_recommended_intervention") or "").strip() or None

    candidate_transition_dict: Mapping[str, Any] | None = None
    candidate_metrics: Dict[str, Any] | None = None
    selected_intervention = core_intervention
    decision_source = "core"

    initial_guard = GuardDecision(True, "schema_confidence_passed")
    if profile in EXTERNAL_PROFILES and gate_decision.called:
        initial_guard = guard_external_choice(
            allowed_interventions=allowed_interventions,
            core_intervention=core_intervention,
            recommended_intervention=recommendation,
            core_metrics=core_metrics,
            candidate_metrics=core_metrics,
            external_ok=external_ok,
            schema_validated=external_schema_validated,
            confidence_proxy=external_confidence_proxy,
            confidence_threshold=confidence_threshold,
            claim=external_claim,
            reasoning_summary=external_reasoning_summary,
        )

    if profile in EXTERNAL_PROFILES and gate_decision.called and initial_guard.accepted:
        candidate_transition = scenario.simulate_counterfactual(
            intervention=recommendation,
            external_input=external_input,
        )
        candidate_transition_dict = scenario.to_transition_dict(candidate_transition)
        candidate_metrics = metric_bundle(
            scenario=scenario,
            observation_dict=observation_dict,
            transition_dict=candidate_transition_dict,
            wall_time_ms=1.0 + (external_latency_s * 1000.0),
            regime=regime,
        )
        if profile in GUARDED_PROFILES:
            guard_decision = guard_external_choice(
                allowed_interventions=allowed_interventions,
                core_intervention=core_intervention,
                recommended_intervention=recommendation,
                core_metrics=core_metrics,
                candidate_metrics=candidate_metrics,
                external_ok=external_ok,
                schema_validated=external_schema_validated,
                confidence_proxy=external_confidence_proxy,
                confidence_threshold=confidence_threshold,
                claim=external_claim,
                reasoning_summary=external_reasoning_summary,
            )
            if guard_decision.accepted:
                selected_intervention = recommendation
                decision_source = "external_guarded"
        else:
            guard_decision = guard_external_choice(
                allowed_interventions=allowed_interventions,
                core_intervention=core_intervention,
                recommended_intervention=recommendation,
                core_metrics=core_metrics,
                candidate_metrics=candidate_metrics,
                external_ok=external_ok,
                schema_validated=external_schema_validated,
                confidence_proxy=external_confidence_proxy,
                confidence_threshold=confidence_threshold,
                claim=external_claim,
                reasoning_summary=external_reasoning_summary,
            )
            if guard_decision.accepted:
                selected_intervention = recommendation
                decision_source = "external"
    elif profile in GUARDED_PROFILES and gate_decision.called:
        guard_decision = initial_guard
    elif profile in EXTERNAL_PROFILES and gate_decision.called:
        guard_decision = initial_guard
    elif profile in EXTERNAL_PROFILES and not gate_decision.called:
        guard_decision = GuardDecision(False, gate_decision.skip_reason or "gate_skipped")

    selected_transition = scenario.simulate_counterfactual(
        intervention=selected_intervention,
        external_input=external_input,
    )
    selected_transition_dict = scenario.to_transition_dict(selected_transition)
    elapsed_ms = max((time.perf_counter() - start) * 1000.0, 1.0)
    selected_metrics = metric_bundle(
        scenario=scenario,
        observation_dict=observation_dict,
        transition_dict=selected_transition_dict,
        wall_time_ms=elapsed_ms,
        regime=regime,
    )
    contribution_proxy = selected_metrics["ioc_proxy"] - core_metrics["ioc_proxy"]
    delta_ivc_r_vs_core = selected_metrics["ivc_r"] - core_metrics["ivc_r"]
    corrected_core_failure = bool(
        gate_decision.called
        and selected_metrics["success_rate"] > core_metrics["success_rate"]
        and selected_metrics["closure_stable"]
    )
    avoided_bad_decision = bool(
        gate_decision.called
        and (
            core_metrics["success_rate"] <= 0.0
            or not core_metrics["closure_stable"]
            or core_metrics["viability_margin"] < 0.0
            or core_metrics["intervention_precision"] <= 0.0
        )
        and selected_metrics["success_rate"] >= core_metrics["success_rate"]
        and selected_metrics["closure_stable"]
    )

    return {
        "profile": profile,
        "regime": regime,
        "episode_index": episode_index,
        "scenario": scenario.config.name,
        "external_input": external_input,
        "core_intervention": core_intervention,
        "alternate_intervention": alternate_intervention,
        "external_recommended_intervention": recommendation,
        "selected_intervention": selected_intervention,
        "decision_source": decision_source,
        "guard_accepted": guard_decision.accepted,
        "guard_reason": guard_decision.reason,
        "external_reasoner_considered": bool(gate_decision.considered),
        "external_reasoner_called": bool(gate_decision.called),
        "external_reasoner_skip_reason": gate_decision.skip_reason,
        "external_reasoner_gate_reason": gate_decision.reason if gate_decision.called else None,
        "external_reasoner_expected_failure_mode": gate_decision.expected_failure_mode,
        "external_reasoner_gate_trigger_signals": dict(gate_decision.trigger_signals),
        "external_reasoner_conflict_trigger": bool(
            gate_decision.trigger_signals.get("explicit_conflict_regime")
            or gate_decision.trigger_signals.get("structural_conflict")
        ),
        "external_reasoner_used": external_used,
        "external_reasoner_ok": external_ok,
        "external_reasoner_status": external_result.get("status"),
        "external_reasoner_failure_mode": external_result.get("failure_mode"),
        "external_reasoner_schema_validated": external_schema_validated,
        "external_reasoner_confidence_proxy": external_confidence_proxy,
        "external_reasoner_latency_s": external_latency_s,
        "external_reasoner_generation_tps": external_generation_tps,
        "external_reasoner_contribution_proxy": contribution_proxy,
        "external_reasoner_net_ivc_delta": delta_ivc_r_vs_core,
        "external_reasoner_corrected_core_failure": corrected_core_failure,
        "external_reasoner_avoided_bad_decision": avoided_bad_decision,
        "external_used": external_used,
        "external_accepted": bool(
            profile in EXTERNAL_PROFILES
            and gate_decision.called
            and guard_decision.accepted
            and decision_source.startswith("external")
        ),
        "external_rejected_reason": (
            None
            if decision_source.startswith("external")
            else (gate_decision.skip_reason if not gate_decision.called else guard_decision.reason)
        ),
        "observation": _jsonable(observation_dict),
        "core_transition": _jsonable(core_transition_dict),
        "alternate_transition": _jsonable(alternate_transition_dict),
        "selected_transition": _jsonable(selected_transition_dict),
        "external_result_excerpt": _jsonable((external_result.get("state_delta") or {})),
        **selected_metrics,
    }


def aggregate_rows(rows: List[Mapping[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    for row in rows:
        key = f"{row['profile']}::{row['regime']}"
        grouped.setdefault(key, []).append(row)

    summary_rows: List[Dict[str, Any]] = []
    for key, block in sorted(grouped.items()):
        profile, regime = key.split("::", 1)
        metrics = {
            "ivc_r",
            "intervention_precision",
            "viability_margin",
            "success_rate",
            "external_reasoner_latency_s",
            "external_reasoner_generation_tps",
            "external_reasoner_contribution_proxy",
            "external_reasoner_net_ivc_delta",
        }
        considered_count = sum(1 for row in block if row.get("external_reasoner_considered"))
        called_rows = [row for row in block if row.get("external_reasoner_called")]
        skipped_rows = [
            row for row in block
            if row.get("external_reasoner_considered") and not row.get("external_reasoner_called")
        ]
        accepted_rows = [row for row in called_rows if row.get("external_accepted")]
        rejected_rows = [row for row in called_rows if not row.get("external_accepted")]
        corrected_rows = [row for row in called_rows if row.get("external_reasoner_corrected_core_failure")]
        avoided_rows = [row for row in called_rows if row.get("external_reasoner_avoided_bad_decision")]
        skip_reason_counts: Dict[str, int] = {}
        for row in skipped_rows:
            reason = str(row.get("external_reasoner_skip_reason") or "unknown")
            skip_reason_counts[reason] = skip_reason_counts.get(reason, 0) + 1
        called_count = len(called_rows)
        out = {
            "profile": profile,
            "regime": regime,
            "episodes": len(block),
            "closure_stable_rate": _mean(1.0 if row.get("closure_stable") else 0.0 for row in block),
            "guard_pass_rate": len(accepted_rows) / called_count if called_count else 0.0,
            "guard_reject_rate": len(rejected_rows) / called_count if called_count else 0.0,
            "selected_intervention_changed_rate": _mean(
                1.0 if row.get("selected_intervention") != row.get("core_intervention") else 0.0
                for row in block
            ),
            "external_reasoner_considered_rate": considered_count / len(block) if block else 0.0,
            "external_reasoner_call_rate": called_count / len(block) if block else 0.0,
            "external_reasoner_skip_rate": len(skipped_rows) / considered_count if considered_count else 0.0,
            "external_reasoner_skip_reason_counts": skip_reason_counts,
            "external_reasoner_conflict_trigger_rate": _mean(
                1.0 if row.get("external_reasoner_conflict_trigger") else 0.0 for row in block
            ),
            "external_reasoner_used_rate": called_count / len(block) if block else 0.0,
            "external_reasoner_success_rate": (
                sum(1 for row in called_rows if row.get("external_reasoner_ok")) / called_count
                if called_count
                else 0.0
            ),
            "external_reasoner_schema_validated_rate": (
                sum(1 for row in called_rows if row.get("external_reasoner_schema_validated")) / called_count
                if called_count
                else 0.0
            ),
            "external_reasoner_accepted_rate": len(accepted_rows) / called_count if called_count else 0.0,
            "external_reasoner_corrected_core_failure_rate": (
                len(corrected_rows) / called_count if called_count else 0.0
            ),
            "avoided_bad_decision_rate": len(avoided_rows) / called_count if called_count else 0.0,
            "external_reasoner_cost_per_corrected_failure": (
                sum(_safe_float(row.get("external_reasoner_latency_s"), 0.0) for row in called_rows)
                / len(corrected_rows)
                if corrected_rows
                else 0.0
            ),
            "external_reasoner_latency_s_p95": _p95(
                _safe_float(row.get("external_reasoner_latency_s"), 0.0) for row in called_rows
            ),
            "external_reasoner_net_ivc_delta_when_called": _mean(
                _safe_float(row.get("external_reasoner_net_ivc_delta"), 0.0) for row in called_rows
            ),
            "external_reasoner_net_ivc_delta_when_skipped": _mean(
                _safe_float(row.get("external_reasoner_net_ivc_delta"), 0.0) for row in skipped_rows
            ),
            "decision_external_rate": _mean(1.0 if str(row.get("decision_source", "")).startswith("external") else 0.0 for row in block),
        }
        for metric in metrics:
            values = [_safe_float(row.get(metric), 0.0) for row in block]
            out[f"{metric}_mean"] = _mean(values)
            out[f"{metric}_std"] = _std(values)
        summary_rows.append(out)

    deltas: List[Dict[str, Any]] = []
    by_pair = {(row["profile"], row["regime"]): row for row in summary_rows}
    for profile in EXTERNAL_PROFILES:
        for regime in regime_specs():
            core = by_pair.get(("core_only", regime))
            candidate = by_pair.get((profile, regime))
            if not core or not candidate:
                continue
            deltas.append(
                {
                    "profile": profile,
                    "regime": regime,
                    "core_ivc_r": core["ivc_r_mean"],
                    "candidate_ivc_r": candidate["ivc_r_mean"],
                    "delta_ivc_r": candidate["ivc_r_mean"] - core["ivc_r_mean"],
                    "core_intervention_precision": core["intervention_precision_mean"],
                    "candidate_intervention_precision": candidate["intervention_precision_mean"],
                    "delta_intervention_precision": (
                        candidate["intervention_precision_mean"] - core["intervention_precision_mean"]
                    ),
                    "core_viability_margin": core["viability_margin_mean"],
                    "candidate_viability_margin": candidate["viability_margin_mean"],
                    "delta_viability_margin": candidate["viability_margin_mean"] - core["viability_margin_mean"],
                    "core_success_rate": core["success_rate_mean"],
                    "candidate_success_rate": candidate["success_rate_mean"],
                    "delta_success_rate": candidate["success_rate_mean"] - core["success_rate_mean"],
                    "core_closure_stable_rate": core["closure_stable_rate"],
                    "closure_stable_rate": candidate["closure_stable_rate"],
                    "delta_closure_stable_rate": candidate["closure_stable_rate"] - core["closure_stable_rate"],
                    "external_reasoner_success_rate": candidate["external_reasoner_success_rate"],
                    "external_reasoner_schema_validated_rate": candidate["external_reasoner_schema_validated_rate"],
                    "external_reasoner_considered_rate": candidate["external_reasoner_considered_rate"],
                    "external_reasoner_call_rate": candidate["external_reasoner_call_rate"],
                    "external_reasoner_skip_rate": candidate["external_reasoner_skip_rate"],
                    "guard_pass_rate": candidate["guard_pass_rate"],
                    "guard_reject_rate": candidate["guard_reject_rate"],
                    "external_reasoner_accepted_rate": candidate["external_reasoner_accepted_rate"],
                    "external_reasoner_corrected_core_failure_rate": candidate[
                        "external_reasoner_corrected_core_failure_rate"
                    ],
                    "avoided_bad_decision_rate": candidate["avoided_bad_decision_rate"],
                    "external_reasoner_cost_per_corrected_failure": candidate[
                        "external_reasoner_cost_per_corrected_failure"
                    ],
                    "selected_intervention_changed_rate": candidate["selected_intervention_changed_rate"],
                    "external_reasoner_latency_s_mean": candidate["external_reasoner_latency_s_mean"],
                    "external_reasoner_latency_s_p95": candidate["external_reasoner_latency_s_p95"],
                    "external_reasoner_generation_tps_mean": candidate["external_reasoner_generation_tps_mean"],
                    "external_reasoner_net_ivc_delta_when_called": candidate[
                        "external_reasoner_net_ivc_delta_when_called"
                    ],
                    "external_reasoner_net_ivc_delta_when_skipped": candidate[
                        "external_reasoner_net_ivc_delta_when_skipped"
                    ],
                    "external_reasoner_contribution_proxy_mean": (
                        candidate["external_reasoner_contribution_proxy_mean"]
                    ),
                }
            )
    return {
        "rows": summary_rows,
        "deltas_vs_core": deltas,
        "dictamen": decide_verdict(summary_rows, deltas),
        "microbenchmark_verdict": decide_microbenchmark_verdict(summary_rows, deltas),
        "gating_verdict": decide_gating_verdict(summary_rows, deltas),
        "stopping_rules": evaluate_stopping_rules(summary_rows, deltas),
        "gating_stopping_rules": evaluate_gating_stopping_rules(summary_rows, deltas),
    }


def decide_verdict(summary_rows: List[Mapping[str, Any]], deltas: List[Mapping[str, Any]]) -> str:
    external_rows = [row for row in summary_rows if row.get("profile") in EXTERNAL_PROFILES]
    if external_rows and max(_safe_float(row.get("external_reasoner_success_rate"), 0.0) for row in external_rows) <= 0.0:
        return "external_reasoner_no_operativo_en_runtime"

    for delta in deltas:
        if delta.get("regime") not in CRITICAL_REGIMES:
            continue
        improves = (
            _safe_float(delta.get("delta_ivc_r"), 0.0) > 0.0
            or _safe_float(delta.get("delta_intervention_precision"), 0.0) > 0.0
        )
        if improves and _safe_float(delta.get("closure_stable_rate"), 0.0) >= 1.0:
            return "external_reasoner_aporta_ganancia_cognitiva"

    if any(_safe_float(row.get("external_reasoner_success_rate"), 0.0) > 0.0 for row in external_rows):
        return "external_reasoner_aporta_solo_estructura"

    return "external_reasoner_no_aporta"


def evaluate_stopping_rules(
    summary_rows: List[Mapping[str, Any]],
    deltas: List[Mapping[str, Any]],
) -> Dict[str, Any]:
    external_rows = [row for row in summary_rows if row.get("profile") in EXTERNAL_PROFILES]
    called_external_rows = [
        row for row in external_rows
        if _safe_float(row.get("external_reasoner_call_rate"), 0.0) > 0.0
    ]
    if not called_external_rows:
        return {"should_abort": False, "violations": []}

    violations: List[str] = []
    min_schema = min(_safe_float(row.get("external_reasoner_schema_validated_rate"), 0.0) for row in called_external_rows)
    min_ok = min(_safe_float(row.get("external_reasoner_success_rate"), 0.0) for row in called_external_rows)
    max_guard_reject = max(_safe_float(row.get("guard_reject_rate"), 0.0) for row in called_external_rows)
    max_latency = max(_safe_float(row.get("external_reasoner_latency_s_mean"), 0.0) for row in called_external_rows)
    if min_schema < 0.90:
        violations.append("external_reasoner_schema_validated_rate_lt_0.90")
    if min_ok < 0.90:
        violations.append("external_reasoner_ok_rate_lt_0.90")
    if max_guard_reject > 0.50:
        violations.append("guard_reject_rate_gt_0.50")
    if max_latency > 100.0:
        violations.append("latency_mean_when_called_gt_100s")
    for delta in deltas:
        if _safe_float(delta.get("delta_closure_stable_rate"), 0.0) < -1e-9:
            violations.append(f"closure_drop:{delta.get('regime')}:{delta.get('profile')}")
    guarded_deltas = [
        delta for delta in deltas
        if delta.get("profile") == GATED_PROFILE
        and delta.get("regime") in CRITICAL_REGIMES
    ]
    if len(guarded_deltas) >= 3 and all(_safe_float(delta.get("delta_ivc_r"), 0.0) < 0.0 for delta in guarded_deltas):
        violations.append("ivc_r_worse_in_all_required_regimes")
    return {
        "should_abort": bool(violations),
        "violations": sorted(set(violations)),
        "min_schema_validated_rate": min_schema,
        "min_external_ok_rate": min_ok,
        "max_guard_reject_rate": max_guard_reject,
        "max_latency_s_mean_when_called": max_latency,
    }


def evaluate_gating_stopping_rules(
    summary_rows: List[Mapping[str, Any]],
    deltas: List[Mapping[str, Any]],
) -> Dict[str, Any]:
    by_pair = {(row.get("profile"), row.get("regime")): row for row in summary_rows}
    gated_rows = [row for row in summary_rows if row.get("profile") == GATED_PROFILE]
    called_rows = [
        row for row in gated_rows
        if _safe_float(row.get("external_reasoner_call_rate"), 0.0) > 0.0
    ]
    violations: List[str] = []
    if not called_rows:
        violations.append("gated_policy_made_no_external_calls")
        min_schema = 0.0
        min_ok = 0.0
        max_guard_reject = 0.0
        max_latency = 0.0
    else:
        min_schema = min(_safe_float(row.get("external_reasoner_schema_validated_rate"), 0.0) for row in called_rows)
        min_ok = min(_safe_float(row.get("external_reasoner_success_rate"), 0.0) for row in called_rows)
        max_guard_reject = max(_safe_float(row.get("guard_reject_rate"), 0.0) for row in called_rows)
        max_latency = max(_safe_float(row.get("external_reasoner_latency_s_mean"), 0.0) for row in called_rows)
        if min_schema < 0.90:
            violations.append("gated_schema_validated_rate_lt_0.90")
        if min_ok < 0.90:
            violations.append("gated_external_ok_rate_lt_0.90")
        if max_guard_reject > 0.50:
            violations.append("gated_guard_reject_rate_gt_0.50")
        if max_latency > 100.0:
            violations.append("gated_latency_mean_when_called_gt_100s")

    causal = by_pair.get((GATED_PROFILE, "causal_counterfactual_conflict"), {})
    viability = by_pair.get((GATED_PROFILE, "viability_edge"), {})
    heterogeneous = by_pair.get((GATED_PROFILE, "heterogeneous_warning"), {})
    causal_call_rate = _safe_float(causal.get("external_reasoner_call_rate"), 0.0)
    non_causal_call_rates = [
        _safe_float(viability.get("external_reasoner_call_rate"), 0.0),
        _safe_float(heterogeneous.get("external_reasoner_call_rate"), 0.0),
    ]
    if causal_call_rate < 0.90:
        violations.append("gated_no_call_in_causal_counterfactual_conflict")
    if all(rate >= 0.90 for rate in [causal_call_rate] + non_causal_call_rates):
        violations.append("gated_calls_indiscriminately_all_regimes")
    if any(rate > 0.25 for rate in non_causal_call_rates):
        violations.append("gated_calls_non_causal_regime_gt_0.25")
    for delta in deltas:
        if delta.get("profile") != GATED_PROFILE:
            continue
        if _safe_float(delta.get("delta_closure_stable_rate"), 0.0) < -1e-9:
            violations.append(f"gated_closure_drop:{delta.get('regime')}")
    return {
        "should_abort": bool(violations),
        "violations": sorted(set(violations)),
        "min_schema_validated_rate_when_called": min_schema,
        "min_external_ok_rate_when_called": min_ok,
        "max_guard_reject_rate_when_called": max_guard_reject,
        "max_latency_s_mean_when_called": max_latency,
        "causal_counterfactual_conflict_call_rate": causal_call_rate,
        "non_causal_call_rates": {
            "viability_edge": non_causal_call_rates[0],
            "heterogeneous_warning": non_causal_call_rates[1],
        },
    }


def decide_microbenchmark_verdict(
    summary_rows: List[Mapping[str, Any]],
    deltas: List[Mapping[str, Any]],
) -> str:
    guarded = [
        delta for delta in deltas
        if delta.get("profile") == GATED_PROFILE
        and delta.get("regime") in CRITICAL_REGIMES
    ]
    if not guarded:
        return "sin_ganancia"

    closure_or_success_drop = any(
        _safe_float(delta.get("delta_closure_stable_rate"), 0.0) < -1e-9
        or _safe_float(delta.get("delta_success_rate"), 0.0) < -1e-9
        for delta in guarded
    )
    relevant_viability_drop = any(
        _safe_float(delta.get("delta_viability_margin"), 0.0) < -0.01
        for delta in guarded
    )
    if closure_or_success_drop or relevant_viability_drop:
        return "external_reasoner_perjudica"
    if all(_safe_float(delta.get("delta_ivc_r"), 0.0) < 0.0 for delta in guarded):
        return "external_reasoner_perjudica"

    positive = [
        delta for delta in guarded
        if (
            _safe_float(delta.get("delta_ivc_r"), 0.0) > 0.0
            and _safe_float(delta.get("guard_pass_rate"), 0.0) > 0.0
            and _safe_float(delta.get("selected_intervention_changed_rate"), 0.0) > 0.0
        )
    ]
    if not positive:
        return "sin_ganancia"

    clear_positive = [
        delta for delta in positive
        if (
            _safe_float(delta.get("delta_ivc_r"), 0.0) >= 0.01
            or _safe_float(delta.get("delta_intervention_precision"), 0.0) >= 0.01
            or _safe_float(delta.get("delta_viability_margin"), 0.0) >= 0.01
        )
    ]
    return "ganancia_cognitiva_inicial" if clear_positive else "señal_marginal"


def _weighted_profile_mean(
    summary_rows: List[Mapping[str, Any]],
    profile: str,
    metric: str,
    *,
    regimes: set[str] | None = None,
) -> float:
    selected = [
        row for row in summary_rows
        if row.get("profile") == profile and (regimes is None or row.get("regime") in regimes)
    ]
    total_episodes = sum(int(row.get("episodes", 0) or 0) for row in selected)
    if total_episodes <= 0:
        return 0.0
    return sum(_safe_float(row.get(metric), 0.0) * int(row.get("episodes", 0) or 0) for row in selected) / total_episodes


def decide_gating_verdict(
    summary_rows: List[Mapping[str, Any]],
    deltas: List[Mapping[str, Any]],
) -> str:
    required = {
        "causal_counterfactual_conflict",
        "viability_edge",
        "heterogeneous_warning",
    }
    by_pair = {(row.get("profile"), row.get("regime")): row for row in summary_rows}
    gated_rows = [row for row in summary_rows if row.get("profile") == GATED_PROFILE]
    if not gated_rows:
        return "gated_policy_mal_calibrada"

    causal = by_pair.get((GATED_PROFILE, "causal_counterfactual_conflict"), {})
    viability = by_pair.get((GATED_PROFILE, "viability_edge"), {})
    heterogeneous = by_pair.get((GATED_PROFILE, "heterogeneous_warning"), {})
    causal_call_rate = _safe_float(causal.get("external_reasoner_call_rate"), 0.0)
    non_causal_call_rates = [
        _safe_float(viability.get("external_reasoner_call_rate"), 0.0),
        _safe_float(heterogeneous.get("external_reasoner_call_rate"), 0.0),
    ]
    if causal_call_rate < 0.90:
        return "gated_policy_mal_calibrada"
    if all(rate >= 0.90 for rate in [causal_call_rate] + non_causal_call_rates):
        return "gated_policy_mal_calibrada"
    if any(rate > 0.25 for rate in non_causal_call_rates):
        return "gated_policy_mal_calibrada"

    called_rows = [
        row for row in gated_rows
        if _safe_float(row.get("external_reasoner_call_rate"), 0.0) > 0.0
    ]
    if not called_rows:
        return "gated_policy_mal_calibrada"
    if min(_safe_float(row.get("external_reasoner_schema_validated_rate"), 0.0) for row in called_rows) < 0.90:
        return "gated_policy_mal_calibrada"
    if min(_safe_float(row.get("external_reasoner_success_rate"), 0.0) for row in called_rows) < 0.90:
        return "gated_policy_mal_calibrada"
    if max(_safe_float(row.get("guard_reject_rate"), 0.0) for row in called_rows) > 0.50:
        return "gated_policy_mal_calibrada"
    if max(_safe_float(row.get("external_reasoner_latency_s_mean"), 0.0) for row in called_rows) > 100.0:
        return "gated_external_reasoner_demasiado_caro"

    for regime in required:
        core = by_pair.get(("core_only", regime), {})
        gated = by_pair.get((GATED_PROFILE, regime), {})
        if _safe_float(gated.get("closure_stable_rate"), 0.0) + 1e-9 < _safe_float(core.get("closure_stable_rate"), 0.0):
            return "gated_external_reasoner_no_supera_core"
        if _safe_float(gated.get("success_rate_mean"), 0.0) + 1e-9 < _safe_float(core.get("success_rate_mean"), 0.0):
            return "gated_external_reasoner_no_supera_core"

    gated_global_ivc = _weighted_profile_mean(summary_rows, GATED_PROFILE, "ivc_r_mean", regimes=required)
    core_global_ivc = _weighted_profile_mean(summary_rows, "core_only", "ivc_r_mean", regimes=required)
    gated_latency = _weighted_profile_mean(
        summary_rows,
        GATED_PROFILE,
        "external_reasoner_latency_s_mean",
        regimes=required,
    )
    causal_delta = next(
        (
            delta for delta in deltas
            if delta.get("profile") == GATED_PROFILE
            and delta.get("regime") == "causal_counterfactual_conflict"
        ),
        {},
    )
    if _safe_float(causal_delta.get("delta_ivc_r"), 0.0) <= 0.0:
        return "gated_external_reasoner_no_supera_core"
    if gated_global_ivc <= core_global_ivc:
        return "gated_external_reasoner_no_supera_core"
    if gated_latency <= 0.0:
        return "gated_policy_mal_calibrada"
    return "gated_external_reasoner_util_condicionado"


def write_outputs(*, rows: List[Dict[str, Any]], summary: Dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    episodes_path = out_dir / "episodes.jsonl"
    with episodes_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(_jsonable(row), sort_keys=True) + "\n")

    (out_dir / "summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (out_dir / "external_reasoner_gain_report.md").write_text(
        render_report(summary=summary, out_dir=out_dir),
        encoding="utf-8",
    )
    (out_dir / "external_reasoner_microbenchmark_report.md").write_text(
        render_report(summary=summary, out_dir=out_dir),
        encoding="utf-8",
    )
    (out_dir / "external_reasoner_gating_report.md").write_text(
        render_report(summary=summary, out_dir=out_dir),
        encoding="utf-8",
    )
    verdict = {
        "campaign_id": summary.get("campaign_id"),
        "dictamen": summary.get("microbenchmark_verdict", summary.get("dictamen")),
        "legacy_dictamen": summary.get("dictamen"),
        "stopping_rules": summary.get("stopping_rules", {}),
        "deltas_vs_core": summary.get("deltas_vs_core", []),
    }
    (out_dir / "external_reasoner_microbenchmark_verdict.json").write_text(
        json.dumps(_jsonable(verdict), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    gating_verdict = {
        "campaign_id": summary.get("campaign_id"),
        "dictamen": summary.get("gating_verdict"),
        "legacy_dictamen": summary.get("dictamen"),
        "microbenchmark_verdict": summary.get("microbenchmark_verdict"),
        "stopping_rules": summary.get("stopping_rules", {}),
        "gating_stopping_rules": summary.get("gating_stopping_rules", {}),
        "deltas_vs_core": summary.get("deltas_vs_core", []),
    }
    (out_dir / "external_reasoner_gating_verdict.json").write_text(
        json.dumps(_jsonable(gating_verdict), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def render_report(*, summary: Mapping[str, Any], out_dir: Path) -> str:
    lines = [
        "# External Reasoner Gain Report",
        "",
        f"- output_dir: `{out_dir}`",
        f"- dictamen: `{summary.get('dictamen')}`",
        f"- microbenchmark_verdict: `{summary.get('microbenchmark_verdict')}`",
        f"- gating_verdict: `{summary.get('gating_verdict')}`",
        "",
        "| profile | regime | episodes | call_rate | ivc_r | precision | viability | success | closure | ext_ok | schema | guard_pass | guard_reject | changed | latency | ext_tps | contribution |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary.get("rows", []):
        lines.append(
            "| {profile} | {regime} | {episodes} | {call_rate:.3f} | {ivc:.6f} | {precision:.6f} | "
            "{margin:.6f} | {success:.3f} | {closure:.3f} | {ext_ok:.3f} | {schema:.3f} | "
            "{guard_pass:.3f} | {guard_reject:.3f} | {changed:.3f} | {latency:.3f} | {tps:.3f} | {contrib:.6f} |".format(
                profile=row["profile"],
                regime=row["regime"],
                episodes=row["episodes"],
                call_rate=_safe_float(row.get("external_reasoner_call_rate"), 0.0),
                ivc=_safe_float(row.get("ivc_r_mean"), 0.0),
                precision=_safe_float(row.get("intervention_precision_mean"), 0.0),
                margin=_safe_float(row.get("viability_margin_mean"), 0.0),
                success=_safe_float(row.get("success_rate_mean"), 0.0),
                closure=_safe_float(row.get("closure_stable_rate"), 0.0),
                ext_ok=_safe_float(row.get("external_reasoner_success_rate"), 0.0),
                schema=_safe_float(row.get("external_reasoner_schema_validated_rate"), 0.0),
                guard_pass=_safe_float(row.get("guard_pass_rate"), 0.0),
                guard_reject=_safe_float(row.get("guard_reject_rate"), 0.0),
                changed=_safe_float(row.get("selected_intervention_changed_rate"), 0.0),
                latency=_safe_float(row.get("external_reasoner_latency_s_mean"), 0.0),
                tps=_safe_float(row.get("external_reasoner_generation_tps_mean"), 0.0),
                contrib=_safe_float(row.get("external_reasoner_contribution_proxy_mean"), 0.0),
            )
        )
    lines.extend(
        [
            "",
            "## Deltas Vs Core",
            "",
            "| profile | regime | delta_ivc_r | delta_precision | delta_viability | closure | ext_ok |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary.get("deltas_vs_core", []):
        lines.append(
            "| {profile} | {regime} | {delta_ivc:.6f} | {delta_precision:.6f} | "
            "{delta_margin:.6f} | {closure:.3f} | {ext_ok:.3f} |".format(
                profile=row["profile"],
                regime=row["regime"],
                delta_ivc=_safe_float(row.get("delta_ivc_r"), 0.0),
                delta_precision=_safe_float(row.get("delta_intervention_precision"), 0.0),
                delta_margin=_safe_float(row.get("delta_viability_margin"), 0.0),
                closure=_safe_float(row.get("closure_stable_rate"), 0.0),
                ext_ok=_safe_float(row.get("external_reasoner_success_rate"), 0.0),
            )
        )
    lines.append("")
    return "\n".join(lines)


def run_campaign(
    *,
    campaign_id: str,
    output_root: Path,
    episodes: int,
    external_input: float,
    backend: str | None,
    allow_cpu_fallback: bool,
    confidence_threshold: float = 0.55,
    profiles: List[str] | None = None,
    regimes: List[str] | None = None,
    external_client: Any | None = None,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    selected_profiles = profiles or list(PROFILES)
    selected_regimes = regimes or list(regime_specs())
    for profile in selected_profiles:
        for regime in selected_regimes:
            for episode_index in range(episodes):
                rows.append(
                    run_episode(
                        profile=profile,
                        regime=regime,
                        episode_index=episode_index,
                        external_input=external_input,
                        backend=backend,
                        allow_cpu_fallback=allow_cpu_fallback,
                        confidence_threshold=confidence_threshold,
                        external_client=external_client,
                    )
                )

    summary = aggregate_rows(rows)
    summary["campaign_id"] = campaign_id
    summary["episodes_total"] = len(rows)
    summary["profiles"] = list(selected_profiles)
    summary["regimes"] = list(selected_regimes)
    summary["external_input"] = external_input
    summary["confidence_threshold"] = confidence_threshold
    out_dir = output_root / campaign_id
    summary["output_dir"] = str(out_dir)
    write_outputs(rows=rows, summary=summary, out_dir=out_dir)
    return summary


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-id", default=f"external_reasoner_gain-{_now_stamp()}")
    parser.add_argument("--output-root", default="data/benchmarks/external_reasoner_gain")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--external-input", type=float, default=0.04)
    parser.add_argument("--backend", choices=["cuda", "cpu"], default=None)
    parser.add_argument("--allow-cpu-fallback", action="store_true")
    parser.add_argument("--confidence-threshold", type=float, default=0.55)
    parser.add_argument("--profiles", nargs="+", choices=PROFILES, default=None)
    parser.add_argument("--regimes", nargs="+", choices=list(regime_specs()), default=None)
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_campaign(
        campaign_id=args.campaign_id,
        output_root=Path(args.output_root),
        episodes=max(1, args.episodes),
        external_input=args.external_input,
        backend=args.backend,
        allow_cpu_fallback=bool(args.allow_cpu_fallback),
        confidence_threshold=args.confidence_threshold,
        profiles=args.profiles,
        regimes=args.regimes,
    )
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
