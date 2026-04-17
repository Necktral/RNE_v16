"""Servicio de validación de realidad operativa del organismo."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict
from typing import Any, Dict, Iterable, List
from uuid import uuid4

from runtime.storage import get_storage
from runtime.storage.records import utc_now_iso
from runtime.world.min_cognitive_episode import MinimalCognitiveEpisodeRunner
from runtime.world.scenario_runner import ScenarioEpisodeRunner

from .collapse import CollapseDetector
from .continuity import continuity_score
from .evaluator import evaluate_episode_closure


GATE_PROFILES = {
    "ci": {
        "min_episodes": 10,
        "closure_rate_min": 0.90,
        "continuity_mean_min": 0.60,
        "collapse_count_max": 1,
    },
    "extended": {
        "min_episodes": 100,
        "closure_rate_min": 0.95,
        "continuity_mean_min": 0.65,
        "collapse_count_max": 3,
    },
}


def _fixed_homeostasis_pack(total: int) -> list[float]:
    base = [0.02, 0.03, 0.05, 0.07, 0.04, 0.06, 0.01, 0.08, 0.03, 0.05]
    values: list[float] = []
    while len(values) < total:
        values.extend(base)
    return values[:total]


class RealityValidationService:
    def __init__(self, *, storage=None):
        self.storage = storage or get_storage()
        self.detector = CollapseDetector()

    def evaluate_episode_result(
        self,
        *,
        run_id: str,
        bench_run_id: str,
        result: Dict[str, Any],
        previous_result: Dict[str, Any] | None,
        recent_assessments,
        scenario_name: str | None = None,
    ):
        closure = evaluate_episode_closure(storage=self.storage, run_id=run_id, result=result)
        continuity = continuity_score(
            previous_episode=(previous_result or {}).get("episode"),
            current_episode=result.get("episode", {}),
            previous_smg_snapshot=(previous_result or {}).get("smg_snapshot"),
            current_smg_snapshot=result.get("smg_snapshot", {}),
            trace_integrity=closure["trace_integrity"],
        )
        collapse = self.detector.detect(
            closure_passed=closure["closure_passed"],
            trace_integrity=closure["trace_integrity"],
            continuity_score=continuity,
            recent_assessments=recent_assessments,
        )
        details = {
            "closure_checks": closure["checks"],
            "reasoning_sequence": result.get("episode", {})
            .get("result", {})
            .get("reasoning_sequence", []),
            "scenario_name": scenario_name or "thermal_homeostasis",  # backward compat
        }
        return self.storage.write_reality_assessment(
            assessment_id=f"assess-{uuid4()}",
            run_id=run_id,
            bench_run_id=bench_run_id,
            episode_id=closure["episode_id"],
            closure_passed=closure["closure_passed"],
            continuity_score=continuity,
            trace_integrity=closure["trace_integrity"],
            collapse_detected=collapse,
            details=details,
        )

    def _evaluate_gate(self, *, gate_profile: str, summary: Dict[str, Any]) -> bool:
        if gate_profile not in GATE_PROFILES:
            raise ValueError(f"Gate profile no soportado: {gate_profile}")
        cfg = GATE_PROFILES[gate_profile]
        return bool(
            summary["total_episodes"] >= cfg["min_episodes"]
            and summary["closure_rate"] >= cfg["closure_rate_min"]
            and summary["continuity_mean"] >= cfg["continuity_mean_min"]
            and summary["collapse_count"] <= cfg["collapse_count_max"]
        )

    def _compute_scenario_metrics(
        self, assessments: List[Any]
    ) -> Dict[str, Dict[str, float]]:
        """Computa métricas agregadas por escenario.

        Args:
            assessments: Lista de assessments con campo details.scenario_name.

        Returns:
            Dict con métricas por escenario: {scenario_name: {closure_rate, continuity_mean, ...}}
        """
        by_scenario = defaultdict(list)
        for assessment in assessments:
            scenario = assessment.details.get("scenario_name", "thermal_homeostasis")
            by_scenario[scenario].append(assessment)

        metrics = {}
        for scenario_name, scenario_assessments in by_scenario.items():
            closure_pass = [
                1.0 if item.closure_passed else 0.0 for item in scenario_assessments
            ]
            continuity_values = [item.continuity_score for item in scenario_assessments]
            collapse_count = sum(1 for item in scenario_assessments if item.collapse_detected)

            metrics[scenario_name] = {
                "total_episodes": len(scenario_assessments),
                "closure_rate": (
                    sum(closure_pass) / len(closure_pass) if closure_pass else 0.0
                ),
                "continuity_mean": (
                    sum(continuity_values) / len(continuity_values)
                    if continuity_values
                    else 0.0
                ),
                "collapse_count": collapse_count,
            }

        return metrics

    def run_benchmark(
        self,
        *,
        run_id: str | None = None,
        gate_profile: str = "ci",
        external_heat_values: Iterable[float] | None = None,
    ) -> Dict[str, Any]:
        profile = gate_profile.strip().lower()
        if profile not in GATE_PROFILES:
            raise ValueError(f"Perfil de gate no soportado: {gate_profile}")
        if run_id is None:
            run_id = f"run-reality-{uuid4()}"
        bench_run_id = f"bench-{uuid4()}"
        target_episodes = GATE_PROFILES[profile]["min_episodes"]
        pack = list(external_heat_values or _fixed_homeostasis_pack(target_episodes))
        if len(pack) < target_episodes:
            raise ValueError("El pack causal no cubre el número mínimo de episodios del gate.")

        runner = MinimalCognitiveEpisodeRunner(storage=self.storage, run_id=run_id)
        previous_result: Dict[str, Any] | None = None
        assessments = []

        for external_heat in pack[:target_episodes]:
            result = runner.run_episode(external_heat=float(external_heat))
            # Extraer scenario_name del episodio si está disponible
            episode = result.get("episode", {})
            scenario_name = episode.get("scenario", "thermal_homeostasis")
            
            assessment = self.evaluate_episode_result(
                run_id=run_id,
                bench_run_id=bench_run_id,
                result=result,
                previous_result=previous_result,
                recent_assessments=assessments[-2:],
                scenario_name=scenario_name,
            )
            assessments.append(assessment)
            previous_result = result

        # Métricas globales
        closure_pass = [1.0 if item.closure_passed else 0.0 for item in assessments]
        continuity_values = [item.continuity_score for item in assessments]
        collapse_count = sum(1 for item in assessments if item.collapse_detected)
        
        # Métricas por escenario
        scenario_metrics = self._compute_scenario_metrics(assessments)
        
        summary = {
            "bench_run_id": bench_run_id,
            "run_id": run_id,
            "total_episodes": len(assessments),
            "closure_rate": (sum(closure_pass) / len(closure_pass)) if closure_pass else 0.0,
            "continuity_mean": (
                sum(continuity_values) / len(continuity_values) if continuity_values else 0.0
            ),
            "collapse_count": collapse_count,
            "gate_profile": profile,
            "scenario_metrics": scenario_metrics,  # Nuevo: métricas por escenario
        }
        passed = self._evaluate_gate(gate_profile=profile, summary=summary)

        bench_record = self.storage.write_reality_bench_run(
            bench_run_id=bench_run_id,
            run_id=run_id,
            total_episodes=summary["total_episodes"],
            closure_rate=summary["closure_rate"],
            continuity_mean=summary["continuity_mean"],
            collapse_count=summary["collapse_count"],
            gate_profile=profile,
            passed=passed,
            summary=summary,
        )
        self.storage.append_event(
            event_type="reality.validation.completed",
            run_id=run_id,
            source="reality_validation_service",
            payload={
                "bench_run_id": bench_run_id,
                "passed": passed,
                "summary": summary,
                "timestamp": utc_now_iso(),
            },
        )
        artifact = self.storage.materialize_artifact(
            run_id=run_id,
            kind="reality_report",
            filename=f"{bench_run_id}.json",
            content=json.dumps(
                {
                    "bench_run": asdict(bench_record),
                    "assessments": [asdict(item) for item in assessments],
                },
                ensure_ascii=True,
                sort_keys=True,
                indent=2,
            ),
            metadata={"bench_run_id": bench_run_id, "gate_profile": profile},
        )
        return {
            "bench_run": asdict(bench_record),
            "assessments": [asdict(item) for item in assessments],
            "artifact": asdict(artifact),
            "passed": passed,
        }
