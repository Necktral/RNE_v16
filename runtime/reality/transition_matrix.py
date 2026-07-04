"""Benchmark matricial de transición NxN entre escenarios.

Para cada par ordenado (S_i → S_j):
1. Correr warmup_episodes en S_i
2. Correr probe_episodes en S_j
3. Medir cierre, trace integrity, continuidad vectorial,
   pureza de memoria, transfer verdict.

Produce un artifact de tipo 'transition_matrix_report' con:
- closure_matrix, continuity_tensor, memory_purity_matrix,
  transfer_verdict_matrix, cell_reports.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict
from typing import Any, Dict, List
from uuid import uuid4

from runtime.storage import get_storage
from runtime.storage.records import utc_now_iso
from runtime.world.compatibility import ScenarioCompatibilityGraph
from runtime.world.registry import get_scenario, list_structural_profiles
from runtime.world.scenario_runner import ScenarioEpisodeRunner
from runtime.reality.continuity import continuity_score
from runtime.reality.evaluator import evaluate_episode_closure
from runtime.reality.collapse import CollapseDetector
from runtime.reality.transition_analysis import (
    TransitionContinuityVector,
    build_continuity_tensor,
    compute_transition_vector,
)
from runtime.certification.transfer_assessment import assess_transfer


def run_transition_matrix_benchmark(
    *,
    run_id: str | None = None,
    scenarios: list[str] | None = None,
    warmup_episodes: int = 3,
    probe_episodes: int = 3,
    memory_mode: str = "strict_same_scenario",
    closure_profile: str = "adaptive_min",
    gate_profile: str = "transition_matrix_ci",
) -> Dict[str, Any]:
    """Ejecuta benchmark matricial NxN de transición entre escenarios.

    Para cada par (source, target), ejecuta warmup en source
    y probe en target, midiendo continuidad, cierre, pureza
    y transferibilidad.

    Args:
        run_id: ID de corrida.
        scenarios: Lista de nombres de escenarios a incluir.
        warmup_episodes: Episodios de warmup por celda.
        probe_episodes: Episodios de probe por celda.
        memory_mode: Modo de filtrado de memoria.
        closure_profile: Perfil de cierre.
        gate_profile: Perfil de gate para evaluación.

    Returns:
        Dict con matrix_meta, matrices, cell_reports, artifact y passed.
    """
    storage = get_storage()
    if run_id is None:
        run_id = f"run-matrix-{uuid4()}"
    bench_run_id = f"bench-matrix-{uuid4()}"

    # Resolve scenarios
    profiles_map = list_structural_profiles()
    if scenarios is None:
        scenarios = list(profiles_map.keys())
    else:
        for s in scenarios:
            if s not in profiles_map:
                raise ValueError(f"Escenario '{s}' no registrado")

    graph = ScenarioCompatibilityGraph()
    detector = CollapseDetector()

    # Pre-compute compatibility matrix
    scenario_profiles = [profiles_map[s] for s in scenarios]
    compat_matrix = graph.matrix(scenario_profiles)

    # Result accumulators
    closure_matrix: Dict[str, Dict[str, float]] = defaultdict(dict)
    purity_matrix: Dict[str, Dict[str, float]] = defaultdict(dict)
    verdict_matrix: Dict[str, Dict[str, str]] = defaultdict(dict)
    all_vectors: List[TransitionContinuityVector] = []
    cell_reports: List[Dict[str, Any]] = []

    for src_name in scenarios:
        for tgt_name in scenarios:
            cell_run_id = f"{run_id}-{src_name[:6]}-{tgt_name[:6]}"
            compat = compat_matrix[src_name][tgt_name]

            # Phase 1: warmup in source
            warmup_results = []
            for i in range(warmup_episodes):
                runner = ScenarioEpisodeRunner(
                    storage=storage,
                    run_id=cell_run_id,
                    scenario=src_name,
                    memory_filter_mode=memory_mode,
                    closure_profile=closure_profile,
                )
                result = runner.run_episode(external_input=0.04 + i * 0.01)
                warmup_results.append(result)

            # Phase 2: probe in target
            probe_results = []
            previous_result = warmup_results[-1] if warmup_results else None
            probe_assessments = []
            probe_vectors = []

            for j in range(probe_episodes):
                runner = ScenarioEpisodeRunner(
                    storage=storage,
                    run_id=cell_run_id,
                    scenario=tgt_name,
                    memory_filter_mode=memory_mode,
                    closure_profile=closure_profile,
                )
                result = runner.run_episode(external_input=0.04 + j * 0.01)
                probe_results.append(result)

                # Evaluate closure
                closure = evaluate_episode_closure(
                    storage=storage,
                    run_id=cell_run_id,
                    result=result,
                    closure_profile=closure_profile,
                )

                # Evaluate continuity (scalar for assessment)
                prev_ep = (previous_result or {}).get("episode", {})
                curr_ep = result.get("episode", {})
                cont = continuity_score(
                    previous_episode=prev_ep if previous_result else None,
                    current_episode=curr_ep,
                    previous_smg_snapshot=(previous_result or {}).get("smg_snapshot"),
                    current_smg_snapshot=result.get("smg_snapshot", {}),
                    trace_integrity=closure["trace_integrity"],
                )

                # Collapse detection
                collapse = detector.detect(
                    closure_passed=closure["closure_passed"],
                    trace_integrity=closure["trace_integrity"],
                    continuity_score=cont,
                    recent_assessments=probe_assessments[-2:],
                )

                # Compute transition vector
                if previous_result is not None:
                    vec = compute_transition_vector(
                        previous_result=previous_result,
                        current_result=result,
                        compatibility=compat,
                    )
                    probe_vectors.append(vec)
                    all_vectors.append(vec)

                # Transfer assessment
                transfer = assess_transfer(
                    episode_result=result,
                    compatibility=compat,
                    transition_vector=probe_vectors[-1] if probe_vectors else None,
                )

                # Write reality assessment
                assessment = storage.write_reality_assessment(
                    assessment_id=f"assess-{uuid4()}",
                    run_id=cell_run_id,
                    bench_run_id=bench_run_id,
                    episode_id=closure["episode_id"],
                    closure_passed=closure["closure_passed"],
                    continuity_score=cont,
                    trace_integrity=closure["trace_integrity"],
                    collapse_detected=collapse,
                    details={
                        "scenario_name": tgt_name,
                        "source_scenario": src_name,
                        "transfer_verdict": transfer.transfer_verdict,
                    },
                )
                probe_assessments.append(assessment)
                previous_result = result

            # Aggregate cell metrics
            cell_closure_pass = [
                1.0 if a.closure_passed else 0.0 for a in probe_assessments
            ]
            cell_closure_rate = (
                sum(cell_closure_pass) / len(cell_closure_pass) if cell_closure_pass else 0.0
            )
            cell_purity = (
                sum(v.memory_purity for v in probe_vectors) / len(probe_vectors)
                if probe_vectors else 1.0
            )
            cell_composite = (
                sum(v.composite_score for v in probe_vectors) / len(probe_vectors)
                if probe_vectors else 0.0
            )

            # Determine dominant transfer verdict
            verdicts = []
            for pr in probe_results:
                t = assess_transfer(
                    episode_result=pr,
                    compatibility=compat,
                    transition_vector=probe_vectors[-1] if probe_vectors else None,
                )
                verdicts.append(t.transfer_verdict)
            dominant_verdict = max(set(verdicts), key=verdicts.count) if verdicts else "certified_local"

            closure_matrix[src_name][tgt_name] = round(cell_closure_rate, 4)
            purity_matrix[src_name][tgt_name] = round(cell_purity, 4)
            verdict_matrix[src_name][tgt_name] = dominant_verdict

            cell_reports.append({
                "source_scenario": src_name,
                "target_scenario": tgt_name,
                "compatibility_class": compat.compatibility_class,
                "warmup_episodes": warmup_episodes,
                "probe_episodes": probe_episodes,
                "closure_rate": round(cell_closure_rate, 4),
                "mean_continuity_composite": round(cell_composite, 4),
                "mean_memory_purity": round(cell_purity, 4),
                "transfer_verdict": dominant_verdict,
                "collapse_count": sum(1 for a in probe_assessments if a.collapse_detected),
            })

    # Build continuity tensor
    tensor = build_continuity_tensor(vectors=all_vectors)
    tensor_serializable = {
        src: {
            tgt: asdict(cell)
            for tgt, cell in inner.items()
        }
        for src, inner in tensor.items()
    }

    # Gate evaluation
    gate_config = GATE_PROFILES.get(gate_profile, GATE_PROFILES["transition_matrix_ci"])
    passed = _evaluate_matrix_gate(
        gate_config=gate_config,
        cell_reports=cell_reports,
    )

    # Summary
    matrix_meta = {
        "scenarios": scenarios,
        "warmup_episodes": warmup_episodes,
        "probe_episodes": probe_episodes,
        "memory_mode": memory_mode,
        "closure_profile": closure_profile,
    }

    summary = {
        "bench_run_id": bench_run_id,
        "run_id": run_id,
        "total_cells": len(cell_reports),
        "gate_profile": gate_profile,
        "passed": passed,
        "matrix_meta": matrix_meta,
    }

    # Persist bench run
    bench_record = storage.write_reality_bench_run(
        bench_run_id=bench_run_id,
        run_id=run_id,
        total_episodes=sum(c["probe_episodes"] for c in cell_reports),
        closure_rate=sum(c["closure_rate"] for c in cell_reports) / len(cell_reports) if cell_reports else 0.0,
        continuity_mean=sum(c["mean_continuity_composite"] for c in cell_reports) / len(cell_reports) if cell_reports else 0.0,
        collapse_count=sum(c["collapse_count"] for c in cell_reports),
        gate_profile=gate_profile,
        passed=passed,
        summary=summary,
    )

    storage.append_event(
        event_type="reality.transition_matrix.completed",
        run_id=run_id,
        source="transition_matrix_benchmark",
        payload={
            "bench_run_id": bench_run_id,
            "passed": passed,
            "summary": summary,
            "timestamp": utc_now_iso(),
        },
    )

    # Artifact
    report_content = {
        "matrix_meta": matrix_meta,
        "closure_matrix": dict(closure_matrix),
        "continuity_tensor": tensor_serializable,
        "memory_purity_matrix": dict(purity_matrix),
        "transfer_verdict_matrix": dict(verdict_matrix),
        "cell_reports": cell_reports,
    }
    artifact = storage.materialize_artifact(
        run_id=run_id,
        kind="transition_matrix_report",
        filename=f"{bench_run_id}.json",
        content=json.dumps(report_content, ensure_ascii=True, sort_keys=True, indent=2),
        metadata={
            "bench_run_id": bench_run_id,
            "gate_profile": gate_profile,
            "benchmark_type": "transition_matrix",
        },
    )

    return {
        "bench_run": asdict(bench_record),
        "cell_reports": cell_reports,
        "closure_matrix": dict(closure_matrix),
        "continuity_tensor": tensor_serializable,
        "memory_purity_matrix": dict(purity_matrix),
        "transfer_verdict_matrix": dict(verdict_matrix),
        "artifact": asdict(artifact),
        "passed": passed,
    }


# ── Gate profiles ────────────────────────────────────────────────────────────

GATE_PROFILES = {
    "transition_matrix_ci": {
        "min_cells": 4,
        "closure_rate_min": 0.90,
        "mean_transition_continuity_min": 0.45,
        "transfer_safe_rate_min": 0.75,
        "collapse_count_max": 1,
    },
}


def _evaluate_matrix_gate(
    *,
    gate_config: Dict[str, Any],
    cell_reports: List[Dict[str, Any]],
) -> bool:
    """Evalúa el gate sobre las celdas de la matriz."""
    if len(cell_reports) < gate_config.get("min_cells", 4):
        return False

    mean_closure = (
        sum(c["closure_rate"] for c in cell_reports) / len(cell_reports)
        if cell_reports else 0.0
    )
    if mean_closure < gate_config.get("closure_rate_min", 0.90):
        return False

    mean_continuity = (
        sum(c["mean_continuity_composite"] for c in cell_reports) / len(cell_reports)
        if cell_reports else 0.0
    )
    if mean_continuity < gate_config.get("mean_transition_continuity_min", 0.45):
        return False

    total_collapse = sum(c["collapse_count"] for c in cell_reports)
    if total_collapse > gate_config.get("collapse_count_max", 1):
        return False

    # Transfer safe rate: fraction of cells that are not rejected
    safe_cells = sum(
        1 for c in cell_reports
        if c["transfer_verdict"] != "rejected_for_transfer"
    )
    safe_rate = safe_cells / len(cell_reports) if cell_reports else 0.0
    if safe_rate < gate_config.get("transfer_safe_rate_min", 0.75):
        return False

    return True
