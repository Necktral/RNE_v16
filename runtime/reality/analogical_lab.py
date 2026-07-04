"""Laboratorio analógico de transferencia + EML shadow.

Compara el rendimiento de la misma matriz de transición bajo dos regímenes:
- strict_same_scenario: solo memoria del mismo escenario
- cross_scenario_analogical: incluye transferencia analógica

Mide deltas de continuidad, cierre, pureza, transfer-safe rate
y concordancia EML (cuando shadow está activo).
"""

from __future__ import annotations

import json
import os
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


def eml_concurrence_score(
    *,
    transfer_verdict: str,
    eml_shadow_payload: dict | None,
    relation_kind: str,
) -> float:
    """Computa concordancia entre EML shadow y transfer verdict.

    Alta concordancia cuando ambos coinciden en estabilidad/regularidad.
    Discordancia crítica cuando EML sube pero certificación cae.

    Args:
        transfer_verdict: Veredicto de transferencia del episodio.
        eml_shadow_payload: Payload EML del episodio (None si deshabilitado).
        relation_kind: Tipo de relación factual/contrafactual.

    Returns:
        Score de concordancia [0.0, 1.0].
    """
    if eml_shadow_payload is None or not eml_shadow_payload.get("enabled"):
        return 0.5  # No evaluable, neutral

    eml_composite = float(eml_shadow_payload.get("top_composite", 0.0))

    # Map transfer verdict to stability signal
    verdict_stable = transfer_verdict in ("certified_local", "certified_transfer_safe")
    verdict_unstable = transfer_verdict == "rejected_for_transfer"

    # EML regularidad alta
    eml_regular = eml_composite >= 0.50

    if verdict_stable and eml_regular:
        # Ambos coinciden en estabilidad → alta concordancia
        return min(1.0, 0.70 + 0.30 * eml_composite)
    if verdict_unstable and not eml_regular:
        # Ambos coinciden en inestabilidad → concordancia moderada
        return 0.60
    if eml_regular and verdict_unstable:
        # Discordancia crítica: EML sube pero certificación cae
        return max(0.0, 0.20 - 0.10 * eml_composite)
    if not eml_regular and verdict_stable:
        # EML baja pero certificación ok → leve discordancia
        return 0.40

    # Default: moderate
    return 0.50


def _run_matrix_under_mode(
    *,
    storage,
    run_id: str,
    bench_run_id: str,
    scenarios: list[str],
    profiles_map: dict,
    graph: ScenarioCompatibilityGraph,
    compat_matrix: dict,
    memory_mode: str,
    closure_profile: str,
    warmup_episodes: int,
    probe_episodes: int,
    collect_eml: bool,
) -> Dict[str, Any]:
    """Ejecuta una matriz completa bajo un modo de memoria dado."""
    detector = CollapseDetector()
    all_vectors: list[TransitionContinuityVector] = []
    cell_reports: list[dict] = []

    for src_name in scenarios:
        for tgt_name in scenarios:
            compat = compat_matrix[src_name][tgt_name]
            cell_run_id = f"{run_id}-{memory_mode[:6]}-{src_name[:6]}-{tgt_name[:6]}"

            # Warmup
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

            # Probe
            previous_result = warmup_results[-1] if warmup_results else None
            probe_assessments = []
            probe_vectors = []
            probe_eml_scores = []

            for j in range(probe_episodes):
                runner = ScenarioEpisodeRunner(
                    storage=storage,
                    run_id=cell_run_id,
                    scenario=tgt_name,
                    memory_filter_mode=memory_mode,
                    closure_profile=closure_profile,
                )
                result = runner.run_episode(external_input=0.04 + j * 0.01)

                closure = evaluate_episode_closure(
                    storage=storage,
                    run_id=cell_run_id,
                    result=result,
                    closure_profile=closure_profile,
                )
                prev_ep = (previous_result or {}).get("episode", {})
                curr_ep = result.get("episode", {})
                cont = continuity_score(
                    previous_episode=prev_ep if previous_result else None,
                    current_episode=curr_ep,
                    previous_smg_snapshot=(previous_result or {}).get("smg_snapshot"),
                    current_smg_snapshot=result.get("smg_snapshot", {}),
                    trace_integrity=closure["trace_integrity"],
                )
                collapse = detector.detect(
                    closure_passed=closure["closure_passed"],
                    trace_integrity=closure["trace_integrity"],
                    continuity_score=cont,
                    recent_assessments=probe_assessments[-2:],
                )

                if previous_result is not None:
                    vec = compute_transition_vector(
                        previous_result=previous_result,
                        current_result=result,
                        compatibility=compat,
                    )
                    probe_vectors.append(vec)
                    all_vectors.append(vec)

                transfer = assess_transfer(
                    episode_result=result,
                    compatibility=compat,
                    transition_vector=probe_vectors[-1] if probe_vectors else None,
                )

                # EML concordance
                eml_payload = result.get("eml_shadow")
                relation = curr_ep.get("result", {}).get("relation_kind", "unknown")
                eml_score = eml_concurrence_score(
                    transfer_verdict=transfer.transfer_verdict,
                    eml_shadow_payload=eml_payload,
                    relation_kind=relation,
                )
                probe_eml_scores.append(eml_score)

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
                        "memory_mode": memory_mode,
                        "transfer_verdict": transfer.transfer_verdict,
                    },
                )
                probe_assessments.append(assessment)
                previous_result = result

            # Aggregate
            cell_closure = [1.0 if a.closure_passed else 0.0 for a in probe_assessments]
            cell_closure_rate = sum(cell_closure) / len(cell_closure) if cell_closure else 0.0
            cell_purity = (
                sum(v.memory_purity for v in probe_vectors) / len(probe_vectors)
                if probe_vectors else 1.0
            )
            cell_composite = (
                sum(v.composite_score for v in probe_vectors) / len(probe_vectors)
                if probe_vectors else 0.0
            )
            verdicts = []
            for pr in probe_assessments:
                v_label = pr.details.get("transfer_verdict", "certified_local")
                verdicts.append(v_label)
            safe_count = sum(1 for v in verdicts if v != "rejected_for_transfer")
            safe_rate = safe_count / len(verdicts) if verdicts else 0.0
            mean_eml = sum(probe_eml_scores) / len(probe_eml_scores) if probe_eml_scores else 0.5

            cell_reports.append({
                "source_scenario": src_name,
                "target_scenario": tgt_name,
                "memory_mode": memory_mode,
                "closure_rate": round(cell_closure_rate, 4),
                "mean_continuity_composite": round(cell_composite, 4),
                "mean_memory_purity": round(cell_purity, 4),
                "transfer_safe_rate": round(safe_rate, 4),
                "eml_concurrence_mean": round(mean_eml, 4),
                "collapse_count": sum(1 for a in probe_assessments if a.collapse_detected),
            })

    tensor = build_continuity_tensor(vectors=all_vectors)
    tensor_ser = {
        src: {tgt: asdict(c) for tgt, c in inner.items()}
        for src, inner in tensor.items()
    }
    return {
        "cell_reports": cell_reports,
        "continuity_tensor": tensor_ser,
    }


def run_analogical_transfer_lab(
    *,
    run_id: str | None = None,
    scenarios: list[str] | None = None,
    warmup_episodes: int = 2,
    probe_episodes: int = 2,
    closure_profile: str = "adaptive_min",
    strict_memory_mode: str = "strict_same_scenario",
    analogical_memory_mode: str = "cross_scenario_analogical",
    eml_shadow: bool = True,
) -> Dict[str, Any]:
    """Ejecuta laboratorio analógico comparando strict vs analogical.

    Args:
        run_id: ID de corrida.
        scenarios: Escenarios a incluir.
        warmup_episodes: Episodios de warmup por celda.
        probe_episodes: Episodios de probe por celda.
        closure_profile: Perfil de cierre.
        strict_memory_mode: Modo estricto de memoria.
        analogical_memory_mode: Modo analógico de memoria.
        eml_shadow: Si activar EML shadow (controla env var).

    Returns:
        Dict con matrices strict/analogical, deltas y artifact.
    """
    storage = get_storage()
    if run_id is None:
        run_id = f"run-analab-{uuid4()}"
    bench_run_id = f"bench-analab-{uuid4()}"

    profiles_map = list_structural_profiles()
    if scenarios is None:
        scenarios = list(profiles_map.keys())

    graph = ScenarioCompatibilityGraph()
    scenario_profiles = [profiles_map[s] for s in scenarios]
    compat_matrix = graph.matrix(scenario_profiles)

    # Optionally enable EML shadow
    old_eml = os.environ.get("RNFE_EML_MODE", "disabled")
    if eml_shadow:
        os.environ["RNFE_EML_MODE"] = "shadow"

    try:
        # Run strict
        strict_result = _run_matrix_under_mode(
            storage=storage,
            run_id=run_id,
            bench_run_id=bench_run_id,
            scenarios=scenarios,
            profiles_map=profiles_map,
            graph=graph,
            compat_matrix=compat_matrix,
            memory_mode=strict_memory_mode,
            closure_profile=closure_profile,
            warmup_episodes=warmup_episodes,
            probe_episodes=probe_episodes,
            collect_eml=eml_shadow,
        )

        # Run analogical
        analogical_result = _run_matrix_under_mode(
            storage=storage,
            run_id=run_id,
            bench_run_id=bench_run_id,
            scenarios=scenarios,
            profiles_map=profiles_map,
            graph=graph,
            compat_matrix=compat_matrix,
            memory_mode=analogical_memory_mode,
            closure_profile=closure_profile,
            warmup_episodes=warmup_episodes,
            probe_episodes=probe_episodes,
            collect_eml=eml_shadow,
        )
    finally:
        os.environ["RNFE_EML_MODE"] = old_eml

    # Compute deltas
    deltas = _compute_deltas(
        strict_cells=strict_result["cell_reports"],
        analogical_cells=analogical_result["cell_reports"],
    )

    # Build report
    report = {
        "matrix_strict": strict_result,
        "matrix_analogical": analogical_result,
        "delta_continuity": deltas["delta_continuity"],
        "delta_purity": deltas["delta_purity"],
        "delta_transfer_safe": deltas["delta_transfer_safe"],
        "delta_closure": deltas["delta_closure"],
        "eml_concurrence_map": deltas["eml_concurrence_map"],
    }

    # Persist
    storage.append_event(
        event_type="reality.analogical_lab.completed",
        run_id=run_id,
        source="analogical_transfer_lab",
        payload={
            "bench_run_id": bench_run_id,
            "timestamp": utc_now_iso(),
            "delta_summary": {
                k: round(v, 4) if isinstance(v, float) else v
                for k, v in deltas.items()
                if k != "eml_concurrence_map"
            },
        },
    )

    artifact = storage.materialize_artifact(
        run_id=run_id,
        kind="analogical_transfer_lab_report",
        filename=f"{bench_run_id}.json",
        content=json.dumps(report, ensure_ascii=True, sort_keys=True, indent=2),
        metadata={
            "bench_run_id": bench_run_id,
            "benchmark_type": "analogical_transfer_lab",
        },
    )

    return {
        **report,
        "artifact": asdict(artifact),
        "bench_run_id": bench_run_id,
    }


def _compute_deltas(
    *,
    strict_cells: list[dict],
    analogical_cells: list[dict],
) -> dict:
    """Computa deltas entre los dos regímenes."""
    # Index by (src, tgt)
    strict_map = {(c["source_scenario"], c["target_scenario"]): c for c in strict_cells}
    analogical_map = {(c["source_scenario"], c["target_scenario"]): c for c in analogical_cells}

    delta_continuity_vals = []
    delta_purity_vals = []
    delta_safe_vals = []
    delta_closure_vals = []
    eml_map = {}

    for key in strict_map:
        s = strict_map[key]
        a = analogical_map.get(key)
        if a is None:
            continue
        delta_continuity_vals.append(a["mean_continuity_composite"] - s["mean_continuity_composite"])
        delta_purity_vals.append(a["mean_memory_purity"] - s["mean_memory_purity"])
        delta_safe_vals.append(a["transfer_safe_rate"] - s["transfer_safe_rate"])
        delta_closure_vals.append(a["closure_rate"] - s["closure_rate"])
        eml_map[f"{key[0]}->{key[1]}"] = {
            "strict_eml": s.get("eml_concurrence_mean", 0.5),
            "analogical_eml": a.get("eml_concurrence_mean", 0.5),
            "delta_eml": a.get("eml_concurrence_mean", 0.5) - s.get("eml_concurrence_mean", 0.5),
        }

    n = len(delta_continuity_vals) or 1
    return {
        "delta_continuity": sum(delta_continuity_vals) / n,
        "delta_purity": sum(delta_purity_vals) / n,
        "delta_transfer_safe": sum(delta_safe_vals) / n,
        "delta_closure": sum(delta_closure_vals) / n,
        "eml_concurrence_map": eml_map,
    }
