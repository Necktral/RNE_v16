#!/usr/bin/env python3
"""Diagnostico E2E del cableado de familias de razonamiento.

Demuestra el flujo runtime -> storage -> benchmark/analyze -> reality validation
con evidencia serializada y clasificacion de conectividad por familia.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, Iterator, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime.reality.service import RealityValidationService
from runtime.reasoning.scheduler_meta.meta_scheduler import MetaScheduler
from runtime.storage import StorageConfig, StorageFactory
from runtime.world.grid_thermal_scenario import GridThermalScenario
from runtime.world.scenario_runner import ScenarioEpisodeRunner
from tests.benchmarks.analysis_report import BenchmarkAnalyzer
from tests.benchmarks.benchmark_runner import (
    BenchmarkConfig,
    BenchmarkRunner,
    adapt_runtime_result_to_benchmark,
)

CORE_FAMILIES = ["ABD", "ANA", "CAU", "CTF", "DED", "PROB"]
OPTIONAL_FAMILIES = ["HEUR", "DIA_ADV", "FAL_GUARD", "IND", "EML_SR"]
ALL_TRACKED_FAMILIES = CORE_FAMILIES + OPTIONAL_FAMILIES


@contextmanager
def _temporary_env(updates: Dict[str, str | None]) -> Iterator[None]:
    previous = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, default=str), encoding="utf-8")


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _classify_families(
    *,
    runtime_sequence: List[str],
    adaptive_sequence: List[str],
    runtime_persisted_families: set[str],
    adaptive_persisted_families: set[str],
    reality_sequences: List[List[str]],
    benchmark_metric_present: bool,
) -> Dict[str, Dict[str, Any]]:
    classes: Dict[str, Dict[str, Any]] = {}
    reality_families = {fam for seq in reality_sequences for fam in seq}

    for family in ALL_TRACKED_FAMILIES:
        executed_runtime = family in runtime_sequence
        executed_adaptive = family in adaptive_sequence
        persisted = family in runtime_persisted_families or family in adaptive_persisted_families
        seen_in_reality = family in reality_families

        if executed_runtime and persisted and seen_in_reality and benchmark_metric_present:
            connectivity = "fuerte"
            reason = "Ejecuta en corrida real, persiste en traces y participa aguas abajo (benchmark/reality)."
        elif (executed_runtime or executed_adaptive or persisted) and (seen_in_reality or benchmark_metric_present):
            connectivity = "debil"
            reason = "Existe ejecucion o persistencia, pero impacto aguas abajo es parcial o indirecto."
        else:
            connectivity = "ausente"
            reason = "Declarada sin evidencia de ejecucion/persistencia/consumo en el pipeline observado."

        classes[family] = {
            "connectivity": connectivity,
            "executed_runtime": executed_runtime,
            "executed_adaptive": executed_adaptive,
            "persisted": persisted,
            "seen_in_reality": seen_in_reality,
            "benchmark_signal_present": benchmark_metric_present,
            "reason": reason,
        }

    return classes


def _dictamen(connectivity: Dict[str, Dict[str, Any]]) -> str:
    core_status = [connectivity[f]["connectivity"] for f in CORE_FAMILIES]
    optional_status = [connectivity[f]["connectivity"] for f in OPTIONAL_FAMILIES]

    if all(status == "fuerte" for status in core_status) and all(
        status == "fuerte" for status in optional_status
    ):
        return "familias conectadas de punta a punta"

    if any(status == "ausente" for status in core_status):
        return "familias presentes pero no integradas de verdad"

    return "familias conectadas parcialmente"


def build_report(args: argparse.Namespace) -> Dict[str, Any]:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_tag = args.run_id or f"reasoning_wiring_{stamp}"
    output_dir = Path(args.output_root) / run_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    db_path = Path(args.db_path)
    storage_cfg = StorageConfig(
        mode="sqlite",
        sqlite_db_path=str(db_path),
        postgres_dsn=None,
        artifact_root=Path(args.artifact_root),
        prefer_postgres_reads=False,
        strict_dual_write=False,
    )

    storage = StorageFactory.create_facade(storage_cfg)

    # 1) Runtime real (ScenarioEpisodeRunner)
    runtime_run_id = f"{run_tag}-runtime"
    runner = ScenarioEpisodeRunner(
        storage=storage,
        run_id=runtime_run_id,
        scenario="grid_thermal_5x5",
    )
    runtime_result = runner.run_episode(external_input=args.external_input)

    runtime_sequence = runtime_result.get("reasoning", {}).get("sequence", [])
    runtime_trace = runtime_result.get("episode", {}).get("trace", [])
    runtime_trace_families = [step.get("family") for step in runtime_trace]

    persisted_runtime = storage.list_reasoning_traces(run_id=runtime_run_id, limit=200)

    # 2) Benchmark bridge + analyzer
    benchmark_root = output_dir / "benchmark_bridge"
    benchmark_runner = BenchmarkRunner(output_root=benchmark_root, storage_config=storage_cfg)

    run_1x1_dir = benchmark_root / "run_1x1"
    run_5x5_dir = benchmark_root / "run_5x5"

    cfg_1x1 = BenchmarkConfig(
        scenario_name="grid_thermal_1x1",
        scenario_class=GridThermalScenario,
        scenario_params={"grid_size": 1},
        episodes=args.benchmark_episodes,
        base_seed=1000,
        max_steps=50,
        output_dir=run_1x1_dir,
        run_id=f"{run_tag}-bench-1x1",
    )
    cfg_5x5 = BenchmarkConfig(
        scenario_name="grid_thermal_5x5_uniform",
        scenario_class=GridThermalScenario,
        scenario_params={"grid_size": 5, "topology": "uniform"},
        episodes=args.benchmark_episodes,
        base_seed=2000,
        max_steps=50,
        output_dir=run_5x5_dir,
        run_id=f"{run_tag}-bench-5x5",
    )

    summary_1x1 = benchmark_runner.run_benchmark(cfg_1x1)
    summary_5x5 = benchmark_runner.run_benchmark(cfg_5x5)

    episodes_1x1 = _read_jsonl(run_1x1_dir / "episodes.jsonl")
    episodes_5x5 = _read_jsonl(run_5x5_dir / "episodes.jsonl")

    analyzer = BenchmarkAnalyzer()
    analyzer.load_results(run_1x1_dir, run_5x5_dir)
    comparison = analyzer.compute_statistical_comparison()

    benchmark_metric_present = "reasoning_trace_length" in comparison

    # 3) Reality validation bridge
    service = RealityValidationService(storage=storage)
    baseline_assessment = service.evaluate_episode_result(
        run_id=runtime_run_id,
        bench_run_id=f"{run_tag}-reality",
        result=runtime_result,
        previous_result=None,
        recent_assessments=[],
        scenario_name="grid_thermal_5x5",
        closure_profile="baseline_fixed",
    )
    adaptive_assessment = service.evaluate_episode_result(
        run_id=runtime_run_id,
        bench_run_id=f"{run_tag}-reality",
        result=runtime_result,
        previous_result=None,
        recent_assessments=[baseline_assessment],
        scenario_name="grid_thermal_5x5",
        closure_profile="adaptive_min",
    )

    # 4) Adaptive controlled mode to evidence optional families
    adaptive_run_id = f"{run_tag}-adaptive"
    with _temporary_env(
        {
            "RNFE_EML_MODE": "shadow",
            "RNFE_META_EXPERIMENTAL_FAMILIES": "eml_sr",
        }
    ):
        scheduler = MetaScheduler(mode="adaptive", trace_store=storage, max_steps=10)
        adaptive_reasoning = scheduler.run(
            {
                "episode_id": "episode-adaptive-controlled",
                "run_id": adaptive_run_id,
                "storage": storage,
                "observation": {"temperature": 0.93, "alarm": True},
                "updated_world": {"temperature": 0.84},
                "counterfactual": {"temperature": 0.96},
                "uncertainty": 0.87,
                "contradiction_signal": 0.79,
                "counterfactual_gap": 0.72,
                "edge_pressure": 0.82,
                "symbolic_regularity": 0.78,
                "law_fit_signal": 0.76,
            }
        )

    adaptive_sequence = adaptive_reasoning.get("sequence", [])
    persisted_adaptive = storage.list_reasoning_traces(run_id=adaptive_run_id, limit=200)

    # Optional family evidence in reality: synthetic assessment with adaptive sequence
    adaptive_result_for_reality = copy.deepcopy(runtime_result)
    adaptive_result_for_reality["episode"]["result"]["reasoning_sequence"] = adaptive_sequence
    adaptive_result_for_reality["episode"]["trace"] = adaptive_reasoning.get("trace", [])

    adaptive_family_assessment = service.evaluate_episode_result(
        run_id=runtime_run_id,
        bench_run_id=f"{run_tag}-reality",
        result=adaptive_result_for_reality,
        previous_result=runtime_result,
        recent_assessments=[baseline_assessment, adaptive_assessment],
        scenario_name="grid_thermal_5x5",
        closure_profile="adaptive_min",
    )

    adapted_payload = adapt_runtime_result_to_benchmark(runtime_result)

    family_connectivity = _classify_families(
        runtime_sequence=runtime_sequence,
        adaptive_sequence=adaptive_sequence,
        runtime_persisted_families={row.family for row in persisted_runtime},
        adaptive_persisted_families={row.family for row in persisted_adaptive},
        reality_sequences=[
            baseline_assessment.details.get("reasoning_sequence", []),
            adaptive_assessment.details.get("reasoning_sequence", []),
            adaptive_family_assessment.details.get("reasoning_sequence", []),
        ],
        benchmark_metric_present=benchmark_metric_present,
    )

    evidence_table = [
        {
            "layer": "runtime",
            "file": "runtime/world/scenario_runner.py",
            "function/class": "ScenarioEpisodeRunner.run_episode",
            "role": "Invoca MetaScheduler y propaga sequence/trace al episodio.",
            "input": {
                "scenario": "grid_thermal_5x5",
                "external_input": args.external_input,
            },
            "output": {
                "reasoning.sequence": runtime_sequence,
                "episode.result.reasoning_sequence": runtime_result["episode"]["result"].get("reasoning_sequence", []),
                "episode.trace_len": len(runtime_trace),
            },
            "evidence": "run_result.runtime+episode payload",
        },
        {
            "layer": "scheduler",
            "file": "runtime/reasoning/scheduler_meta/meta_scheduler.py",
            "function/class": "MetaScheduler.run",
            "role": "Selecciona secuencia, ejecuta familias y persiste trace.",
            "input": {
                "mode": "adaptive",
                "run_id": adaptive_run_id,
            },
            "output": {
                "adaptive_sequence": adaptive_sequence,
                "trace_len": len(adaptive_reasoning.get("trace", [])),
            },
            "evidence": "adaptive controlled run",
        },
        {
            "layer": "storage",
            "file": "runtime/storage/facade.py",
            "function/class": "append_reasoning_trace/list_reasoning_traces",
            "role": "Persistencia de familias por step_index y detail.sequence.",
            "input": {
                "runtime_run_id": runtime_run_id,
                "adaptive_run_id": adaptive_run_id,
            },
            "output": {
                "runtime_rows": len(persisted_runtime),
                "adaptive_rows": len(persisted_adaptive),
            },
            "evidence": "reasoning_traces rows",
        },
        {
            "layer": "benchmark",
            "file": "tests/benchmarks/benchmark_runner.py",
            "function/class": "adapt_runtime_result_to_benchmark / run_single_episode",
            "role": "Convierte reasoning.sequence -> reasoning_sequence -> reasoning_trace_length.",
            "input": {
                "episodes_per_run": args.benchmark_episodes,
            },
            "output": {
                "adapted_reasoning_sequence_len": len(adapted_payload.get("reasoning_sequence", [])),
                "episodes_jsonl_1x1": len(episodes_1x1),
                "episodes_jsonl_5x5": len(episodes_5x5),
            },
            "evidence": "episodes.jsonl + summary.json",
        },
        {
            "layer": "analysis",
            "file": "tests/benchmarks/analysis_report.py",
            "function/class": "BenchmarkAnalyzer.compute_statistical_comparison",
            "role": "Consume reasoning_trace_length en comparacion estadistica.",
            "input": {
                "dir_1x1": str(run_1x1_dir),
                "dir_5x5": str(run_5x5_dir),
            },
            "output": {
                "metric_present": benchmark_metric_present,
                "net_cognitive_gain": comparison.get("net_cognitive_gain", {}).get("net_gain"),
            },
            "evidence": "comparison dict",
        },
        {
            "layer": "reality_validation",
            "file": "runtime/reality/service.py",
            "function/class": "RealityValidationService.evaluate_episode_result",
            "role": "Persiste details.reasoning_sequence y sequence_validation por profile.",
            "input": {
                "profiles": ["baseline_fixed", "adaptive_min"],
            },
            "output": {
                "baseline_validation": baseline_assessment.details.get("sequence_validation"),
                "adaptive_validation": adaptive_family_assessment.details.get("sequence_validation"),
            },
            "evidence": "reality_assessments.details",
        },
        {
            "layer": "tests/regression",
            "file": "tests/regression/test_meta_scheduler_storage_trace.py",
            "function/class": "test_meta_scheduler_persists_trace_via_storage",
            "role": "Regresion de persistencia de trace META.",
            "input": {"run_id": "run-meta-1"},
            "output": {"expected_sequence": CORE_FAMILIES},
            "evidence": "suite regression",
        },
    ]

    summary = {
        "runtime": {
            "run_id": runtime_run_id,
            "sequence": runtime_sequence,
            "trace_families": runtime_trace_families,
            "persisted_rows": len(persisted_runtime),
            "artifact_path": runtime_result.get("artifact", {}).get("abs_path"),
        },
        "benchmark": {
            "run_1x1_dir": str(run_1x1_dir),
            "run_5x5_dir": str(run_5x5_dir),
            "summary_1x1": summary_1x1,
            "summary_5x5": summary_5x5,
            "comparison_has_reasoning_trace_length": benchmark_metric_present,
            "comparison_reasoning_trace_length": comparison.get("reasoning_trace_length"),
            "episodes_1x1_reasoning_trace_len_mean": _mean(
                float(row.get("reasoning_trace_length", 0.0)) for row in episodes_1x1
            ),
            "episodes_5x5_reasoning_trace_len_mean": _mean(
                float(row.get("reasoning_trace_length", 0.0)) for row in episodes_5x5
            ),
        },
        "reality": {
            "baseline": {
                "reasoning_sequence": baseline_assessment.details.get("reasoning_sequence", []),
                "sequence_validation": baseline_assessment.details.get("sequence_validation"),
            },
            "adaptive": {
                "reasoning_sequence": adaptive_family_assessment.details.get("reasoning_sequence", []),
                "sequence_validation": adaptive_family_assessment.details.get("sequence_validation"),
            },
        },
        "adaptive_controlled": {
            "run_id": adaptive_run_id,
            "sequence": adaptive_sequence,
            "persisted_rows": len(persisted_adaptive),
        },
    }

    report = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "run_tag": run_tag,
            "db_path": str(db_path),
            "artifact_root": str(Path(args.artifact_root)),
            "proxy_mapping": {
                "closure_rate": "success_rate",
                "continuity_mean": "viability_margin",
            },
        },
        "summary": summary,
        "family_connectivity": family_connectivity,
        "dictamen": _dictamen(family_connectivity),
        "evidence_table": evidence_table,
    }

    storage.close()
    return report


def parse_args() -> argparse.Namespace:
    default_db = os.environ.get("AEON_EVENT_DB", "aeon_event_log.db")
    parser = argparse.ArgumentParser(description="Inspeccion E2E de cableado de familias de razonamiento")
    parser.add_argument("--run-id", default="", help="Identificador del reporte de diagnostico")
    parser.add_argument(
        "--output-root",
        default="data/diagnostics/reasoning_wiring",
        help="Directorio raiz para artifacts de diagnostico",
    )
    parser.add_argument("--db-path", default=default_db, help="SQLite DB path (dedicada)")
    parser.add_argument(
        "--artifact-root",
        default="data/artifacts",
        help="Root para artifacts de storage",
    )
    parser.add_argument("--external-input", type=float, default=0.04, help="External input para corrida runtime")
    parser.add_argument(
        "--benchmark-episodes",
        type=int,
        default=2,
        help="Episodios por corrida benchmark minima",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(args)

    run_tag = report["metadata"]["run_tag"]
    out_dir = Path(args.output_root) / run_tag
    out_file = out_dir / "reasoning_wiring_report.json"
    _write_json(out_file, report)

    print("REASONING_WIRING_DIAGNOSTIC_DONE")
    print(f"report={out_file}")
    print(f"dictamen={report['dictamen']}")

    core = {fam: report["family_connectivity"][fam]["connectivity"] for fam in CORE_FAMILIES}
    optional = {fam: report["family_connectivity"][fam]["connectivity"] for fam in OPTIONAL_FAMILIES}
    print(f"core_connectivity={core}")
    print(f"optional_connectivity={optional}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
