#!/usr/bin/env python3
"""Campaña fuerte de validación arquitectónica 1x1 vs 5x5.

Secuencia:
1) Gate A (integridad + robustez base)
2) Matriz A (baseline controlado fuerte)
3) Matriz B (heterogeneidad espacial fuerte)
4) Matriz C (borde y estrés)
5) Matriz D (repetibilidad fuerte)

Artefactos:
- manifest.json
- runs_index.jsonl
- cell_master.csv
- failure_surface.csv
- final_report.md
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import os
import random
import sqlite3
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime.world.grid_thermal_scenario import GridThermalScenario
from tests.benchmarks.analysis_report import BenchmarkAnalyzer
from tests.benchmarks.benchmark_runner import BenchmarkConfig, BenchmarkRunner


REMOVED_METRICS = {
    "memory_pressure_mb",
    "scheduler_cpu_time_ms",
    "counterfactual_overhead_ratio",
    "factual_cf_divergence",
    "world_level_transitions",
    "spatial_coherence_index",
}

REQUIRED_EPISODE_FIELDS = {
    "episode_id",
    "outcome",
    "success_rate",
    "viability_margin",
    "intervention_precision",
    "proposition_diversity",
    "wall_time_ms",
    "artifact_size_bytes",
    "reasoning_trace_length",
    "ivc_r",
}


class CampaignAbort(RuntimeError):
    """Abort global inmediato por reglas duras."""


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _as_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def _mean(values: Iterable[float]) -> float:
    data = [float(v) for v in values]
    return sum(data) / len(data) if data else 0.0


def _sign(x: float) -> int:
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _assert_finite_tree(obj: Any, context: str) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            _assert_finite_tree(v, f"{context}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _assert_finite_tree(v, f"{context}[{i}]")
    elif isinstance(obj, float):
        if not math.isfinite(obj):
            raise CampaignAbort(f"NaN/Inf detectado en {context}: {obj}")


@dataclass(frozen=True)
class ScenarioSpec:
    label: str
    scenario_name: str
    scenario_params: Dict[str, Any]


class ArchValidationCampaign:
    def __init__(self, campaign_id: str, root_dir: Path, db_path: Path):
        self.campaign_id = campaign_id
        self.root_dir = root_dir
        self.db_path = db_path
        self.logs_dir = self.root_dir / "logs"
        self.aggregates_dir = self.root_dir / "aggregates"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.aggregates_dir.mkdir(parents=True, exist_ok=True)
        self.runs_index_path = self.root_dir / "runs_index.jsonl"

        self.runner = BenchmarkRunner(output_root=self.root_dir)

        self.manifest: Dict[str, Any] = {}
        self.gate_a_results: List[Dict[str, Any]] = []
        self.run_records: List[Dict[str, Any]] = []
        self.cell_master_rows: List[Dict[str, Any]] = []
        self.failure_surface_rows: List[Dict[str, Any]] = []
        self.comparison_records: List[Dict[str, Any]] = []
        self.matrix_d_rows: List[Dict[str, Any]] = []

        self.group_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.group_meta: Dict[str, Dict[str, Any]] = {}

        self.gate_b: Dict[str, Any] = {}
        self.gate_c: Dict[str, Any] = {}
        self.gate_d: Dict[str, Any] = {}
        self.final_verdict = "valor marginal bajo"
        self.start_ts = time.time()

    # ---------------------------- Manifest ----------------------------
    def write_manifest(self) -> None:
        manifest = {
            "campaign_id": self.campaign_id,
            "created_at": datetime.now().isoformat(),
            "db_path": str(self.db_path),
            "objective": "Validación arquitectónica fuerte 1x1 vs 5x5",
            "invariants": {
                "no_runtime_contract_changes": True,
                "no_scenario_episode_runner_changes": True,
                "no_removed_metrics_reintroduced": True,
                "proxy_semantics_explicit": {
                    "closure_rate_proxy": "success_rate",
                    "continuity_proxy": "viability_margin",
                },
            },
            "stopping_rules": {
                "abort_global": [
                    "inconsistencia episodes.jsonl/summary.json/reality_bench_runs",
                    "NaN/Inf/math-domain/ZeroDivision",
                    "reapertura de fallo de contrato",
                ],
                "quarantine_cell": [
                    "error_rate > 0.05",
                    "artifact_missing_rate > 0.01",
                    "data incompleta",
                    "output no cargable por BenchmarkAnalyzer",
                ],
            },
            "matrices": {
                "A": {
                    "episodes": 512,
                    "scenarios": ["grid_thermal_1x1", "grid_thermal_5x5_uniform"],
                    "levels": ["SAFE", "ELEVATED", "WARNING", "CRITICAL"],
                    "replicas_per_cell": 64,
                    "seed_ranges": {
                        "SAFE": [100000, 100063],
                        "ELEVATED": [100100, 100163],
                        "WARNING": [100200, 100263],
                        "CRITICAL": [100300, 100363],
                    },
                },
                "B": {
                    "episodes": 960,
                    "topologies": [
                        "uniform",
                        "hotspot_center",
                        "gradient_ns",
                        "gradient_ew",
                        "checkerboard",
                    ],
                    "levels": ["SAFE", "ELEVATED", "WARNING", "CRITICAL"],
                    "replicas_per_cell": 48,
                },
                "C": {
                    "episodes": 576,
                    "scenarios": [
                        "grid_thermal_1x1",
                        "grid_thermal_5x5_uniform",
                        "grid_thermal_5x5_hotspot_center",
                    ],
                    "conditions": [
                        "tight_margin",
                        "weak_cooling",
                        "near_alarm",
                        "high_temp_tight",
                    ],
                    "replicas_per_cell": 48,
                },
                "D": {
                    "episodes": 320,
                    "selected_cells": 10,
                    "episodes_per_selected_cell": 32,
                    "paired_seed_policy": "16 seed-pairs -> 32 episodios por celda",
                },
            },
            "gates": {
                "B": {
                    "wall_time_ratio_lt": 2.0,
                    "artifact_size_ratio_lt": 3.0,
                    "success_rate_degradation_pp_le": 3.0,
                },
                "C": {
                    "net_cognitive_gain_gt": 0.08,
                    "intervention_precision_delta_pct_ge": 8.0,
                    "improvement_in_levels_ge": 3,
                },
            },
            "outputs": {
                "manifest_json": "manifest.json",
                "runs_index_jsonl": "runs_index.jsonl",
                "cell_master_csv": "cell_master.csv",
                "failure_surface_csv": "failure_surface.csv",
                "final_report_md": "final_report.md",
            },
        }
        self.manifest = manifest
        (self.root_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=True)
        )

    # ---------------------------- Gate A ----------------------------
    def run_gate_a(self) -> None:
        tests = [
            ("contrato", "tests/benchmarks/test_metrics_consistency.py"),
            ("persistencia_analisis", "tests/benchmarks/test_benchmark_pipeline_resilience.py"),
            ("world_semantics", "tests/world/test_grid_thermal_scenario.py"),
            ("integration_5x5", "tests/integration/test_grid_5x5_episode.py"),
            ("benchmark_sanity", "tests/benchmarks/test_1x1_vs_5x5_benchmark.py"),
        ]
        for tag, test_path in tests:
            result = self._run_pytest_case(tag, test_path)
            self.gate_a_results.append(result)
            if not result["passed"]:
                reruns = []
                for i in range(1, 4):
                    rer = self._run_pytest_case(f"{tag}_rerun{i}", test_path, rerun=True)
                    reruns.append(rer["passed"])
                result["reruns"] = reruns
                if all(not x for x in reruns):
                    result["failure_mode"] = "deterministic"
                elif any(reruns) and any(not x for x in reruns):
                    result["failure_mode"] = "flaky"
                else:
                    result["failure_mode"] = "unknown"
                raise CampaignAbort(f"Gate A falló en {test_path} ({result['failure_mode']})")

    def _run_pytest_case(self, tag: str, test_path: str, rerun: bool = False) -> Dict[str, Any]:
        t0 = time.perf_counter()
        proc = subprocess.run(
            ["pytest", "-q", test_path, "-vv"],
            cwd=Path.cwd(),
            text=True,
            capture_output=True,
            check=False,
        )
        dt = time.perf_counter() - t0
        out_file = self.logs_dir / f"gateA_{tag}.log"
        out_file.write_text(proc.stdout + "\n\nSTDERR:\n" + proc.stderr)
        return {
            "tag": tag,
            "test_path": test_path,
            "passed": proc.returncode == 0,
            "duration_sec": dt,
            "rerun": rerun,
            "log_file": str(out_file),
        }

    # ---------------------------- Matrices ----------------------------
    def run_matrix_a(self) -> None:
        levels = [
            ("SAFE", {"initial_temperature": 0.60, "alarm_threshold": 0.85, "cooling_effect": 0.07}, 100000),
            ("ELEVATED", {"initial_temperature": 0.78, "alarm_threshold": 0.85, "cooling_effect": 0.07}, 100100),
            ("WARNING", {"initial_temperature": 0.87, "alarm_threshold": 0.85, "cooling_effect": 0.07}, 100200),
            ("CRITICAL", {"initial_temperature": 0.95, "alarm_threshold": 0.85, "cooling_effect": 0.07}, 100300),
        ]
        scenarios = [
            ScenarioSpec("1x1", "grid_thermal_1x1", {"grid_size": 1}),
            ScenarioSpec("5x5_uniform", "grid_thermal_5x5_uniform", {"grid_size": 5, "topology": "uniform"}),
        ]

        for lidx, (level_name, base_params, seed_base) in enumerate(levels):
            for block_idx in range(8):
                seed_start = seed_base + block_idx * 8
                ordered = scenarios[:]
                random.Random(20260420 + lidx * 100 + block_idx).shuffle(ordered)
                for spec in ordered:
                    params = {**base_params, **spec.scenario_params}
                    run_id = (
                        f"{self.campaign_id}-A-{level_name}-{spec.label}"
                        f"-b{block_idx:02d}-s{seed_start}"
                    )
                    output_dir = (
                        self.root_dir
                        / "matrix_a"
                        / level_name.lower()
                        / spec.label
                        / f"block_{block_idx:02d}"
                    )
                    rows = self._execute_chunk(
                        matrix="A",
                        run_id=run_id,
                        scenario_name=spec.scenario_name,
                        scenario_params=params,
                        episodes=8,
                        base_seed=seed_start,
                        output_dir=output_dir,
                        meta={
                            "level": level_name,
                            "scenario": spec.label,
                            "topology": params.get("topology", "uniform" if params["grid_size"] == 5 else "n/a"),
                            "seed_block": f"{seed_start}-{seed_start+7}",
                        },
                    )
                    group_key = f"A|{level_name}|{spec.label}"
                    self.group_rows[group_key].extend(rows)
                    self.group_meta[group_key] = {
                        "matrix": "A",
                        "level": level_name,
                        "scenario": spec.label,
                        "topology": params.get("topology", "n/a"),
                        "seed_block": "paired_blocks_8",
                    }

        self._finalize_groups_for_matrix("A")
        self._compare_matrix_a_levels()

    def run_matrix_b(self) -> None:
        levels = [
            ("SAFE", {"initial_temperature": 0.60, "alarm_threshold": 0.85, "cooling_effect": 0.07}, 200000),
            ("ELEVATED", {"initial_temperature": 0.78, "alarm_threshold": 0.85, "cooling_effect": 0.07}, 200100),
            ("WARNING", {"initial_temperature": 0.87, "alarm_threshold": 0.85, "cooling_effect": 0.07}, 200200),
            ("CRITICAL", {"initial_temperature": 0.95, "alarm_threshold": 0.85, "cooling_effect": 0.07}, 200300),
        ]
        topologies = ["uniform", "hotspot_center", "gradient_ns", "gradient_ew", "checkerboard"]

        for lidx, (level_name, base_params, seed_base) in enumerate(levels):
            for block_idx in range(6):
                seed_start = seed_base + block_idx * 8
                ordered_topos = topologies[:]
                random.Random(20260420 + 1000 + lidx * 100 + block_idx).shuffle(ordered_topos)
                for topo in ordered_topos:
                    params = {**base_params, "grid_size": 5, "topology": topo}
                    run_id = (
                        f"{self.campaign_id}-B-{level_name}-{topo}"
                        f"-b{block_idx:02d}-s{seed_start}"
                    )
                    output_dir = (
                        self.root_dir
                        / "matrix_b"
                        / level_name.lower()
                        / topo
                        / f"block_{block_idx:02d}"
                    )
                    rows = self._execute_chunk(
                        matrix="B",
                        run_id=run_id,
                        scenario_name=f"grid_thermal_5x5_{topo}",
                        scenario_params=params,
                        episodes=8,
                        base_seed=seed_start,
                        output_dir=output_dir,
                        meta={
                            "level": level_name,
                            "scenario": "5x5",
                            "topology": topo,
                            "seed_block": f"{seed_start}-{seed_start+7}",
                        },
                    )
                    group_key = f"B|{level_name}|{topo}"
                    self.group_rows[group_key].extend(rows)
                    self.group_meta[group_key] = {
                        "matrix": "B",
                        "level": level_name,
                        "scenario": "5x5",
                        "topology": topo,
                        "seed_block": "paired_blocks_8",
                    }

        self._finalize_groups_for_matrix("B")
        self._compare_matrix_b_topologies()

    def run_matrix_c(self) -> None:
        conditions = [
            ("tight_margin", {"initial_temperature": 0.82, "alarm_threshold": 0.85, "cooling_effect": 0.07}, 300000),
            ("weak_cooling", {"initial_temperature": 0.82, "alarm_threshold": 0.85, "cooling_effect": 0.04}, 300100),
            ("near_alarm", {"initial_temperature": 0.84, "alarm_threshold": 0.85, "cooling_effect": 0.07}, 300200),
            ("high_temp_tight", {"initial_temperature": 0.96, "alarm_threshold": 0.98, "cooling_effect": 0.04}, 300300),
        ]
        scenarios = [
            ScenarioSpec("1x1", "grid_thermal_1x1", {"grid_size": 1}),
            ScenarioSpec("5x5_uniform", "grid_thermal_5x5_uniform", {"grid_size": 5, "topology": "uniform"}),
            ScenarioSpec(
                "5x5_hotspot_center",
                "grid_thermal_5x5_hotspot_center",
                {"grid_size": 5, "topology": "hotspot_center"},
            ),
        ]

        for cidx, (cond_name, base_params, seed_base) in enumerate(conditions):
            for block_idx in range(6):
                seed_start = seed_base + block_idx * 8
                ordered = scenarios[:]
                random.Random(20260420 + 2000 + cidx * 100 + block_idx).shuffle(ordered)
                for spec in ordered:
                    params = {**base_params, **spec.scenario_params}
                    run_id = (
                        f"{self.campaign_id}-C-{cond_name}-{spec.label}"
                        f"-b{block_idx:02d}-s{seed_start}"
                    )
                    output_dir = (
                        self.root_dir
                        / "matrix_c"
                        / cond_name
                        / spec.label
                        / f"block_{block_idx:02d}"
                    )
                    rows = self._execute_chunk(
                        matrix="C",
                        run_id=run_id,
                        scenario_name=spec.scenario_name,
                        scenario_params=params,
                        episodes=8,
                        base_seed=seed_start,
                        output_dir=output_dir,
                        meta={
                            "condition": cond_name,
                            "scenario": spec.label,
                            "topology": params.get("topology", "n/a"),
                            "seed_block": f"{seed_start}-{seed_start+7}",
                        },
                    )
                    group_key = f"C|{cond_name}|{spec.label}"
                    self.group_rows[group_key].extend(rows)
                    self.group_meta[group_key] = {
                        "matrix": "C",
                        "condition": cond_name,
                        "scenario": spec.label,
                        "topology": params.get("topology", "n/a"),
                        "seed_block": "paired_blocks_8",
                    }

        self._finalize_groups_for_matrix("C")
        self._compare_matrix_c_conditions()

    def run_matrix_d(self) -> None:
        selected = self._select_cells_for_matrix_d()
        for idx, record in enumerate(selected):
            seed_base = 400000 + idx * 1000
            left = record["left_spec"]
            right = record["right_spec"]
            left_rows: List[Dict[str, Any]] = []
            right_rows: List[Dict[str, Any]] = []

            # 16 seed-pairs => 32 episodios/celda (total D=320)
            for block_idx in range(2):
                seed_start = seed_base + block_idx * 8
                pair = [left, right]
                random.Random(20260420 + 3000 + idx * 10 + block_idx).shuffle(pair)
                for side_idx, spec in enumerate(pair):
                    run_id = (
                        f"{self.campaign_id}-D-cell{idx:02d}-{spec['label']}"
                        f"-b{block_idx:02d}-s{seed_start}"
                    )
                    output_dir = (
                        self.root_dir
                        / "matrix_d"
                        / f"cell_{idx:02d}"
                        / spec["label"]
                        / f"block_{block_idx:02d}"
                    )
                    rows = self._execute_chunk(
                        matrix="D",
                        run_id=run_id,
                        scenario_name=spec["scenario_name"],
                        scenario_params=spec["scenario_params"],
                        episodes=8,
                        base_seed=seed_start,
                        output_dir=output_dir,
                        meta={
                            "selected_from": record["record_id"],
                            "cell_idx": idx,
                            "side": spec["label"],
                            "seed_block": f"{seed_start}-{seed_start+7}",
                        },
                    )
                    if spec["label"] == left["label"]:
                        left_rows.extend(rows)
                    else:
                        right_rows.extend(rows)

            comp = self._compute_comparison_record(
                matrix="D",
                record_id=f"D-cell-{idx:02d}",
                left_spec=left,
                right_spec=right,
                left_rows=left_rows,
                right_rows=right_rows,
                context={
                    "source_record_id": record["record_id"],
                    "source_matrix": record["matrix"],
                    "selected_cell": idx,
                },
            )
            source_sign = _sign(record["net_gain"])
            rep_sign = _sign(comp["net_gain"])
            stable = (
                source_sign == rep_sign
                or abs(record["net_gain"]) < 0.03
                or abs(comp["net_gain"]) < 0.03
            )
            self.matrix_d_rows.append(
                {
                    "matrix": "D",
                    "cell_id": f"D-cell-{idx:02d}",
                    "source_record_id": record["record_id"],
                    "source_net_gain": record["net_gain"],
                    "repeat_net_gain": comp["net_gain"],
                    "source_sign": source_sign,
                    "repeat_sign": rep_sign,
                    "stable_direction": stable,
                    "left_label": left["label"],
                    "right_label": right["label"],
                }
            )

        stable_count = sum(1 for r in self.matrix_d_rows if r["stable_direction"])
        strong_reversals = sum(
            1
            for r in self.matrix_d_rows
            if abs(r["source_net_gain"]) >= 0.08
            and abs(r["repeat_net_gain"]) >= 0.03
            and _sign(r["source_net_gain"]) != _sign(r["repeat_net_gain"])
        )
        self.gate_d = {
            "stable_count": stable_count,
            "total": len(self.matrix_d_rows),
            "stable_ratio": stable_count / len(self.matrix_d_rows) if self.matrix_d_rows else 0.0,
            "strong_reversals": strong_reversals,
            "passed": stable_count >= 8 and strong_reversals <= 2,
        }

    # ---------------------------- Core execution ----------------------------
    def _execute_chunk(
        self,
        *,
        matrix: str,
        run_id: str,
        scenario_name: str,
        scenario_params: Dict[str, Any],
        episodes: int,
        base_seed: int,
        output_dir: Path,
        meta: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        output_dir.mkdir(parents=True, exist_ok=True)
        cfg = BenchmarkConfig(
            scenario_name=scenario_name,
            scenario_class=GridThermalScenario,
            scenario_params=scenario_params,
            episodes=episodes,
            base_seed=base_seed,
            max_steps=50,
            output_dir=output_dir,
            run_id=run_id,
        )
        log_file = self.logs_dir / f"{run_id}.log"
        t0 = time.perf_counter()
        with log_file.open("w") as lf, contextlib.redirect_stdout(lf):
            summary = self.runner.run_benchmark(cfg)
        duration = time.perf_counter() - t0

        rows = self._validate_run_output(
            run_id=run_id,
            output_dir=output_dir,
            expected_episodes=episodes,
            summary_obj=summary,
        )
        self._validate_db_record(run_id)

        record = {
            "matrix": matrix,
            "run_id": run_id,
            "scenario_name": scenario_name,
            "scenario_params": scenario_params,
            "episodes": episodes,
            "base_seed": base_seed,
            "seed_range": [base_seed, base_seed + episodes - 1],
            "output_dir": str(output_dir),
            "summary_file": str(output_dir / "summary.json"),
            "episodes_file": str(output_dir / "episodes.jsonl"),
            "duration_sec": duration,
            "meta": meta,
        }
        self.run_records.append(record)
        with self.runs_index_path.open("a") as f:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")
        return rows

    def _validate_run_output(
        self,
        *,
        run_id: str,
        output_dir: Path,
        expected_episodes: int,
        summary_obj: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        episodes_file = output_dir / "episodes.jsonl"
        summary_file = output_dir / "summary.json"
        if not episodes_file.exists() or not summary_file.exists():
            raise CampaignAbort(f"{run_id}: faltan artifacts canónicos")

        with episodes_file.open() as f:
            rows = [json.loads(line) for line in f if line.strip()]
        with summary_file.open() as f:
            summary_disk = json.load(f)

        if len(rows) != expected_episodes:
            raise CampaignAbort(
                f"{run_id}: episodes.jsonl incompleto "
                f"(esperado={expected_episodes}, obtenido={len(rows)})"
            )
        if summary_disk.get("total_episodes") != expected_episodes:
            raise CampaignAbort(f"{run_id}: summary total_episodes inconsistente")

        # Coherencia summary memoria/disco
        if summary_obj.get("total_episodes") != summary_disk.get("total_episodes"):
            raise CampaignAbort(f"{run_id}: inconsistencia summary memoria vs disco")

        summary_dump = _as_json(summary_disk)
        if any(metric in summary_dump for metric in REMOVED_METRICS):
            raise CampaignAbort(f"{run_id}: métricas removidas detectadas en summary")

        for idx, row in enumerate(rows):
            missing = [k for k in REQUIRED_EPISODE_FIELDS if k not in row]
            if missing:
                raise CampaignAbort(f"{run_id}: fila {idx} incompleta, faltan {missing}")
            row_dump = _as_json(row)
            if any(metric in row_dump for metric in REMOVED_METRICS):
                raise CampaignAbort(f"{run_id}: métricas removidas detectadas en episodes")
            _assert_finite_tree(row, f"{run_id}.episodes[{idx}]")

        _assert_finite_tree(summary_disk, f"{run_id}.summary")
        return rows

    def _validate_db_record(self, run_id: str) -> None:
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute(
            "SELECT summary FROM reality_bench_runs WHERE run_id = ? ORDER BY created_at DESC LIMIT 1",
            (run_id,),
        )
        row = cur.fetchone()
        conn.close()
        if not row:
            raise CampaignAbort(f"{run_id}: fila faltante en reality_bench_runs")

        payload = json.loads(row[0]) if isinstance(row[0], str) else row[0]
        pm = payload.get("proxy_mapping", {})
        if pm.get("closure_rate") != "success_rate" or pm.get("continuity_mean") != "viability_margin":
            raise CampaignAbort(f"{run_id}: proxy_mapping inconsistente en DB")

        dump = _as_json(payload)
        if any(metric in dump for metric in REMOVED_METRICS):
            raise CampaignAbort(f"{run_id}: métricas removidas detectadas en DB summary")
        _assert_finite_tree(payload, f"{run_id}.db_summary")

    # ---------------------------- Aggregation ----------------------------
    def _finalize_groups_for_matrix(self, matrix: str) -> None:
        for group_key, rows in sorted(self.group_rows.items()):
            if not group_key.startswith(f"{matrix}|"):
                continue
            meta = self.group_meta[group_key]
            aggregate = self._aggregate_group_rows(rows)
            status = self._cell_status(aggregate)

            row = {
                "matrix": meta.get("matrix"),
                "scenario": meta.get("scenario"),
                "topology": meta.get("topology", "n/a"),
                "level": meta.get("level", ""),
                "condition": meta.get("condition", ""),
                "seed_block": meta.get("seed_block", ""),
                "n_episodes": aggregate["n_episodes"],
                "success_rate": aggregate["success_rate"],
                "viability_margin_mean": aggregate["viability_margin_mean"],
                "intervention_precision_mean": aggregate["intervention_precision_mean"],
                "ivc_r_mean": aggregate["ivc_r_mean"],
                "wall_time_ms_mean": aggregate["wall_time_ms_mean"],
                "artifact_size_bytes_mean": aggregate["artifact_size_bytes_mean"],
                "reasoning_trace_length_mean": aggregate["reasoning_trace_length_mean"],
                "error_rate": aggregate["error_rate"],
                "artifact_missing_rate": aggregate["artifact_missing_rate"],
                "failure_distribution": json.dumps(aggregate["failure_distribution"], ensure_ascii=True),
                "status": status,
                "group_key": group_key,
            }
            self.cell_master_rows.append(row)

            total = aggregate["n_episodes"] if aggregate["n_episodes"] > 0 else 1
            for failure_primary, count in aggregate["failure_distribution"].items():
                self.failure_surface_rows.append(
                    {
                        "matrix": matrix,
                        "scenario": row["scenario"],
                        "topology": row["topology"],
                        "level": row["level"],
                        "condition": row["condition"],
                        "failure_primary": failure_primary,
                        "count": count,
                        "rate": count / total,
                        "wall_time_ms_mean": row["wall_time_ms_mean"],
                        "artifact_size_bytes_mean": row["artifact_size_bytes_mean"],
                        "ivc_r_mean": row["ivc_r_mean"],
                        "viability_margin_mean": row["viability_margin_mean"],
                        "status": status,
                    }
                )

    def _aggregate_group_rows(self, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        n = len(rows)
        failures = Counter()
        missing_artifacts = 0
        non_error_rows = 0
        incomplete_rows = 0
        error_rows = 0

        viability = []
        precision = []
        ivc_r = []
        wall_time = []
        artifact = []
        trace_len = []
        success_proxy = []

        for row in rows:
            for k in REQUIRED_EPISODE_FIELDS:
                if k not in row:
                    incomplete_rows += 1
                    break

            outcome = row.get("outcome")
            if outcome == "error":
                error_rows += 1
            else:
                non_error_rows += 1
                artifact_path = row.get("artifact_path")
                if not artifact_path or not Path(str(artifact_path)).exists():
                    missing_artifacts += 1

            failure_primary = row.get("failure_primary")
            if failure_primary:
                failures[str(failure_primary)] += 1

            success_proxy.append(_safe_float(row.get("success_rate")))
            if row.get("viability_margin") is not None:
                viability.append(_safe_float(row.get("viability_margin")))
            if row.get("intervention_precision") is not None:
                precision.append(_safe_float(row.get("intervention_precision")))
            if row.get("ivc_r") is not None:
                ivc_r.append(_safe_float(row.get("ivc_r")))
            if row.get("wall_time_ms") is not None:
                wall_time.append(_safe_float(row.get("wall_time_ms")))
            if row.get("artifact_size_bytes") is not None:
                artifact.append(_safe_float(row.get("artifact_size_bytes")))
            if row.get("reasoning_trace_length") is not None:
                trace_len.append(_safe_float(row.get("reasoning_trace_length")))

        artifact_missing_rate = (
            missing_artifacts / non_error_rows if non_error_rows > 0 else 0.0
        )
        return {
            "n_episodes": n,
            "success_rate": _mean(success_proxy),
            "viability_margin_mean": _mean(viability),
            "intervention_precision_mean": _mean(precision),
            "ivc_r_mean": _mean(ivc_r),
            "wall_time_ms_mean": _mean(wall_time),
            "artifact_size_bytes_mean": _mean(artifact),
            "reasoning_trace_length_mean": _mean(trace_len),
            "error_rate": (error_rows / n) if n else 0.0,
            "artifact_missing_rate": artifact_missing_rate,
            "incomplete_rows": incomplete_rows,
            "failure_distribution": dict(failures),
        }

    def _cell_status(self, aggregate: Dict[str, Any]) -> str:
        if (
            aggregate["error_rate"] > 0.05
            or aggregate["artifact_missing_rate"] > 0.01
            or aggregate["incomplete_rows"] > 0
        ):
            return "quarantined"
        if aggregate["error_rate"] > 0.0 or aggregate["artifact_missing_rate"] > 0.0:
            return "degraded"
        return "passed"

    # ---------------------------- Comparisons ----------------------------
    def _write_aggregate_episodes(self, name: str, rows: List[Dict[str, Any]]) -> Path:
        out_dir = self.aggregates_dir / name
        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / "episodes.jsonl"
        with target.open("w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=True) + "\n")
        return out_dir

    def _compute_comparison_record(
        self,
        *,
        matrix: str,
        record_id: str,
        left_spec: Dict[str, Any],
        right_spec: Dict[str, Any],
        left_rows: List[Dict[str, Any]],
        right_rows: List[Dict[str, Any]],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        left_dir = self._write_aggregate_episodes(f"{record_id}_left", left_rows)
        right_dir = self._write_aggregate_episodes(f"{record_id}_right", right_rows)

        analyzer = BenchmarkAnalyzer()
        analyzer.load_results(left_dir, right_dir)
        comparison = analyzer.compute_statistical_comparison()
        _assert_finite_tree(comparison, f"comparison.{record_id}")

        net = comparison.get("net_cognitive_gain", {})
        precision_stats = comparison.get("intervention_precision", {})
        success_stats = comparison.get("success_rate", {})
        fail_stats = comparison.get("failure_analysis", {})
        dist_right = fail_stats.get("distribution_5x5", {})  # second dataset

        top_failure = None
        if dist_right:
            top_failure = max(dist_right.items(), key=lambda kv: kv[1])[0]

        rec = {
            "matrix": matrix,
            "record_id": record_id,
            "left_label": left_spec["label"],
            "right_label": right_spec["label"],
            "left_spec": left_spec,
            "right_spec": right_spec,
            "context": context,
            "net_gain": _safe_float(net.get("ganancia_neta")),
            "net_gain_bruta": _safe_float(net.get("ganancia_bruta")),
            "cost_penalty": _safe_float(net.get("penalizacion_costo")),
            "wall_time_ratio": _safe_float(net.get("wall_time_ratio"), 1.0),
            "artifact_ratio": _safe_float(net.get("artifact_ratio"), 1.0),
            "precision_delta": _safe_float(precision_stats.get("delta")),
            "precision_delta_pct": _safe_float(precision_stats.get("delta_pct")),
            "success_delta": _safe_float(success_stats.get("delta")),
            "success_delta_pct": _safe_float(success_stats.get("delta_pct")),
            "top_failure_right": top_failure,
            "comparison": comparison,
        }
        self.comparison_records.append(rec)
        return rec

    def _compare_matrix_a_levels(self) -> None:
        levels = ["SAFE", "ELEVATED", "WARNING", "CRITICAL"]
        level_records = []
        for level in levels:
            left_rows = self.group_rows[f"A|{level}|1x1"]
            right_rows = self.group_rows[f"A|{level}|5x5_uniform"]
            rec = self._compute_comparison_record(
                matrix="A",
                record_id=f"A-{level}",
                left_spec={
                    "label": "1x1",
                    "scenario_name": "grid_thermal_1x1",
                    "scenario_params": {"grid_size": 1},
                },
                right_spec={
                    "label": "5x5_uniform",
                    "scenario_name": "grid_thermal_5x5_uniform",
                    "scenario_params": {"grid_size": 5, "topology": "uniform"},
                },
                left_rows=left_rows,
                right_rows=right_rows,
                context={"level": level},
            )
            level_records.append(rec)

        # Gate B
        gate_b_rows = []
        for rec in level_records:
            pass_row = (
                rec["wall_time_ratio"] < 2.0
                and rec["artifact_ratio"] < 3.0
                and rec["success_delta"] >= -0.03
            )
            gate_b_rows.append(
                {
                    "level": rec["context"]["level"],
                    "wall_time_ratio": rec["wall_time_ratio"],
                    "artifact_ratio": rec["artifact_ratio"],
                    "success_delta_pp": rec["success_delta"] * 100,
                    "passed": pass_row,
                }
            )
        self.gate_b = {
            "rows": gate_b_rows,
            "passed": all(r["passed"] for r in gate_b_rows),
        }

        # Gate C
        overall_net = _mean([r["net_gain"] for r in level_records])
        overall_precision_pct = _mean([r["precision_delta_pct"] for r in level_records])
        levels_with_signal = sum(
            1
            for r in level_records
            if r["net_gain"] > 0.08 and r["precision_delta_pct"] >= 8.0
        )

        # nueva categoría dominante crítica en 5x5 vs 1x1
        dist_1x1 = Counter()
        dist_5x5 = Counter()
        for row in self.group_rows["A|SAFE|1x1"] + self.group_rows["A|ELEVATED|1x1"] + self.group_rows["A|WARNING|1x1"] + self.group_rows["A|CRITICAL|1x1"]:
            if row.get("failure_primary"):
                dist_1x1[row["failure_primary"]] += 1
        for row in self.group_rows["A|SAFE|5x5_uniform"] + self.group_rows["A|ELEVATED|5x5_uniform"] + self.group_rows["A|WARNING|5x5_uniform"] + self.group_rows["A|CRITICAL|5x5_uniform"]:
            if row.get("failure_primary"):
                dist_5x5[row["failure_primary"]] += 1

        n1 = len(self.group_rows["A|SAFE|1x1"] + self.group_rows["A|ELEVATED|1x1"] + self.group_rows["A|WARNING|1x1"] + self.group_rows["A|CRITICAL|1x1"])
        n5 = len(self.group_rows["A|SAFE|5x5_uniform"] + self.group_rows["A|ELEVATED|5x5_uniform"] + self.group_rows["A|WARNING|5x5_uniform"] + self.group_rows["A|CRITICAL|5x5_uniform"])
        critical_failures = {"viability_failed", "both_failed"}
        critical_dominant = False
        for failure_name in critical_failures:
            r1 = dist_1x1.get(failure_name, 0) / n1 if n1 else 0.0
            r5 = dist_5x5.get(failure_name, 0) / n5 if n5 else 0.0
            if r5 - r1 > 0.05 and dist_5x5.get(failure_name, 0) >= max(dist_5x5.values(), default=0):
                critical_dominant = True
                break

        self.gate_c = {
            "overall_net_gain": overall_net,
            "overall_precision_delta_pct": overall_precision_pct,
            "levels_with_signal": levels_with_signal,
            "critical_failure_dominant": critical_dominant,
            "passed": (
                overall_net > 0.08
                and overall_precision_pct >= 8.0
                and levels_with_signal >= 3
                and not critical_dominant
            ),
        }

    def _compare_matrix_b_topologies(self) -> None:
        levels = ["SAFE", "ELEVATED", "WARNING", "CRITICAL"]
        for level in levels:
            uniform_rows = self.group_rows[f"B|{level}|uniform"]
            for topo in ["hotspot_center", "gradient_ns", "gradient_ew", "checkerboard"]:
                topo_rows = self.group_rows[f"B|{level}|{topo}"]
                self._compute_comparison_record(
                    matrix="B",
                    record_id=f"B-{level}-{topo}-vs-uniform",
                    left_spec={
                        "label": "uniform",
                        "scenario_name": "grid_thermal_5x5_uniform",
                        "scenario_params": {"grid_size": 5, "topology": "uniform"},
                    },
                    right_spec={
                        "label": topo,
                        "scenario_name": f"grid_thermal_5x5_{topo}",
                        "scenario_params": {"grid_size": 5, "topology": topo},
                    },
                    left_rows=uniform_rows,
                    right_rows=topo_rows,
                    context={"level": level, "topology": topo},
                )

    def _compare_matrix_c_conditions(self) -> None:
        for cond in ["tight_margin", "weak_cooling", "near_alarm", "high_temp_tight"]:
            rows_1x1 = self.group_rows[f"C|{cond}|1x1"]
            rows_uniform = self.group_rows[f"C|{cond}|5x5_uniform"]
            rows_hotspot = self.group_rows[f"C|{cond}|5x5_hotspot_center"]

            self._compute_comparison_record(
                matrix="C",
                record_id=f"C-{cond}-uniform-vs-1x1",
                left_spec={
                    "label": "1x1",
                    "scenario_name": "grid_thermal_1x1",
                    "scenario_params": {"grid_size": 1},
                },
                right_spec={
                    "label": "5x5_uniform",
                    "scenario_name": "grid_thermal_5x5_uniform",
                    "scenario_params": {"grid_size": 5, "topology": "uniform"},
                },
                left_rows=rows_1x1,
                right_rows=rows_uniform,
                context={"condition": cond, "contrast": "uniform_vs_1x1"},
            )
            self._compute_comparison_record(
                matrix="C",
                record_id=f"C-{cond}-hotspot-vs-1x1",
                left_spec={
                    "label": "1x1",
                    "scenario_name": "grid_thermal_1x1",
                    "scenario_params": {"grid_size": 1},
                },
                right_spec={
                    "label": "5x5_hotspot_center",
                    "scenario_name": "grid_thermal_5x5_hotspot_center",
                    "scenario_params": {"grid_size": 5, "topology": "hotspot_center"},
                },
                left_rows=rows_1x1,
                right_rows=rows_hotspot,
                context={"condition": cond, "contrast": "hotspot_vs_1x1"},
            )

    def _select_cells_for_matrix_d(self) -> List[Dict[str, Any]]:
        pool = [r for r in self.comparison_records if r["matrix"] in {"A", "B", "C"}]
        fav_1x1 = sorted([r for r in pool if r["net_gain"] <= -0.08], key=lambda x: x["net_gain"])[:3]
        fav_5x5 = sorted([r for r in pool if r["net_gain"] >= 0.08], key=lambda x: -x["net_gain"])[:3]
        ambiguous = [
            r
            for r in pool
            if abs(r["net_gain"]) < 0.03
            or (
                r["net_gain"] > 0
                and (r["wall_time_ratio"] >= 2.0 or r["artifact_ratio"] >= 3.0 or r["success_delta"] < -0.03)
            )
        ]
        ambiguous = sorted(ambiguous, key=lambda x: abs(x["net_gain"]))[:4]

        selected: List[Dict[str, Any]] = []
        seen = set()
        for chunk in (fav_1x1, fav_5x5, ambiguous):
            for rec in chunk:
                if rec["record_id"] not in seen:
                    selected.append(rec)
                    seen.add(rec["record_id"])

        if len(selected) < 10:
            fillers = sorted(pool, key=lambda x: abs(abs(x["net_gain"]) - 0.08))
            for rec in fillers:
                if rec["record_id"] not in seen:
                    selected.append(rec)
                    seen.add(rec["record_id"])
                if len(selected) >= 10:
                    break
        return selected[:10]

    # ---------------------------- Outputs ----------------------------
    def write_csv_outputs(self) -> None:
        cell_master_path = self.root_dir / "cell_master.csv"
        failure_surface_path = self.root_dir / "failure_surface.csv"

        cell_fields = [
            "matrix",
            "scenario",
            "topology",
            "level",
            "condition",
            "seed_block",
            "n_episodes",
            "success_rate",
            "viability_margin_mean",
            "intervention_precision_mean",
            "ivc_r_mean",
            "wall_time_ms_mean",
            "artifact_size_bytes_mean",
            "reasoning_trace_length_mean",
            "error_rate",
            "artifact_missing_rate",
            "failure_distribution",
            "status",
            "group_key",
        ]
        with cell_master_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cell_fields)
            w.writeheader()
            for row in self.cell_master_rows:
                w.writerow(row)

        failure_fields = [
            "matrix",
            "scenario",
            "topology",
            "level",
            "condition",
            "failure_primary",
            "count",
            "rate",
            "wall_time_ms_mean",
            "artifact_size_bytes_mean",
            "ivc_r_mean",
            "viability_margin_mean",
            "status",
        ]
        with failure_surface_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=failure_fields)
            w.writeheader()
            for row in self.failure_surface_rows:
                w.writerow(row)

    def decide_verdict(self) -> None:
        gate_a_pass = all(r["passed"] for r in self.gate_a_results)
        gate_b_pass = self.gate_b.get("passed", False)
        gate_c_pass = self.gate_c.get("passed", False)
        gate_d_pass = self.gate_d.get("passed", False)

        any_strong_positive = any(
            r["net_gain"] > 0.08 for r in self.comparison_records if r["matrix"] in {"A", "B", "C"}
        )

        if gate_a_pass and gate_b_pass and gate_c_pass and gate_d_pass:
            self.final_verdict = "valor arquitectónico demostrado"
        elif gate_a_pass and gate_b_pass and any_strong_positive:
            self.final_verdict = "avance parcial con señal condicionada"
        elif gate_a_pass and gate_b_pass:
            self.final_verdict = "valor marginal bajo"
        else:
            self.final_verdict = "congelar rama experimental"

    def write_final_report(self) -> None:
        report_path = self.root_dir / "final_report.md"
        gate_a_pass = all(r["passed"] for r in self.gate_a_results)

        lines: List[str] = []
        lines.append("# Campaña Fuerte de Validación Arquitectónica")
        lines.append("")
        lines.append(f"- campaign_id: `{self.campaign_id}`")
        lines.append(f"- duración_total_s: `{time.time() - self.start_ts:.2f}`")
        lines.append(f"- db: `{self.db_path}`")
        lines.append("")

        lines.append("## A. Estado del Gate A")
        lines.append("| Test | Passed | Duración (s) | Log |")
        lines.append("|---|---:|---:|---|")
        for r in self.gate_a_results:
            lines.append(
                f"| `{r['test_path']}` | `{r['passed']}` | `{r['duration_sec']:.2f}` | `{r['log_file']}` |"
            )
        lines.append(f"- Gate A passed: `{gate_a_pass}`")
        lines.append("")

        lines.append("## B. Resultados de la Matriz A")
        lines.append("| Nivel | Net Gain | Precision Δ% | Success Δpp | Wall Ratio | Artifact Ratio |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for rec in [x for x in self.comparison_records if x["matrix"] == "A"]:
            lines.append(
                f"| {rec['context']['level']} | {rec['net_gain']:.4f} | {rec['precision_delta_pct']:.2f} | "
                f"{rec['success_delta']*100:.2f} | {rec['wall_time_ratio']:.3f} | {rec['artifact_ratio']:.3f} |"
            )
        lines.append(f"- Gate B passed: `{self.gate_b.get('passed', False)}`")
        lines.append("")

        lines.append("## C. Resultados de la Matriz B")
        b_records = [x for x in self.comparison_records if x["matrix"] == "B"]
        topologies_signal = sorted(
            {
                r["context"]["topology"]
                for r in b_records
                if r["net_gain"] > 0.08 and r["precision_delta_pct"] >= 8.0
            }
        )
        topologies_no_signal = sorted(
            {
                r["context"]["topology"]
                for r in b_records
                if not (r["net_gain"] > 0.08 and r["precision_delta_pct"] >= 8.0)
            }
        )
        lines.append(f"- topologías_con_señal: `{topologies_signal}`")
        lines.append(f"- topologías_sin_señal: `{topologies_no_signal}`")
        lines.append("")

        lines.append("## D. Resultados de la Matriz C")
        c_records = [x for x in self.comparison_records if x["matrix"] == "C"]
        lines.append("| Condición | Contraste | Net Gain | Top Failure (right) |")
        lines.append("|---|---|---:|---|")
        for rec in c_records:
            lines.append(
                f"| {rec['context']['condition']} | {rec['context']['contrast']} | "
                f"{rec['net_gain']:.4f} | {rec.get('top_failure_right')} |"
            )
        lines.append("")

        lines.append("## E. Resultados de la Matriz D")
        lines.append("| Cell | Source Record | Source Gain | Repeat Gain | Stable |")
        lines.append("|---|---|---:|---:|---:|")
        for row in self.matrix_d_rows:
            lines.append(
                f"| {row['cell_id']} | {row['source_record_id']} | "
                f"{row['source_net_gain']:.4f} | {row['repeat_net_gain']:.4f} | {row['stable_direction']} |"
            )
        lines.append(
            f"- Gate D passed: `{self.gate_d.get('passed', False)}` "
            f"(stable={self.gate_d.get('stable_count', 0)}/{self.gate_d.get('total', 0)})"
        )
        lines.append("")

        lines.append("## F. Dictamen final")
        lines.append(f"- `{self.final_verdict}`")
        lines.append("")

        lines.append("## G. Riesgos residuales")
        lines.append("- semánticos: success_rate/viability_margin siguen siendo proxies explícitos.")
        lines.append("- numéricos: robustez validada en escenarios degenerados, mantener vigilancia en campañas mayores.")
        lines.append("- metodológicos: señal puede ser local por topología/nivel; dictamen depende de repetibilidad.")
        lines.append("")

        report_path.write_text("\n".join(lines))

    # ---------------------------- Pipeline ----------------------------
    def run(self) -> None:
        self.write_manifest()
        self.run_gate_a()
        self.run_matrix_a()
        self.run_matrix_b()
        self.run_matrix_c()
        self.run_matrix_d()
        self.write_csv_outputs()
        self.decide_verdict()
        self.write_final_report()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Campaña fuerte de validación arquitectónica 1x1 vs 5x5.")
    parser.add_argument(
        "--campaign-id",
        default=f"archval_{_now_stamp()}",
        help="Identificador de campaña.",
    )
    parser.add_argument(
        "--output-root",
        default="data/benchmarks/arch_validation",
        help="Directorio raíz de salida.",
    )
    parser.add_argument(
        "--db-path",
        default=os.environ.get("AEON_EVENT_DB", "aeon_event_log.db"),
        help="Ruta de DB SQLite para reality_bench_runs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.output_root) / args.campaign_id
    root.mkdir(parents=True, exist_ok=True)
    db_path = Path(args.db_path)
    if not db_path.exists():
        raise SystemExit(f"DB no encontrada: {db_path}")

    campaign = ArchValidationCampaign(
        campaign_id=args.campaign_id,
        root_dir=root,
        db_path=db_path,
    )
    campaign.run()

    print("ARCH_VALIDATION_DONE")
    print(f"campaign_id={args.campaign_id}")
    print(f"output_root={root}")
    print(f"final_report={root / 'final_report.md'}")
    print(f"cell_master={root / 'cell_master.csv'}")
    print(f"failure_surface={root / 'failure_surface.csv'}")
    print(f"runs_index={root / 'runs_index.jsonl'}")
    print(f"verdict={campaign.final_verdict}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CampaignAbort as exc:
        print(f"ARCH_VALIDATION_ABORT: {exc}")
        raise SystemExit(2)
