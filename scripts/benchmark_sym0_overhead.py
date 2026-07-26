#!/usr/bin/env python3
"""Benchmark pareado del overhead de SYM-0 con Z3 real.

El runner instrumenta siempre, por contrato. Para obtener un baseline sin cambiar
el código productivo, este script sustituye temporalmente y solo en proceso los
adaptadores SYM-0 por no-ops. DED deja entonces de reconocer el registry y usa su
Solver Z3 desnudo compatible con el camino anterior.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime.storage import StorageConfig, StorageFactory
from runtime.world import ScenarioEpisodeRunner, ThermalScenario
from runtime.world import scenario_runner as runner_module


class _BaselineRegistry:
    """Registry no-op cuya clase no activa TrackedSolver en DED."""

    latest_core_report = None

    def __init__(self, **_: Any):
        pass

    def get_core_report(self, **_: Any):
        return None

    def export_jsonl(self, _path: str | Path) -> Path:
        return Path(_path)


class _BaselineCollector:
    """Collector no-op que satisface únicamente el contrato del runner."""

    trace_status = "baseline_disabled"

    def __init__(self, *, trace_dir: str | Path, **_: Any):
        self.trace_dir = Path(trace_dir)

    @property
    def paths(self) -> dict[str, str]:
        return {
            name: str(self.trace_dir / f"{name}.disabled")
            for name in ("constraints", "core_report", "decision_trace", "outcome_link")
        }

    def set_baseline_action(self, _action: str) -> None:
        pass

    def set_ded_report(self, _report: object) -> None:
        pass

    def set_opt_report(self, _report: object) -> None:
        pass

    def set_guard_report(self, _report: object) -> None:
        pass

    def mark_persistence_degraded(self) -> None:
        pass

    def seal(self, **_: Any):
        return SimpleNamespace(sealed_hash=None)

    def link_outcome(self, **_: Any):
        return SimpleNamespace(to_dict=lambda: {})


class _BaselineTrackedSolver:
    def __init__(self, **_: Any):
        pass


@contextmanager
def _baseline_adapters() -> Iterator[None]:
    replacements = {
        "ConstraintRegistry": _BaselineRegistry,
        "TrackedSolver": _BaselineTrackedSolver,
        "ActingTraceCollector": _BaselineCollector,
        "persist_core_report": lambda report, path: Path(path),
        "build_optimization_report": lambda *args, **kwargs: None,
        "CausalGuardReport": lambda *args, **kwargs: None,
        "sealed_sha256": lambda payload: "",
    }
    originals = {name: getattr(runner_module, name) for name in replacements}
    try:
        for name, replacement in replacements.items():
            setattr(runner_module, name, replacement)
        yield
    finally:
        for name, original in originals.items():
            setattr(runner_module, name, original)


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(percentile * len(ordered)) - 1))
    return ordered[index]


def _one_episode(root: Path, *, index: int, sym0: bool) -> float:
    sample_root = root / ("sym0" if sym0 else "baseline") / str(index)
    sample_root.mkdir(parents=True)
    storage = StorageFactory.create_facade(
        StorageConfig(
            mode="sqlite",
            sqlite_db_path=str(sample_root / "runtime.db"),
            postgres_dsn=None,
            artifact_root=sample_root / "artifacts",
            prefer_postgres_reads=True,
            strict_dual_write=False,
        )
    )
    runner = ScenarioEpisodeRunner(
        storage=storage,
        run_id=f"sym0-benchmark-{index}",
        scenario=ThermalScenario(
            initial_temperature=0.88,
            alarm_threshold=0.85,
            cooling_effect=0.07,
        ),
    )
    started = time.perf_counter()
    runner.run_episode(
        external_input=0.0,
        replay_unit_id=f"thermal/bench-sym0/ep-{index}",
        trace_dir=sample_root / "traces",
    )
    elapsed = time.perf_counter() - started
    storage.close()
    return elapsed


def run_benchmark(iterations: int) -> dict[str, Any]:
    os.environ["RNFE_REASONING_MODE"] = "fixed"
    os.environ["RNFE_REASONING_FAMILY_PROFILE"] = "core_plus_opt"
    os.environ.pop("RNFE_REASONING_ACTUATES", None)
    sym0_times: list[float] = []
    baseline_times: list[float] = []
    with tempfile.TemporaryDirectory(prefix="rnfe-sym0-bench-") as temporary:
        root = Path(temporary)
        for index in range(iterations):
            # Alternar el orden reduce sesgo por calentamiento o frecuencia del host.
            if index % 2 == 0:
                with _baseline_adapters():
                    baseline_times.append(_one_episode(root, index=index, sym0=False))
                sym0_times.append(_one_episode(root, index=index, sym0=True))
            else:
                sym0_times.append(_one_episode(root, index=index, sym0=True))
                with _baseline_adapters():
                    baseline_times.append(_one_episode(root, index=index, sym0=False))

    sym0_mean = statistics.mean(sym0_times)
    baseline_mean = statistics.mean(baseline_times)
    paired_deltas = [
        sym0_elapsed - baseline_elapsed
        for sym0_elapsed, baseline_elapsed in zip(sym0_times, baseline_times)
    ]
    return {
        "iterations_per_arm": iterations,
        "python": os.sys.version.split()[0],
        "z3": __import__("z3").get_version_string(),
        "sym0": {
            "mean_seconds": sym0_mean,
            "median_seconds": statistics.median(sym0_times),
            "p95_seconds": _percentile(sym0_times, 0.95),
            "stdev_seconds": statistics.stdev(sym0_times),
        },
        "baseline": {
            "mean_seconds": baseline_mean,
            "median_seconds": statistics.median(baseline_times),
            "p95_seconds": _percentile(baseline_times, 0.95),
            "stdev_seconds": statistics.stdev(baseline_times),
        },
        "paired_delta_mean_seconds": statistics.mean(paired_deltas),
        "paired_delta_median_seconds": statistics.median(paired_deltas),
        "overhead_mean_percent": (sym0_mean - baseline_mean) / baseline_mean * 100.0,
        "overhead_p95_percent": (
            _percentile(sym0_times, 0.95) - _percentile(baseline_times, 0.95)
        )
        / _percentile(baseline_times, 0.95)
        * 100.0,
        "baseline_method": "in-process no-op SYM-0 adapters; runner unchanged",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.iterations < 2:
        parser.error("--iterations debe ser al menos 2")
    result = run_benchmark(args.iterations)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
