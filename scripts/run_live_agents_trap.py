#!/usr/bin/env python3
"""Corrida VIVA de agentes — A11+A12 cableados en el organismo, sobre deferred_load_trap.

Corre una trayectoria multi-episodio a través del `ScenarioEpisodeRunner` COMPLETO
(memoria, certificación, reward, override), comparando dos configuraciones sobre el
mismo escenario-trampa:

  - baseline   : core_only, sin actuación → el greedy elige boost (trampa).
  - agents_live: core_plus_imagination_a12 + RNFE_REASONING_ACTUATES → A11 prevé el
                 breach, A12 retracta por Bayes-factor, el override aplica shed.

Mide, a nivel del organismo: breaches de alarma, carga media/terminal, overrides.
Determinista, sin GPU. Escribe un reporte JSON y una tabla.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime.storage import StorageConfig, StorageFactory  # noqa: E402
from runtime.world import ScenarioEpisodeRunner  # noqa: E402
from runtime.world.deferred_load_scenario import DeferredLoadScenario  # noqa: E402

EXT = 0.04

_BASE_ENV = {
    "RNFE_REASONING_MODE": "fixed",
    "RNFE_REASONING_MAX_STEPS": "10",
}
_CONFIGS = {
    "baseline_core_only": {
        "RNFE_REASONING_FAMILY_PROFILE": "core_only",
        "RNFE_REASONING_ACTUATES": None,   # None ⇒ borrar (sombra)
        "RNFE_IMAGINATION_DEEP": None,
        "RNFE_A12_DEEP": None,
    },
    "agents_live_a11_a12": {
        "RNFE_REASONING_FAMILY_PROFILE": "core_plus_imagination_a12",
        "RNFE_REASONING_ACTUATES": "1",
        "RNFE_IMAGINATION_DEEP": "1",
        "RNFE_A12_DEEP": "1",
    },
}


def _apply_env(cfg: dict) -> None:
    for k, v in {**_BASE_ENV, **cfg}.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def run_config(name: str, cfg: dict, *, steps: int, storage_dir: Path, initial_load: float) -> dict:
    _apply_env(cfg)
    storage_dir.mkdir(parents=True, exist_ok=True)
    storage = StorageFactory.create_facade(StorageConfig(
        mode="sqlite", sqlite_db_path=str(storage_dir / "live.db"), postgres_dsn=None,
        artifact_root=storage_dir / "art", prefer_postgres_reads=False, strict_dual_write=False,
    ))
    runner = ScenarioEpisodeRunner(
        scenario=DeferredLoadScenario(initial_load=initial_load, alarm_threshold=0.85),
        storage=storage, run_id=f"live-{name}", closure_profile="adaptive_min",
    )
    breaches = overrides = 0
    loads = []
    for _ in range(steps):
        res = runner.run_episode(external_input=EXT)
        if (res.get("intervention_override") or {}).get("fired"):
            overrides += 1
        obs = runner.scenario.observe()
        loads.append(obs.state["load"])
        if obs.alarm:
            breaches += 1
    return {
        "breaches": breaches,
        "overrides": overrides,
        "steps": steps,
        "final_load": round(loads[-1], 4),
        "mean_load": round(sum(loads) / len(loads), 4),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Corrida viva de A11+A12 en el mundo-trampa.")
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--initial-load", type=float, default=0.70)
    parser.add_argument("--output-root", default="data/reports/live_agents_trap")
    parser.add_argument("--stamp", default=None)
    args = parser.parse_args(argv)

    root = Path(args.output_root)
    results = {}
    for name, cfg in _CONFIGS.items():
        results[name] = run_config(
            name, cfg, steps=args.steps,
            storage_dir=root / "_work" / name, initial_load=args.initial_load,
        )

    print(f"\nCorrida viva A11+A12 @ deferred_load_trap ({args.steps} episodios)\n")
    print(f"{'config':26s} {'breaches':>12s} {'overrides':>10s} {'carga media':>12s}")
    print("-" * 64)
    for name, r in results.items():
        print(f"{name:26s} {r['breaches']:>5d}/{r['steps']:<6d} {r['overrides']:>10d} {r['mean_load']:>12.4f}")

    base = results["baseline_core_only"]["breaches"]
    live = results["agents_live_a11_a12"]["breaches"]
    print(f"\nVeredicto: agentes vivos breaches={live} vs baseline={base} "
          f"→ {'GANANCIA' if live < base else 'sin ganancia'}")

    stamp = args.stamp or datetime.now().strftime("%Y%m%dT%H%M%S")
    root.mkdir(parents=True, exist_ok=True)
    out = root / f"{stamp}_live_agents_run.json"
    out.write_text(json.dumps({
        "scenario": "deferred_load_trap", "steps": args.steps,
        "initial_load": args.initial_load, "generated_at": datetime.now().isoformat(),
        "results": results,
    }, indent=2, ensure_ascii=True))
    print(f"Reporte: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
