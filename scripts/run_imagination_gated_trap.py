#!/usr/bin/env python3
"""Ejecutador de corrida física — A11 (imaginación) gated en `deferred_load_trap`.

Corre episodios REALES del escenario a través del A11 real (`imagination.execute`) +
la compuerta gated (`imagination.gate`). Compara 4 modos que demuestran que la
compuerta abre SÓLO bajo condiciones seguras (tipo R1) y que, cuando abre, la
previsión elimina los breaches de la trampa:

  - baseline_gate_closed      : advisory (no override) → reactivo cae en la trampa.
  - gated_checkpoint_healthy  : checkpoint sano + riesgo bajo → override → seguro.
  - gated_but_no_checkpoint   : sin checkpoint sano → la gate se niega → trampa (R1).
  - gated_but_high_risk       : riesgo alto → la gate se niega → trampa.

Determinista y sin GPU. Escribe un reporte JSON y una tabla legible.
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

# Corrida de imaginación: activar la variante deep de A11.
os.environ.setdefault("RNFE_IMAGINATION_DEEP", "1")

from runtime.world.deferred_load_scenario import DeferredLoadScenario  # noqa: E402
from runtime.reasoning.families import imagination as IMAG  # noqa: E402

EXT = 0.04
THRESHOLD = 0.85


def _reactive_choice(sig) -> str:
    """Elección del núcleo reactivo: mayor reducción inmediata en la firma lineal."""
    linear = {
        e.intervention_name: (-e.expected_magnitude if e.expected_direction == "-" else e.expected_magnitude)
        for e in sig.intervention_effects
    }
    return min(linear, key=lambda k: linear[k])


def _state_for(sc, obs, chosen: str) -> dict:
    """Estado de razonamiento mínimo que el A11 real consume (resuelve la firma registrada)."""
    return {
        "scenario": sc.config.name,
        "scenario_metadata": {
            "scenario_name": sc.config.name,
            "main_variable": sc.config.main_variable,
            "alarm_threshold": sc.config.alarm_threshold,
            "interventions": list(sc.config.interventions),
        },
        "observation": dict(obs.state),
        "intervention": chosen,
    }


def run_episode(*, initial_load: float, steps: int, gate_open: bool,
                checkpoint_healthy: bool, risk: float) -> dict:
    sc = DeferredLoadScenario(initial_load=initial_load, alarm_threshold=THRESHOLD)
    sig = sc.causal_signature
    breaches = overrides = 0
    loads = []
    for _ in range(steps):
        obs = sc.observe()
        reactive = _reactive_choice(sig)                       # lo que haría el reactivo
        result = IMAG.execute(_state_for(sc, obs, reactive))   # A11 REAL end-to-end
        sd = result.get("state_delta", {})
        final = reactive
        if gate_open:
            decision = IMAG.gate(sd, checkpoint_healthy=checkpoint_healthy, risk=risk)
            if decision["override"] and decision["intervention"]:
                final = decision["intervention"]
                overrides += 1
        tr = sc.factual_transition(intervention=final, external_input=EXT)
        loads.append(tr.state["load"])
        if tr.alarm:
            breaches += 1
    return {
        "initial_load": initial_load,
        "breaches": breaches,
        "overrides": overrides,
        "final_load": round(loads[-1], 4),
        "mean_load": round(sum(loads) / len(loads), 4),
    }


_MODES = {
    "baseline_gate_closed": dict(gate_open=False, checkpoint_healthy=False, risk=0.2),
    "gated_checkpoint_healthy": dict(gate_open=True, checkpoint_healthy=True, risk=0.2),
    "gated_but_no_checkpoint": dict(gate_open=True, checkpoint_healthy=False, risk=0.2),
    "gated_but_high_risk": dict(gate_open=True, checkpoint_healthy=True, risk=0.9),
}


def run_campaign(*, steps: int, initial_loads: list[float]) -> dict:
    out = {}
    for name, cfg in _MODES.items():
        episodes = [run_episode(initial_load=il, steps=steps, **cfg) for il in initial_loads]
        n_steps = steps * len(episodes)
        out[name] = {
            "config": cfg,
            "episodes": episodes,
            "total_breaches": sum(e["breaches"] for e in episodes),
            "total_steps": n_steps,
            "total_overrides": sum(e["overrides"] for e in episodes),
            "mean_load": round(sum(e["mean_load"] for e in episodes) / len(episodes), 4),
        }
    return out


def _print_table(results: dict, steps: int, initial_loads: list[float]) -> None:
    print(f"\nCorrida física A11 gated @ deferred_load_trap "
          f"({len(initial_loads)} episodios × {steps} pasos)\n")
    print(f"{'modo':28s} {'breaches':>12s} {'overrides':>10s} {'carga media':>12s}")
    print("-" * 66)
    for name, r in results.items():
        print(f"{name:28s} {r['total_breaches']:>5d}/{r['total_steps']:<6d} "
              f"{r['total_overrides']:>10d} {r['mean_load']:>12.4f}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Corrida física de A11 gated en el mundo-trampa.")
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--initial-loads", type=float, nargs="+", default=[0.60, 0.65, 0.70, 0.75])
    parser.add_argument("--output-root", default="data/reports/imagination_gated_trap")
    parser.add_argument("--stamp", default=None, help="Sello para el nombre del reporte (default: ahora).")
    args = parser.parse_args(argv)

    results = run_campaign(steps=args.steps, initial_loads=args.initial_loads)
    _print_table(results, args.steps, args.initial_loads)

    stamp = args.stamp or datetime.now().strftime("%Y%m%dT%H%M%S")
    out_dir = Path(args.output_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "scenario": "deferred_load_trap",
        "steps": args.steps,
        "initial_loads": args.initial_loads,
        "generated_at": datetime.now().isoformat(),
        "results": results,
    }
    out_path = out_dir / f"{stamp}_gated_trap_run.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=True))
    print(f"\nReporte: {out_path}")

    # Veredicto simple: la gate sana debe eliminar breaches; las gates negadas, no.
    healthy = results["gated_checkpoint_healthy"]["total_breaches"]
    baseline = results["baseline_gate_closed"]["total_breaches"]
    print(f"\nVeredicto: gate sana breaches={healthy} vs baseline={baseline} "
          f"→ {'GANANCIA' if healthy < baseline else 'sin ganancia'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
