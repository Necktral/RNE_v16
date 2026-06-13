"""Life-loop: el organismo RNFE viviendo en el tiempo (R2 del roadmap canon).

Corre episodios cognitivos de forma continua alternando regímenes (escenarios),
con un calendario determinista de perturbaciones que ejercita alarmas y
degradación — y por tanto el lazo de autoevolución (ρₜ) y el linaje (μₜ).
El MISMO organismo (OrganismState + LineageState) se hereda entre bloques de
régimen: la vida es continua aunque el mundo cambie.

Persiste todo (episodios, certificados con risk_plus, eventos de autoevolución)
al storage por defecto (PostgreSQL vía .env, o SQLite con RNFE_STORAGE_MODE).

Pensado para hardware modesto: un solo hilo, pausa entre episodios
(``--interval``, default 1 s), sin GPU, sin dependencias nuevas. Ctrl-C para
terminar con resumen de vida.

Uso:
    PYTHONPATH=. python scripts/life_loop.py                       # ∞, 1 ep/s
    PYTHONPATH=. python scripts/life_loop.py --max-episodes 40
    PYTHONPATH=. python scripts/life_loop.py --interval 0.2 --block 6 \\
        --scenarios thermal_homeostasis,resource_management
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from datetime import datetime, timezone

from runtime.world import ScenarioEpisodeRunner


def _perturbation(step: int, *, base: float = 0.04, spike: float = 0.14, period: int = 7) -> float:
    """Calendario determinista de perturbaciones.

    Entrada externa suave con un pico cada ``period`` episodios: suficiente
    para provocar alarmas/degradación de vez en cuando sin volver el mundo
    caótico. Determinista ⇒ una vida es reproducible dado el mismo arranque.
    """
    if step % period == period - 1:
        return spike
    return base + 0.01 * (step % 3)


_STOP = False


def _handle_sigint(signum, frame):  # noqa: ARG001
    global _STOP
    _STOP = True
    print("\n[life] señal de parada recibida; cerrando con gracia…", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Life-loop del organismo RNFE")
    ap.add_argument("--scenarios", default="thermal_homeostasis,resource_management",
                    help="regímenes a alternar, separados por coma")
    ap.add_argument("--block", type=int, default=8, help="episodios por bloque de régimen")
    ap.add_argument("--interval", type=float, default=1.0, help="pausa entre episodios (s)")
    ap.add_argument("--max-episodes", type=int, default=0, help="0 = sin límite")
    ap.add_argument("--memory-mode", default="strict_same_scenario",
                    choices=["strict_same_scenario", "cross_scenario_analogical"])
    args = ap.parse_args()

    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    run_id = f"life-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    signal.signal(signal.SIGINT, _handle_sigint)

    print(f"[life] run_id={run_id} regímenes={scenarios} bloque={args.block} "
          f"intervalo={args.interval}s máx={args.max_episodes or '∞'}", flush=True)

    organism_state = None
    lineage = None
    episode_total = 0
    block_idx = 0
    actions_seen: dict[str, int] = {}

    try:
        while not _STOP:
            scenario = scenarios[block_idx % len(scenarios)]
            runner = ScenarioEpisodeRunner(
                run_id=run_id,
                scenario=scenario,
                memory_filter_mode=args.memory_mode,
                organism_state=organism_state,
                lineage=lineage,
            )
            for _ in range(args.block):
                if _STOP or (args.max_episodes and episode_total >= args.max_episodes):
                    break
                result = runner.run_episode(
                    external_input=_perturbation(episode_total)
                )
                episode_total += 1

                cert = result["certification"]
                via = result.get("viability_assessment") or {}
                evo = result.get("autoevolution") or {}
                lin = result.get("lineage") or {}
                action = evo.get("action", "none")
                actions_seen[action] = actions_seen.get(action, 0) + 1

                rp = None
                try:
                    certs = runner.storage.list_episode_certificates(run_id=run_id, limit=1)
                    rp = (certs[0].metadata or {}).get("risk_plus") or {}
                except Exception:
                    rp = {}

                print(
                    f"[life] ep={episode_total:5d} {scenario[:18]:18s} "
                    f"verdict={cert['verdict']:9s} ioc={rp.get('ioc', '—')} "
                    f"Δ={rp.get('delta_ioc', '—')} sie={rp.get('sie_verdict', '—'):8s} "
                    f"margen={via.get('viability_margin', '—')} "
                    f"evo={action} gen={lin.get('generation', 0)}",
                    flush=True,
                )
                if action not in {"none", "disabled", "monitoring"}:
                    print(f"[life]   ↳ evolución: {evo}", flush=True)

                if args.interval > 0:
                    time.sleep(args.interval)

            # Heredar el organismo al siguiente bloque de régimen.
            organism_state = runner.organism_state
            lineage = runner.lineage
            block_idx += 1
            if args.max_episodes and episode_total >= args.max_episodes:
                break
    finally:
        print("\n[life] ── resumen de vida ──", flush=True)
        print(f"[life] episodios vividos: {episode_total}", flush=True)
        print(f"[life] acciones evolutivas: {actions_seen}", flush=True)
        if lineage is not None:
            d = lineage.to_dict()
            print(f"[life] linaje: generación={d['generation']} "
                  f"rollbacks={len(d['rollback_ancestry'])} "
                  f"consistencia={d['consistency_score']}", flush=True)
            for entry in lineage.history[-6:]:
                print(f"[life]   {entry.entry_type:24s} {entry.description[:70]}", flush=True)
        if organism_state is not None:
            print(f"[life] organismo: episodios={organism_state.episode_count} "
                  f"margen={organism_state.viability.viability_margin:.3f} "
                  f"drift={organism_state.policy.accumulated_drift:.3f} "
                  f"vivo={organism_state.is_viable}", flush=True)
        print(f"[life] run_id={run_id} (certificados y eventos persistidos en storage)", flush=True)


if __name__ == "__main__":
    sys.exit(main())
