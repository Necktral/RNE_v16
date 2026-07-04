#!/usr/bin/env python3
"""Experimento: ¿qué modo de ecología MULTIPLICA la ganancia cognitiva (sin engaños)?

A/B honesto sobre seeds held-out de los 4 modos de transferencia:
    isolated · inheritance_only · reasoning_policy · reasoning_policy_plus_rules

Cada modo corre la MISMA población inicial (mismas semillas, mismos regímenes,
mismo presupuesto de episodios). Métrica de multiplicación = Δ de la recompensa
canon media GUARDADA (solo episodios certificados con cierre estable) de cada modo
vs ``isolated``, con bootstrap CI. Guard anti-engaño: el cierre debe quedar intacto
en TODOS los brazos — un modo que "gana" rompiendo cierre queda descalificado.

Diseñado para correr DESPUÉS de la campaña fuerte (compite por CPU). Python puro.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime.organism.ecology import OrganismEcology, TransferMode, build_member
from runtime.storage import StorageConfig, StorageFactory
from scripts.intelligence_campaign_lib import _regime_specs, bootstrap_ci_delta

MODES = [
    TransferMode.ISOLATED,
    TransferMode.INHERITANCE_ONLY,
    TransferMode.REASONING_POLICY,
    TransferMode.REASONING_POLICY_PLUS_RULES,
]

# Regímenes held-out: una mezcla heterogénea donde la transferencia PUEDE importar
# (distintos regímenes ⇒ políticas distintas que compartir/heredar).
HELD_OUT_REGIMES = ["heterogeneous_elevated", "heterogeneous_warning", "viability_edge"]


def _storage(root: Path, mode: str):
    config = StorageConfig(
        mode="sqlite",
        sqlite_db_path=str(root / f"{mode}.db"),
        postgres_dsn=None,
        artifact_root=root / f"art_{mode}",
        prefer_postgres_reads=False,
        strict_dual_write=False,
    )
    return StorageFactory.create_facade(config)


def _run_mode(
    *,
    mode: TransferMode,
    root: Path,
    members_per_regime: int,
    generations: int,
    episodes_per_member: int,
    seed: int,
    regimes: List[str],
) -> Dict[str, Any]:
    storage = _storage(root, mode.value)
    specs = _regime_specs()
    members = []
    for regime in regimes:
        params = dict(specs[regime]["scenario_params"])
        for i in range(members_per_regime):
            members.append(
                build_member(
                    member_id=f"{mode.value}-{regime}-{i}",
                    scenario="grid_thermal_5x5",
                    scenario_kwargs=params,
                    storage=storage,
                )
            )
    eco = OrganismEcology(members, storage=storage, transfer_mode=mode, seed=seed)
    for _ in range(generations):
        eco.run_generation(episodes_per_member=episodes_per_member)

    # Recompensa canon guardada por episodio certificado (la muestra para el CI).
    guarded_rewards: List[float] = []
    closure_breaks = 0
    total = 0
    for member in eco.members:
        certified = eco._certified_episode_ids(member.run_id)
        try:
            events = storage.list_events(run_id=member.run_id, limit=2048)
        except Exception:
            events = []
        for item in events:
            if getattr(item, "event_type", None) != "reasoning.reward":
                continue
            total += 1
            payload = getattr(item, "payload", None) or {}
            if payload.get("episode_id") in certified:
                guarded_rewards.append(float(payload.get("reward", 0.0)))
        try:
            certs = storage.list_episode_certificates(run_id=member.run_id, limit=2048)
        except Exception:
            certs = []
        for cert in certs:
            meta = getattr(cert, "metadata", None) or {}
            if meta.get("closure_passed") is False:
                closure_breaks += 1

    n = len(guarded_rewards)
    mean_reward = sum(guarded_rewards) / n if n else float("nan")
    return {
        "mode": mode.value,
        "population_summary": eco.population_summary(),
        "guarded_rewards": guarded_rewards,
        "mean_guarded_reward": mean_reward,
        "n_certified": n,
        "n_total": total,
        "closure_break_count": closure_breaks,  # guard anti-engaño: debe ser 0
    }


def run_experiment(
    *,
    output_root: str | Path = "data/benchmarks/ecology",
    experiment_id: str | None = None,
    members_per_regime: int = 2,
    generations: int = 3,
    episodes_per_member: int = 6,
    seed: int = 990000,
    regimes: List[str] | None = None,
) -> Dict[str, Any]:
    from datetime import datetime

    regimes = list(regimes or HELD_OUT_REGIMES)
    experiment_id = experiment_id or f"ecology_multiplication_{datetime.now():%Y%m%d_%H%M%S}"
    root = Path(output_root) / experiment_id
    root.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Dict[str, Any]] = {}
    for mode in MODES:
        print(f"[ecology] corriendo modo {mode.value}...", flush=True)
        results[mode.value] = _run_mode(
            mode=mode,
            root=root,
            regimes=regimes,
            members_per_regime=members_per_regime,
            generations=generations,
            episodes_per_member=episodes_per_member,
            seed=seed,
        )

    baseline = results[TransferMode.ISOLATED.value]
    base_rewards = baseline["guarded_rewards"]
    comparisons: Dict[str, Any] = {}
    for mode in MODES:
        if mode == TransferMode.ISOLATED:
            continue
        cand = results[mode.value]
        delta = cand["mean_guarded_reward"] - baseline["mean_guarded_reward"]
        ci = bootstrap_ci_delta(base_rewards, cand["guarded_rewards"], n_bootstrap=1000, seed=seed)
        comparisons[mode.value] = {
            "delta_mean_guarded_reward_vs_isolated": round(delta, 6),
            "ci_lower": round(ci[0], 6),
            "ci_upper": round(ci[1], 6),
            "multiplies": delta > 0 and ci[0] > 0,  # ganancia con CI por encima de 0
            "closure_intact": cand["closure_break_count"] == 0,
        }

    # Veredicto: el modo que multiplica con CI>0 y cierre intacto, de mayor Δ.
    eligible = [
        (data["delta_mean_guarded_reward_vs_isolated"], name)
        for name, data in comparisons.items()
        if data["multiplies"] and data["closure_intact"]
    ]
    eligible.sort(reverse=True)
    if eligible:
        verdict = (
            f"el modo '{eligible[0][1]}' multiplica la ganancia cognitiva "
            f"(+{eligible[0][0]:.6f} recompensa guardada vs isolated, CI>0, cierre intacto)"
        )
    else:
        verdict = (
            "ningún modo de transferencia supera a 'isolated' con CI por encima de 0 "
            "en este diseño (la colonia no multiplica la ganancia aquí — sin engaños)"
        )

    payload = {
        "experiment_id": experiment_id,
        "primary_verdict": verdict,
        "design": {
            "regimes": regimes,
            "members_per_regime": members_per_regime,
            "generations": generations,
            "episodes_per_member": episodes_per_member,
            "seed": seed,
        },
        "per_mode": {
            name: {k: v for k, v in data.items() if k != "guarded_rewards"}
            for name, data in results.items()
        },
        "comparisons_vs_isolated": comparisons,
    }
    (root / "ecology_multiplication_verdict.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=True, default=str), encoding="utf-8"
    )
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Experimento de multiplicación de ganancia por ecología.")
    p.add_argument("--experiment-id", default=None)
    p.add_argument("--output-root", default="data/benchmarks/ecology")
    p.add_argument("--members-per-regime", type=int, default=2)
    p.add_argument("--generations", type=int, default=3)
    p.add_argument("--episodes-per-member", type=int, default=6)
    p.add_argument("--seed", type=int, default=990000)
    p.add_argument(
        "--regimes",
        default=None,
        help="Coma-separados; por defecto la mezcla heterogénea held-out.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = run_experiment(
        output_root=args.output_root,
        experiment_id=args.experiment_id,
        members_per_regime=args.members_per_regime,
        generations=args.generations,
        episodes_per_member=args.episodes_per_member,
        seed=args.seed,
        regimes=[r.strip() for r in args.regimes.split(",")] if args.regimes else None,
    )
    print(json.dumps(payload["primary_verdict"], ensure_ascii=True))
    print(json.dumps(payload["comparisons_vs_isolated"], indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
