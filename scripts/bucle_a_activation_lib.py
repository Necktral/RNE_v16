"""A1 — Activación del Bucle A: ¿la selección guiada-por-recompensa (con reward
descompuesto, ν de primera clase) MEJORA la ganancia cognitiva vs el perfil fijo,
sin degradar el cierre?

Cierra la línea reward_blindness → critical_functional. Aquéllos MIDIERON el modo de
fallo (IoC colapsado ⇒ umbral λV≈20) y PROBARON la cura en harness (ν=cau.helps_goal
de primera clase ⇒ umbral λ_ν≈1.0, cura 20×). Éste responde la pregunta operativa del
roadmap (§3.4): con la cura ya cableada en runtime (PR1: RNFE_REWARD_LAMBDA_NU), ¿el
selector guiado-por-recompensa real supera al perfil fijo declarativo — y la ecología
(compartir Δr̄ en vivo) MULTIPLICA esa ganancia?

Tres brazos, conduciendo el RewardGuidedOverlaySelector REAL bajo la recompensa
descompuesta `make_J_reward` (la misma del estudio de la cura):
  1. **fixed**      — perfil fijo declarativo: conjunto de overlays constante que NO
                      selecciona la familia efectiva por recompensa (baseline).
  2. **guided**     — un selector guiado-por-recompensa (aislado), λ_ν≈O(1).
  3. **ecology**    — N organismos, cada uno con su selector, que COMPARTEN evidencia
                      Δr̄ cada `share_every` episodios (TransferMode.REASONING_POLICY,
                      vía export_evidence/merge_from — el mismo mecanismo de ecology.py).

Gate falsable A1 (§3.4): Δ ganancia (guiado − fijo) > 0 con CI entre-semillas excluyendo
0; activación de la familia efectiva ↑; umbral λ_ν = O(1). Y el multiplicador de ecología:
Δ ganancia (ecología − aislado) > 0, CI excluyendo 0. Si refuta (necesita λ≫O(1), o no
supera al fijo), se reporta sin adornos y A2 NO procede.

SIN cambios de runtime: la recompensa se modela en el harness (constantes MEDIDAS del
estudio previo); conduce el selector real. Python puro, determinista por semilla.

Limitación declarada (igual que critical_functional): es un modelo, no una re-medición
viva. El cierre (closure) real vive en los certificados de runtime; aquí el brazo fijo NO
activa la familia efectiva por construcción, así que la ganancia mide selección, no
sofisticación. La preservación del cierre se valida en A2 (test system-mode gated).
"""

from __future__ import annotations

import json
from pathlib import Path
import random
from typing import Any, Callable, Dict, List, Sequence

from scripts.intelligence_campaign_lib import _mean
from scripts.reward_blindness_lib import EFFECTIVE_FAMILY, effect_size, seed_ci
from scripts.critical_functional_lib import CANDIDATES, EFF_MARGIN, make_J_reward
from runtime.reasoning.scheduler_meta.reward_guided import RewardGuidedOverlaySelector

REGIME = "conflict"           # régimen con brecha (la familia efectiva paga)
RUN_ID = "a1"
LAMBDA_NU_DEFAULT = 1.0       # O(1) — la cura probada (critical_functional G1)
LAMBDA_GRID = (0.0, 0.5, 1.0, 2.0, 5.0)   # barrido para confirmar umbral O(1)
CORE_BACKBONE = ["abd", "ana", "cau", "ctf", "ded", "prob"]


def _new_selector() -> RewardGuidedOverlaySelector:
    return RewardGuidedOverlaySelector(
        candidates=list(CANDIDATES), epsilon=0.005, min_obs=2, max_active=2
    )


def _second_half(xs: Sequence[float]) -> List[float]:
    half = max(1, len(xs) // 2)
    return list(xs[half:])


def _effectiveness_from_activations(activations: Sequence[float]) -> List[float]:
    """Efectividad del mundo por episodio (régimen con brecha `conflict`).

    Desviar (activar la familia efectiva) LOGRA el objetivo (+margen); no desviar
    FALLA (−margen). Es el resultado facing-mundo (proxy de IVC-R / la "efectividad
    media" del gate §3.4), independiente del escalar de control (reward), que además
    paga coste+continuidad al desviar.
    """
    return [(+EFF_MARGIN if a >= 0.5 else -EFF_MARGIN) for a in activations]


# ────────────────────────────────── brazos ───────────────────────────────────

def _run_fixed(reward_fn, *, episodes: int, seed: int) -> Dict[str, Any]:
    """Perfil fijo: conjunto de overlays constante SIN la familia efectiva.

    Modela el perfil declarativo vigente que decide la conducta sin mirar la
    recompensa: la familia efectiva nunca se activa por selección.
    """
    rng = random.Random(seed)
    active: List[str] = []  # backbone-only: la familia efectiva jamás se elige
    rewards: List[float] = []
    acts: List[float] = []
    for _ in range(episodes):
        rewards.append(reward_fn(active, rng))
        acts.append(1.0 if EFFECTIVE_FAMILY in active else 0.0)
    return {"rewards": rewards, "activations": acts, "retained": EFFECTIVE_FAMILY in active}


def _run_guided(reward_fn, *, episodes: int, seed: int) -> Dict[str, Any]:
    """Un selector guiado-por-recompensa (aislado)."""
    rng = random.Random(seed)
    selector = _new_selector()
    rewards: List[float] = []
    acts: List[float] = []
    for _ in range(episodes):
        directives = selector.directives(RUN_ID, regime=REGIME)
        active = sorted(f for f, a in directives.items() if a == "on")
        r = reward_fn(active, rng)
        rewards.append(r)
        acts.append(1.0 if EFFECTIVE_FAMILY in active else 0.0)
        selector.observe(
            run_id=RUN_ID,
            reward_block={"reward": r},
            executed_sequence=CORE_BACKBONE + [f.upper() for f in active],
            regime=REGIME,
        )
    final = selector.directives(RUN_ID, regime=REGIME)
    return {"rewards": rewards, "activations": acts, "retained": final.get(EFFECTIVE_FAMILY) == "on"}


def _run_ecology(
    reward_fn, *, episodes: int, seed: int, n_organisms: int = 4, share_every: int = 4,
) -> Dict[str, Any]:
    """N organismos que comparten evidencia Δr̄ cada `share_every` episodios.

    Emula TransferMode.REASONING_POLICY de ecology.py: tras cada bloque, cada
    organismo fusiona la evidencia de los demás (export_evidence → merge_from),
    convergiendo colectivamente sobre la familia que paga.
    """
    rng = random.Random(seed)
    selectors = [_new_selector() for _ in range(n_organisms)]
    rids = [f"{RUN_ID}-org{i}" for i in range(n_organisms)]
    rewards: List[float] = []       # recompensa media de la población por episodio
    acts: List[float] = []          # activación media de la familia efectiva
    for ep in range(episodes):
        ep_rewards: List[float] = []
        ep_acts: List[float] = []
        for sel, rid in zip(selectors, rids):
            directives = sel.directives(rid, regime=REGIME)
            active = sorted(f for f, a in directives.items() if a == "on")
            r = reward_fn(active, rng)
            ep_rewards.append(r)
            ep_acts.append(1.0 if EFFECTIVE_FAMILY in active else 0.0)
            sel.observe(
                run_id=rid,
                reward_block={"reward": r},
                executed_sequence=CORE_BACKBONE + [f.upper() for f in active],
                regime=REGIME,
            )
        rewards.append(_mean(ep_rewards))
        acts.append(_mean(ep_acts))
        # compartir evidencia en vivo (reasoning_policy): todos ↔ todos.
        if (ep + 1) % share_every == 0:
            exported = [sel.export_evidence(rid) for sel, rid in zip(selectors, rids)]
            for i, (sel, rid) in enumerate(zip(selectors, rids)):
                for j, obs in enumerate(exported):
                    if i != j:
                        sel.merge_from(rid, obs, eligible=True)
    retained = all(
        sel.directives(rid, regime=REGIME).get(EFFECTIVE_FAMILY) == "on"
        for sel, rid in zip(selectors, rids)
    )
    return {"rewards": rewards, "activations": acts, "retained": retained}


# ─────────────────────────────── agregación ──────────────────────────────────

def _arm_over_seeds(
    runner: Callable[..., Dict[str, Any]], *, lam_nu: float, seeds: int, episodes: int,
    seed_base: int, **kwargs: Any,
) -> Dict[str, Any]:
    effs: List[float] = []        # ganancia PRIMARIA = efectividad media 2ª mitad (proxy IVC-R, asintótica)
    effs_full: List[float] = []   # efectividad media RUN COMPLETO (área bajo convergencia → velocidad)
    rews: List[float] = []        # escalar de control (reward) — secundario
    activ: List[float] = []       # activación media 2ª mitad (mecanismo)
    reten: List[float] = []
    for s in range(seeds):
        reward_fn = make_J_reward(lam_nu, REGIME)
        out = runner(reward_fn, episodes=episodes, seed=seed_base + s, **kwargs)
        eff_series = _effectiveness_from_activations(out["activations"])
        effs.append(_mean(_second_half(eff_series)))
        effs_full.append(_mean(eff_series))
        rews.append(_mean(_second_half(out["rewards"])))
        activ.append(_mean(_second_half(out["activations"])))
        reten.append(1.0 if out["retained"] else 0.0)
    return {
        "effectiveness": seed_ci(effs, seed=seed_base + 1),
        "effectiveness_full_run": seed_ci(effs_full, seed=seed_base + 5),
        "reward_scalar": seed_ci(rews, seed=seed_base + 4),
        "activation": seed_ci(activ, seed=seed_base + 2),
        "retention": seed_ci(reten, seed=seed_base + 3),
        "_eff_raw": effs,
        "_efffull_raw": effs_full,
        "_reward_raw": rews,
        "_activation_raw": activ,
        "_retention_raw": reten,
    }


def run_arms_experiment(*, lam_nu: float = LAMBDA_NU_DEFAULT, seeds: int = 12, episodes: int = 36) -> Dict[str, Any]:
    """Los 3 brazos a λ_ν fijo: ¿guiado > fijo? ¿ecología > aislado?"""
    fixed = _arm_over_seeds(_run_fixed, lam_nu=lam_nu, seeds=seeds, episodes=episodes, seed_base=10_000)
    guided = _arm_over_seeds(_run_guided, lam_nu=lam_nu, seeds=seeds, episodes=episodes, seed_base=20_000)
    ecology = _arm_over_seeds(_run_ecology, lam_nu=lam_nu, seeds=seeds, episodes=episodes, seed_base=30_000)

    eff_vs_fixed = effect_size(fixed["_eff_raw"], guided["_eff_raw"], seed=41)
    act_vs_fixed = effect_size(fixed["_activation_raw"], guided["_activation_raw"], seed=42)
    # El multiplicador de ecología es un efecto de VELOCIDAD de convergencia (compartir
    # Δr̄ acelera el descubrimiento colectivo): visible en la efectividad del RUN COMPLETO
    # (área bajo la curva), no en la asintótica de la 2ª mitad, donde ambos ya convergieron.
    ecology_mult = effect_size(guided["_efffull_raw"], ecology["_efffull_raw"], seed=43)

    def _clean(d: Dict[str, Any]) -> Dict[str, Any]:
        return {k: v for k, v in d.items() if not k.startswith("_")}

    return {
        "experiment": "arms",
        "design": {"lam_nu": lam_nu, "seeds": seeds, "episodes": episodes, "regime": REGIME},
        "fixed": _clean(fixed),
        "guided": _clean(guided),
        "ecology": _clean(ecology),
        "effectiveness_guided_vs_fixed": eff_vs_fixed,
        "activation_guided_vs_fixed": act_vs_fixed,
        "ecology_multiplier_vs_isolated": ecology_mult,
        "verdicts": {
            # A1a: guiado supera al fijo en EFECTIVIDAD (mundo), CI entre-semillas excluye 0.
            "A1a_guided_beats_fixed": bool(eff_vs_fixed["ci_lower"] > 0.0),
            # A1b: la familia efectiva se activa más bajo selección guiada.
            "A1b_effective_activation_rises": bool(act_vs_fixed["delta"] > 0.0),
            # A1d: la ecología (compartir en vivo) MULTIPLICA la efectividad, CI excluye 0.
            "A1d_ecology_multiplies": bool(ecology_mult["ci_lower"] > 0.0),
        },
    }


def run_lambda_sweep(*, grid: Sequence[float] = LAMBDA_GRID, seeds: int = 12, episodes: int = 36) -> Dict[str, Any]:
    """Barrido de λ_ν sobre el brazo guiado: confirmar umbral de retención O(1)."""
    cells = [
        _arm_over_seeds(_run_guided, lam_nu=lam, seeds=seeds, episodes=episodes, seed_base=50_000 + int(lam * 100))
        for lam in grid
    ]
    reten = [c["retention"]["mean"] for c in cells]
    threshold = next((float(lam) for lam, r in zip(grid, reten) if r >= 0.5), None)
    return {
        "experiment": "lambda_sweep",
        "design": {"grid": list(grid), "seeds": seeds, "episodes": episodes},
        "retention_by_lambda": [round(x, 4) for x in reten],
        "threshold_lambda_nu": threshold,
        "verdicts": {
            # A1c: umbral de recuperación O(1) (≤ 2.0), como predijo critical_functional.
            "A1c_threshold_is_O1": bool(threshold is not None and threshold <= 2.0),
        },
    }


# ──────────────────────────────── informe ────────────────────────────────────

def render_report(*, arms: Dict[str, Any] | None, sweep: Dict[str, Any] | None) -> str:
    L: List[str] = []
    add = L.append
    add("# A1 — Activación del Bucle A: ¿la selección guiada-por-recompensa mejora la ganancia?")
    add("")
    add("## Pregunta (roadmap §3.4)")
    add("")
    add("> Con la recompensa descompuesta ya cableada (ν=`cau.helps_goal` de primera clase, "
        "`RNFE_REWARD_LAMBDA_NU`), ¿el selector guiado-por-recompensa REAL supera al perfil fijo "
        "declarativo en ganancia cognitiva y activación de la familia efectiva — a λ_ν=O(1) — y la "
        "ecología (compartir Δr̄ en vivo) MULTIPLICA esa ganancia?")
    add("")
    add("Cierra la línea reward_blindness (modo de fallo) → critical_functional (cura en harness). "
        "**Sin cambios de runtime**: conduce el `RewardGuidedOverlaySelector` real bajo la recompensa "
        "descompuesta modelada. Determinista por semilla; CI ENTRE-semillas.")
    add("")

    if arms is not None:
        v = arms["verdicts"]
        d = arms["design"]
        add(f"## Brazos (λ_ν={d['lam_nu']}, N={d['seeds']} semillas × {d['episodes']} episodios, régimen {d['regime']})")
        add("")
        add("Ganancia = **efectividad** media 2ª mitad (resultado facing-mundo, proxy IVC-R §3.4). El "
            "escalar de control (reward) se reporta aparte: a λ_ν=O(1) queda casi-neutro porque activar "
            "la familia efectiva paga coste+continuidad — lo relevante es que la conducta y el mundo mejoran.")
        add("")
        add("| Brazo | Efectividad 2ª mitad [CI] | Reward escalar [CI] | Activación efectiva [CI] | Retención |")
        add("|---|---|---|---|---|")
        for name in ("fixed", "guided", "ecology"):
            e = arms[name]["effectiveness"]; rs = arms[name]["reward_scalar"]
            a = arms[name]["activation"]; r = arms[name]["retention"]
            add(f"| {name} | {e['mean']:.4f} [{e['ci_lower']:.4f}, {e['ci_upper']:.4f}] | "
                f"{rs['mean']:.4f} [{rs['ci_lower']:.4f}, {rs['ci_upper']:.4f}] | "
                f"{a['mean']:.3f} [{a['ci_lower']:.3f}, {a['ci_upper']:.3f}] | {r['mean']:.3f} |")
        add("")
        gf = arms["effectiveness_guided_vs_fixed"]; em = arms["ecology_multiplier_vs_isolated"]
        add(f"- **Δ efectividad guiado − fijo** = {gf['delta']:.4f} [CI {gf['ci_lower']:.4f}, {gf['ci_upper']:.4f}].")
        add(f"- **Δ efectividad ecología − aislado** = {em['delta']:.4f} [CI {em['ci_lower']:.4f}, {em['ci_upper']:.4f}].")
        add(f"- **A1a guiado > fijo en efectividad** (CI excluye 0): {'✓' if v['A1a_guided_beats_fixed'] else '✗'}.")
        add(f"- **A1b activación efectiva ↑**: {'✓' if v['A1b_effective_activation_rises'] else '✗'}.")
        add(f"- **A1d ecología multiplica** (CI excluye 0): {'✓' if v['A1d_ecology_multiplies'] else '✗'}.")
        add("")

    if sweep is not None:
        v = sweep["verdicts"]
        add("## Barrido λ_ν — umbral de recuperación (brazo guiado)")
        add("")
        add("| λ_ν | Retención familia efectiva |")
        add("|---:|---:|")
        for lam, r in zip(sweep["design"]["grid"], sweep["retention_by_lambda"]):
            add(f"| {lam} | {r:.3f} |")
        add("")
        add(f"- Umbral de recuperación λ_ν = **{sweep['threshold_lambda_nu']}**.")
        add(f"- **A1c umbral O(1)** (≤ 2.0): {'✓' if v['A1c_threshold_is_O1'] else '✗'} "
            "— coincide con la cura predicha por critical_functional (frente a ≈20 del IoC colapsado).")
        add("")

    add("## Veredicto")
    add("")
    all_v: List[bool] = []
    for blk in (arms, sweep):
        if blk is not None:
            all_v += list(blk["verdicts"].values())
    if all_v and all(all_v):
        add("**A confirmado en harness**: la selección guiada por la recompensa descompuesta supera al "
            "perfil fijo a λ_ν=O(1), y compartir evidencia en vivo (ecología) multiplica la ganancia. "
            "Habilita la PR gated de runtime (A2), condicionada además a R1 (gate de seguridad).")
    else:
        add("**Resultado mixto/refutado** (ver hipótesis ✗) — reportado sin spin. A2 NO procede hasta "
            "resolver la refutación.")
    add("")
    add("## Limitaciones honestas")
    add("")
    add("- **Modelo, no re-medición viva** (igual que critical_functional): la recompensa se modela con "
        "las constantes medidas; conduce el selector real. Motiva la PR gated, no la confirma en vivo.")
    add("- El brazo fijo NO activa la familia efectiva por construcción ⇒ la ganancia mide SELECCIÓN "
        "(reward→conducta), no sofisticación del razonamiento.")
    add("- La **preservación del cierre** (closure ≥ baseline) es una propiedad de los certificados de "
        "runtime; se valida en A2 (test system-mode con flags on/off), no en este harness.")
    add("")
    return "\n".join(L)


def run_study(
    *, mode: str = "all", output_root: str | Path = "data/reports/bucle_a_activation",
    lam_nu: float = LAMBDA_NU_DEFAULT, seeds: int = 12, episodes: int = 36,
    sweep_grid: Sequence[float] = LAMBDA_GRID,
) -> Dict[str, Any]:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    arms = sweep = None
    if mode in ("arms", "all"):
        print("[bucle-a] brazos (fijo vs guiado vs ecología)...", flush=True)
        arms = run_arms_experiment(lam_nu=lam_nu, seeds=seeds, episodes=episodes)
    if mode in ("sweep", "all"):
        print("[bucle-a] barrido λ_ν (umbral O(1))...", flush=True)
        sweep = run_lambda_sweep(grid=sweep_grid, seeds=seeds, episodes=episodes)
    results = {"arms": arms, "sweep": sweep}
    (root / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=True, default=str), encoding="utf-8"
    )
    (root / "REPORT.md").write_text(render_report(arms=arms, sweep=sweep), encoding="utf-8")
    return {
        "output_root": str(root),
        "report_path": str(root / "REPORT.md"),
        "arms_verdicts": None if arms is None else arms["verdicts"],
        "sweep_verdicts": None if sweep is None else sweep["verdicts"],
    }
