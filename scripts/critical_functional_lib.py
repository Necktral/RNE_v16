"""Experimento del Funcional Crítico J(h|X): ¿la descomposición cura la ceguera?

Continúa la línea del estudio de ceguera de recompensa. Aquél MIDIÓ el modo de fallo:
el reward de coherencia colapsa la evaluación en un escalar IoC dominado por
*continuidad* que PENALIZA la desviación efectiva (umbral real λV≈20, ~40× el
idealizado). El ADR propuso —sin implementar— descomponer IoC en canales ortogonales.

El documento `algoritmo_pensamiento_critico_objetivista.tex` da la forma general:
    J(h|X) = w1·κ + w2·σ + w3·ρ + w4·α + w5·ν − w6·u,   Ψ = argmax_{h∈H_adm} J.

Afirmación falsable de ESTE experimento:
    Tratar la viabilidad/coherencia-causal (ν) como criterio ADITIVO de primera clase
    en J recupera la familia efectiva con un peso pequeño λ_ν=O(1), eliminando la
    inflación ~40× del umbral; y, con un control target_band (óptimo interior), activa
    la familia efectiva SOLO cuando hay brecha de viabilidad (especificidad limpia).

Mecanismo honesto de la cura (medido en el estudio previo, no inventado):
    - colapsado: la efectividad entra como MARGEN crudo de resultado (~±0.012, diminuto
      y ruidoso) bajo un IoC dominado por continuidad (peso 0.45, caída 0.24 al desviar)
      ⇒ hay que escalar el término ~20× para superar la penalización + el ruido.
    - J: la viabilidad entra como el CANAL booleano de coherencia causal
      `cau.helps_goal` ∈ {0,1} (limpio, alto-SNR; ver core_inference.py) con peso
      co-igual ⇒ registra a λ_ν=O(1). La cura es RE-PONDERAR (degradar continuidad de
      dominante a co-igual) + usar el veredicto causal limpio en vez del margen ruidoso.

SIN cambios de runtime: J, H_adm y target_band viven SOLO en este harness; conduce el
RewardGuidedOverlaySelector existente. Python puro, determinista por semilla.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from scripts.intelligence_campaign_lib import _mean
from scripts.reward_blindness_lib import (
    EFFECTIVE_FAMILY,
    _monotone_nondecreasing,
    effect_size,
    seed_ci,
)
from runtime.reasoning.scheduler_meta.reward_guided import RewardGuidedOverlaySelector

# ───────────────────────── constantes MEDIDAS (estudio previo) ────────────────
# No son ajustables a conveniencia: provienen de data/reports/reward_blindness y del
# proxy real, para que el baseline colapsado REPRODUZCA el umbral real ~20.
IOC_CONTINUITY_WEIGHT = 0.45   # ioc_proxy.py: peso dominante de `continuity`
IOC_DEVIATION_DROP = 0.24      # IoC 0.888 → 0.646 al ejecutar el override efectivo
EFF_MARGIN = 0.012             # margen de efectividad real (~+0.01: diminuto y ruidoso)
IOC_NOISE = 0.06               # ruido por-episodio del escalar de recompensa (ΔIoC* oscila)
COST = 0.06                    # coste por familia activa

CANDIDATES: Tuple[str, ...] = ("opt", "heur", "dia_adv", "ind")
N_CHANNELS = 7                 # κ, σ, ρ, α, ν, identidad, (−u)
BALANCED_W = 1.0 / N_CHANNELS  # ningún canal domina (vs continuidad 0.45 en el colapsado)

CURE_GRID_DEFAULT: Tuple[float, ...] = (0.0, 0.5, 1.0, 2.0, 5.0, 20.0, 50.0)
DEFAULT_TAUS = {"kappa": 0.5, "sigma": 0.5, "rho": 0.5, "nu": 0.0}  # admisibilidad laxa

REGIONS = ("conflict", "band_out", "band_in")


@dataclass(frozen=True)
class Channels:
    """Canales tipados del funcional J, en [0,1]. `margin` es el resultado crudo."""
    kappa: float       # coherencia estructural (closure/trace)
    sigma: float       # soporte evidencial (ind/prob)
    rho: float         # robustez de supuestos
    alpha: float       # resistencia adversarial (fal_guard/dia_adv)
    nu: float          # viabilidad / coherencia causal = cau.helps_goal ∈ {0,1}
    identity: float    # continuidad del auto-modelo (el término dominante de IoC)
    u: float           # incertidumbre residual
    margin: float      # margen de resultado crudo (la señal de efectividad ruidosa)


def _region_signals(region: str, deviates: bool) -> Tuple[float, float]:
    """Devuelve (helps_goal∈{0,1}, margen_crudo) por región.

    - conflict/band_out: hay BRECHA. La desviación (override) ayuda (helps=1, +margen);
      el greedy falla (helps=0, −margen).
    - band_in (óptimo interior, SIN-BRECHA): el mundo ya está en banda; desviar NO ayuda
      (helps=0, margen≈0) y quedarse tampoco aporta margen ⇒ ninguna acción tiene ventaja.
    """
    if region in ("conflict", "band_out"):
        if deviates:
            return 1.0, +EFF_MARGIN
        return 0.0, -EFF_MARGIN
    # band_in: sin brecha de viabilidad para ninguna acción.
    return 0.0, 0.0


def _channels(region: str, deviates: bool) -> Channels:
    helps, margin = _region_signals(region, deviates)
    identity = 1.0 - (IOC_DEVIATION_DROP if deviates else 0.0)  # desviar rompe continuidad
    # κ, σ, ρ, α, u ~ constantes (iguales en ambas acciones): el contraste vive en ν/identidad.
    return Channels(
        kappa=0.85, sigma=0.80, rho=0.80, alpha=0.85,
        nu=helps, identity=identity, u=0.15, margin=margin,
    )


def _admissible(ch: Channels, taus: Dict[str, float]) -> bool:
    """H_adm: umbrales mínimos de admisibilidad (doc §8). No gatea la viabilidad (τ_ν=0)."""
    return (
        ch.kappa >= taus.get("kappa", 0.0)
        and ch.sigma >= taus.get("sigma", 0.0)
        and ch.rho >= taus.get("rho", 0.0)
        and ch.nu >= taus.get("nu", 0.0)
    )


# ───────────────────────────── funciones de recompensa ───────────────────────

def make_collapsed_reward(lam_v: float, region: str) -> Callable[[Sequence[str], random.Random], float]:
    """Baseline IoC-colapsado: escalar dominado por continuidad + λV·margen_crudo.

    Reproduce el reward real: la efectividad es el MARGEN crudo (diminuto, ruidoso) y la
    continuidad domina (0.45). Por eso exige λV grande.
    """
    def reward(active: Sequence[str], rng: random.Random) -> float:
        deviates = EFFECTIVE_FAMILY in active
        ch = _channels(region, deviates)
        ioc = (
            IOC_CONTINUITY_WEIGHT * ch.identity
            + 0.25 * ch.kappa
            + 0.20 * ch.alpha
            - 0.06 * ch.u
        )
        ioc += rng.gauss(0.0, IOC_NOISE)
        return ioc - COST * len(active) + lam_v * ch.margin
    return reward


def make_J_reward(
    lam_nu: float,
    region: str,
    *,
    drop_channel: Optional[str] = None,
    taus: Optional[Dict[str, float]] = None,
) -> Callable[[Sequence[str], random.Random], float]:
    """Funcional J descompuesto: canales co-iguales, ν (coherencia causal) de 1ª clase.

    `drop_channel ∈ {None,'nu','sigma'}` para la ablación (G3). Mismo ruido por-episodio
    que el colapsado: la diferencia es la SEÑAL (canal booleano limpio vs margen diminuto)
    y el PESO (continuidad degradada de 0.45 a 1/7).
    """
    taus = taus or DEFAULT_TAUS

    def reward(active: Sequence[str], rng: random.Random) -> float:
        deviates = EFFECTIVE_FAMILY in active
        ch = _channels(region, deviates)
        if not _admissible(ch, taus):
            return -1e9  # inadmisible ⇒ fuera de H_adm
        w = BALANCED_W
        base = w * (ch.kappa + ch.rho + ch.alpha + ch.identity) - w * ch.u
        if drop_channel != "sigma":
            base += w * ch.sigma
        nu_term = 0.0 if drop_channel == "nu" else lam_nu * w * ch.nu
        j = base + nu_term + rng.gauss(0.0, IOC_NOISE)
        return j - COST * len(active)
    return reward


# ──────────────────────────────── motor del selector ─────────────────────────

def _run_selector(
    reward_fn: Callable[[Sequence[str], random.Random], float],
    *,
    episodes: int,
    seed: int,
) -> Dict[str, Any]:
    """Conduce el RewardGuidedOverlaySelector bajo una recompensa dada (determinista)."""
    rng = random.Random(seed)
    selector = RewardGuidedOverlaySelector(
        candidates=list(CANDIDATES), epsilon=0.005, min_obs=2, max_active=2
    )
    run_id, regime = "cf", "task"
    hist: List[bool] = []
    for _ in range(episodes):
        directives = selector.directives(run_id, regime=regime)
        active = sorted(f for f, a in directives.items() if a == "on")
        hist.append(EFFECTIVE_FAMILY in active)
        r = reward_fn(active, rng)
        executed = ["abd", "ana", "cau", "ctf", "ded", "prob"] + [f.upper() for f in active]
        selector.observe(
            run_id=run_id, reward_block={"reward": r}, executed_sequence=executed, regime=regime
        )
    final = selector.directives(run_id, regime=regime)
    half = max(1, episodes // 2)
    return {
        "retained": final.get(EFFECTIVE_FAMILY) == "on",
        "activation_rate_2nd_half": _mean([1.0 if x else 0.0 for x in hist[half:]]),
    }


def _retention(
    make_reward: Callable[[float], Callable[[Sequence[str], random.Random], float]],
    lam: float,
    *,
    seeds: int,
    episodes: int,
    seed_base: int,
) -> Dict[str, Any]:
    ret: List[float] = []
    act: List[float] = []
    for s in range(seeds):
        out = _run_selector(make_reward(lam), episodes=episodes, seed=seed_base + s)
        ret.append(1.0 if out["retained"] else 0.0)
        act.append(out["activation_rate_2nd_half"])
    return {
        "lambda": lam,
        "retention_rate": seed_ci(ret, seed=int(lam * 1000) + 1),
        "activation_rate": seed_ci(act, seed=int(lam * 1000) + 2),
        "_ret_raw": ret,
        "_act_raw": act,
    }


def _threshold(grid: Sequence[float], retention: Sequence[float], *, level: float = 0.5) -> Optional[float]:
    """Menor λ del grid con retención ≥ level (umbral de recuperación). None si nunca."""
    for lam, r in zip(grid, retention):
        if r >= level:
            return float(lam)
    return None


# ─────────────────────────── (G1) experimento de CURA ────────────────────────

def run_cure_experiment(
    *, grid: Sequence[float] = CURE_GRID_DEFAULT, seeds: int = 400, episodes: int = 30,
) -> Dict[str, Any]:
    """G1: umbral de recuperación J (ν 1ª clase) vs IoC-colapsado, en conflicto.

    Espera: umbral(J)=O(1) ≪ umbral(colapsado)≈20 ⇒ la descomposición CURA la inflación.
    """
    region = "conflict"
    collapsed = [
        _retention(lambda l: make_collapsed_reward(l, region), lam,
                   seeds=seeds, episodes=episodes, seed_base=100_000 + int(lam * 100))
        for lam in grid
    ]
    jdec = [
        _retention(lambda l: make_J_reward(l, region), lam,
                   seeds=seeds, episodes=episodes, seed_base=200_000 + int(lam * 100))
        for lam in grid
    ]
    col_ret = [c["retention_rate"]["mean"] for c in collapsed]
    j_ret = [c["retention_rate"]["mean"] for c in jdec]
    th_col = _threshold(grid, col_ret)
    th_j = _threshold(grid, j_ret)
    ratio = (th_col / th_j) if (th_col and th_j and th_j > 0) else None
    return {
        "experiment": "cure",
        "design": {"grid": list(grid), "seeds": seeds, "episodes": episodes, "region": region},
        "collapsed": [{k: v for k, v in c.items() if not k.startswith("_")} for c in collapsed],
        "decomposed_J": [{k: v for k, v in c.items() if not k.startswith("_")} for c in jdec],
        "collapsed_retention_by_lambda": [round(x, 4) for x in col_ret],
        "J_retention_by_lambda": [round(x, 4) for x in j_ret],
        "threshold_collapsed": th_col,
        "threshold_J": th_j,
        "threshold_ratio_collapsed_over_J": (round(ratio, 2) if ratio else None),
        "verdicts": {
            # G1: J recupera con λ pequeño y MUY por debajo del colapsado.
            "G1_J_recovers_at_small_lambda": bool(th_j is not None and th_j <= 2.0),
            "G1_collapsed_threshold_much_higher": bool(
                th_col is not None and th_j is not None and th_col >= 5.0 * max(th_j, 0.5)
            ),
        },
    }


# ───────────────────── (G2) experimento de ESPECIFICIDAD ──────────────────────

def run_specificity_experiment(
    *, lam_nu: float = 1.0, seeds: int = 400, episodes: int = 30,
    regions: Sequence[str] = REGIONS,
) -> Dict[str, Any]:
    """G2: con target_band, J/ν activa la familia efectiva SOLO donde hay brecha.

    Es el control sin-brecha que faltó en el estudio (H2). band_in = óptimo interior.
    """
    by_region: Dict[str, Any] = {}
    for region in regions:
        cell = _retention(lambda l: make_J_reward(l, region), lam_nu,
                          seeds=seeds, episodes=episodes, seed_base=300_000 + hash(region) % 9973)
        by_region[region] = {k: v for k, v in cell.items() if not k.startswith("_")}
    act = {r: by_region[r]["activation_rate"]["mean"] for r in regions}
    gap_regions = [r for r in regions if r in ("conflict", "band_out")]
    nogap = "band_in"
    return {
        "experiment": "specificity",
        "design": {"lam_nu": lam_nu, "seeds": seeds, "episodes": episodes, "regions": list(regions)},
        "by_region": by_region,
        "activation_by_region": {r: round(act[r], 4) for r in regions},
        "verdicts": {
            # G2: activa donde hay brecha, NO en banda (sin-brecha) ⇒ especificidad limpia.
            "G2_active_in_gap": bool(all(act[r] > 0.7 for r in gap_regions)),
            "G2_inactive_in_band": bool(nogap not in act or act[nogap] < 0.2),
        },
    }


# ───────────────────────── (G3) experimento de ABLACIÓN ───────────────────────

def run_ablation_experiment(
    *, lam_nu: float = 2.0, seeds: int = 400, episodes: int = 30,
) -> Dict[str, Any]:
    """G3: quitar ν reproduce la supresión; quitar σ (no-causal) no ⇒ ν-específico.

    Análogo estructural del control de ruido (H3): el efecto es del canal de viabilidad,
    no de "cualquier canal extra".
    """
    region = "conflict"
    variants = {
        "J_full": None,
        "J_sin_nu": "nu",
        "J_sin_sigma": "sigma",
    }
    cells: Dict[str, Any] = {}
    ret: Dict[str, float] = {}
    for name, drop in variants.items():
        cell = _retention(
            lambda l, d=drop: make_J_reward(l, region, drop_channel=d), lam_nu,
            seeds=seeds, episodes=episodes, seed_base=400_000 + (hash(name) % 9973),
        )
        cells[name] = {k: v for k, v in cell.items() if not k.startswith("_")}
        ret[name] = cell["retention_rate"]["mean"]
    return {
        "experiment": "ablation",
        "design": {"lam_nu": lam_nu, "seeds": seeds, "episodes": episodes, "region": region},
        "variants": cells,
        "retention_by_variant": {k: round(v, 4) for k, v in ret.items()},
        "verdicts": {
            # G3: ν necesario; σ no. El efecto es específico del canal de viabilidad.
            "G3_nu_necessary": bool(ret["J_full"] > 0.7 and ret["J_sin_nu"] < 0.2),
            "G3_sigma_not_necessary": bool(ret["J_sin_sigma"] > 0.7),
        },
    }


# ───────────────────────────── informe + orquestación ────────────────────────

def render_report(
    *, cure: Optional[Dict[str, Any]], specificity: Optional[Dict[str, Any]],
    ablation: Optional[Dict[str, Any]],
) -> str:
    L: List[str] = []
    add = L.append
    add("# Funcional Crítico J(h|X): ¿la descomposición cura la ceguera de la recompensa?")
    add("")
    add("## Afirmación")
    add("")
    add("> Tratar la viabilidad/coherencia-causal (ν) como criterio aditivo de PRIMERA CLASE en un "
        "funcional multicriterio `J(h|X) = w1·κ + w2·σ + w3·ρ + w4·α + w5·ν − w6·u` recupera la familia "
        "efectiva con un peso pequeño λ_ν=O(1), eliminando la inflación ~40× del umbral que sufre el "
        "reward de coherencia colapsado; y, con un control `target_band`, activa la familia efectiva "
        "SOLO cuando hay brecha de viabilidad (especificidad limpia).")
    add("")
    add("Continúa el estudio de ceguera de recompensa (modo de fallo MEDIDO). Aquí se prueba en el "
        "harness si la DESCOMPOSICIÓN propuesta por el ADR cura el fallo. **Sin cambios de runtime.**")
    add("")
    add("## Hipótesis (pre-registradas)")
    add("")
    add("- **G1 (cura / sin inflación):** umbral de recuperación bajo J ≈ O(1), MUY por debajo del "
        "umbral del IoC-colapsado (≈20). Dosis-respuesta de retención vs λ.")
    add("- **G2 (especificidad, el H2 con control limpio):** con `target_band`, J/ν activa la familia "
        "efectiva fuera de banda (con brecha) y NO en banda (sin brecha = óptimo interior).")
    add("- **G3 (necesidad del canal):** quitar ν reproduce la supresión; quitar σ (no-causal) no ⇒ "
        "el efecto es específico del canal de viabilidad (análogo estructural del control de ruido H3).")
    add("- **Falsación:** si J también exige λ≈20, o activa en banda, o cualquier canal extra recupera, "
        "la afirmación de descomposición es FALSA y se reporta así.")
    add("")
    add("Constantes MEDIDAS (del estudio previo, no ajustadas a conveniencia): continuidad domina IoC "
        f"con peso {IOC_CONTINUITY_WEIGHT} (`ioc_proxy.py`); desviar baja IoC {IOC_DEVIATION_DROP} "
        f"(0.888→0.646); el margen de efectividad real es diminuto (~{EFF_MARGIN}); ν es el canal "
        "booleano `cau.helps_goal`∈{0,1} (`core_inference.py`).")
    add("")

    if cure is not None:
        v = cure["verdicts"]
        add("## G1 — Cura: umbral de recuperación J vs IoC-colapsado (conflicto)")
        add("")
        add(f"N={cure['design']['seeds']} semillas × {cure['design']['episodes']} episodios por celda. "
            "Mismo ruido por-episodio en ambos; difiere la SEÑAL (canal booleano limpio vs margen "
            "diminuto) y el PESO (continuidad 0.45 → 1/7).")
        add("")
        add("| λ | Retención IoC-colapsado | Retención J (ν 1ª clase) |")
        add("|---:|---:|---:|")
        for i, lam in enumerate(cure["design"]["grid"]):
            add(f"| {lam} | {cure['collapsed_retention_by_lambda'][i]:.3f} | "
                f"{cure['J_retention_by_lambda'][i]:.3f} |")
        add("")
        add(f"- Umbral de recuperación **colapsado** = {cure['threshold_collapsed']}; "
            f"**J** = {cure['threshold_J']}; **ratio** = {cure['threshold_ratio_collapsed_over_J']}×.")
        add(f"- **G1 J recupera con λ pequeño**: {'✓' if v['G1_J_recovers_at_small_lambda'] else '✗'}.")
        add(f"- **G1 umbral colapsado ≫ J**: {'✓' if v['G1_collapsed_threshold_much_higher'] else '✗'}.")
        add("")

    if specificity is not None:
        v = specificity["verdicts"]
        add("## G2 — Especificidad: control `target_band` (el H2 que faltó)")
        add("")
        add(f"J a λ_ν={specificity['design']['lam_nu']} sobre regiones con/ sin brecha. `band_in` = "
            "óptimo interior (mundo ya en banda ⇒ desviar no ayuda ⇒ ninguna acción tiene ventaja).")
        add("")
        add("| Región | Activación familia efectiva |")
        add("|---|---:|")
        for r, a in specificity["activation_by_region"].items():
            add(f"| {r} | {a:.3f} |")
        add("")
        add(f"- **G2 activa donde hay brecha**: {'✓' if v['G2_active_in_gap'] else '✗'}.")
        add(f"- **G2 inactiva en banda (sin-brecha)**: {'✓' if v['G2_inactive_in_band'] else '✗'} "
            "— este es el control limpio que el térmico-minimize no ofrecía.")
        add("")

    if ablation is not None:
        v = ablation["verdicts"]
        add("## G3 — Necesidad del canal: ablación ν vs σ (conflicto)")
        add("")
        add("| Variante | Retención familia efectiva |")
        add("|---|---:|")
        for k, val in ablation["retention_by_variant"].items():
            add(f"| {k} | {val:.3f} |")
        add("")
        add(f"- **G3 ν necesario** (quitarlo suprime): {'✓' if v['G3_nu_necessary'] else '✗'}.")
        add(f"- **G3 σ no necesario** (quitarlo no suprime): {'✓' if v['G3_sigma_not_necessary'] else '✗'} "
            "⇒ el efecto es específico del canal de viabilidad.")
        add("")

    add("## Veredicto")
    add("")
    all_v: List[bool] = []
    for blk in (cure, specificity, ablation):
        if blk is not None:
            all_v += list(blk["verdicts"].values())
    if all_v and all(all_v):
        add("Las hipótesis se **confirman** en el harness: la descomposición (ν de primera clase, "
            "continuidad degradada) cura la inflación del umbral, es específica del canal de viabilidad, "
            "y respeta la especificidad en el control sin-brecha.")
    else:
        add("Resultado **mixto/refutado** (ver hipótesis marcadas ✗) — reportado sin adornos.")
    add("")
    add("## Limitaciones honestas")
    add("")
    add("- **Es un modelo, no una re-medición viva.** El harness conduce el selector real con canales "
        "MODELADOS usando las constantes medidas en el estudio previo (continuidad 0.45, caída 0.24, "
        "margen ~0.012, ν booleano). La cura es una PREDICCIÓN del modelo que motiva la PR gated de "
        "runtime (descomponer IoC), no una medición de la arquitectura viva.")
    add("- La afirmación es sobre **especificación/forma de la evaluación** (re-ponderar + usar el canal "
        "causal limpio), no sobre sofisticación del razonamiento.")
    add("- `target_band` está modelado en el harness; un escenario de óptimo interior en runtime sigue "
        "siendo trabajo futuro para confirmarlo en vivo.")
    add("- Una sola familia efectiva y un conjunto de canales; generalización a tareas ricas pendiente.")
    add("")
    return "\n".join(L)


def run_study(
    *,
    mode: str = "all",
    output_root: str | Path = "data/reports/critical_functional",
    cure_grid: Sequence[float] = CURE_GRID_DEFAULT,
    seeds: int = 400,
    episodes: int = 30,
    spec_lam_nu: float = 1.0,
    ablation_lam_nu: float = 2.0,
) -> Dict[str, Any]:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    cure = specificity = ablation = None
    if mode in ("cure", "all"):
        print("[critical-J] G1 cura (umbral J vs colapsado)...", flush=True)
        cure = run_cure_experiment(grid=cure_grid, seeds=seeds, episodes=episodes)
    if mode in ("specificity", "all"):
        print("[critical-J] G2 especificidad (target_band)...", flush=True)
        specificity = run_specificity_experiment(lam_nu=spec_lam_nu, seeds=seeds, episodes=episodes)
    if mode in ("ablation", "all"):
        print("[critical-J] G3 ablación (ν vs σ)...", flush=True)
        ablation = run_ablation_experiment(lam_nu=ablation_lam_nu, seeds=seeds, episodes=episodes)
    results = {"cure": cure, "specificity": specificity, "ablation": ablation}
    (root / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=True, default=str), encoding="utf-8"
    )
    (root / "REPORT.md").write_text(
        render_report(cure=cure, specificity=specificity, ablation=ablation), encoding="utf-8"
    )
    return {
        "output_root": str(root),
        "results_path": str(root / "results.json"),
        "report_path": str(root / "REPORT.md"),
        "cure_verdicts": None if cure is None else cure["verdicts"],
        "specificity_verdicts": None if specificity is None else specificity["verdicts"],
        "ablation_verdicts": None if ablation is None else ablation["verdicts"],
    }
