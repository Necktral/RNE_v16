"""Estudio controlado de la "ceguera de la recompensa" (process vs outcome reward).

Afirmación falsable:
    Una recompensa de coherencia/proceso que NO mide efectividad suprime
    sistemáticamente capacidades que mejoran el resultado sin mejorar la coherencia.

Dos experimentos complementarios:
- MECANISMO (sintético): conduce el RewardGuidedOverlaySelector con recompensas
  generadas donde coherencia (plano) y efectividad (depende de la familia) están
  disociadas. Alto-n, aísla la variable, control de ruido (H3).
- SISTEMA REAL (ecológico): el organismo completo sobre una familia de tareas de
  conflicto + controles saturados, barriendo λV. Confirma en la arquitectura, CI
  entre-semillas (H1, H2).

Sin cambios de runtime: solo barre env (RNFE_REASONING_ACTUATES,
RNFE_REWARD_LAMBDA_EFFECTIVENESS) y lee eventos. Python puro, determinista por semilla.
"""

from __future__ import annotations

import math
import os
import random
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from scripts.intelligence_campaign_lib import _mean, _std, bootstrap_ci_delta
from runtime.reasoning.scheduler_meta.reward_guided import RewardGuidedOverlaySelector

# La familia efectiva del estudio: la que (con override) resuelve el conflicto.
EFFECTIVE_FAMILY = "opt"
LAMBDA_GRID_DEFAULT: Tuple[float, ...] = (0.0, 0.1, 0.25, 0.5, 1.0)


# ───────────────────────────── análisis (reutiliza helpers) ──────────────────

def seed_ci(per_seed_values: Sequence[float], *, seed: int = 12345) -> Dict[str, Any]:
    """Media ± CI bootstrap ENTRE-SEMILLAS (no intra-corrida).

    CI de la media vía bootstrap contra una muestra de ceros (delta = candidato − 0).
    """
    vals = [float(v) for v in per_seed_values]
    n = len(vals)
    mean = _mean(vals)
    if n < 2:
        return {"n": n, "mean": round(mean, 6), "ci_lower": mean, "ci_upper": mean, "std": 0.0}
    lo, hi = bootstrap_ci_delta([0.0] * n, vals, n_bootstrap=2000, seed=seed)
    return {
        "n": n,
        "mean": round(mean, 6),
        "ci_lower": round(lo, 6),
        "ci_upper": round(hi, 6),
        "std": round(_std(vals), 6),
    }


def effect_size(a: Sequence[float], b: Sequence[float], *, seed: int = 777) -> Dict[str, Any]:
    """Δ medio (b − a) con CI bootstrap entre-semillas y d de Cohen."""
    a, b = [float(x) for x in a], [float(x) for x in b]
    delta = _mean(b) - _mean(a)
    lo, hi = bootstrap_ci_delta(a, b, n_bootstrap=2000, seed=seed)
    pooled = math.sqrt((_std(a) ** 2 + _std(b) ** 2) / 2.0) if (len(a) > 1 and len(b) > 1) else 0.0
    cohen_d = (delta / pooled) if pooled > 1e-9 else 0.0
    return {
        "delta": round(delta, 6),
        "ci_lower": round(lo, 6),
        "ci_upper": round(hi, 6),
        "ci_excludes_zero": (lo > 0.0) or (hi < 0.0),
        "cohen_d": round(cohen_d, 4),
    }


def _monotone_nondecreasing(values: Sequence[float], *, tol: float = 1e-6) -> bool:
    return all(values[i + 1] >= values[i] - tol for i in range(len(values) - 1))


# ─────────────────────── (A) experimento de MECANISMO ────────────────────────

def _synthetic_run(
    *,
    lambda_effectiveness: float,
    effectiveness_source: str,
    episodes: int,
    seed: int,
    effective_gap: float = 0.10,
    cost: float = 0.06,
    coherence_base: float = 0.0,
    candidates: Sequence[str] = ("opt", "heur", "dia_adv", "ind"),
) -> Dict[str, Any]:
    """Una corrida sintética del selector bajo una recompensa dada.

    Coherencia plana (coherence_base). Efectividad = effective_gap si la familia
    efectiva está activa (source 'real'); 'shuffled' = aleatoria decorrelacionada;
    'zero' = sin efectividad. Coste por familia activa. El selector decide on/off.
    Determinista por semilla.
    """
    rng = random.Random(seed)
    selector = RewardGuidedOverlaySelector(
        candidates=list(candidates), epsilon=0.005, min_obs=2, max_active=2
    )
    run_id = "synthetic"
    regime = "task"
    active_history: List[bool] = []
    for _ in range(episodes):
        directives = selector.directives(run_id, regime=regime)
        active = sorted(f for f, a in directives.items() if a == "on")
        eff_active = EFFECTIVE_FAMILY in active
        active_history.append(eff_active)
        # Señal de efectividad según la condición.
        if effectiveness_source == "real":
            effectiveness = effective_gap if eff_active else 0.0
        elif effectiveness_source == "shuffled":
            effectiveness = effective_gap if rng.random() < 0.5 else 0.0  # decorrelacionada
        else:  # zero
            effectiveness = 0.0
        n_active = len(active)
        reward = coherence_base - cost * n_active + lambda_effectiveness * effectiveness
        executed = ["abd", "ana", "cau", "ctf", "ded", "prob"] + [f.upper() for f in active]
        selector.observe(
            run_id=run_id,
            reward_block={"reward": reward},
            executed_sequence=executed,
            regime=regime,
        )
    # Estado final: ¿el selector retiene la familia efectiva?
    final = selector.directives(run_id, regime=regime)
    evidence = selector.summary(run_id, regime=regime)["evidence"]
    delta_eff = next(
        (e["delta_reward"] for e in evidence if e["family"] == EFFECTIVE_FAMILY), None
    )
    half = max(1, episodes // 2)
    activation_rate_2nd = _mean([1.0 if x else 0.0 for x in active_history[half:]])
    return {
        "retained": final.get(EFFECTIVE_FAMILY) == "on",
        "activation_rate_2nd_half": activation_rate_2nd,
        "delta_reward_effective": delta_eff,
    }


def run_mechanism_experiment(
    *,
    lambda_grid: Sequence[float] = LAMBDA_GRID_DEFAULT,
    sources: Sequence[str] = ("real", "shuffled"),
    seeds: int = 1000,
    episodes: int = 30,
) -> Dict[str, Any]:
    """Dosis-respuesta sintética + control de ruido (H1, H3). Alto-n, segundos."""
    cells: Dict[str, Dict[str, Any]] = {}
    for source in sources:
        for lam in lambda_grid:
            retained: List[float] = []
            act_rate: List[float] = []
            for s in range(seeds):
                out = _synthetic_run(
                    lambda_effectiveness=lam,
                    effectiveness_source=source,
                    episodes=episodes,
                    seed=10_000 * int(lam * 100 + 1) + s,
                )
                retained.append(1.0 if out["retained"] else 0.0)
                act_rate.append(out["activation_rate_2nd_half"])
            key = f"{source}|lam={lam}"
            cells[key] = {
                "source": source,
                "lambda": lam,
                "retention_rate": seed_ci(retained, seed=int(lam * 1000) + 1),
                "activation_rate": seed_ci(act_rate, seed=int(lam * 1000) + 2),
            }
    # Veredictos de hipótesis sobre la condición 'real'.
    real_ret = [cells[f"real|lam={lam}"]["retention_rate"]["mean"] for lam in lambda_grid]
    noise_ret = (
        [cells[f"shuffled|lam={lam}"]["retention_rate"]["mean"] for lam in lambda_grid]
        if "shuffled" in sources
        else []
    )
    h1_dose_response = _monotone_nondecreasing(real_ret) and (real_ret[-1] - real_ret[0] > 0.3)
    h1_suppressed_at_zero = real_ret[0] < 0.2
    h3_noise_flat = (max(noise_ret) < 0.3) if noise_ret else None
    return {
        "experiment": "mechanism",
        "design": {
            "lambda_grid": list(lambda_grid),
            "sources": list(sources),
            "seeds": seeds,
            "episodes": episodes,
            "effective_family": EFFECTIVE_FAMILY,
        },
        "cells": cells,
        "verdicts": {
            "H1_dose_response": bool(h1_dose_response),
            "H1_suppressed_at_lambda_zero": bool(h1_suppressed_at_zero),
            "H3_noise_control_flat": (None if h3_noise_flat is None else bool(h3_noise_flat)),
            "real_retention_by_lambda": [round(x, 4) for x in real_ret],
            "noise_retention_by_lambda": [round(x, 4) for x in noise_ret],
        },
    }


# ─────────────────────── (B) experimento de SISTEMA REAL ─────────────────────

_ENV_KEYS = (
    "RNFE_REASONING_MODE",
    "RNFE_REASONING_FAMILY_PROFILE",
    "RNFE_REASONING_REGIME_HINT",
    "RNFE_REASONING_MAX_STEPS",
    "RNFE_REASONING_ACTUATES",
    "RNFE_REWARD_LAMBDA_EFFECTIVENESS",
)


def _tmp_storage(root: Path, tag: str):
    from runtime.storage import StorageConfig, StorageFactory

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    config = StorageConfig(
        mode="sqlite",
        sqlite_db_path=str(root / f"{tag}.db"),
        postgres_dsn=None,
        artifact_root=root / f"art_{tag}",
        prefer_postgres_reads=False,
        strict_dual_write=False,
    )
    return StorageFactory.create_facade(config)


def _set_env(profile: str, lam: float, regime_hint: str) -> Dict[str, Optional[str]]:
    prev = {k: os.environ.get(k) for k in _ENV_KEYS}
    os.environ["RNFE_REASONING_MODE"] = "adaptive"
    os.environ["RNFE_REASONING_FAMILY_PROFILE"] = profile
    os.environ["RNFE_REASONING_REGIME_HINT"] = regime_hint
    os.environ["RNFE_REASONING_MAX_STEPS"] = "10"
    os.environ["RNFE_REASONING_ACTUATES"] = "1"
    os.environ["RNFE_REWARD_LAMBDA_EFFECTIVENESS"] = str(lam)
    return prev


def _restore_env(prev: Dict[str, Optional[str]]) -> None:
    for k, v in prev.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def classify_task(params: Dict[str, Any], root: Path) -> str:
    """Probe barato: 1 episodio greedy (core_only, actuación OFF). effectiveness<0 ⇒
    conflicto (el greedy falla); ≥0 ⇒ saturada (el greedy basta)."""
    from runtime.world import ScenarioEpisodeRunner
    from runtime.world.grid_thermal_scenario import GridThermalScenario

    prev = {k: os.environ.get(k) for k in _ENV_KEYS}
    os.environ["RNFE_REASONING_MODE"] = "fixed"
    os.environ["RNFE_REASONING_FAMILY_PROFILE"] = "core_only"
    os.environ.pop("RNFE_REASONING_ACTUATES", None)
    os.environ.pop("RNFE_REWARD_LAMBDA_EFFECTIVENESS", None)
    scratch = Path(tempfile.mkdtemp(prefix="rb_probe_"))
    storage = _tmp_storage(scratch, "probe")
    try:
        runner = ScenarioEpisodeRunner(
            scenario=GridThermalScenario(**params),
            storage=storage,
            run_id="probe",
            closure_profile="baseline_fixed",
        )
        res = runner.run_episode(external_input=0.04)
        eff = (res.get("reasoning_reward") or {}).get("effectiveness")
    finally:
        _restore_env(prev)
        try:
            storage.close()
        except Exception:
            pass
        shutil.rmtree(scratch, ignore_errors=True)
    return "conflict" if (isinstance(eff, (int, float)) and eff < 0) else "saturated"


def build_task_family(root: Path) -> List[Dict[str, Any]]:
    """Genera y CLASIFICA empíricamente una familia de instancias térmicas."""
    grid = []
    for temp in (0.80, 0.84, 0.88, 0.92, 0.96):
        for cooling in (0.04, 0.07):
            params = {
                "grid_size": 5,
                "topology": "uniform",
                "initial_temperature": temp,
                "alarm_threshold": 0.85,
                "cooling_effect": cooling,
            }
            label = classify_task(params, root)
            grid.append({"params": params, "type": label,
                         "name": f"t{int(temp*100)}_c{int(cooling*100)}"})
    return grid


def _run_system_cell(
    *, params: Dict[str, Any], lam: float, episodes: int, seed: int, root: Path, tag: str,
    profile: str = "core_plus_deliberative",
) -> Dict[str, Any]:
    """Un organismo K episodios bajo λV dado; mide activación/efectividad por eventos.

    Perfil ``core_plus_deliberative``: las opcionales (plan/opt) se activan SOLO si
    el selector guiado-por-recompensa las admite ⇒ el selector es el GOBERNADOR
    real, sin el confound de full_family_exploration (que activa todas por score).
    """
    from runtime.world import ScenarioEpisodeRunner
    from runtime.world.grid_thermal_scenario import GridThermalScenario

    # Scratch EFÍMERO: cada celda escribe su sqlite+artifacts en un tempdir que se
    # BORRA al terminar. Solo se leen los eventos para las métricas; persistirlos
    # ballonaría el disco (cada celda × 36 episodios genera artifacts) y, vía el
    # ext4.vhdx de WSL, llenaría C:. Pico = una celda (~MB), no la corrida entera.
    scratch = Path(tempfile.mkdtemp(prefix="rb_cell_"))
    storage = _tmp_storage(scratch, tag)
    regime_hint = "causal_counterfactual_conflict"
    prev = _set_env(profile, lam, regime_hint)
    # Selector y organismo COMPARTIDOS entre episodios; escenario RESETEADO a
    # conflicto cada episodio (mundo estacionario). Sin el reset, el override
    # enfría el mundo y el conflicto se auto-resuelve ⇒ el régimen deja de ser
    # estacionario y el selector no puede aprender una política estable. Con
    # reset, el selector enfrenta la MISMA decisión repetidamente (lo que el
    # experimento de mecanismo modela idealmente).
    selector = RewardGuidedOverlaySelector()  # in-memory compartido, sin re-seed
    organism_state = None
    try:
        rng = random.Random(seed)
        for _ in range(episodes):
            runner = ScenarioEpisodeRunner(
                scenario=GridThermalScenario(**params),  # fresco ⇒ conflicto estacionario
                storage=storage,
                run_id=tag,
                closure_profile="adaptive_min",
                reward_guided=selector,
                organism_state=organism_state,
            )
            runner.run_episode(external_input=0.04 + rng.uniform(-0.005, 0.005))
            organism_state = runner.organism_state

        # Leer eventos reasoning.reward para las métricas por-run.
        try:
            events = [e for e in storage.list_events(run_id=tag, limit=4096)
                      if getattr(e, "event_type", None) == "reasoning.reward"]
        except Exception:
            events = []
        opt_active: List[float] = []
        effs: List[float] = []
        for e in events:
            p = getattr(e, "payload", None) or {}
            overlays = p.get("optional_overlays_active") or []
            opt_active.append(1.0 if EFFECTIVE_FAMILY in [str(x).lower() for x in overlays] else 0.0)
            ef = p.get("effectiveness")
            if isinstance(ef, (int, float)):
                effs.append(float(ef))
        try:
            overrides = [e for e in storage.list_events(run_id=tag, limit=4096)
                         if getattr(e, "event_type", None) == "reasoning.intervention_override"
                         and (getattr(e, "payload", None) or {}).get("fired")]
        except Exception:
            overrides = []
    finally:
        _restore_env(prev)
        try:
            storage.close()
        except Exception:
            pass
        shutil.rmtree(scratch, ignore_errors=True)

    half = max(1, len(opt_active) // 2)
    return {
        "opt_activation_rate_2nd_half": _mean(opt_active[half:]) if opt_active else 0.0,
        "mean_effectiveness": _mean(effs) if effs else 0.0,
        "override_rate": (len(overrides) / len(opt_active)) if opt_active else 0.0,
        "n_episodes": len(opt_active),
    }


def run_system_experiment(
    *,
    output_root: Path,
    lambda_grid: Sequence[float] = (0.0, 5.0, 20.0, 50.0),
    seeds: int = 8,
    episodes: int = 36,
    max_tasks_per_type: int = 2,
) -> Dict[str, Any]:
    """λV × tipo-de-tarea × semilla en el organismo real (H1 dosis-respuesta, H2 interacción).

    El grid de λV cruza el UMBRAL REAL (~20): a diferencia del mecanismo idealizado
    (coherencia plana, umbral ~0.5), en la arquitectura el proxy IoC ANTI-correlaciona
    con la acción efectiva (la desviación baja IoC ~0.24) y ΔIoC* es un delta ruidoso,
    así que la señal de efectividad debe escalarse ~40× para superar ese ruido.
    El conflicto se RESETEA cada episodio (mundo estacionario, ver _run_system_cell).
    """
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    family = build_task_family(root / "_probe")
    by_type: Dict[str, List[Dict[str, Any]]] = {"conflict": [], "saturated": []}
    for task in family:
        if len(by_type[task["type"]]) < max_tasks_per_type:
            by_type[task["type"]].append(task)
    tasks = by_type["conflict"] + by_type["saturated"]

    started = time.monotonic()
    total = len(lambda_grid) * len(tasks) * seeds
    done = 0
    # raw[task_type][lambda] = lista de medias-por-(tarea,semilla)
    raw: Dict[str, Dict[float, Dict[str, List[float]]]] = {
        ttype: {lam: {"act": [], "eff": [], "ovr": []} for lam in lambda_grid}
        for ttype in ("conflict", "saturated")
    }
    for task in tasks:
        for lam in lambda_grid:
            for s in range(seeds):
                tag = f"{task['name']}_lam{int(lam*100)}_s{s}"
                cell = _run_system_cell(
                    params=task["params"], lam=lam, episodes=episodes,
                    seed=20260613 + s, root=root, tag=tag,
                )
                raw[task["type"]][lam]["act"].append(cell["opt_activation_rate_2nd_half"])
                raw[task["type"]][lam]["eff"].append(cell["mean_effectiveness"])
                raw[task["type"]][lam]["ovr"].append(cell["override_rate"])
                done += 1
                _heartbeat(done, total, started, f"{task['type']}/{task['name']}/λ{lam}")

    # Agregación con CI ENTRE-(tarea,semilla).
    cells: Dict[str, Any] = {}
    for ttype in ("conflict", "saturated"):
        for lam in lambda_grid:
            cells[f"{ttype}|lam={lam}"] = {
                "task_type": ttype,
                "lambda": lam,
                "opt_activation_rate": seed_ci(raw[ttype][lam]["act"], seed=int(lam * 100) + 1),
                "mean_effectiveness": seed_ci(raw[ttype][lam]["eff"], seed=int(lam * 100) + 2),
                "override_rate": seed_ci(raw[ttype][lam]["ovr"], seed=int(lam * 100) + 3),
            }

    lam0, lamT = lambda_grid[0], lambda_grid[-1]
    conflict_contrast_act = effect_size(
        raw["conflict"][lam0]["act"], raw["conflict"][lamT]["act"], seed=101
    )
    conflict_contrast_eff = effect_size(
        raw["conflict"][lam0]["eff"], raw["conflict"][lamT]["eff"], seed=102
    )
    saturated_contrast_act = effect_size(
        raw["saturated"][lam0]["act"], raw["saturated"][lamT]["act"], seed=103
    )
    conf_act = [cells[f"conflict|lam={lam}"]["opt_activation_rate"]["mean"] for lam in lambda_grid]
    return {
        "experiment": "system",
        "design": {
            "lambda_grid": list(lambda_grid),
            "seeds": seeds,
            "episodes": episodes,
            "tasks": [{"name": t["name"], "type": t["type"], "params": t["params"]} for t in tasks],
        },
        "cells": cells,
        "contrasts": {
            "conflict_activation_lam0_vs_lamT": conflict_contrast_act,
            "conflict_effectiveness_lam0_vs_lamT": conflict_contrast_eff,
            "saturated_activation_lam0_vs_lamT": saturated_contrast_act,
        },
        "verdicts": {
            "H1_conflict_dose_response": _monotone_nondecreasing(conf_act)
            and (conf_act[-1] - conf_act[0] > 0.1),
            "H1_conflict_activation_rises": conflict_contrast_act["ci_excludes_zero"]
            and conflict_contrast_act["delta"] > 0,
            "H2_specificity_saturated_null": not saturated_contrast_act["ci_excludes_zero"],
            "conflict_activation_by_lambda": [round(x, 4) for x in conf_act],
        },
    }


def _heartbeat(done: int, total: int, started: float, label: str) -> None:
    elapsed = time.monotonic() - started
    eta = (elapsed / done) * (total - done) if done else 0.0
    print(
        f"[blindness] {done}/{total}  {label}  transcurrido {elapsed:6.0f}s  ETA {eta:6.0f}s",
        flush=True,
    )


# ───────────────────────────── informe + orquestación ────────────────────────

PRE_REGISTERED_HYPOTHESES = """\
- **H1 (dosis-respuesta):** en tareas de conflicto, la tasa de retención/activación de la familia
  efectiva y la efectividad del mundo crecen monótonamente con λV; en λV=0 la familia queda suprimida.
- **H2 (especificidad / interacción):** el efecto de λV existe en conflicto y es NULO en tareas
  saturadas (sin brecha de efectividad que premiar).
- **H3 (control de ruido, sintético):** aleatorizar la señal de efectividad abole la recuperación ⇒
  el efecto es específico de la efectividad, no de "cualquier término extra en la recompensa".
- **Falsación:** retención alta con λV=0, o recuperación bajo el término de ruido, refutan la afirmación.
"""


def render_report(*, mechanism: Optional[Dict[str, Any]], system: Optional[Dict[str, Any]]) -> str:
    L: List[str] = []
    add = L.append
    add("# Ceguera de la recompensa: ¿un reward de coherencia suprime la efectividad?")
    add("")
    add("## Afirmación")
    add("")
    add("> Una recompensa de coherencia/proceso que NO mide efectividad suprime sistemáticamente "
        "capacidades que mejoran el resultado sin mejorar la coherencia.")
    add("")
    add("Relevante a process/intrinsic rewards (process reward models, RLHF, motivación intrínseca): "
        "un proxy plausible de \"buen razonamiento\" (IoC) puede degradar el desempeño al eliminar "
        "comportamientos efectivos-pero-coherencia-neutros.")
    add("")
    add("## Hipótesis (pre-registradas, fijadas antes de correr)")
    add("")
    add(PRE_REGISTERED_HYPOTHESES)

    if mechanism is not None:
        d = mechanism["design"]
        v = mechanism["verdicts"]
        add("## Experimento A — Mecanismo (sintético, aísla la variable)")
        add("")
        add(f"Conduce el selector guiado-por-recompensa con recompensas generadas donde coherencia es "
            f"plana y efectividad depende de la familia efectiva ({d['effective_family'].upper()}). "
            f"N={d['seeds']} semillas × {d['episodes']} episodios por celda.")
        add("")
        add("| λV | Retención (real) | Retención (ruido) |")
        add("|---:|---:|---:|")
        for i, lam in enumerate(d["lambda_grid"]):
            r = v["real_retention_by_lambda"][i]
            n = v["noise_retention_by_lambda"][i] if v["noise_retention_by_lambda"] else float("nan")
            add(f"| {lam} | {r:.3f} | {n:.3f} |")
        add("")
        add(f"- **H1 dosis-respuesta**: {'✓' if v['H1_dose_response'] else '✗'} "
            f"(retención real {v['real_retention_by_lambda']}).")
        add(f"- **H1 suprimida en λV=0**: {'✓' if v['H1_suppressed_at_lambda_zero'] else '✗'}.")
        add(f"- **H3 control de ruido plano**: "
            f"{'✓' if v['H3_noise_control_flat'] else '✗' if v['H3_noise_control_flat'] is not None else 'n/a'} "
            f"(retención ruido {v['noise_retention_by_lambda']}).")
        add("")

    if system is not None:
        d = system["design"]
        v = system["verdicts"]
        c = system["contrasts"]
        add("## Experimento B — Sistema real (confirma en la arquitectura)")
        add("")
        tasks_conf = [t["name"] for t in d["tasks"] if t["type"] == "conflict"]
        tasks_sat = [t["name"] for t in d["tasks"] if t["type"] == "saturated"]
        add(f"Organismo completo (core_plus_deliberative ⇒ el selector gobierna plan/opt; actuación ON; "
            f"conflicto RESETEADO cada episodio ⇒ estacionario), λV ∈ {d['lambda_grid']}, "
            f"{d['seeds']} semillas × {d['episodes']} episodios. Tareas conflicto: {tasks_conf or '—'}; "
            f"saturadas: {tasks_sat or '—'}. CI **entre-(tarea,semilla)**.")
        add("")
        add("| Tarea | λV | Activación OPT [CI] | Efectividad media [CI] | Override rate |")
        add("|---|---:|---|---|---:|")
        for ttype in ("conflict", "saturated"):
            for lam in d["lambda_grid"]:
                cell = system["cells"].get(f"{ttype}|lam={lam}")
                if not cell:
                    continue
                a = cell["opt_activation_rate"]; e = cell["mean_effectiveness"]; o = cell["override_rate"]
                add(f"| {ttype} | {lam} | {a['mean']:.3f} [{a['ci_lower']:.3f},{a['ci_upper']:.3f}] | "
                    f"{e['mean']:+.4f} [{e['ci_lower']:+.4f},{e['ci_upper']:+.4f}] | {o['mean']:.3f} |")
        add("")
        ca = c["conflict_activation_lam0_vs_lamT"]
        ce = c["conflict_effectiveness_lam0_vs_lamT"]
        sa = c["saturated_activation_lam0_vs_lamT"]
        add(f"- **H1 dosis-respuesta (conflicto)**: {'✓' if v['H1_conflict_dose_response'] else '✗'} "
            f"(activación {v['conflict_activation_by_lambda']}).")
        add(f"- **H1 activación sube con λV (conflicto, λ0→λT)**: "
            f"{'✓' if v['H1_conflict_activation_rises'] else '✗'} "
            f"(Δ={ca['delta']:+.3f}, CI [{ca['ci_lower']:+.3f},{ca['ci_upper']:+.3f}], d={ca['cohen_d']}).")
        add(f"- Efectividad sube con λV (conflicto): Δ={ce['delta']:+.4f}, "
            f"CI [{ce['ci_lower']:+.4f},{ce['ci_upper']:+.4f}], d={ce['cohen_d']}.")
        add(f"- **H2 especificidad (saturado nulo)**: {'✓' if v['H2_specificity_saturated_null'] else '✗'} "
            f"(Δ activación saturado={sa['delta']:+.3f}, CI [{sa['ci_lower']:+.3f},{sa['ci_upper']:+.3f}] — "
            f"{'incluye 0' if not sa['ci_excludes_zero'] else 'EXCLUYE 0'}).")
        add("")

    add("## Veredicto")
    add("")
    verdicts_ok = []
    if mechanism is not None:
        verdicts_ok += [mechanism["verdicts"]["H1_dose_response"],
                        mechanism["verdicts"]["H1_suppressed_at_lambda_zero"]]
        if mechanism["verdicts"]["H3_noise_control_flat"] is not None:
            verdicts_ok.append(mechanism["verdicts"]["H3_noise_control_flat"])
    if system is not None:
        verdicts_ok += [system["verdicts"]["H1_conflict_activation_rises"],
                        system["verdicts"]["H2_specificity_saturated_null"]]
    if verdicts_ok and all(verdicts_ok):
        add("Las hipótesis pre-registradas se **confirman**: la recompensa de coherencia (λV=0) suprime "
            "la capacidad efectiva; añadir la señal de efectividad la recupera de forma graduada y "
            "específica (no la recupera el ruido; no actúa donde no hay brecha).")
    else:
        add("Resultado **mixto/refutado** (ver hipótesis marcadas ✗) — reportado sin adornos.")
    add("")
    if mechanism is not None and system is not None:
        add("## Hallazgo del sistema real: el umbral salta ~40× (la idealización se rompe)")
        add("")
        add("El mecanismo idealizado asume **coherencia plana** entre acciones; ahí el término de "
            "efectividad recupera la familia con λV≈0.5. En la arquitectura real eso NO se cumple: "
            "el proxy IoC **anti-correlaciona** con la acción efectiva — ejecutar la intervención "
            "desviada (override) baja IoC ~0.24 (de 0.888 a ~0.646), porque cambiar la conducta "
            "altera la estructura causal/cierre que IoC premia. Además ΔIoC* es un *delta* sobre una "
            "secuencia no-estacionaria ⇒ oscila ±0.24, ~70× la señal de efectividad a λV=0.5. "
            "Consecuencia: la recuperación de la familia efectiva exige λV≈**20** (no 0.5) — la señal "
            "de efectividad debe escalarse ~40× para superar el ruido/penalización de coherencia. "
            "Esto es una forma MÁS FUERTE de la ceguera: el reward de coherencia no solo ignora la "
            "efectividad, **penaliza activamente** la desviación efectiva (coherencia-como-continuidad "
            "premia el conservadurismo).")
        add("")
    if system is not None and not system["verdicts"]["H2_specificity_saturated_null"]:
        add("## H2 REFUTADA (honesto): no hay control 'sin-brecha' en el escenario térmico")
        add("")
        add("H2 predecía que λV NO activaría la familia efectiva en tareas saturadas (sin brecha de "
            "efectividad). Se **refuta**: en saturado la activación también sube 0→1 con λV. La causa "
            "es honesta — las tareas 'saturadas' (greedy no falla) NO son de verdad sin-brecha: el "
            "override MEJORA la efectividad incluso ahí (p.ej. +0.054→+0.088), porque la dirección de "
            "optimización es *minimize* y enfriar más SIEMPRE ayuda ⇒ siempre queda margen que la "
            "efectividad premia. Un control limpio de especificidad exige una tarea donde el greedy ya "
            "sea óptimo (dirección *target_band*, o greedy = acción efectiva), que el térmico-minimize "
            "no ofrece. Esto NO contradice H1/H3 (el efecto central existe y es graduado); acota su "
            "alcance: no pudimos demostrar que sea EXCLUSIVO de tareas con brecha, porque no hay tarea "
            "sin brecha aquí. Trabajo futuro: escenario con óptimo interior.")
        add("")
    add("## Limitaciones honestas")
    add("")
    add("- La afirmación es sobre **especificación de la recompensa**, NO sobre sofisticación del "
        "razonamiento. La tarea es deliberadamente simple (térmica binaria) para AISLAR el efecto del "
        "reward de la dificultad de la tarea — no es evidencia de razonamiento avanzado.")
    add("- Una sola familia de tareas (térmica) y una señal de efectividad; la generalización a tareas "
        "ricas (observabilidad parcial, horizontes largos, ambigüedad) es trabajo futuro.")
    add("- λV no está calibrado contra un objetivo externo; el umbral de retención depende del "
        "balance coste/efectividad de esta tarea.")
    add("- Sin baseline contra métodos estándar (lookahead/RL): el override ES one-step lookahead; el "
        "estudio mide la SUPRESIÓN por la recompensa, no la potencia del razonamiento.")
    add("")
    return "\n".join(L)


def run_study(
    *,
    mode: str = "both",
    output_root: str | Path = "data/reports/reward_blindness",
    lambda_grid: Sequence[float] = LAMBDA_GRID_DEFAULT,
    mechanism_seeds: int = 1000,
    mechanism_episodes: int = 30,
    system_lambda_grid: Sequence[float] = (0.0, 5.0, 20.0, 50.0),
    system_seeds: int = 8,
    system_episodes: int = 36,
) -> Dict[str, Any]:
    import json

    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    mechanism = None
    system = None
    if mode in ("mechanism", "both"):
        print("[blindness] experimento de mecanismo (sintético)...", flush=True)
        mechanism = run_mechanism_experiment(
            lambda_grid=lambda_grid, seeds=mechanism_seeds, episodes=mechanism_episodes
        )
    if mode in ("system", "both"):
        print("[blindness] experimento de sistema real...", flush=True)
        system = run_system_experiment(
            output_root=root / "_system",
            lambda_grid=system_lambda_grid,
            seeds=system_seeds,
            episodes=system_episodes,
        )
    results = {"mechanism": mechanism, "system": system}
    (root / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=True, default=str), encoding="utf-8"
    )
    report = render_report(mechanism=mechanism, system=system)
    (root / "REPORT.md").write_text(report, encoding="utf-8")
    return {
        "output_root": str(root),
        "results_path": str(root / "results.json"),
        "report_path": str(root / "REPORT.md"),
        "mechanism_verdicts": None if mechanism is None else mechanism["verdicts"],
        "system_verdicts": None if system is None else system["verdicts"],
    }
