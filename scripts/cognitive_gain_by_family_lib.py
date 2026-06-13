"""Campaña de ganancia cognitiva POR TIPO DE RAZONAMIENTO (post-R3/Ω).

La campaña de abril 2026 (`cognitive_gain_v2`) midió HEUR/DIA_ADV/FAL_GUARD
contra `core_only` y dictaminó "no hay ganancia cognitiva suficiente" — antes
de que IND/PLAN/OPT fueran reales y sin las señales del canon (IoC*, Ωₜ,
recompensa semi-Markov r = ΔIoC* − λE·ΔE − λB·B_safe, risk_plus).

Esta campaña re-plantea el experimento con la matriz completa de tipos de
razonamiento y tres líneas de evidencia por familia:

1. **Ablación entre perfiles**: core+FAMILIA vs core_only (bootstrap CI,
   repetibilidad por bloque, clase de ganancia) — maquinaria reutilizada de
   `intelligence_campaign_lib`.
2. **Contrafactual intra-episodio**: `family_delta_ivc_r` /
   `family_contribution_proxy` (¿qué aportó la familia DENTRO del episodio?).
   Única vía para el núcleo ABD/ANA/CAU/CTF/DED/PROB, que los floors prohíben
   ablacionar.
3. **Economía del canon**: la recompensa semi-Markov ya internaliza el coste;
   Δrecompensa vs core es "ganancia neta por coste" en la escala del canon.

Python puro, sin dependencias nuevas; storage sqlite autocontenido por campaña.
"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Mapping, Sequence

from scripts.intelligence_campaign_lib import (
    OPTIONAL_FAMILY_ORDER,
    REGIME_ORDER,
    _episode_ioc_proxy,
    ProtocolSpec,
    _aggregate_cell_summary,
    _as_float,
    _block_seed,
    _mean,
    _now_stamp,
    _read_jsonl,
    _run_single_block,
    _serialize_cell_summary,
    _std,
    _write_json,
    _write_jsonl,
    bootstrap_ci_delta,
    build_storage_config,
    classify_family_role,
    classify_regime_gain,
    compute_family_synergy_delta,
    family_redundancy_flag,
    _compare_cells,
)
from runtime.storage import StorageConfig
from tests.benchmarks.benchmark_runner import BenchmarkRunner


# Diseño de protocolos (INVIERTE el de abril, con razón): el protocolo natural
# (presupuesto 6) expulsa a DED en cuanto una opcional se activa — el dictamen
# de abril estaba parcialmente contaminado por ese artefacto. El primario aquí
# da presupuesto holgado (10 = HARD_MAX_STEPS) para que cada familia se exprese;
# la recompensa canon ya internaliza el coste extra, así que el holgado no
# sesga a favor de los overlays. El natural queda como protocolo de sensibilidad
# ("¿sobrevive la ganancia al presupuesto real por defecto?").
CGF_PRIMARY_PROTOCOL = "steps10"
CGF_SENSITIVITY_PROTOCOL = "natural"
CGF_PROTOCOL_SPECS: List[ProtocolSpec] = [
    ProtocolSpec(name=CGF_PRIMARY_PROTOCOL, reasoning_max_steps=10),
    ProtocolSpec(name=CGF_SENSITIVITY_PROTOCOL, reasoning_max_steps=None),
]

# Matriz de tipos de razonamiento: baseline, una familia opcional por perfil,
# combo deliberativo, y los dos perfiles compuestos de referencia.
FAMILY_MATRIX_PROFILES: List[str] = [
    "core_only",
    "core_plus_heur",
    "core_plus_dialectic",
    "core_plus_guard",
    "core_plus_ind",
    "core_plus_plan",
    "core_plus_opt",
    "core_plus_deliberative",
    "adaptive_family_ecology_v2",
    "full_family_exploration",
]

SINGLE_FAMILY_PROFILES: Dict[str, str] = {
    "HEUR": "core_plus_heur",
    "DIA_ADV": "core_plus_dialectic",
    "FAL_GUARD": "core_plus_guard",
    "IND": "core_plus_ind",
    "PLAN": "core_plus_plan",
    "OPT": "core_plus_opt",
}

# Perfil compuesto donde cada familia puede activarse adaptativamente.
ADAPTIVE_REFERENCE_PROFILE: Dict[str, str] = {
    "HEUR": "adaptive_family_ecology_v2",
    "DIA_ADV": "adaptive_family_ecology_v2",
    "FAL_GUARD": "adaptive_family_ecology_v2",
    "IND": "adaptive_family_ecology_v2",
    "PLAN": "full_family_exploration",
    "OPT": "full_family_exploration",
}

COMBO_PROFILES_BY_FAMILY: Dict[str, List[str]] = {
    "HEUR": ["adaptive_family_ecology_v2", "full_family_exploration"],
    "DIA_ADV": ["adaptive_family_ecology_v2", "full_family_exploration"],
    "FAL_GUARD": ["adaptive_family_ecology_v2", "full_family_exploration"],
    "IND": ["adaptive_family_ecology_v2", "full_family_exploration"],
    "PLAN": ["core_plus_deliberative", "full_family_exploration"],
    "OPT": ["core_plus_deliberative", "full_family_exploration"],
}

EXTENDED_FAMILY_ORDER: List[str] = ["HEUR", "DIA_ADV", "FAL_GUARD", "IND", "PLAN", "OPT"]
CORE_FAMILY_ORDER: List[str] = ["ABD", "ANA", "CAU", "CTF", "DED", "PROB"]

# Señales del canon capturadas por episodio (benchmark_runner Grupo 7).
CANON_METRICS: List[str] = [
    "reasoning_reward",
    "ioc_star",
    "omega",
    "omega_cycle_error",
    "reward_reasoning_cost",
    "ioc_proxy",
]

APRIL_VERDICTS_PATH = Path(
    "data/benchmarks/cognitive_gain/adaptive_v2_intelligence_full_20260421_prompt1/"
    "regime_cognitive_verdicts.json"
)

PROFILE_LABELS: Dict[str, str] = {
    "core_only": "núcleo solo",
    "core_plus_heur": "núcleo + HEUR",
    "core_plus_dialectic": "núcleo + DIA_ADV",
    "core_plus_guard": "núcleo + FAL_GUARD",
    "core_plus_ind": "núcleo + IND",
    "core_plus_plan": "núcleo + PLAN",
    "core_plus_opt": "núcleo + OPT",
    "core_plus_deliberative": "núcleo + PLAN + OPT",
    "adaptive_family_ecology_v2": "adaptativo v2",
    "full_family_exploration": "exploración total",
}


def _canon_values(episodes: Sequence[Mapping[str, Any]], metric: str) -> List[float]:
    values: List[float] = []
    for row in episodes:
        value = row.get(metric)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def _active_families_from_episodes(episodes: Sequence[Mapping[str, Any]]) -> List[str]:
    active: List[str] = []
    for row in episodes:
        counts = row.get("family_activation_counts") or {}
        if not isinstance(counts, Mapping):
            continue
        for family, count in counts.items():
            name = str(family).upper()
            if name in EXTENDED_FAMILY_ORDER and _as_float(count, 0.0) > 0 and name not in active:
                active.append(name)
    return [family for family in EXTENDED_FAMILY_ORDER if family in active]


def _aggregate_family_dict(
    episodes: Sequence[Mapping[str, Any]],
    key: str,
) -> Dict[str, Dict[str, float]]:
    """Media/soporte por familia de un dict métrico por episodio."""
    sums: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for row in episodes:
        payload = row.get(key) or {}
        if not isinstance(payload, Mapping):
            continue
        for family, value in payload.items():
            name = str(family).upper()
            number = _as_float(value, 0.0)
            sums[name] = sums.get(name, 0.0) + number
            counts[name] = counts.get(name, 0) + 1
    return {
        name: {"mean": sums[name] / counts[name], "n": counts[name]}
        for name in sums
        if counts.get(name)
    }


def _augment_cell_with_canon(cell: Dict[str, Any], episodes: List[Dict[str, Any]],
                             block_episode_lists: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Añade medias/std + muestras por episodio y por bloque de las señales canon."""
    for metric in CANON_METRICS:
        values = _canon_values(episodes, metric)
        cell[f"{metric}_mean"] = _mean(values)
        cell[f"{metric}_std"] = _std(values)
        cell["_episode_metric_samples"][metric] = values
        cell["_block_metric_samples"][metric] = [
            _mean(_canon_values(block_rows, metric)) for block_rows in block_episode_lists
        ]
    cell["omega_cross_context_rate"] = _mean(_canon_values(episodes, "omega_cross_context"))
    cell["active_families_extended"] = _active_families_from_episodes(episodes)
    cell["family_delta_ivc_r_agg"] = _aggregate_family_dict(episodes, "family_delta_ivc_r")
    cell["family_contribution_proxy_agg"] = _aggregate_family_dict(
        episodes, "family_contribution_proxy"
    )
    return cell


def _compare_canon_cells(
    *,
    candidate: Mapping[str, Any],
    baseline: Mapping[str, Any],
    bootstrap_samples: int,
    seed: int,
) -> Dict[str, Any]:
    """Extensión canon de `_compare_cells`: Δ + CI + repetibilidad para reward/IoC*/Ω."""
    out: Dict[str, Any] = {}
    baseline_blocks = baseline.get("_block_metric_samples", {}) or {}
    candidate_blocks = candidate.get("_block_metric_samples", {}) or {}
    for metric in CANON_METRICS:
        delta = _as_float(candidate.get(f"{metric}_mean"), 0.0) - _as_float(
            baseline.get(f"{metric}_mean"), 0.0
        )
        out[f"delta_{metric}"] = delta
        left = list(baseline_blocks.get(metric, []))
        right = list(candidate_blocks.get(metric, []))
        comparable = min(len(left), len(right))
        positive = sum(1 for idx in range(comparable) if right[idx] > left[idx])
        out[f"{metric}_repeatability_rate"] = (positive / comparable) if comparable else 0.0
    for metric, offset in (("reasoning_reward", 31), ("ioc_star", 47)):
        ci = bootstrap_ci_delta(
            list((baseline.get("_episode_metric_samples", {}) or {}).get(metric, [])),
            list((candidate.get("_episode_metric_samples", {}) or {}).get(metric, [])),
            n_bootstrap=bootstrap_samples,
            seed=seed + offset,
        )
        out[f"{metric}_ci_lower"] = ci[0]
        out[f"{metric}_ci_upper"] = ci[1]
    return out


def _load_block_record_from_disk(
    *,
    root_dir: Path,
    campaign_id: str,
    protocol: ProtocolSpec,
    profile: str,
    regime: str,
    block_index: int,
    episodes_per_block: int,
    seed_start: int,
) -> Dict[str, Any] | None:
    """Reanudación: reconstruye el registro de un bloque ya corrido (espejo de
    `_run_single_block`, leyendo summary.json/episodes.jsonl en vez de ejecutar)."""
    output_dir = root_dir / "runs" / protocol.name / profile / regime / f"block_{block_index:02d}"
    episodes_path = output_dir / "episodes.jsonl"
    summary_path = output_dir / "summary.json"
    if not episodes_path.exists() or not summary_path.exists():
        return None
    episodes = _read_jsonl(episodes_path)
    if len(episodes) != episodes_per_block:
        return None
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    ivc_values = [_as_float(row.get("ivc_r"), 0.0) for row in episodes]
    precision_values = [_as_float(row.get("intervention_precision"), 0.0) for row in episodes]
    viability_values = [_as_float(row.get("viability_margin"), 0.0) for row in episodes]
    spatial_values = [_as_float(row.get("spatial_information_usage"), 0.0) for row in episodes]
    ioc_values = [_episode_ioc_proxy(row, regime=regime) for row in episodes]
    family_counts = summary.get("family_specific_activation_counts", {}) or {}
    active_optional_families = [
        family
        for family in OPTIONAL_FAMILY_ORDER
        if int(_as_float(family_counts.get(family, 0), 0.0)) > 0
    ]
    avg = summary.get("avg_metrics", {}) or {}

    return {
        "campaign_id": campaign_id,
        "protocol": protocol.name,
        "reasoning_max_steps": protocol.reasoning_max_steps,
        "profile": profile,
        "profile_label": PROFILE_LABELS.get(profile, profile),
        "regime": regime,
        "block_index": block_index,
        "episodes": episodes_per_block,
        "seed_start": seed_start,
        "summary_path": str(summary_path),
        "episodes_path": str(episodes_path),
        "success_rate": _as_float(summary.get("success_rate"), 0.0),
        "closure_break_rate": _as_float(summary.get("closure_break_rate"), 0.0),
        "backbone_floor_satisfied_rate": _as_float(
            summary.get("backbone_floor_satisfied_rate"), 0.0
        ),
        "optional_family_usage_rate": _as_float(summary.get("optional_family_usage_rate"), 0.0),
        "ivc_r_mean": _as_float(avg.get("ivc_r"), 0.0),
        "ivc_r_std": _std(ivc_values),
        "intervention_precision_mean": _as_float(avg.get("intervention_precision"), 0.0),
        "intervention_precision_std": _std(precision_values),
        "viability_margin_mean": _as_float(avg.get("viability_margin"), 0.0),
        "viability_margin_std": _std(viability_values),
        "spatial_information_usage_mean": _as_float(avg.get("spatial_information_usage"), 0.0),
        "spatial_information_usage_std": _std(spatial_values),
        "ioc_proxy_gain_mean": _mean(ioc_values),
        "ioc_proxy_gain_std": _std(ioc_values),
        "wall_time_ms_mean": _as_float(avg.get("wall_time_ms"), 0.0),
        "artifact_size_bytes_mean": _as_float(avg.get("artifact_size_bytes"), 0.0),
        "reasoning_trace_length_mean": _as_float(avg.get("reasoning_trace_length"), 0.0),
        "family_mix_entropy_mean": _as_float(avg.get("family_mix_entropy"), 0.0),
        "active_optional_families": active_optional_families,
        "resumed_from_disk": True,
    }


def _heartbeat(done: int, total: int, started: float, label: str) -> None:
    elapsed = time.monotonic() - started
    eta = (elapsed / done) * (total - done) if done else 0.0
    print(
        f"[campaña] bloque {done}/{total}  {label}  "
        f"transcurrido {elapsed:7.0f}s  ETA {eta:7.0f}s",
        flush=True,
    )


def run_family_matrix(
    *,
    campaign_id: str,
    output_root: str | Path,
    storage_config: StorageConfig,
    profiles: Sequence[str],
    regimes: Sequence[str],
    blocks: int,
    episodes_per_block: int,
    seed_base: int,
    protocols: Sequence[ProtocolSpec] | None = None,
) -> Dict[str, Any]:
    """Matriz protocolo × perfil × régimen × bloque con heartbeat y señales canon."""
    protocols = list(protocols or CGF_PROTOCOL_SPECS)
    root_dir = Path(output_root) / campaign_id
    root_dir.mkdir(parents=True, exist_ok=True)

    runner = BenchmarkRunner(output_root=root_dir, storage_config=storage_config)
    block_records: List[Dict[str, Any]] = []
    cell_rows: Dict[tuple, Dict[str, Any]] = {}
    total_blocks = len(protocols) * len(profiles) * len(regimes) * blocks
    started = time.monotonic()
    done = 0

    for protocol_index, protocol in enumerate(protocols):
        for profile_index, profile in enumerate(profiles):
            for regime_index, regime in enumerate(regimes):
                key = (protocol.name, profile, regime)
                cell_rows[key] = {"blocks": [], "episode_lists": []}
                for block_index in range(blocks):
                    seed_start = _block_seed(
                        seed_base=seed_base,
                        protocol_index=protocol_index,
                        profile_index=profile_index,
                        regime_index=regime_index,
                        block_index=block_index,
                        episodes_per_block=episodes_per_block,
                    )
                    block_record = _load_block_record_from_disk(
                        root_dir=root_dir,
                        campaign_id=campaign_id,
                        protocol=protocol,
                        profile=profile,
                        regime=regime,
                        block_index=block_index,
                        episodes_per_block=episodes_per_block,
                        seed_start=seed_start,
                    )
                    if block_record is None:
                        block_record = _run_single_block(
                            runner=runner,
                            root_dir=root_dir,
                            campaign_id=campaign_id,
                            protocol=protocol,
                            profile=profile,
                            regime=regime,
                            block_index=block_index,
                            episodes_per_block=episodes_per_block,
                            seed_start=seed_start,
                        )
                    block_episodes = _read_jsonl(Path(block_record["episodes_path"]))
                    for metric in CANON_METRICS:
                        block_record[f"{metric}_mean"] = _mean(
                            _canon_values(block_episodes, metric)
                        )
                    block_records.append(block_record)
                    cell_rows[key]["blocks"].append(block_record)
                    cell_rows[key]["episode_lists"].append(block_episodes)
                    done += 1
                    _heartbeat(done, total_blocks, started, f"{protocol.name}/{profile}/{regime}")

    _write_jsonl(root_dir / "block_metrics.jsonl", block_records)

    protocol_by_name = {spec.name: spec for spec in protocols}
    cell_summaries: Dict[tuple, Dict[str, Any]] = {}
    for (protocol_name, profile, regime), payload in cell_rows.items():
        episodes = [row for rows in payload["episode_lists"] for row in rows]
        cell = _aggregate_cell_summary(
            protocol=protocol_by_name[protocol_name],
            profile=profile,
            regime=regime,
            block_records=list(payload["blocks"]),
            episodes=episodes,
        )
        cell_summaries[(protocol_name, profile, regime)] = _augment_cell_with_canon(
            cell, episodes, payload["episode_lists"]
        )

    return {
        "root_dir": root_dir,
        "block_metrics_path": str(root_dir / "block_metrics.jsonl"),
        "block_records": block_records,
        "cell_summaries": cell_summaries,
    }


def _load_april_baseline() -> Dict[str, Any]:
    try:
        if APRIL_VERDICTS_PATH.exists():
            payload = json.loads(APRIL_VERDICTS_PATH.read_text(encoding="utf-8"))
            return {
                "available": True,
                "campaign_id": payload.get("campaign_id"),
                "primary_verdict": payload.get("primary_verdict"),
                "regime_gain_classes": {
                    str(record.get("regime")): str(record.get("regime_gain_class"))
                    for record in payload.get("regime_records", [])
                    if isinstance(record, Mapping)
                },
            }
    except Exception:
        pass
    return {"available": False}


def analyze_family_matrix(
    *,
    cell_summaries: Mapping[tuple, Mapping[str, Any]],
    regimes: Sequence[str],
    profiles: Sequence[str],
    bootstrap_samples: int,
    seed: int = 20260611,
) -> Dict[str, Any]:
    """Análisis por régimen y por familia sobre las celdas del protocolo primario."""
    regime_analysis: Dict[str, Any] = {}
    family_analysis: Dict[str, Any] = {family: {"per_regime": {}} for family in EXTENDED_FAMILY_ORDER}
    primary = CGF_PRIMARY_PROTOCOL

    for regime_index, regime in enumerate(regimes):
        core = cell_summaries[(primary, "core_only", regime)]
        profile_rows: Dict[str, Any] = {}
        for profile_index, profile in enumerate(profiles):
            cell = cell_summaries.get((primary, profile, regime))
            if cell is None:
                continue
            entry: Dict[str, Any] = {"cell": _serialize_cell_summary(cell)}
            if profile != "core_only":
                comparison = _compare_cells(
                    candidate=cell,
                    baseline=core,
                    bootstrap_samples=bootstrap_samples,
                    seed=seed + regime_index * 100 + profile_index,
                )
                comparison.update(
                    _compare_canon_cells(
                        candidate=cell,
                        baseline=core,
                        bootstrap_samples=bootstrap_samples,
                        seed=seed + regime_index * 100 + profile_index,
                    )
                )
                entry["vs_core_only"] = comparison
                entry["gain_class"] = classify_regime_gain(
                    candidate=cell, baseline=core, comparison=comparison
                )
            profile_rows[profile] = entry

        budget_sensitivity: Dict[str, Any] = {}
        for profile in profiles:
            natural_cell = cell_summaries.get((CGF_SENSITIVITY_PROTOCOL, profile, regime))
            primary_cell = cell_summaries.get((primary, profile, regime))
            if natural_cell is None or primary_cell is None:
                continue
            budget_sensitivity[profile] = {
                "success_rate_primary": _as_float(primary_cell.get("success_rate"), 0.0),
                "success_rate_natural": _as_float(natural_cell.get("success_rate"), 0.0),
                "closure_break_rate_natural": _as_float(
                    natural_cell.get("closure_break_rate"), 0.0
                ),
                "reward_natural": _as_float(natural_cell.get("reasoning_reward_mean"), 0.0),
                "reward_primary": _as_float(primary_cell.get("reasoning_reward_mean"), 0.0),
            }

        regime_analysis[regime] = {
            "profiles": profile_rows,
            "budget_sensitivity": budget_sensitivity,
            "core_family_counterfactual": {
                "family_delta_ivc_r": core.get("family_delta_ivc_r_agg", {}),
                "family_contribution_proxy": core.get("family_contribution_proxy_agg", {}),
            },
        }

        # Sinergia deliberativa PLAN+OPT en este régimen.
        plan_entry = profile_rows.get("core_plus_plan", {})
        opt_entry = profile_rows.get("core_plus_opt", {})
        delib_entry = profile_rows.get("core_plus_deliberative", {})
        if plan_entry.get("vs_core_only") and opt_entry.get("vs_core_only") and delib_entry.get("vs_core_only"):
            single_deltas = [
                _as_float(plan_entry["vs_core_only"].get("delta_ivc_r"), 0.0),
                _as_float(opt_entry["vs_core_only"].get("delta_ivc_r"), 0.0),
            ]
            combo_delta = _as_float(delib_entry["vs_core_only"].get("delta_ivc_r"), 0.0)
            synergy = compute_family_synergy_delta(
                single_deltas=single_deltas, combo_delta=combo_delta
            )
            regime_analysis[regime]["deliberative_synergy"] = {
                "plan_delta_ivc_r": single_deltas[0],
                "opt_delta_ivc_r": single_deltas[1],
                "combo_delta_ivc_r": combo_delta,
                "synergy_delta": synergy,
                "redundant": family_redundancy_flag(
                    single_delta=max(single_deltas),
                    combo_delta=combo_delta,
                    synergy_delta=synergy,
                ),
                "delta_reward_combo": _as_float(
                    delib_entry["vs_core_only"].get("delta_reasoning_reward"), 0.0
                ),
            }

        for family in EXTENDED_FAMILY_ORDER:
            single_profile = SINGLE_FAMILY_PROFILES[family]
            single_entry = profile_rows.get(single_profile)
            if not single_entry or "vs_core_only" not in single_entry:
                continue
            adaptive_profile = ADAPTIVE_REFERENCE_PROFILE[family]
            adaptive_entry = profile_rows.get(adaptive_profile, {})
            adaptive_cell = adaptive_entry.get("cell", {})
            adaptive_active = family in (adaptive_cell.get("active_families_extended") or [])
            adaptive_positive = str(adaptive_entry.get("gain_class") or "") in {
                "ganancia cognitiva fuerte",
                "ganancia cognitiva condicionada",
                "ganancia marginal",
            }
            combo_comparisons = [
                profile_rows[combo]["vs_core_only"]
                for combo in COMBO_PROFILES_BY_FAMILY[family]
                if profile_rows.get(combo, {}).get("vs_core_only")
            ]
            best_combo = max(
                combo_comparisons,
                key=lambda comp: _as_float(comp.get("delta_ioc_proxy_gain"), 0.0),
                default=None,
            )
            single_cell = single_entry["cell"]
            within_episode = (single_cell.get("family_delta_ivc_r_agg") or {}).get(family, {})
            family_analysis[family]["per_regime"][regime] = {
                "single_profile": single_profile,
                "gain_class": single_entry.get("gain_class"),
                "role": classify_family_role(
                    single_comparison=single_entry["vs_core_only"],
                    single_closure_stable=bool(single_cell.get("closure_stable")),
                    adaptive_active=adaptive_active,
                    adaptive_positive=adaptive_positive,
                    best_combo_comparison=best_combo,
                ),
                "delta_ivc_r": _as_float(single_entry["vs_core_only"].get("delta_ivc_r"), 0.0),
                "delta_ioc_proxy_gain": _as_float(
                    single_entry["vs_core_only"].get("delta_ioc_proxy_gain"), 0.0
                ),
                "delta_reasoning_reward": _as_float(
                    single_entry["vs_core_only"].get("delta_reasoning_reward"), 0.0
                ),
                "reward_ci": [
                    _as_float(single_entry["vs_core_only"].get("reasoning_reward_ci_lower"), 0.0),
                    _as_float(single_entry["vs_core_only"].get("reasoning_reward_ci_upper"), 0.0),
                ],
                "delta_ioc_star": _as_float(
                    single_entry["vs_core_only"].get("delta_ioc_star"), 0.0
                ),
                "delta_omega": _as_float(single_entry["vs_core_only"].get("delta_omega"), 0.0),
                "delta_reasoning_cost": _as_float(
                    single_entry["vs_core_only"].get("delta_reward_reasoning_cost"), 0.0
                ),
                "within_episode_delta_ivc_r": within_episode,
                "activated_in_single_profile": family
                in (single_cell.get("active_families_extended") or []),
                "success_rate": _as_float(single_cell.get("success_rate"), 0.0),
                "closure_stable": bool(single_cell.get("closure_stable")),
            }

    # Rol global por familia: el rol más favorable sostenido y conteos.
    role_rank = ["aporta", "aporta condicionado", "neutral", "perjudica"]
    for family, payload in family_analysis.items():
        roles = [entry["role"] for entry in payload["per_regime"].values()]
        payload["regimes_evaluated"] = len(roles)
        payload["role_counts"] = {role: roles.count(role) for role in role_rank if role in roles}
        payload["positive_regimes"] = [
            regime
            for regime, entry in payload["per_regime"].items()
            if entry["role"] in {"aporta", "aporta condicionado"}
        ]
        payload["harmful_regimes"] = [
            regime
            for regime, entry in payload["per_regime"].items()
            if entry["role"] == "perjudica"
        ]
        payload["mean_delta_reward"] = _mean(
            [entry["delta_reasoning_reward"] for entry in payload["per_regime"].values()]
        )
        payload["overall_role"] = next(
            (role for role in role_rank if roles and roles.count(role) >= max(1, len(roles) // 2)),
            roles and sorted(roles, key=role_rank.index)[0] or "sin evidencia",
        )

    return {"regimes": regime_analysis, "families": family_analysis}


def _primary_verdict(analysis: Mapping[str, Any]) -> str:
    families = analysis.get("families", {})
    contributing = [
        family
        for family, payload in families.items()
        if payload.get("positive_regimes")
    ]
    strong = [
        family
        for family, payload in families.items()
        if any(
            entry.get("gain_class") in {"ganancia cognitiva fuerte"}
            for entry in payload.get("per_regime", {}).values()
        )
    ]
    if strong:
        return (
            "hay ganancia cognitiva fuerte en al menos un tipo de razonamiento: "
            + ", ".join(sorted(strong))
        )
    if contributing:
        return (
            "hay ganancia cognitiva condicionada por tipo de razonamiento: "
            + ", ".join(sorted(contributing))
        )
    return "no hay ganancia cognitiva suficiente en ningún tipo de razonamiento"


def render_report(
    *,
    campaign_id: str,
    analysis: Mapping[str, Any],
    regimes: Sequence[str],
    profiles: Sequence[str],
    blocks: int,
    episodes_per_block: int,
    bootstrap_samples: int,
    april_baseline: Mapping[str, Any],
) -> str:
    lines: List[str] = []
    add = lines.append
    families = analysis["families"]
    regimes_payload = analysis["regimes"]
    verdict = _primary_verdict(analysis)

    add(f"# Ganancia cognitiva por tipo de razonamiento — {campaign_id}")
    add("")
    add(f"Generado: {datetime.now().isoformat(timespec='seconds')}  ")
    add(
        f"Diseño: {len(profiles)} perfiles × {len(regimes)} regímenes × "
        f"{blocks} bloques × {episodes_per_block} episodios. Protocolo primario "
        f"`{CGF_PRIMARY_PROTOCOL}` (presupuesto 10 = tope duro: cada familia puede "
        f"expresarse; la recompensa canon internaliza el coste extra) + protocolo de "
        f"sensibilidad `{CGF_SENSITIVITY_PROTOCOL}` (presupuesto 6 por defecto). "
        f"Bootstrap {bootstrap_samples}."
    )
    add("")
    add("## 1. Dictamen primario")
    add("")
    add(f"`{verdict}`")
    add("")

    add("## 2. Matriz régimen × perfil (protocolo primario)")
    add("")
    for regime in regimes:
        add(f"### {regime}")
        add("")
        add(
            "| Perfil | IVC-R | ΔIVC-R | IoC | IoC* | Ω | Recompensa r | ΔRecompensa | "
            "Coste | Éxito | Cierre | Clase |"
        )
        add("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|")
        for profile in profiles:
            entry = regimes_payload[regime]["profiles"].get(profile)
            if not entry:
                continue
            cell = entry["cell"]
            comp = entry.get("vs_core_only") or {}
            add(
                "| {label} | {ivc:.4f} | {divc} | {ioc:.4f} | {iocs:.4f} | {omega:.4f} | "
                "{rew:.4f} | {drew} | {cost:.1f} | {succ:.2f} | {closure} | {cls} |".format(
                    label=PROFILE_LABELS.get(profile, profile),
                    ivc=_as_float(cell.get("ivc_r_mean"), 0.0),
                    divc=(
                        f"{_as_float(comp.get('delta_ivc_r'), 0.0):+.4f}" if comp else "—"
                    ),
                    ioc=_as_float(cell.get("ioc_proxy_mean"), 0.0),
                    iocs=_as_float(cell.get("ioc_star_mean"), 0.0),
                    omega=_as_float(cell.get("omega_mean"), 0.0),
                    rew=_as_float(cell.get("reasoning_reward_mean"), 0.0),
                    drew=(
                        f"{_as_float(comp.get('delta_reasoning_reward'), 0.0):+.4f}"
                        if comp
                        else "—"
                    ),
                    cost=_as_float(cell.get("reward_reasoning_cost_mean"), 0.0),
                    succ=_as_float(cell.get("success_rate"), 0.0),
                    closure="sí" if cell.get("closure_stable") else "no",
                    cls=entry.get("gain_class", "baseline"),
                )
            )
        add("")

    add("## 3. Aislamiento por familia (core+X vs core_only)")
    add("")
    add(
        "| Familia | Rol global | Regímenes positivos | Regímenes dañinos | "
        "Δr̄ (recompensa) | Evidencia por régimen |"
    )
    add("|---|---|---|---|---:|---|")
    for family in EXTENDED_FAMILY_ORDER:
        payload = families[family]
        per_regime = payload["per_regime"]
        details = "; ".join(
            f"{regime}: {entry['role']} (ΔIVC-R {entry['delta_ivc_r']:+.4f}, "
            f"Δr {entry['delta_reasoning_reward']:+.4f}, CI r [{entry['reward_ci'][0]:+.4f},"
            f"{entry['reward_ci'][1]:+.4f}])"
            for regime, entry in per_regime.items()
        )
        add(
            f"| {family} | {payload.get('overall_role')} | "
            f"{', '.join(payload.get('positive_regimes') or []) or '—'} | "
            f"{', '.join(payload.get('harmful_regimes') or []) or '—'} | "
            f"{payload.get('mean_delta_reward', 0.0):+.4f} | {details or '—'} |"
        )
    add("")

    add("## 4. Sinergia deliberativa (PLAN+OPT)")
    add("")
    any_synergy = False
    for regime in regimes:
        synergy = regimes_payload[regime].get("deliberative_synergy")
        if not synergy:
            continue
        any_synergy = True
        add(
            f"- **{regime}**: ΔIVC-R PLAN {synergy['plan_delta_ivc_r']:+.4f}, "
            f"OPT {synergy['opt_delta_ivc_r']:+.4f}, combo {synergy['combo_delta_ivc_r']:+.4f} "
            f"⇒ sinergia {synergy['synergy_delta']:+.4f}"
            f" ({'redundante' if synergy['redundant'] else 'no redundante'}); "
            f"Δr combo {synergy['delta_reward_combo']:+.4f}."
        )
    if not any_synergy:
        add("- Sin datos de sinergia (faltan celdas PLAN/OPT/deliberative).")
    add("")

    add("## 5. Núcleo ABD/ANA/CAU/CTF/DED/PROB — contrafactual intra-episodio")
    add("")
    add(
        "El núcleo no es ablacionable (los floors de cierre lo protegen); su aporte se mide "
        "con el contrafactual intra-episodio `family_delta_ivc_r` sobre la celda `core_only`."
    )
    add("")
    add("| Régimen | " + " | ".join(CORE_FAMILY_ORDER) + " |")
    add("|---|" + "---:|" * len(CORE_FAMILY_ORDER))
    for regime in regimes:
        agg = regimes_payload[regime]["core_family_counterfactual"]["family_delta_ivc_r"]
        cells = []
        for family in CORE_FAMILY_ORDER:
            entry = agg.get(family) or {}
            cells.append(f"{_as_float(entry.get('mean'), 0.0):+.5f}" if entry else "—")
        add(f"| {regime} | " + " | ".join(cells) + " |")
    add("")

    add("## 6. Economía del razonamiento (recompensa canon r = ΔIoC* − λE·coste)")
    add("")
    add(
        "La recompensa semi-Markov internaliza el coste: Δr vs core es ganancia neta "
        "en la escala del canon. Ranking global (media de Δr sobre regímenes):"
    )
    add("")
    ranked = sorted(
        EXTENDED_FAMILY_ORDER,
        key=lambda fam: families[fam].get("mean_delta_reward", 0.0),
        reverse=True,
    )
    for index, family in enumerate(ranked, start=1):
        payload = families[family]
        add(
            f"{index}. **{family}** — Δr̄ {payload.get('mean_delta_reward', 0.0):+.4f} "
            f"(rol: {payload.get('overall_role')})"
        )
    add("")

    add("## 7. Coherencia multi-contexto (Ω / IoC*)")
    add("")
    for family in EXTENDED_FAMILY_ORDER:
        per_regime = families[family]["per_regime"]
        if not per_regime:
            continue
        mean_domega = _mean([entry["delta_omega"] for entry in per_regime.values()])
        mean_dstar = _mean([entry["delta_ioc_star"] for entry in per_regime.values()])
        add(
            f"- **{family}**: ΔΩ̄ {mean_domega:+.4f}, ΔIoC*̄ {mean_dstar:+.4f} "
            f"({'reduce obstrucción' if mean_domega < -1e-4 else 'no reduce obstrucción'})."
        )
    add("")

    add("## 8. Sensibilidad al presupuesto natural (6 pasos)")
    add("")
    add(
        "Bajo el presupuesto por defecto, la inserción legacy de overlays expulsa a DED "
        "(el validador Z3) y rompe el cierre — el artefacto que contaminó el dictamen de "
        "abril. Tasas de éxito por perfil (primario `steps10` vs `natural`):"
    )
    add("")
    add("| Régimen | Perfil | Éxito steps10 | Éxito natural | Cierre roto natural | r natural |")
    add("|---|---|---:|---:|---:|---:|")
    for regime in regimes:
        sensitivity = regimes_payload[regime].get("budget_sensitivity") or {}
        for profile, entry in sensitivity.items():
            if (
                abs(entry["success_rate_primary"] - entry["success_rate_natural"]) < 1e-9
                and entry["closure_break_rate_natural"] <= 0.0
            ):
                continue  # solo filas donde el presupuesto cambia algo
            add(
                f"| {regime} | {PROFILE_LABELS.get(profile, profile)} | "
                f"{entry['success_rate_primary']:.2f} | {entry['success_rate_natural']:.2f} | "
                f"{entry['closure_break_rate_natural']:.2f} | {entry['reward_natural']:+.4f} |"
            )
    add("")

    add("## 9. Comparación con la campaña de abril 2026")
    add("")
    if april_baseline.get("available"):
        add(
            f"- Abril (`{april_baseline.get('campaign_id')}`): "
            f"`{april_baseline.get('primary_verdict')}` — medía solo HEUR/DIA_ADV/FAL_GUARD, "
            "sin IND/PLAN/OPT reales ni IoC*/Ω/recompensa."
        )
        add(
            "- Dos artefactos corregidos desde abril: (a) el perfil de cierre `adaptive_min` "
            "no reconocía PLAN/OPT como opcionales legítimas (rechazo automático de toda "
            "secuencia deliberativa); (b) el protocolo primario de abril (presupuesto 6) "
            "expulsaba a DED al activarse cualquier overlay."
        )
        add(f"- Ahora: `{verdict}`.")
        classes = april_baseline.get("regime_gain_classes") or {}
        if classes:
            add("- Clases por régimen en abril: " + "; ".join(
                f"{regime}: {gain}" for regime, gain in classes.items()
            ) + ".")
    else:
        add("- No se encontró el verdict de abril; comparación omitida.")
    add("")

    add("## 10. Guardas y riesgos residuales")
    add("")
    add(
        "- Ningún dictamen se sostiene sin `success_rate`, `closure_break_rate` y "
        "`backbone_floor_satisfied_rate` estables (columna Cierre de la matriz §2)."
    )
    add(
        "- La señal primaria sigue siendo bootstrap sobre episodios; la recompensa canon "
        "añade la dimensión de coste pero su Dₜ (disipación física) es 0 hasta R4."
    )
    add(
        "- Los contrafactuales intra-episodio (`family_delta_ivc_r`) son proxies "
        "aditivos, no ablaciones reales del núcleo."
    )
    add(
        "- PLAN/OPT operan sobre el modelo de efectos declarado de la firma causal; "
        "su valor depende de la fidelidad de esa firma al mundo."
    )
    add(
        "- `full_family_exploration` no cabe ni en el tope duro de 10 pasos (6 núcleo + "
        "6 opcionales): su celda mide al perfil bajo desbordamiento real de presupuesto, "
        "no el valor ideal de sus familias."
    )
    add("")
    return "\n".join(lines)


def run_cognitive_gain_by_family_campaign(
    *,
    campaign_id: str | None = None,
    output_root: str | Path = "data/benchmarks/cognitive_gain",
    db_path: str | Path | None = None,
    artifact_root: str | Path | None = None,
    blocks: int = 8,
    episodes_per_block: int = 8,
    seed_base: int = 970000,
    bootstrap_samples: int = 1000,
    profiles: Sequence[str] | None = None,
    regimes: Sequence[str] | None = None,
    protocols: Sequence[ProtocolSpec] | None = None,
) -> Dict[str, Any]:
    campaign_id = campaign_id or f"cognitive_gain_by_family_{_now_stamp()}"
    profiles = list(profiles or FAMILY_MATRIX_PROFILES)
    regimes = list(regimes or REGIME_ORDER)
    root_dir = Path(output_root) / campaign_id
    root_dir.mkdir(parents=True, exist_ok=True)
    # Storage autocontenido por campaña (no engordar bases compartidas).
    storage_config = build_storage_config(
        db_path=db_path or (root_dir / "campaign.db"),
        artifact_root=artifact_root or (root_dir / "artifacts"),
    )

    matrix = run_family_matrix(
        campaign_id=campaign_id,
        output_root=output_root,
        storage_config=storage_config,
        profiles=profiles,
        regimes=regimes,
        blocks=blocks,
        episodes_per_block=episodes_per_block,
        seed_base=seed_base,
        protocols=protocols,
    )

    analysis = analyze_family_matrix(
        cell_summaries=matrix["cell_summaries"],
        regimes=regimes,
        profiles=profiles,
        bootstrap_samples=bootstrap_samples,
    )
    april_baseline = _load_april_baseline()
    verdict = _primary_verdict(analysis)

    report = render_report(
        campaign_id=campaign_id,
        analysis=analysis,
        regimes=regimes,
        profiles=profiles,
        blocks=blocks,
        episodes_per_block=episodes_per_block,
        bootstrap_samples=bootstrap_samples,
        april_baseline=april_baseline,
    )
    report_path = root_dir / "cognitive_gain_by_family_report.md"
    report_path.write_text(report, encoding="utf-8")

    verdicts_payload = {
        "campaign_id": campaign_id,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "primary_verdict": verdict,
        "blocks": blocks,
        "episodes_per_block": episodes_per_block,
        "bootstrap_samples": bootstrap_samples,
        "profiles": profiles,
        "regimes": regimes,
        "april_baseline": april_baseline,
        "families": analysis["families"],
        "regimes_analysis": {
            regime: {
                "profiles": {
                    profile: {
                        key: value
                        for key, value in entry.items()
                        if key != "cell"
                    }
                    | {"cell": entry.get("cell", {})}
                    for profile, entry in payload["profiles"].items()
                },
                "core_family_counterfactual": payload["core_family_counterfactual"],
                "deliberative_synergy": payload.get("deliberative_synergy"),
                "budget_sensitivity": payload.get("budget_sensitivity", {}),
            }
            for regime, payload in analysis["regimes"].items()
        },
    }
    verdicts_path = root_dir / "family_verdicts.json"
    _write_json(verdicts_path, verdicts_payload)

    return {
        "campaign_id": campaign_id,
        "root_dir": str(root_dir),
        "primary_verdict": verdict,
        "report_path": str(report_path),
        "verdicts_path": str(verdicts_path),
        "block_metrics_path": matrix["block_metrics_path"],
    }
