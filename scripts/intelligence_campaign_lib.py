"""Libreria reusable para campañas de ganancia cognitiva e inteligencia."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Mapping, Sequence
import json
import random

from runtime.reasoning.scheduler_meta.family_profiles import resolve_family_profile
from runtime.storage import StorageConfig
from runtime.world.grid_thermal_scenario import GridThermalScenario
from tests.benchmarks.benchmark_runner import BenchmarkConfig, BenchmarkRunner


COGNITIVE_PROFILES: List[str] = [
    "core_only",
    "core_plus_guard",
    "core_plus_dialectic",
    "core_plus_heur",
    "adaptive_family_ecology_v2",
    "adaptive_family_ecology",
]
FIXED_BASELINE_PROFILES: List[str] = [
    "core_only",
    "core_plus_guard",
    "core_plus_dialectic",
    "core_plus_heur",
]
CAUSAL_PROFILES: List[str] = [
    "core_only",
    "core_plus_heur",
    "core_plus_dialectic",
    "core_plus_guard",
    "core_plus_heur_guard",
    "core_plus_heur_dialectic",
    "core_plus_guard_dialectic",
    "core_plus_triple_optional",
    "adaptive_family_ecology_v2",
]
PRIMARY_PROTOCOL = "natural"
SENSITIVITY_PROTOCOL = "sensitivity_10"
OPTIONAL_FAMILY_ORDER: List[str] = ["HEUR", "DIA_ADV", "FAL_GUARD"]
REGIME_ORDER: List[str] = [
    "homogeneous_safe",
    "heterogeneous_elevated",
    "heterogeneous_warning",
    "viability_edge",
    "vram_favorable",
]
FAMILY_TO_SINGLE_PROFILE: Dict[str, str] = {
    "HEUR": "core_plus_heur",
    "DIA_ADV": "core_plus_dialectic",
    "FAL_GUARD": "core_plus_guard",
}
PROFILE_TO_LABEL: Dict[str, str] = {
    "core_only": "backbone solo",
    "core_plus_heur": "backbone + HEUR",
    "core_plus_dialectic": "backbone + DIA_ADV",
    "core_plus_guard": "backbone + FAL_GUARD",
    "core_plus_heur_guard": "backbone + HEUR + FAL_GUARD",
    "core_plus_heur_dialectic": "backbone + HEUR + DIA_ADV",
    "core_plus_guard_dialectic": "backbone + FAL_GUARD + DIA_ADV",
    "core_plus_triple_optional": "backbone + triple opcional",
    "adaptive_family_ecology": "adaptive v1",
    "adaptive_family_ecology_v2": "adaptive v2",
}


@dataclass(frozen=True)
class ProtocolSpec:
    name: str
    reasoning_max_steps: int | None


PROTOCOL_SPECS: List[ProtocolSpec] = [
    ProtocolSpec(name=PRIMARY_PROTOCOL, reasoning_max_steps=None),
    ProtocolSpec(name=SENSITIVITY_PROTOCOL, reasoning_max_steps=10),
]


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _regime_specs() -> Dict[str, Dict[str, Any]]:
    return {
        "homogeneous_safe": {
            "scenario_params": {
                "grid_size": 1,
                "initial_temperature": 0.62,
                "alarm_threshold": 0.85,
                "cooling_effect": 0.07,
            }
        },
        "heterogeneous_elevated": {
            "scenario_params": {
                "grid_size": 5,
                "topology": "gradient_ns",
                "initial_temperature": 0.78,
                "alarm_threshold": 0.85,
                "cooling_effect": 0.07,
            }
        },
        "heterogeneous_warning": {
            "scenario_params": {
                "grid_size": 5,
                "topology": "checkerboard",
                "initial_temperature": 0.88,
                "alarm_threshold": 0.85,
                "cooling_effect": 0.06,
            }
        },
        "viability_edge": {
            "scenario_params": {
                "grid_size": 5,
                "topology": "hotspot_center",
                "initial_temperature": 0.95,
                "alarm_threshold": 0.90,
                "cooling_effect": 0.04,
            }
        },
        "vram_favorable": {
            "scenario_params": {
                "grid_size": 5,
                "topology": "uniform",
                "initial_temperature": 0.80,
                "alarm_threshold": 0.85,
                "cooling_effect": 0.07,
            }
        },
        # Conflicto causal-contrafactual: el núcleo greedy elige mal
        # (deactivate_cooling con alarma activa) mientras activate_cooling triunfa.
        # Único régimen con error residual ⇒ el razonamiento deliberativo paga
        # (validado para ext_open_thinker; reusado para el override determinista).
        "causal_counterfactual_conflict": {
            "scenario_params": {
                "grid_size": 5,
                "topology": "uniform",
                "initial_temperature": 0.88,
                "alarm_threshold": 0.85,
                "cooling_effect": 0.07,
            }
        },
    }


def build_storage_config(
    *,
    db_path: str | Path,
    artifact_root: str | Path,
) -> StorageConfig:
    return StorageConfig(
        mode="sqlite",
        sqlite_db_path=str(Path(db_path)),
        postgres_dsn=None,
        artifact_root=Path(artifact_root),
        prefer_postgres_reads=False,
        strict_dual_write=False,
    )


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True, default=str), encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=True, default=str) + "\n")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    center = _mean(values)
    variance = sum((value - center) ** 2 for value in values) / (len(values) - 1)
    return variance ** 0.5


def _safe_median(values: Iterable[float]) -> float:
    values = list(values)
    return float(median(values)) if values else 0.0


def _spatial_applicable(regime: str) -> bool:
    return regime != "homogeneous_safe"


def compute_ioc_proxy_gain(
    *,
    ivc_r: float,
    intervention_precision: float,
    viability_margin: float,
    spatial_information_usage: float | None,
    spatial_applicable: bool,
) -> float:
    components = {
        "ivc_r": (0.45, _as_float(ivc_r, 0.0)),
        "intervention_precision": (0.25, _as_float(intervention_precision, 0.0)),
        "viability_margin": (0.20, _as_float(viability_margin, 0.0)),
    }
    if spatial_applicable and spatial_information_usage is not None:
        components["spatial_information_usage"] = (
            0.10,
            _as_float(spatial_information_usage, 0.0),
        )

    total_weight = sum(weight for weight, _ in components.values())
    if total_weight <= 0.0:
        return 0.0
    return sum((weight / total_weight) * value for weight, value in components.values())


def _episode_ioc_proxy(row: Mapping[str, Any], *, regime: str) -> float:
    spatial_value = row.get("spatial_information_usage")
    return compute_ioc_proxy_gain(
        ivc_r=_as_float(row.get("ivc_r"), 0.0),
        intervention_precision=_as_float(row.get("intervention_precision"), 0.0),
        viability_margin=_as_float(row.get("viability_margin"), 0.0),
        spatial_information_usage=(
            None if spatial_value is None else _as_float(spatial_value, 0.0)
        ),
        spatial_applicable=_spatial_applicable(regime),
    )


def bootstrap_ci_delta(
    values_left: Sequence[float],
    values_right: Sequence[float],
    *,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 20260421,
) -> tuple[float, float]:
    if not values_left or not values_right:
        return (0.0, 0.0)
    rng = random.Random(seed)
    deltas: List[float] = []
    left = list(values_left)
    right = list(values_right)
    for _ in range(max(n_bootstrap, 50)):
        sample_left = [rng.choice(left) for _ in range(len(left))]
        sample_right = [rng.choice(right) for _ in range(len(right))]
        deltas.append(_mean(sample_right) - _mean(sample_left))
    deltas.sort()
    alpha = 1.0 - confidence
    lower_idx = int(len(deltas) * (alpha / 2))
    upper_idx = int(len(deltas) * (1 - alpha / 2))
    upper_idx = min(max(upper_idx, 0), len(deltas) - 1)
    lower_idx = min(max(lower_idx, 0), len(deltas) - 1)
    return deltas[lower_idx], deltas[upper_idx]


def _profile_optional_families(profile_name: str) -> List[str]:
    profile = resolve_family_profile(profile_name, mode="adaptive" if "adaptive" in profile_name else "fixed")
    families = []
    for family in profile.optional_families:
        normalized = family.strip().upper()
        if normalized in OPTIONAL_FAMILY_ORDER and normalized not in families:
            families.append(normalized)
    return families


def _block_seed(
    *,
    seed_base: int,
    protocol_index: int,
    profile_index: int,
    regime_index: int,
    block_index: int,
    episodes_per_block: int,
) -> int:
    return (
        seed_base
        + (protocol_index * 1_000_000)
        + (profile_index * 100_000)
        + (regime_index * 10_000)
        + (block_index * max(episodes_per_block, 1) * 10)
    )


def _run_single_block(
    *,
    runner: BenchmarkRunner,
    root_dir: Path,
    campaign_id: str,
    protocol: ProtocolSpec,
    profile: str,
    regime: str,
    block_index: int,
    episodes_per_block: int,
    seed_start: int,
) -> Dict[str, Any]:
    output_dir = root_dir / "runs" / protocol.name / profile / regime / f"block_{block_index:02d}"
    cfg = BenchmarkConfig(
        scenario_name=f"{campaign_id}__{protocol.name}__{regime}__{profile}__block_{block_index:02d}",
        scenario_class=GridThermalScenario,
        scenario_params=dict(_regime_specs()[regime]["scenario_params"]),
        episodes=episodes_per_block,
        base_seed=seed_start,
        max_steps=50,
        output_dir=output_dir,
        run_id=f"{campaign_id}-{protocol.name}-{profile}-{regime}-b{block_index:02d}",
        reasoning_mode="adaptive" if "adaptive" in profile else "fixed",
        family_profile=profile,
        regime_label=regime,
        reasoning_max_steps=protocol.reasoning_max_steps,
    )
    summary = runner.run_benchmark(cfg)
    episodes = _read_jsonl(output_dir / "episodes.jsonl")

    ivc_values = [_as_float(row.get("ivc_r"), 0.0) for row in episodes]
    precision_values = [_as_float(row.get("intervention_precision"), 0.0) for row in episodes]
    viability_values = [_as_float(row.get("viability_margin"), 0.0) for row in episodes]
    spatial_values = [_as_float(row.get("spatial_information_usage"), 0.0) for row in episodes]
    ioc_values = [_episode_ioc_proxy(row, regime=regime) for row in episodes]

    family_counts = summary.get("family_specific_activation_counts", {}) or {}
    active_optional_families = [
        family for family in OPTIONAL_FAMILY_ORDER if int(_as_float(family_counts.get(family, 0), 0.0)) > 0
    ]
    avg = summary.get("avg_metrics", {}) or {}

    return {
        "campaign_id": campaign_id,
        "protocol": protocol.name,
        "reasoning_max_steps": protocol.reasoning_max_steps,
        "profile": profile,
        "profile_label": PROFILE_TO_LABEL.get(profile, profile),
        "regime": regime,
        "block_index": block_index,
        "episodes": episodes_per_block,
        "seed_start": seed_start,
        "summary_path": str(output_dir / "summary.json"),
        "episodes_path": str(output_dir / "episodes.jsonl"),
        "success_rate": _as_float(summary.get("success_rate"), 0.0),
        "closure_break_rate": _as_float(summary.get("closure_break_rate"), 0.0),
        "backbone_floor_satisfied_rate": _as_float(summary.get("backbone_floor_satisfied_rate"), 0.0),
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
    }


def _aggregate_cell_summary(
    *,
    protocol: ProtocolSpec,
    profile: str,
    regime: str,
    block_records: List[Dict[str, Any]],
    episodes: List[Dict[str, Any]],
) -> Dict[str, Any]:
    ivc_values = [_as_float(row.get("ivc_r"), 0.0) for row in episodes]
    precision_values = [_as_float(row.get("intervention_precision"), 0.0) for row in episodes]
    viability_values = [_as_float(row.get("viability_margin"), 0.0) for row in episodes]
    spatial_values = [_as_float(row.get("spatial_information_usage"), 0.0) for row in episodes]
    success_values = [_as_float(row.get("success_rate"), 0.0) for row in episodes]
    closure_values = [_as_float(row.get("closure_break_flag"), 0.0) for row in episodes]
    floor_values = [_as_float(row.get("backbone_floor_satisfied_flag"), 0.0) for row in episodes]
    wall_values = [_as_float(row.get("wall_time_ms"), 0.0) for row in episodes]
    artifact_values = [_as_float(row.get("artifact_size_bytes"), 0.0) for row in episodes]
    trace_values = [_as_float(row.get("reasoning_trace_length"), 0.0) for row in episodes]
    ioc_values = [_episode_ioc_proxy(row, regime=regime) for row in episodes]

    active_optional_families: List[str] = []
    for record in block_records:
        for family in record.get("active_optional_families", []):
            if family not in active_optional_families:
                active_optional_families.append(family)

    block_metric_samples = {
        "ivc_r": [record["ivc_r_mean"] for record in block_records],
        "intervention_precision": [record["intervention_precision_mean"] for record in block_records],
        "viability_margin": [record["viability_margin_mean"] for record in block_records],
        "spatial_information_usage": [record["spatial_information_usage_mean"] for record in block_records],
        "ioc_proxy_gain": [record["ioc_proxy_gain_mean"] for record in block_records],
        "success_rate": [record["success_rate"] for record in block_records],
        "closure_break_rate": [record["closure_break_rate"] for record in block_records],
    }

    return {
        "protocol": protocol.name,
        "reasoning_max_steps": protocol.reasoning_max_steps,
        "profile": profile,
        "profile_label": PROFILE_TO_LABEL.get(profile, profile),
        "regime": regime,
        "blocks": len(block_records),
        "episodes": len(episodes),
        "active_optional_families": active_optional_families,
        "success_rate": _mean(success_values),
        "closure_break_rate": _mean(closure_values),
        "backbone_floor_satisfied_rate": _mean(floor_values),
        "optional_family_usage_rate": _mean(
            [_as_float(row.get("family_optional_used_flag"), 0.0) for row in episodes]
        ),
        "ivc_r_mean": _mean(ivc_values),
        "ivc_r_median_by_block": _safe_median(block_metric_samples["ivc_r"]),
        "intervention_precision_mean": _mean(precision_values),
        "intervention_precision_median_by_block": _safe_median(block_metric_samples["intervention_precision"]),
        "viability_margin_mean": _mean(viability_values),
        "viability_margin_median_by_block": _safe_median(block_metric_samples["viability_margin"]),
        "spatial_information_usage_mean": _mean(spatial_values),
        "spatial_information_usage_median_by_block": _safe_median(block_metric_samples["spatial_information_usage"]),
        "ioc_proxy_gain_mean": _mean(ioc_values),
        "ioc_proxy_gain_median_by_block": _safe_median(block_metric_samples["ioc_proxy_gain"]),
        "wall_time_ms_mean": _mean(wall_values),
        "artifact_size_bytes_mean": _mean(artifact_values),
        "reasoning_trace_length_mean": _mean(trace_values),
        "closure_stable": (
            _mean(success_values) >= 0.95
            and _mean(closure_values) <= 0.0
            and _mean(floor_values) >= 1.0
        ),
        "_episode_metric_samples": {
            "ivc_r": ivc_values,
            "intervention_precision": precision_values,
            "viability_margin": viability_values,
            "spatial_information_usage": spatial_values,
            "ioc_proxy_gain": ioc_values,
            "success_rate": success_values,
            "closure_break_rate": closure_values,
            "backbone_floor_satisfied_rate": floor_values,
        },
        "_block_metric_samples": block_metric_samples,
    }


def _serialize_cell_summary(cell: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in cell.items() if not str(key).startswith("_")}


def _compare_cells(
    *,
    candidate: Mapping[str, Any],
    baseline: Mapping[str, Any],
    bootstrap_samples: int,
    seed: int,
) -> Dict[str, Any]:
    metric_aliases = {
        "ivc_r": "delta_ivc_r",
        "intervention_precision": "delta_intervention_precision",
        "viability_margin": "delta_viability_margin",
        "spatial_information_usage": "delta_spatial_information_usage",
        "ioc_proxy_gain": "delta_ioc_proxy_gain",
        "success_rate": "delta_success_rate",
        "closure_break_rate": "delta_closure_break_rate",
    }
    baseline_blocks = baseline.get("_block_metric_samples", {}) or {}
    candidate_blocks = candidate.get("_block_metric_samples", {}) or {}
    repeatability: Dict[str, Dict[str, float]] = {}
    out: Dict[str, Any] = {}

    for metric, field_name in metric_aliases.items():
        baseline_mean = _as_float(baseline.get(f"{metric}_mean"), 0.0)
        candidate_mean = _as_float(candidate.get(f"{metric}_mean"), 0.0)
        out[field_name] = candidate_mean - baseline_mean

        left_block_values = list(baseline_blocks.get(metric, []))
        right_block_values = list(candidate_blocks.get(metric, []))
        comparable = min(len(left_block_values), len(right_block_values))
        positive = 0
        for idx in range(comparable):
            if right_block_values[idx] > left_block_values[idx]:
                positive += 1
        rate = (positive / comparable) if comparable > 0 else 0.0
        repeatability[metric] = {
            "positive_blocks": positive,
            "total_blocks": comparable,
            "rate": rate,
        }

    ivc_ci = bootstrap_ci_delta(
        list((baseline.get("_episode_metric_samples", {}) or {}).get("ivc_r", [])),
        list((candidate.get("_episode_metric_samples", {}) or {}).get("ivc_r", [])),
        n_bootstrap=bootstrap_samples,
        seed=seed,
    )
    ioc_ci = bootstrap_ci_delta(
        list((baseline.get("_episode_metric_samples", {}) or {}).get("ioc_proxy_gain", [])),
        list((candidate.get("_episode_metric_samples", {}) or {}).get("ioc_proxy_gain", [])),
        n_bootstrap=bootstrap_samples,
        seed=seed + 17,
    )

    out["ivc_r_ci_lower"] = ivc_ci[0]
    out["ivc_r_ci_upper"] = ivc_ci[1]
    out["ioc_proxy_ci_lower"] = ioc_ci[0]
    out["ioc_proxy_ci_upper"] = ioc_ci[1]
    out["block_median_delta_ivc_r"] = (
        _as_float(candidate.get("ivc_r_median_by_block"), 0.0)
        - _as_float(baseline.get("ivc_r_median_by_block"), 0.0)
    )
    out["repeatability_blocks_positive"] = repeatability

    secondary_metrics = ["intervention_precision", "viability_margin"]
    if _spatial_applicable(str(candidate.get("regime"))):
        secondary_metrics.append("spatial_information_usage")
    out["secondary_positive_repeatable_metrics"] = [
        metric
        for metric in secondary_metrics
        if out[f"delta_{metric}"] > 0.0 and repeatability[metric]["rate"] >= 0.625
    ]
    return out


def classify_regime_gain(
    *,
    candidate: Mapping[str, Any],
    baseline: Mapping[str, Any],
    comparison: Mapping[str, Any],
) -> str:
    if not bool(candidate.get("closure_stable")):
        return "sin ganancia"
    if _as_float(comparison.get("delta_ivc_r"), 0.0) <= 0.0:
        return "sin ganancia"

    ivc_repeatability = (
        (comparison.get("repeatability_blocks_positive", {}) or {})
        .get("ivc_r", {})
        .get("rate", 0.0)
    )
    has_secondary = bool(comparison.get("secondary_positive_repeatable_metrics"))
    if (
        _as_float(candidate.get("success_rate"), 0.0) >= 0.95
        and _as_float(candidate.get("closure_break_rate"), 1.0) <= 0.0
        and _as_float(candidate.get("backbone_floor_satisfied_rate"), 0.0) >= 1.0
        and _as_float(comparison.get("block_median_delta_ivc_r"), 0.0) > 0.0
        and _as_float(comparison.get("ivc_r_ci_lower"), 0.0) > 0.0
        and ivc_repeatability >= 0.625
        and has_secondary
    ):
        return "ganancia cognitiva fuerte"
    if (
        _as_float(comparison.get("delta_ivc_r"), 0.0) > 0.0
        and ivc_repeatability >= 0.375
    ):
        if has_secondary or _as_float(comparison.get("ivc_r_ci_upper"), 0.0) > 0.0:
            return "ganancia cognitiva condicionada"
    if _as_float(comparison.get("delta_ivc_r"), 0.0) > 0.0:
        return "ganancia marginal"
    return "sin ganancia"


def _rank_profiles_for_regime(
    *,
    cell_summaries: Mapping[tuple[str, str, str], Mapping[str, Any]],
    regime: str,
    protocol_name: str,
    profiles: Sequence[str],
) -> List[str]:
    candidates = []
    for profile in profiles:
        cell = cell_summaries.get((protocol_name, profile, regime))
        if cell is None:
            continue
        candidates.append(
            (
                1 if bool(cell.get("closure_stable")) else 0,
                _as_float(cell.get("ioc_proxy_gain_mean"), 0.0),
                _as_float(cell.get("ivc_r_mean"), 0.0),
                _as_float(cell.get("intervention_precision_mean"), 0.0),
                -_as_float(cell.get("closure_break_rate"), 0.0),
                profile,
            )
        )
    candidates.sort(reverse=True)
    return [profile for *_, profile in candidates]


def _optional_family_evidence(regime_records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    evidence: Dict[str, Dict[str, Any]] = {
        family: {
            "positive_regimes": [],
            "best_fixed_alignment": [],
            "active_regimes": [],
        }
        for family in OPTIONAL_FAMILY_ORDER
    }
    baseline_map = {
        "core_plus_heur": "HEUR",
        "core_plus_dialectic": "DIA_ADV",
        "core_plus_guard": "FAL_GUARD",
    }
    for record in regime_records:
        gain_class = str(record.get("regime_gain_class") or "")
        active = list(record.get("active_optional_families") or [])
        best_fixed = str(record.get("best_fixed_baseline_by_regime") or "")
        for family in active:
            evidence[family]["active_regimes"].append(record["regime"])
            if gain_class in {"ganancia cognitiva fuerte", "ganancia cognitiva condicionada"}:
                evidence[family]["positive_regimes"].append(record["regime"])
            if baseline_map.get(best_fixed) == family:
                evidence[family]["best_fixed_alignment"].append(record["regime"])
    return evidence


def run_block_matrix(
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
    protocols = list(protocols or PROTOCOL_SPECS)
    root_dir = Path(output_root) / campaign_id
    root_dir.mkdir(parents=True, exist_ok=True)

    runner = BenchmarkRunner(output_root=root_dir, storage_config=storage_config)
    block_records: List[Dict[str, Any]] = []
    cell_rows: Dict[tuple[str, str, str], Dict[str, Any]] = {}

    for protocol_index, protocol in enumerate(protocols):
        for profile_index, profile in enumerate(profiles):
            for regime_index, regime in enumerate(regimes):
                key = (protocol.name, profile, regime)
                cell_rows[key] = {"blocks": [], "episodes": []}
                for block_index in range(blocks):
                    seed_start = _block_seed(
                        seed_base=seed_base,
                        protocol_index=protocol_index,
                        profile_index=profile_index,
                        regime_index=regime_index,
                        block_index=block_index,
                        episodes_per_block=episodes_per_block,
                    )
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
                    block_records.append(block_record)
                    cell_rows[key]["blocks"].append(block_record)
                    cell_rows[key]["episodes"].extend(_read_jsonl(Path(block_record["episodes_path"])))

    _write_jsonl(root_dir / "block_metrics.jsonl", block_records)

    cell_summaries: Dict[tuple[str, str, str], Dict[str, Any]] = {}
    protocol_by_name = {spec.name: spec for spec in protocols}
    for (protocol_name, profile, regime), payload in cell_rows.items():
        cell_summaries[(protocol_name, profile, regime)] = _aggregate_cell_summary(
            protocol=protocol_by_name[protocol_name],
            profile=profile,
            regime=regime,
            block_records=list(payload["blocks"]),
            episodes=list(payload["episodes"]),
        )

    return {
        "root_dir": root_dir,
        "block_metrics_path": str(root_dir / "block_metrics.jsonl"),
        "block_records": block_records,
        "cell_summaries": cell_summaries,
    }


def run_cognitive_gain_campaign(
    *,
    campaign_id: str | None = None,
    output_root: str | Path = "data/benchmarks/cognitive_gain",
    db_path: str | Path = "aeon_event_log.db",
    artifact_root: str | Path = "data/artifacts",
    blocks: int = 8,
    episodes_per_block: int = 8,
    seed_base: int = 910000,
    bootstrap_samples: int = 1000,
) -> Dict[str, Any]:
    campaign_id = campaign_id or f"cognitive_gain_v2_{_now_stamp()}"
    storage_config = build_storage_config(db_path=db_path, artifact_root=artifact_root)
    matrix = run_block_matrix(
        campaign_id=campaign_id,
        output_root=output_root,
        storage_config=storage_config,
        profiles=COGNITIVE_PROFILES,
        regimes=REGIME_ORDER,
        blocks=blocks,
        episodes_per_block=episodes_per_block,
        seed_base=seed_base,
        protocols=PROTOCOL_SPECS,
    )

    cell_summaries = matrix["cell_summaries"]
    primary_regime_records: List[Dict[str, Any]] = []

    for regime in REGIME_ORDER:
        candidate = cell_summaries[(PRIMARY_PROTOCOL, "adaptive_family_ecology_v2", regime)]
        core = cell_summaries[(PRIMARY_PROTOCOL, "core_only", regime)]
        best_fixed_ranking = _rank_profiles_for_regime(
            cell_summaries=cell_summaries,
            regime=regime,
            protocol_name=PRIMARY_PROTOCOL,
            profiles=FIXED_BASELINE_PROFILES,
        )
        best_fixed_profile = best_fixed_ranking[0]
        best_fixed = cell_summaries[(PRIMARY_PROTOCOL, best_fixed_profile, regime)]
        best_profile = _rank_profiles_for_regime(
            cell_summaries=cell_summaries,
            regime=regime,
            protocol_name=PRIMARY_PROTOCOL,
            profiles=COGNITIVE_PROFILES,
        )[0]

        vs_core = _compare_cells(
            candidate=candidate,
            baseline=core,
            bootstrap_samples=bootstrap_samples,
            seed=20260421 + REGIME_ORDER.index(regime),
        )
        vs_best_fixed = _compare_cells(
            candidate=candidate,
            baseline=best_fixed,
            bootstrap_samples=bootstrap_samples,
            seed=20260521 + REGIME_ORDER.index(regime),
        )
        regime_gain_class = classify_regime_gain(
            candidate=candidate,
            baseline=core,
            comparison=vs_core,
        )

        sensitivity_candidate = cell_summaries[(SENSITIVITY_PROTOCOL, "adaptive_family_ecology_v2", regime)]
        sensitivity_core = cell_summaries[(SENSITIVITY_PROTOCOL, "core_only", regime)]
        sensitivity_best_fixed = cell_summaries[(SENSITIVITY_PROTOCOL, best_fixed_profile, regime)]
        sensitivity_vs_core = _compare_cells(
            candidate=sensitivity_candidate,
            baseline=sensitivity_core,
            bootstrap_samples=bootstrap_samples,
            seed=20260621 + REGIME_ORDER.index(regime),
        )
        sensitivity_vs_best_fixed = _compare_cells(
            candidate=sensitivity_candidate,
            baseline=sensitivity_best_fixed,
            bootstrap_samples=bootstrap_samples,
            seed=20260721 + REGIME_ORDER.index(regime),
        )
        sensitivity_gain_class = classify_regime_gain(
            candidate=sensitivity_candidate,
            baseline=sensitivity_core,
            comparison=sensitivity_vs_core,
        )

        primary_regime_records.append(
            {
                "regime": regime,
                "regime_gain_class": regime_gain_class,
                "best_profile_by_regime": best_profile,
                "best_fixed_baseline_by_regime": best_fixed_profile,
                "active_optional_families": list(candidate.get("active_optional_families", [])),
                "primary_protocol": {
                    "candidate": _serialize_cell_summary(candidate),
                    "vs_core_only": vs_core,
                    "vs_best_fixed": vs_best_fixed,
                },
                "sensitivity_protocol": {
                    "candidate": _serialize_cell_summary(sensitivity_candidate),
                    "vs_core_only": sensitivity_vs_core,
                    "vs_best_fixed": sensitivity_vs_best_fixed,
                    "regime_gain_class": sensitivity_gain_class,
                },
            }
        )

    strong_regimes = [
        row["regime"] for row in primary_regime_records if row["regime_gain_class"] == "ganancia cognitiva fuerte"
    ]
    conditioned_regimes = [
        row["regime"] for row in primary_regime_records if row["regime_gain_class"] == "ganancia cognitiva condicionada"
    ]
    closure_regressions = [
        row["regime"]
        for row in primary_regime_records
        if not bool(row["primary_protocol"]["candidate"].get("closure_stable"))
    ]
    if len(strong_regimes) >= 3 and not closure_regressions:
        primary_verdict = "sí hay ganancia cognitiva"
    elif strong_regimes or conditioned_regimes:
        primary_verdict = "hay ganancia cognitiva condicionada"
    else:
        primary_verdict = "no hay ganancia cognitiva suficiente"

    family_evidence = _optional_family_evidence(primary_regime_records)
    should_run_prompt_2 = (
        primary_verdict != "no hay ganancia cognitiva suficiente"
        and (
            primary_verdict == "hay ganancia cognitiva condicionada"
            or len({row["regime_gain_class"] for row in primary_regime_records}) > 1
        )
    )

    regime_payload = {
        "campaign_id": campaign_id,
        "generated_at": datetime.now().isoformat(),
        "primary_protocol": PRIMARY_PROTOCOL,
        "sensitivity_protocol": SENSITIVITY_PROTOCOL,
        "blocks": blocks,
        "episodes_per_block": episodes_per_block,
        "bootstrap_samples": bootstrap_samples,
        "primary_verdict": primary_verdict,
        "strong_regimes": strong_regimes,
        "conditioned_regimes": conditioned_regimes,
        "closure_regressions": closure_regressions,
        "should_run_family_causal": should_run_prompt_2,
        "family_evidence": family_evidence,
        "regime_records": primary_regime_records,
    }

    root_dir = Path(matrix["root_dir"])
    regime_json_path = root_dir / "regime_cognitive_verdicts.json"
    _write_json(regime_json_path, regime_payload)

    report_lines = [
        "# Ganancia Cognitiva v2",
        "",
        "## 1. Dictamen primario",
        "",
        f"`{primary_verdict}`",
        "",
        "## 2. Evidencia por régimen",
        "",
        "| Régimen | Clase | Mejor perfil | Mejor baseline fijo | ΔIVC-R vs core | ΔPrecisión | ΔViability | ΔIoC proxy | success_rate | cierre estable | opcionales activas |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in primary_regime_records:
        comp = row["primary_protocol"]["vs_core_only"]
        candidate = row["primary_protocol"]["candidate"]
        report_lines.append(
            "| {regime} | {klass} | {best_profile} | {best_fixed} | {delta_ivc_r:.4f} | "
            "{delta_intervention_precision:.4f} | {delta_viability_margin:.4f} | {delta_ioc_proxy_gain:.4f} | "
            "{success_rate:.4f} | {closure_stable} | {families} |".format(
                regime=row["regime"],
                klass=row["regime_gain_class"],
                best_profile=row["best_profile_by_regime"],
                best_fixed=row["best_fixed_baseline_by_regime"],
                delta_ivc_r=_as_float(comp.get("delta_ivc_r"), 0.0),
                delta_intervention_precision=_as_float(comp.get("delta_intervention_precision"), 0.0),
                delta_viability_margin=_as_float(comp.get("delta_viability_margin"), 0.0),
                delta_ioc_proxy_gain=_as_float(comp.get("delta_ioc_proxy_gain"), 0.0),
                success_rate=_as_float(candidate.get("success_rate"), 0.0),
                closure_stable="sí" if candidate.get("closure_stable") else "no",
                families=", ".join(row["active_optional_families"]) or "-",
            )
        )
    report_lines.extend(
        [
            "",
            "## 3. Evidencia por perfil",
            "",
            "La lectura principal se hace contra `core_only`; la comparación contra el mejor baseline fijo por régimen se deja en `regime_cognitive_verdicts.json` para aislar si el valor viene de adaptividad o de un overlay fijo.",
            "",
            "## 4. Familias que parecen aportar",
            "",
        ]
    )
    for family in OPTIONAL_FAMILY_ORDER:
        payload = family_evidence[family]
        report_lines.append(
            f"- `{family}`: activo en {len(payload['active_regimes'])} regímenes, aparece en {len(payload['positive_regimes'])} regímenes positivos y alinea con el mejor baseline fijo en {len(payload['best_fixed_alignment'])}."
        )
    report_lines.extend(
        [
            "",
            "## 5. Coste y guardas",
            "",
            "Los costos operativos se dejan como contexto. Ningún dictamen se sostiene sin `success_rate`, `closure_break_rate` y `backbone_floor_satisfied_rate` estables.",
            "",
            "## 6. Riesgos residuales",
            "",
            "- La señal primaria depende de bootstrap sobre episodios, no de tests pareados externos.",
            "- `ioc_proxy_gain` es analítico y no sustituye a `ivc_r`.",
            "- La sensibilidad con `reasoning_max_steps=10` se reporta, pero no gobierna el dictamen principal.",
        ]
    )
    report_path = root_dir / "cognitive_gain_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    return {
        "campaign_id": campaign_id,
        "root_dir": str(root_dir),
        "block_metrics_path": matrix["block_metrics_path"],
        "regime_cognitive_verdicts_path": str(regime_json_path),
        "cognitive_gain_report_path": str(report_path),
        "primary_verdict": primary_verdict,
        "should_run_family_causal": should_run_prompt_2,
    }


def compute_family_synergy_delta(
    *,
    single_deltas: Sequence[float],
    combo_delta: float,
) -> float:
    if not single_deltas:
        return combo_delta
    return combo_delta - max(single_deltas)


def family_redundancy_flag(
    *,
    single_delta: float,
    combo_delta: float,
    synergy_delta: float,
    epsilon: float = 0.005,
) -> bool:
    return combo_delta <= (single_delta + epsilon) and synergy_delta <= epsilon


def _profile_family_set(profile_name: str) -> List[str]:
    return _profile_optional_families(profile_name)


def classify_family_role(
    *,
    single_comparison: Mapping[str, Any],
    single_closure_stable: bool,
    adaptive_active: bool,
    adaptive_positive: bool,
    best_combo_comparison: Mapping[str, Any] | None,
) -> str:
    single_delta_ivc = _as_float(single_comparison.get("delta_ivc_r"), 0.0)
    single_secondary = bool(single_comparison.get("secondary_positive_repeatable_metrics"))
    if single_closure_stable and single_delta_ivc > 0.0 and single_secondary:
        return "aporta"
    if single_delta_ivc < -0.005 and not adaptive_positive:
        return "perjudica"
    if adaptive_active and adaptive_positive:
        combo_delta = _as_float((best_combo_comparison or {}).get("delta_ivc_r"), 0.0)
        if combo_delta > 0.0 or single_delta_ivc > 0.0:
            return "aporta condicionado"
    if single_delta_ivc < -0.005:
        return "perjudica"
    return "neutral"


def run_family_causal_gain_campaign(
    *,
    campaign_id: str | None = None,
    output_root: str | Path = "data/benchmarks/family_causal_gain",
    db_path: str | Path = "aeon_event_log.db",
    artifact_root: str | Path = "data/artifacts",
    blocks: int = 8,
    episodes_per_block: int = 8,
    seed_base: int = 960000,
    bootstrap_samples: int = 1000,
) -> Dict[str, Any]:
    campaign_id = campaign_id or f"family_causal_gain_{_now_stamp()}"
    storage_config = build_storage_config(db_path=db_path, artifact_root=artifact_root)
    target_regimes = REGIME_ORDER[1:]
    matrix = run_block_matrix(
        campaign_id=campaign_id,
        output_root=output_root,
        storage_config=storage_config,
        profiles=CAUSAL_PROFILES,
        regimes=target_regimes,
        blocks=blocks,
        episodes_per_block=episodes_per_block,
        seed_base=seed_base,
        protocols=PROTOCOL_SPECS,
    )
    cell_summaries = matrix["cell_summaries"]

    regime_matrix: Dict[str, Dict[str, Any]] = {}
    family_overall: Dict[str, Dict[str, Any]] = {family: {"per_regime": {}} for family in OPTIONAL_FAMILY_ORDER}

    for regime in target_regimes:
        core = cell_summaries[(PRIMARY_PROTOCOL, "core_only", regime)]
        adaptive = cell_summaries[(PRIMARY_PROTOCOL, "adaptive_family_ecology_v2", regime)]
        adaptive_vs_core = _compare_cells(
            candidate=adaptive,
            baseline=core,
            bootstrap_samples=bootstrap_samples,
            seed=20270421 + target_regimes.index(regime),
        )
        adaptive_positive = classify_regime_gain(
            candidate=adaptive,
            baseline=core,
            comparison=adaptive_vs_core,
        ) in {"ganancia cognitiva fuerte", "ganancia cognitiva condicionada", "ganancia marginal"}
        regime_matrix[regime] = {
            "adaptive_v2": {
                "cell": _serialize_cell_summary(adaptive),
                "vs_core_only": adaptive_vs_core,
            },
            "families": {},
        }

        profile_comparisons: Dict[str, Dict[str, Any]] = {}
        for profile in CAUSAL_PROFILES:
            if profile == "core_only":
                continue
            profile_cell = cell_summaries[(PRIMARY_PROTOCOL, profile, regime)]
            profile_comparisons[profile] = _compare_cells(
                candidate=profile_cell,
                baseline=core,
                bootstrap_samples=bootstrap_samples,
                seed=20271421 + len(profile_comparisons) + target_regimes.index(regime),
            )

        for family in OPTIONAL_FAMILY_ORDER:
            single_profile = FAMILY_TO_SINGLE_PROFILE[family]
            single_cell = cell_summaries[(PRIMARY_PROTOCOL, single_profile, regime)]
            single_comp = profile_comparisons[single_profile]
            combo_profiles = [
                profile
                for profile in CAUSAL_PROFILES
                if profile not in {"core_only", single_profile}
                and family in _profile_family_set(profile)
            ]
            combo_candidates = []
            for profile in combo_profiles:
                cell = cell_summaries[(PRIMARY_PROTOCOL, profile, regime)]
                comparison = profile_comparisons[profile]
                combo_candidates.append(
                    (
                        _as_float(comparison.get("delta_ioc_proxy_gain"), 0.0),
                        profile,
                        cell,
                        comparison,
                    )
                )
            combo_candidates.sort(reverse=True)
            best_combo_profile = combo_candidates[0][1] if combo_candidates else None
            best_combo_cell = combo_candidates[0][2] if combo_candidates else None
            best_combo_comp = combo_candidates[0][3] if combo_candidates else None

            synergy_values: List[float] = []
            for profile in combo_profiles:
                comparison = profile_comparisons[profile]
                family_members = _profile_family_set(profile)
                single_deltas = [
                    _as_float(profile_comparisons[FAMILY_TO_SINGLE_PROFILE[member]].get("delta_ioc_proxy_gain"), 0.0)
                    for member in family_members
                    if member in FAMILY_TO_SINGLE_PROFILE
                ]
                synergy_values.append(
                    compute_family_synergy_delta(
                        single_deltas=single_deltas,
                        combo_delta=_as_float(comparison.get("delta_ioc_proxy_gain"), 0.0),
                    )
                )
            best_synergy = max(synergy_values) if synergy_values else 0.0
            redundancy = family_redundancy_flag(
                single_delta=_as_float(single_comp.get("delta_ioc_proxy_gain"), 0.0),
                combo_delta=_as_float((best_combo_comp or {}).get("delta_ioc_proxy_gain"), 0.0),
                synergy_delta=best_synergy,
            )
            adaptive_active = family in list(adaptive.get("active_optional_families", []))
            family_class = classify_family_role(
                single_comparison=single_comp,
                single_closure_stable=bool(single_cell.get("closure_stable")),
                adaptive_active=adaptive_active,
                adaptive_positive=adaptive_positive,
                best_combo_comparison=best_combo_comp,
            )

            family_record = {
                "dictamen": family_class,
                "single_profile": single_profile,
                "single_profile_label": PROFILE_TO_LABEL[single_profile],
                "single_profile_closure_stable": bool(single_cell.get("closure_stable")),
                "single_profile_vs_core": single_comp,
                "best_combo_profile": best_combo_profile,
                "best_combo_profile_label": PROFILE_TO_LABEL.get(best_combo_profile or "", best_combo_profile),
                "best_combo_vs_core": best_combo_comp or {},
                "adaptive_active": adaptive_active,
                "adaptive_v2_positive": adaptive_positive,
                "family_synergy_delta": best_synergy,
                "family_redundancy_flag": redundancy,
            }
            regime_matrix[regime]["families"][family] = family_record
            family_overall[family]["per_regime"][regime] = family_record

    recommendations: Dict[str, str] = {}
    overall_dictamen: Dict[str, str] = {}
    for family in OPTIONAL_FAMILY_ORDER:
        regime_dicta = Counter(
            str(payload.get("dictamen") or "")
            for payload in family_overall[family]["per_regime"].values()
        )
        if regime_dicta.get("aporta", 0) >= 2 and regime_dicta.get("perjudica", 0) == 0:
            overall = "aporta"
            recommendation = "mantener"
        elif regime_dicta.get("perjudica", 0) >= 2 and regime_dicta.get("aporta", 0) == 0:
            overall = "perjudica"
            recommendation = "sacar de adaptive"
        elif regime_dicta.get("aporta", 0) + regime_dicta.get("aporta condicionado", 0) > 0:
            overall = "aporta condicionado"
            recommendation = "restringir por régimen"
        else:
            overall = "neutral"
            recommendation = "dejar solo como experimental"
        overall_dictamen[family] = overall
        recommendations[family] = recommendation

    payload = {
        "campaign_id": campaign_id,
        "generated_at": datetime.now().isoformat(),
        "primary_protocol": PRIMARY_PROTOCOL,
        "sensitivity_protocol": SENSITIVITY_PROTOCOL,
        "blocks": blocks,
        "episodes_per_block": episodes_per_block,
        "bootstrap_samples": bootstrap_samples,
        "family_overall_dictamen": overall_dictamen,
        "family_recommendations": recommendations,
        "regime_matrix": regime_matrix,
    }

    root_dir = Path(matrix["root_dir"])
    matrix_path = root_dir / "family_regime_matrix.json"
    _write_json(matrix_path, payload)

    lines = [
        "# Diagnóstico Causal por Familia",
        "",
        "## 1. Dictamen por familia",
        "",
    ]
    for family in OPTIONAL_FAMILY_ORDER:
        lines.append(
            f"- `{family}`: `{overall_dictamen[family]}` -> recomendación `{recommendations[family]}`."
        )
    lines.extend(
        [
            "",
            "## 2. Evidencia por régimen",
            "",
            "| Régimen | Familia | Dictamen | ΔIVC-R single | ΔPrecisión single | ΔViability single | cierre single | mejor combo | synergy | redundancia | activa en v2 |",
            "|---|---|---|---:|---:|---:|---|---|---:|---|---|",
        ]
    )
    for regime in target_regimes:
        for family in OPTIONAL_FAMILY_ORDER:
            record = regime_matrix[regime]["families"][family]
            single = record["single_profile_vs_core"]
            lines.append(
                "| {regime} | {family} | {dictamen} | {delta_ivc_r:.4f} | {delta_intervention_precision:.4f} | "
                "{delta_viability_margin:.4f} | {closure} | {combo} | {synergy:.4f} | {redundancy} | {active} |".format(
                    regime=regime,
                    family=family,
                    dictamen=record["dictamen"],
                    delta_ivc_r=_as_float(single.get("delta_ivc_r"), 0.0),
                    delta_intervention_precision=_as_float(single.get("delta_intervention_precision"), 0.0),
                    delta_viability_margin=_as_float(single.get("delta_viability_margin"), 0.0),
                    closure="sí" if record["single_profile_closure_stable"] else "no",
                    combo=record["best_combo_profile"] or "-",
                    synergy=_as_float(record.get("family_synergy_delta"), 0.0),
                    redundancy="sí" if record.get("family_redundancy_flag") else "no",
                    active="sí" if record.get("adaptive_active") else "no",
                )
            )
    lines.extend(
        [
            "",
            "## 3. Señal cognitiva",
            "",
            "La matriz reporta impacto en `ivc_r`, `intervention_precision`, `viability_margin` y robustez de cierre para cada familia y régimen.",
            "",
            "## 4. Recomendación operativa",
            "",
        ]
    )
    for family in OPTIONAL_FAMILY_ORDER:
        lines.append(f"- `{family}` -> `{recommendations[family]}`.")
    report_path = root_dir / "family_regime_matrix.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    return {
        "campaign_id": campaign_id,
        "root_dir": str(root_dir),
        "block_metrics_path": matrix["block_metrics_path"],
        "family_regime_matrix_path": str(matrix_path),
        "family_regime_matrix_md_path": str(report_path),
        "family_overall_dictamen": overall_dictamen,
    }


def render_intelligence_verdict(
    *,
    cognitive_verdicts_path: str | Path,
    output_root: str | Path = "data/reports/intelligence_verdict",
    campaign_id: str | None = None,
    family_causal_path: str | Path | None = None,
) -> Dict[str, Any]:
    cognitive_payload = json.loads(Path(cognitive_verdicts_path).read_text(encoding="utf-8"))
    family_payload = (
        json.loads(Path(family_causal_path).read_text(encoding="utf-8"))
        if family_causal_path
        else None
    )

    primary_verdict = str(cognitive_payload.get("primary_verdict") or "")
    family_overall = (family_payload or {}).get("family_overall_dictamen", {})
    family_positive = [
        family for family, verdict in family_overall.items() if verdict in {"aporta", "aporta condicionado"}
    ]
    closure_regressions = list(cognitive_payload.get("closure_regressions") or [])

    if primary_verdict == "sí hay ganancia cognitiva" and family_positive:
        final_verdict = "hubo progreso real en inteligencia"
    elif primary_verdict == "hay ganancia cognitiva condicionada" and family_positive:
        final_verdict = "hubo progreso parcial"
    elif primary_verdict == "no hay ganancia cognitiva suficiente" and not closure_regressions:
        final_verdict = "hubo mejora estructural pero no cognitiva"
    else:
        final_verdict = "no hubo progreso cognitivo suficiente"

    if final_verdict == "hubo progreso real en inteligencia":
        what_really_won = ["discriminación por régimen", "cierre estable", "capacidad composicional útil"]
        next_priority = "comprimir la política adaptativa con evidencia causal por familia para reducir complejidad sin perder ganancia."
    elif final_verdict == "hubo progreso parcial":
        what_really_won = ["capacidad de régimen", "mejor cierre adaptativo"]
        next_priority = "restringir overlays por régimen con la matriz causal para convertir ganancia condicionada en ganancia fuerte."
    elif final_verdict == "hubo mejora estructural pero no cognitiva":
        what_really_won = ["estructura", "cierre", "disciplina composicional"]
        next_priority = "podar familias neutrales y volver a medir antes de añadir más complejidad."
    else:
        what_really_won = ["nada sustantivo"]
        next_priority = "volver a un baseline más simple y buscar una fuente cognitiva nueva, no más instrumentación."

    campaign_id = campaign_id or f"intelligence_verdict_{_now_stamp()}"
    root_dir = Path(output_root) / campaign_id
    root_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "campaign_id": campaign_id,
        "generated_at": datetime.now().isoformat(),
        "dictamen_principal": final_verdict,
        "justificacion_breve": {
            "primary_verdict": primary_verdict,
            "strong_regimes": cognitive_payload.get("strong_regimes", []),
            "conditioned_regimes": cognitive_payload.get("conditioned_regimes", []),
            "closure_regressions": closure_regressions,
            "family_positive": family_positive,
        },
        "what_really_won": what_really_won,
        "next_priority": next_priority,
        "sources": {
            "cognitive_verdicts_path": str(Path(cognitive_verdicts_path)),
            "family_causal_path": str(Path(family_causal_path)) if family_causal_path else None,
        },
    }
    json_path = root_dir / "intelligence_verdict.json"
    _write_json(json_path, payload)

    lines = [
        "# Dictamen de Inteligencia",
        "",
        "## A. Dictamen principal",
        "",
        f"`{final_verdict}`",
        "",
        "## B. Justificación breve",
        "",
        f"- Dictamen cognitivo primario: `{primary_verdict}`.",
        f"- Regímenes fuertes: {', '.join(cognitive_payload.get('strong_regimes', [])) or '-'}",
        f"- Regímenes condicionados: {', '.join(cognitive_payload.get('conditioned_regimes', [])) or '-'}",
        f"- Regresiones de cierre: {', '.join(closure_regressions) or '-'}",
        f"- Familias con señal positiva: {', '.join(family_positive) or '-'}",
        "",
        "## C. Qué fue lo que realmente ganó",
        "",
    ]
    for item in what_really_won:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## D. Qué sigue",
            "",
            f"- {next_priority}",
        ]
    )
    md_path = root_dir / "intelligence_verdict.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")

    return {
        "campaign_id": campaign_id,
        "intelligence_verdict_path": str(json_path),
        "intelligence_verdict_md_path": str(md_path),
        "dictamen_principal": final_verdict,
    }


def run_adaptive_v2_intelligence_campaign(
    *,
    campaign_id: str | None = None,
    cognitive_output_root: str | Path = "data/benchmarks/cognitive_gain",
    family_output_root: str | Path = "data/benchmarks/family_causal_gain",
    verdict_output_root: str | Path = "data/reports/intelligence_verdict",
    db_path: str | Path = "aeon_event_log.db",
    artifact_root: str | Path = "data/artifacts",
    blocks: int = 8,
    episodes_per_block: int = 8,
    bootstrap_samples: int = 1000,
    seed_base: int = 910000,
) -> Dict[str, Any]:
    campaign_id = campaign_id or f"adaptive_v2_intelligence_{_now_stamp()}"

    cognitive_result = run_cognitive_gain_campaign(
        campaign_id=f"{campaign_id}_prompt1",
        output_root=cognitive_output_root,
        db_path=db_path,
        artifact_root=artifact_root,
        blocks=blocks,
        episodes_per_block=episodes_per_block,
        seed_base=seed_base,
        bootstrap_samples=bootstrap_samples,
    )

    family_result = None
    if cognitive_result["should_run_family_causal"]:
        family_result = run_family_causal_gain_campaign(
            campaign_id=f"{campaign_id}_prompt2",
            output_root=family_output_root,
            db_path=db_path,
            artifact_root=artifact_root,
            blocks=blocks,
            episodes_per_block=episodes_per_block,
            seed_base=seed_base + 500_000,
            bootstrap_samples=bootstrap_samples,
        )

    verdict_result = render_intelligence_verdict(
        cognitive_verdicts_path=cognitive_result["regime_cognitive_verdicts_path"],
        output_root=verdict_output_root,
        campaign_id=f"{campaign_id}_prompt3",
        family_causal_path=(family_result or {}).get("family_regime_matrix_path"),
    )

    manifest = {
        "campaign_id": campaign_id,
        "generated_at": datetime.now().isoformat(),
        "prompt_1": cognitive_result,
        "prompt_2": family_result,
        "prompt_3": verdict_result,
    }
    manifest_path = Path(verdict_output_root) / f"{campaign_id}_manifest.json"
    _write_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    return manifest
