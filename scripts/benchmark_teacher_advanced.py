"""Campaña pareada avanzada: sin docente vs 7B local vs docente Codex.

Cada trial ejecuta primero un episodio base real y después repite la misma situación
con el mismo organismo y storage. Las ramas se aíslan por escenario, seed y variante.
La campaña mide semántica docente, latencia, cambio de intervención y severidad; no
promueve currículo ni entrenamiento.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime.organism.experience import ExperienceStore
from runtime.organism.teacher import Teacher
from runtime.storage import StorageConfig, StorageFactory
from runtime.world.registry import get_scenario
from runtime.world.scenario_runner import ScenarioEpisodeRunner


SCENARIOS: dict[str, dict[str, Any]] = {
    "thermal_homeostasis": {
        "codex_prefer": "activate_cooling",
        "codex_lesson": "Activa enfriamiento ante calor externo y verifica que la temperatura descienda.",
    },
    "resource_management": {
        "codex_prefer": "start_production",
        "codex_lesson": "Inicia producción antes de cruzar escasez y mide la recuperación del stock.",
    },
    "deferred_load_trap": {
        "codex_prefer": "shed_load",
        "codex_lesson": "Evita alivio inmediato con deuda; reduce carga de forma sostenible y mide el rebote.",
    },
}
PERTURBATION_PROFILES: dict[str, dict[str, tuple[tuple[str, float], ...]]] = {
    "pilot": {
        "thermal_homeostasis": (("pilot", 0.08),),
        "resource_management": (("pilot", 0.10),),
        "deferred_load_trap": (("pilot", 0.14),),
    },
    "heldout_v1": {
        "thermal_homeostasis": (("heldout-low", 0.05), ("heldout-high", 0.13)),
        "resource_management": (("heldout-low", 0.06), ("heldout-high", 0.16)),
        "deferred_load_trap": (("heldout-low", 0.09), ("heldout-high", 0.20)),
    },
}
VARIANTS = ("no_teacher", "local_7b", "codex_frontier")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_evidence_manifest(campaign_dir: Path) -> Path:
    """Hash de los cinco artefactos soberanos; excluye DBs y temporales."""
    names = ["manifest.json", "trials.json", "summary.json", "verdict.json", "REPORT.md"]
    if (campaign_dir / "stratified_reanalysis.json").exists():
        names.append("stratified_reanalysis.json")
    artifacts = {}
    for name in names:
        path = campaign_dir / name
        artifacts[name] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
        }
    target = campaign_dir / "evidence_manifest.json"
    _write_json(
        target,
        {
            "schema_version": "rnfe-teacher-evidence-manifest-v1",
            "campaign_id": campaign_dir.name,
            "artifacts": artifacts,
        },
    )
    return target


def reconcile_stratified_evidence(campaign_dir: Path) -> Path:
    """Hace soberana una reanálisis estratificada sin reejecutar inferencias.

    La función sólo acepta artefactos de la misma campaña, conserva los trials y
    vuelve a generar summary, verdict, reporte y hashes como una unidad atómica
    de evidencia. Nunca puede autorizar entrenamiento ni promoción.
    """

    summary_path = campaign_dir / "summary.json"
    reanalysis_path = campaign_dir / "stratified_reanalysis.json"
    if not summary_path.is_file() or not reanalysis_path.is_file():
        raise FileNotFoundError("teacher_stratified_reconciliation_inputs_missing")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    reanalysis = json.loads(reanalysis_path.read_text(encoding="utf-8"))
    campaign_id = campaign_dir.name
    if (
        reanalysis.get("schema_version")
        != "rnfe-teacher-stratified-reanalysis-v1"
        or reanalysis.get("campaign_id") != campaign_id
    ):
        raise ValueError("teacher_stratified_reanalysis_identity_mismatch")
    if reanalysis.get("training_authorized") is not False:
        raise ValueError("teacher_stratified_reanalysis_training_must_be_false")
    if reanalysis.get("promotion_authorized") is not False:
        raise ValueError("teacher_stratified_reanalysis_promotion_must_be_false")

    comparisons = dict(summary.get("comparisons") or {})
    comparisons.update(dict(reanalysis.get("comparisons") or {}))
    summary = {
        **summary,
        "comparisons": comparisons,
        "scenario_comparisons": dict(reanalysis.get("scenario_comparisons") or {}),
        "stratum_comparisons": dict(reanalysis.get("stratum_comparisons") or {}),
        "stratified_reanalysis": {
            "schema_version": reanalysis["schema_version"],
            "reason": reanalysis.get("reason"),
            "supersedes_aggregate_candidate_claim": bool(
                reanalysis.get("supersedes_aggregate_candidate_claim", False)
            ),
        },
    }
    cross_scenario = bool(comparisons.get("codex_cross_scenario_gate_passed", False))
    verdict = {
        "schema_version": "rnfe-teacher-campaign-verdict-v1",
        "campaign_id": campaign_id,
        "verdict": str(reanalysis.get("verdict") or "retain_local_7b_as_supervised_student"),
        "codex_teacher_candidate": False,
        "codex_cross_scenario_gate_passed": cross_scenario,
        "curriculum_promotion_authorized": False,
        "training_authorized": False,
        "reason": str(reanalysis.get("reason") or "stratified_tradeoff_detected"),
        "evidence_basis": "stratified_reanalysis.json",
    }
    _write_json(summary_path, summary)
    _write_json(campaign_dir / "verdict.json", verdict)
    (campaign_dir / "REPORT.md").write_text(
        _report(campaign_id, summary, verdict), encoding="utf-8"
    )
    return build_evidence_manifest(campaign_dir)


def _experience(result: Mapping[str, Any]) -> dict[str, Any]:
    direct = result.get("experience")
    if isinstance(direct, Mapping):
        return dict(direct)
    episode = result.get("episode")
    if isinstance(episode, Mapping) and isinstance(episode.get("experience"), Mapping):
        return dict(episode["experience"])
    return {}


def _intervention(result: Mapping[str, Any]) -> str:
    return str(result.get("episode", {}).get("context", {}).get("intervention") or "")


def _episode_id(result: Mapping[str, Any]) -> str:
    episode = result.get("episode", {})
    return str(episode.get("episode_id") or episode.get("id") or "")


def _reward(result: Mapping[str, Any]) -> float | None:
    payload = result.get("reasoning_reward")
    value = payload.get("reward") if isinstance(payload, Mapping) else None
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _extension_class(result: Mapping[str, Any], role: str, field: str) -> Any:
    bundle = result.get("neural_symbiosis_trace", {}).get("neural_agent_extensions", {})
    for report in bundle.get("reports", ()) if isinstance(bundle, Mapping) else ():
        if report.get("role") == role:
            return report.get("outputs", {}).get("stages", {}).get("classify", {}).get(field)
    return None


def _configure_native_runtime(seed: int, *, max_tokens: int, timeout_s: float, temperature: float) -> None:
    root = "/home/wis/rnfe_models"
    os.environ.update(
        {
            "RNFE_EXPERIENCE": "1",
            "RNFE_TEACHER": "1",
            "RNFE_NEURAL_MODE": "shadow",
            "RNFE_REASONING_GGUF": f"{root}/gguf/OpenThinker3-7B/OpenThinker3-7B-Q4_K_M.gguf",
            "RNFE_LLAMA_CLI_CUDA": f"{root}/tools/llama.cpp-src/build-cuda/bin/llama-cli",
            "RNFE_EXTERNAL_REASONER_BACKEND": "cuda",
            "RNFE_EXTERNAL_REASONER_NGL": "99",
            "RNFE_EXTERNAL_REASONER_MAX_TOKENS": str(max_tokens),
            "RNFE_EXTERNAL_REASONER_TIMEOUT_S": str(timeout_s),
            "RNFE_EXTERNAL_REASONER_TEMPERATURE": str(temperature),
            "RNFE_EXTERNAL_REASONER_SEED": str(seed),
            "RNFE_EXTERNAL_REASONER_REASONING_BUDGET": "0",
        }
    )


def _storage(branch_dir: Path, storage_config: StorageConfig | None = None):
    branch_dir.mkdir(parents=True, exist_ok=True)
    if storage_config is not None:
        if storage_config.mode != "postgres":
            raise ValueError("teacher_campaign_official_storage_must_be_postgres")
        return StorageFactory.create_facade(storage_config)
    return StorageFactory.create_facade(
        StorageConfig(
            mode="sqlite",
            sqlite_db_path=str(branch_dir / "campaign.db"),
            postgres_dsn=None,
            artifact_root=branch_dir / "artifacts",
            prefer_postgres_reads=False,
            strict_dual_write=False,
        )
    )


def _semantic_metrics(lesson: Mapping[str, Any] | None, *, baseline: str, interventions: Iterable[str]) -> dict[str, Any]:
    if not lesson:
        return {"available": False, "semantic_pass": None}
    allowed = set(interventions)
    avoid = str(lesson.get("avoid") or "")
    prefer = str(lesson.get("prefer") or "")
    text = str(lesson.get("lesson") or "")
    checks = {
        "avoid_matches_observed_action": avoid == baseline,
        "prefer_in_catalog": prefer in allowed and prefer != avoid,
        "lesson_informative": len(text.split()) >= 6,
    }
    return {
        "available": True,
        **checks,
        "semantic_pass": all(checks.values()),
        "raw_semantic_valid": lesson.get("teacher_raw_semantic_valid"),
        "repairs": list(lesson.get("teacher_repairs") or ()),
        "latency_s": lesson.get("teacher_latency_s"),
        "prompt_tps": lesson.get("teacher_prompt_tps"),
        "generation_tps": lesson.get("teacher_generation_tps"),
    }


def _run_trial(*, campaign_id: str, campaign_dir: Path, scenario_name: str, perturbation_id: str, external_input: float, seed: int, variant: str, max_tokens: int, timeout_s: float, temperature: float, horizon: int, storage_config: StorageConfig | None = None) -> dict[str, Any]:
    _configure_native_runtime(seed, max_tokens=max_tokens, timeout_s=timeout_s, temperature=temperature)
    spec = SCENARIOS[scenario_name]
    pair_id = f"{scenario_name}-{perturbation_id}-seed-{seed}"
    branch_dir = campaign_dir / "work" / pair_id / variant
    storage = _storage(branch_dir, storage_config)
    organism_id = "organism-" + hashlib.sha256(
        f"{campaign_id}:{pair_id}:{variant}".encode("utf-8")
    ).hexdigest()[:20]
    try:
        baseline_runner = ScenarioEpisodeRunner(
            storage=storage,
            run_id=f"{campaign_id}-{pair_id}-{variant}-baseline",
            scenario=scenario_name,
        )
        baseline_runner.set_organism_id(organism_id)
        baseline = baseline_runner.run_episode(external_input=float(external_input))
        baseline_exp = _experience(baseline)
        baseline_iv = _intervention(baseline)
        scenario = get_scenario(scenario_name)
        interventions = list(scenario.config.interventions)
        lesson: dict[str, Any] | None = None
        teacher = Teacher(storage=storage, experience=ExperienceStore(storage=storage))
        if variant == "local_7b":
            lesson = teacher._reflect_one(
                organism_id=organism_id,
                wound={
                    "episode_id": _episode_id(baseline),
                    "situation_key": baseline_exp.get("situation_key"),
                    "scenario": scenario_name,
                    "regime": "calm",
                    "intervention": baseline_iv,
                    "severity": baseline_exp.get("severity", 0.0),
                    "viability_margin": baseline.get("viability_assessment", {}).get("viability_margin", 0.0),
                    "ioc": 0.0,
                    "risk": 0.0,
                },
                valid_interventions=interventions,
            )
        elif variant == "codex_frontier":
            prefer = str(spec["codex_prefer"])
            if prefer == baseline_iv or prefer not in interventions:
                prefer = next(item for item in interventions if item != baseline_iv)
            lesson = teacher.register_external_lesson(
                teacher_source="codex_frontier",
                lesson={
                    "organism_id": organism_id,
                    "situation_key": baseline_exp.get("situation_key"),
                    "scenario": scenario_name,
                    "regime": "calm",
                    "avoid": baseline_iv,
                    "prefer": prefer,
                    "lesson": spec["codex_lesson"],
                    "from_severity": baseline_exp.get("severity", 0.0),
                    "source_wound_episode_id": _episode_id(baseline),
                    "teacher_raw_semantic_valid": True,
                    "teacher_generation_mode": "offline_curated_by_codex",
                },
            )
        if lesson is not None:
            lesson["evaluation_pair_id"] = pair_id
            lesson["evaluation_variant"] = variant

        evaluation_runner = ScenarioEpisodeRunner(
            storage=storage,
            run_id=f"{campaign_id}-{pair_id}-{variant}-evaluation",
            scenario=scenario_name,
        )
        evaluation_runner.set_organism_id(organism_id)
        evaluation_runner.set_experience_lessons([lesson] if lesson else [])
        evaluated_episodes = [
            evaluation_runner.run_episode(external_input=float(external_input))
            for _ in range(horizon)
        ]
        evaluated = evaluated_episodes[0]
        final_evaluated = evaluated_episodes[-1]
        evaluated_exp = _experience(evaluated)
        final_exp = _experience(final_evaluated)
        episode_rewards = [_reward(item) for item in evaluated_episodes]
        numeric_rewards = [float(item) for item in episode_rewards if item is not None]
        episode_severities = [float(_experience(item).get("severity", 0.0) or 0.0) for item in evaluated_episodes]
        baseline_severity = float(baseline_exp.get("severity", 0.0) or 0.0)
        evaluated_severity = float(evaluated_exp.get("severity", 0.0) or 0.0)
        return {
            "schema_version": "rnfe-teacher-trial-v1",
            "campaign_id": campaign_id,
            "evaluation_pair_id": pair_id,
            "scenario": scenario_name,
            "perturbation_id": perturbation_id,
            "seed": seed,
            "variant": variant,
            "external_input": external_input,
            "baseline": {
                "episode_id": _episode_id(baseline),
                "intervention": baseline_iv,
                "severity": baseline_severity,
                "reward": _reward(baseline),
            },
            "lesson": lesson,
            "teacher_semantics": _semantic_metrics(lesson, baseline=baseline_iv, interventions=interventions),
            "evaluation": {
                "episode_id": _episode_id(evaluated),
                "intervention": _intervention(evaluated),
                "severity": evaluated_severity,
                "severity_reduction": round(baseline_severity - evaluated_severity, 6),
                "reward": _reward(evaluated),
                "reward_delta": (
                    round(float(_reward(evaluated)) - float(_reward(baseline)), 6)
                    if _reward(evaluated) is not None and _reward(baseline) is not None
                    else None
                ),
                "horizon": horizon,
                "interventions": [_intervention(item) for item in evaluated_episodes],
                "severities": episode_severities,
                "rewards": episode_rewards,
                "cumulative_reward": round(sum(numeric_rewards), 6),
                "mean_severity": round(statistics.fmean(episode_severities), 6),
                "final_severity": float(final_exp.get("severity", 0.0) or 0.0),
                "origin_to_final_severity_reduction": round(
                    baseline_severity - float(final_exp.get("severity", 0.0) or 0.0), 6
                ),
                "behavior_changed": _intervention(evaluated) != baseline_iv,
                "experience_bias": evaluated_exp.get("biased"),
                "pedagogical_class": _extension_class(evaluated, "pedagogical_teacher", "pedagogical_class"),
                "curriculum_class": _extension_class(evaluated, "curriculum_learning", "curriculum_class"),
            },
        }
    finally:
        storage.close()


def _mean(rows: list[dict[str, Any]], path: tuple[str, ...]) -> float | None:
    values = []
    for row in rows:
        value: Any = row
        for key in path:
            value = value.get(key) if isinstance(value, Mapping) else None
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            values.append(float(value))
    return round(statistics.fmean(values), 6) if values else None


def _summary(trials: list[dict[str, Any]]) -> dict[str, Any]:
    variants = {}
    for variant in VARIANTS:
        rows = [row for row in trials if row["variant"] == variant]
        semantics = [row["teacher_semantics"] for row in rows if row["teacher_semantics"]["available"]]
        variants[variant] = {
            "trial_count": len(rows),
            "behavior_change_rate": round(statistics.fmean(float(row["evaluation"]["behavior_changed"]) for row in rows), 6) if rows else None,
            "mean_severity_reduction": _mean(rows, ("evaluation", "severity_reduction")),
            "mean_reward": _mean(rows, ("evaluation", "reward")),
            "mean_reward_delta": _mean(rows, ("evaluation", "reward_delta")),
            "mean_cumulative_reward": _mean(rows, ("evaluation", "cumulative_reward")),
            "mean_episode_severity": _mean(rows, ("evaluation", "mean_severity")),
            "semantic_pass_rate": round(statistics.fmean(float(row["semantic_pass"]) for row in semantics), 6) if semantics else None,
            "raw_semantic_pass_rate": round(statistics.fmean(float(row.get("raw_semantic_valid") is True) for row in semantics), 6) if semantics else None,
            "mean_teacher_latency_s": round(statistics.fmean(float(row["latency_s"]) for row in semantics if isinstance(row.get("latency_s"), (int, float))), 6) if any(isinstance(row.get("latency_s"), (int, float)) for row in semantics) else None,
            "mean_generation_tps": round(statistics.fmean(float(row["generation_tps"]) for row in semantics if isinstance(row.get("generation_tps"), (int, float))), 6) if any(isinstance(row.get("generation_tps"), (int, float)) for row in semantics) else None,
            "unique_lesson_rate": (
                round(
                    len({str(row.get("lesson", {}).get("lesson") or "") for row in rows})
                    / len(rows),
                    6,
                )
                if semantics and rows
                else None
            ),
        }
    local_semantic = variants["local_7b"]["semantic_pass_rate"]
    control_gain = variants["no_teacher"]["mean_severity_reduction"] or 0.0
    codex_gain = variants["codex_frontier"]["mean_severity_reduction"] or 0.0
    control_reward = variants["no_teacher"]["mean_reward_delta"] or 0.0
    codex_reward = variants["codex_frontier"]["mean_reward_delta"] or 0.0
    scenario_comparisons = {}
    stratum_comparisons = {}
    for scenario in sorted({row["scenario"] for row in trials}):
        scenario_rows = [row for row in trials if row["scenario"] == scenario]
        control_rows = [row for row in scenario_rows if row["variant"] == "no_teacher"]
        codex_rows = [row for row in scenario_rows if row["variant"] == "codex_frontier"]
        control_cumulative = _mean(control_rows, ("evaluation", "cumulative_reward")) or 0.0
        codex_cumulative = _mean(codex_rows, ("evaluation", "cumulative_reward")) or 0.0
        control_severity = _mean(control_rows, ("evaluation", "mean_severity")) or 0.0
        codex_severity = _mean(codex_rows, ("evaluation", "mean_severity")) or 0.0
        scenario_comparisons[scenario] = {
            "codex_minus_control_cumulative_reward": round(codex_cumulative - control_cumulative, 6),
            "codex_minus_control_mean_severity": round(codex_severity - control_severity, 6),
        }
        for perturbation in sorted({str(row.get("perturbation_id") or "pilot") for row in scenario_rows}):
            stratum_rows = [
                row for row in scenario_rows
                if str(row.get("perturbation_id") or "pilot") == perturbation
            ]
            stratum_control = [row for row in stratum_rows if row["variant"] == "no_teacher"]
            stratum_codex = [row for row in stratum_rows if row["variant"] == "codex_frontier"]
            key = f"{scenario}:{perturbation}"
            stratum_comparisons[key] = {
                "codex_minus_control_cumulative_reward": round(
                    (_mean(stratum_codex, ("evaluation", "cumulative_reward")) or 0.0)
                    - (_mean(stratum_control, ("evaluation", "cumulative_reward")) or 0.0),
                    6,
                ),
                "codex_minus_control_mean_severity": round(
                    (_mean(stratum_codex, ("evaluation", "mean_severity")) or 0.0)
                    - (_mean(stratum_control, ("evaluation", "mean_severity")) or 0.0),
                    6,
                ),
            }
    cross_scenario_gate = all(
        row["codex_minus_control_cumulative_reward"] > 0.0
        and row["codex_minus_control_mean_severity"] <= 0.0
        for row in stratum_comparisons.values()
    )
    return {
        "schema_version": "rnfe-teacher-campaign-summary-v1",
        "trial_count": len(trials),
        "pair_count": len({row["evaluation_pair_id"] for row in trials}),
        "variants": variants,
        "scenario_comparisons": scenario_comparisons,
        "stratum_comparisons": stratum_comparisons,
        "comparisons": {
            "codex_minus_control_severity_reduction": round(codex_gain - control_gain, 6),
            "codex_minus_control_reward_delta": round(codex_reward - control_reward, 6),
            "local_7b_semantic_gate_passed": bool(local_semantic is not None and local_semantic >= 0.8),
            "codex_cross_scenario_gate_passed": cross_scenario_gate,
        },
    }


def _report(campaign_id: str, summary: Mapping[str, Any], verdict: Mapping[str, Any]) -> str:
    lines = [
        f"# Campaña docente avanzada — {campaign_id}",
        "",
        f"Trials: {summary['trial_count']} · pares escenario/seed: {summary['pair_count']}.",
        "",
        "| Variante | N | cambio conducta | Δ severidad | Δ reward | reward acumulado | severidad media | semántica | sin reparación | diversidad | latencia s | tok/s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant, row in summary["variants"].items():
        lines.append(
            f"| {variant} | {row['trial_count']} | {row['behavior_change_rate']} | "
            f"{row['mean_severity_reduction']} | {row['mean_reward_delta']} | {row['mean_cumulative_reward']} | {row['mean_episode_severity']} | {row['semantic_pass_rate']} | "
            f"{row['raw_semantic_pass_rate']} | {row['unique_lesson_rate']} | {row['mean_teacher_latency_s']} | "
            f"{row['mean_generation_tps']} |"
        )
    lines.extend(
        [
            "",
            "## Consistencia por escenario",
            "",
            "| Escenario | Codex-control reward acumulado | Codex-control severidad media |",
            "|---|---:|---:|",
            *[
                f"| {scenario} | {row['codex_minus_control_cumulative_reward']} | {row['codex_minus_control_mean_severity']} |"
                for scenario, row in summary["scenario_comparisons"].items()
            ],
            "",
            f"Veredicto: **{verdict['verdict']}**.",
            "",
            "No se autoriza promoción curricular ni entrenamiento. El efecto causal sólo se considera candidato hasta repetición held-out y control de la reparación semántica.",
            "",
        ]
    )
    return "\n".join(lines)


def run_campaign(*, campaign_id: str, output_root: Path, scenarios: Iterable[str], seeds: Iterable[int], max_tokens: int, timeout_s: float, temperature: float, horizon: int, profile: str = "pilot", storage_config: StorageConfig | None = None) -> Path:
    campaign_dir = output_root / campaign_id
    campaign_dir.mkdir(parents=True, exist_ok=False)
    started = time.time()
    trials = []
    if profile not in PERTURBATION_PROFILES:
        raise ValueError(f"teacher_campaign_unknown_profile:{profile}")
    for scenario in scenarios:
        if scenario not in SCENARIOS:
            raise ValueError(f"teacher_campaign_unknown_scenario:{scenario}")
        perturbations = PERTURBATION_PROFILES[profile].get(scenario)
        if not perturbations:
            raise ValueError(f"teacher_campaign_profile_missing_scenario:{profile}:{scenario}")
        for perturbation_id, external_input in perturbations:
            for seed in seeds:
                for variant in VARIANTS:
                    trials.append(
                        _run_trial(
                            campaign_id=campaign_id,
                            campaign_dir=campaign_dir,
                            scenario_name=scenario,
                            perturbation_id=perturbation_id,
                            external_input=external_input,
                            seed=int(seed),
                            variant=variant,
                            max_tokens=max_tokens,
                            timeout_s=timeout_s,
                            temperature=temperature,
                            horizon=horizon,
                            storage_config=storage_config,
                        )
                    )
    summary = _summary(trials)
    local_gate = summary["comparisons"]["local_7b_semantic_gate_passed"]
    codex_delta = summary["comparisons"]["codex_minus_control_severity_reduction"]
    codex_cross_scenario = summary["comparisons"]["codex_cross_scenario_gate_passed"]
    verdict_name = (
        "retain_local_7b_as_supervised_student"
        if not local_gate or not codex_cross_scenario
        else "local_7b_teacher_candidate_requires_held_out"
    )
    verdict = {
        "schema_version": "rnfe-teacher-campaign-verdict-v1",
        "campaign_id": campaign_id,
        "verdict": verdict_name,
        "codex_teacher_candidate": codex_delta > 0.0 and codex_cross_scenario,
        "codex_cross_scenario_gate_passed": codex_cross_scenario,
        "curriculum_promotion_authorized": False,
        "training_authorized": False,
        "reason": (
            "stratified_tradeoff_or_semantic_gate_failed"
            if not local_gate or not codex_cross_scenario
            else "paired_evidence_requires_independent_post_training_evaluation"
        ),
    }
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = "unavailable"
    manifest = {
        "schema_version": "rnfe-teacher-campaign-manifest-v1",
        "campaign_id": campaign_id,
        "commit": commit,
        "scenarios": list(scenarios),
        "seeds": list(seeds),
        "variants": list(VARIANTS),
        "max_tokens": max_tokens,
        "timeout_s": timeout_s,
        "temperature": temperature,
        "horizon": horizon,
        "perturbation_profile": profile,
        "elapsed_s": round(time.time() - started, 3),
        "native_models_root": "/home/wis/rnfe_models",
        "experimental": True,
        "storage_mode": storage_config.mode if storage_config is not None else "sqlite",
        "sqlite_official_evidence": False if storage_config is not None else None,
    }
    _write_json(campaign_dir / "manifest.json", manifest)
    _write_json(campaign_dir / "trials.json", {"trials": trials})
    _write_json(campaign_dir / "summary.json", summary)
    _write_json(campaign_dir / "verdict.json", verdict)
    (campaign_dir / "REPORT.md").write_text(
        _report(campaign_id, summary, verdict), encoding="utf-8"
    )
    shutil.rmtree(campaign_dir / "work", ignore_errors=True)
    build_evidence_manifest(campaign_dir)
    return campaign_dir


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--campaign-id")
    target.add_argument("--reconcile-existing", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("data/reports/teacher_advanced"))
    parser.add_argument("--scenarios", default=",".join(SCENARIOS))
    parser.add_argument("--seeds", default="42,101,202")
    parser.add_argument("--max-tokens", type=int, default=160)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--profile", choices=sorted(PERTURBATION_PROFILES), default="pilot")
    args = parser.parse_args(argv)
    if args.reconcile_existing is not None:
        manifest = reconcile_stratified_evidence(args.reconcile_existing)
        print(manifest)
        return 0
    campaign_dir = run_campaign(
        campaign_id=args.campaign_id,
        output_root=args.output_root,
        scenarios=_parse_csv(args.scenarios),
        seeds=[int(item) for item in _parse_csv(args.seeds)],
        max_tokens=args.max_tokens,
        timeout_s=args.timeout_s,
        temperature=args.temperature,
        horizon=max(1, args.horizon),
        profile=args.profile,
    )
    print(campaign_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
