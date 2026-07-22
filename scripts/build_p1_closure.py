#!/usr/bin/env python3
"""Build the deterministic, derived and non-authoritative final P1 closure."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "rnfe-p1-closure-audit-v1"
CAMPAIGN_ID = "neural-p1-final-20260721-06e95c8"
EXPERIMENT_COMMIT = "06e95c8f45c132be87f81f03c19a966674dfb51b"
CLOSURE_FIX_COMMIT = "5f22227a9fee8584be7c740af65b7f6b41a2e47e"
EVIDENCE_HEAD = "060101975a752793c4b5bae6872002ce69c0f8ec"
N3_AUDIT_HEAD = "d2d54a649309e54d9cc434871e8a4e40c6325ad0"
EXPECTED_SEEDS = (
    911001,
    911102,
    911203,
    911304,
    911405,
    911506,
    911607,
    911708,
    911809,
    911910,
    912011,
    912112,
)
EXPECTED_PROFILES = frozenset(
    {
        "off",
        "shadow-none",
        "only-n2",
        "only-n3-reference",
        "only-n3-trained",
        "only-n4-v2",
        "p1-all",
        "p1-without-n2",
        "p1-without-n3",
        "p1-without-n4",
    }
)
NO_N3_PROFILES = frozenset(
    {"off", "shadow-none", "only-n2", "only-n4-v2", "p1-without-n3"}
)
FORBIDDEN_INFERENCES = (
    "AGI",
    "general intelligence",
    "operational autonomy",
    "autopoiesis",
    "effective self-evolution",
    "improved final decision",
    "improved actuation",
    "improved scheduler",
    "top-rank improvement",
    "conventional nDCG",
    "decomposed calibration",
    "trained-reference global superiority",
    "sufficient OOD robustness",
    "live safety",
    "production readiness",
    "P2 authorization",
    "main merge authorization",
)


class ClosureError(RuntimeError):
    """Fail-closed source or semantic validation error."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_sha256s(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not (line := raw.strip()):
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise ClosureError(f"invalid_sha256s_line:{line_number}")
        digest, relative = parts[0].lower(), parts[1].lstrip("*")
        candidate = Path(relative)
        if (
            any(character not in "0123456789abcdef" for character in digest)
            or candidate.is_absolute()
            or ".." in candidate.parts
            or relative in entries
        ):
            raise ClosureError(f"invalid_sha256s_entry:{line_number}:{relative}")
        entries[relative] = digest
    return entries


def validate_hashes(
    *,
    matrix_path: Path,
    audit_v2_path: Path,
    n3_attribution_path: Path,
    sha256s_path: Path,
) -> dict[str, str]:
    entries = read_sha256s(sha256s_path)
    required = {
        matrix_path.name: matrix_path,
        audit_v2_path.name: audit_v2_path,
        n3_attribution_path.name: n3_attribution_path,
    }
    for name, source in required.items():
        expected = entries.get(name)
        if expected is None:
            raise ClosureError(f"required_hash_missing:{name}")
        observed = sha256_file(source)
        if observed != expected:
            raise ClosureError(f"sha256_mismatch:{name}:{expected}:{observed}")
    for relative, expected in entries.items():
        candidate = sha256s_path.parent / relative
        if not candidate.is_file():
            raise ClosureError(f"hashed_file_missing:{relative}")
        observed = sha256_file(candidate)
        if observed != expected:
            raise ClosureError(f"sha256_mismatch:{relative}:{expected}:{observed}")
    return entries


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ClosureError(f"invalid_json:{path.name}:{type(exc).__name__}") from exc
    if not isinstance(payload, dict):
        raise ClosureError(f"json_root_not_object:{path.name}")
    reject_nonfinite(payload)
    return payload


def reject_nonfinite(value: Any, path: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ClosureError(f"nonfinite_value:{path}")
    if isinstance(value, Mapping):
        for key, nested in value.items():
            reject_nonfinite(nested, f"{path}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            reject_nonfinite(nested, f"{path}[{index}]")


def profile_map(matrix: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = matrix.get("profiles")
    if not isinstance(rows, list):
        raise ClosureError("profiles_not_list")
    profiles: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ClosureError("profile_not_object")
        profile_id = str(row.get("profile_id") or "")
        if not profile_id or profile_id in profiles:
            raise ClosureError(f"profile_missing_or_duplicate:{profile_id}")
        profiles[profile_id] = row
    if frozenset(profiles) != EXPECTED_PROFILES:
        raise ClosureError(f"profile_set_invalid:{sorted(profiles)}")
    return profiles


def lane_map(profile: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    profile_id = str(profile.get("profile_id") or "")
    rows = profile.get("lanes")
    if not isinstance(rows, list):
        raise ClosureError(f"lanes_not_list:{profile_id}")
    lanes: dict[int, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or isinstance(row.get("seed"), bool):
            raise ClosureError(f"invalid_lane:{profile_id}")
        try:
            seed = int(row["seed"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ClosureError(f"invalid_seed:{profile_id}") from exc
        if seed in lanes:
            raise ClosureError(f"duplicate_seed:{profile_id}:{seed}")
        lanes[seed] = row
    if tuple(sorted(lanes)) != EXPECTED_SEEDS:
        raise ClosureError(f"seed_set_invalid:{profile_id}:{sorted(lanes)}")
    return lanes


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ClosureError(f"numeric_value_missing:{label}")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ClosureError(f"nonfinite_value:{label}")
    return resolved


def validate_sources(
    matrix: Mapping[str, Any],
    audit_v2: Mapping[str, Any],
    n3: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    if matrix.get("campaign_id") != CAMPAIGN_ID:
        raise ClosureError("matrix_campaign_invalid")
    if audit_v2.get("campaign_id") != CAMPAIGN_ID:
        raise ClosureError("audit_v2_campaign_invalid")
    if (n3.get("campaign_integrity") or {}).get("campaign_id") != CAMPAIGN_ID:
        raise ClosureError("n3_campaign_invalid")
    if matrix.get("commit") != EXPERIMENT_COMMIT:
        raise ClosureError("experiment_commit_invalid")
    if audit_v2.get("audit_commit") != CLOSURE_FIX_COMMIT:
        raise ClosureError("closure_fix_commit_invalid")
    if (n3.get("source") or {}).get("head") != EVIDENCE_HEAD:
        raise ClosureError("evidence_head_invalid")
    if (n3.get("campaign_integrity") or {}).get("audit_v2_commit") != CLOSURE_FIX_COMMIT:
        raise ClosureError("n3_audit_v2_commit_invalid")
    if n3.get("schema_version") != "rnfe-p1-n3-attribution-audit-v1":
        raise ClosureError("n3_schema_invalid")
    if n3.get("verdict") != "n3_attribution_supported_limited":
        raise ClosureError("n3_verdict_invalid")
    if n3.get("p2_authorized") is not False or n3.get("runtime_modified") is not False:
        raise ClosureError("n3_authority_invariant_invalid")

    if matrix.get("steps_per_lane") != 32:
        raise ClosureError("steps_per_lane_invalid")
    if matrix.get("warmup_visits_per_scenario") != 2:
        raise ClosureError("warmup_invalid")
    if tuple(matrix.get("seed_order") or ()) != EXPECTED_SEEDS:
        raise ClosureError("seed_order_invalid")
    integrity = audit_v2.get("integrity") or {}
    parity = integrity.get("canonical_behavior_parity") or {}
    if (
        integrity.get("emitted_episodes") != 3250
        or integrity.get("matched_closures") != 3250
        or integrity.get("certification_rate") != 1.0
        or integrity.get("safety_violations") != 0
        or parity.get("step_comparison_count") != 3456
        or parity.get("mismatch_count") != 0
    ):
        raise ClosureError("campaign_integrity_incomplete")
    if (
        matrix.get("authority_ceiling") != "shadow"
        or (audit_v2.get("gates") or {}).get("canonical_behavior_identical") is not True
        or matrix.get("external_reasoner_enabled") is not False
        or matrix.get("staging_authorized") is not False
        or matrix.get("promotion_authorized") is not False
    ):
        raise ClosureError("authority_isolation_invalid")

    profiles = profile_map(matrix)
    for profile_id, profile in profiles.items():
        if int(profile.get("seed_count", -1)) != 12:
            raise ClosureError(f"profile_seed_count_invalid:{profile_id}")
        lanes = lane_map(profile)
        for seed, lane in lanes.items():
            summary = lane.get("summary") or {}
            if summary.get("total_steps") != 32:
                raise ClosureError(f"lane_steps_invalid:{profile_id}:{seed}")
        if profile_id in NO_N3_PROFILES:
            for lane in lanes.values():
                n3_lane = (lane.get("summary") or {}).get("n3") or {}
                for key in (
                    "mean_ndcg_delta",
                    "mean_mrr_delta",
                    "mean_risk_brier",
                    "balanced_accuracy",
                ):
                    if n3_lane.get(key) is not None:
                        raise ClosureError(
                            f"invalid_metric_missingness_semantics:{profile_id}:{key}"
                        )
    return profiles


def _n3_metric(n3: Mapping[str, Any], path: Sequence[str]) -> Mapping[str, Any]:
    current: Any = n3
    for key in path:
        if not isinstance(current, Mapping):
            raise ClosureError(f"n3_metric_path_invalid:{'.'.join(path)}")
        current = current.get(key)
    if not isinstance(current, Mapping):
        raise ClosureError(f"n3_metric_missing:{'.'.join(path)}")
    return current


def build_closure_model(
    matrix: Mapping[str, Any],
    audit_v2: Mapping[str, Any],
    n3: Mapping[str, Any],
    *,
    matrix_sha256: str,
    audit_v2_sha256: str,
    n3_sha256: str,
) -> dict[str, Any]:
    profiles = validate_sources(matrix, audit_v2, n3)
    all_on = profiles["p1-all"]
    n2 = all_on.get("n2_totals") or {}
    if (
        n2.get("retry_false_accepts") != 0
        or n2.get("valid_corrections") != 0
        or n2.get("final_false_rejections") != 70
    ):
        raise ClosureError("n2_frozen_result_mismatch")
    n4_metrics = all_on.get("metrics") or {}
    coverage = _number((n4_metrics.get("n4_coverage") or {}).get("mean"), "n4.coverage")
    top1 = _number(
        (n4_metrics.get("n4_top1_accuracy") or {}).get("mean"), "n4.top1_accuracy"
    )
    isolated_coverage = _number(
        ((profiles["only-n4-v2"].get("metrics") or {}).get("n4_coverage") or {}).get(
            "mean"
        ),
        "n4.isolated_coverage",
    )
    prior = n4_metrics.get("n4_regret_delta_vs_prior") or {}
    prior_ci = prior.get("ci95") or []
    if len(prior_ci) != 2 or _number(prior_ci[0], "n4.prior_ci_lower") > 0.0:
        raise ClosureError("n4_incremental_value_unexpected")

    isolated = _n3_metric(n3, ("contrasts", "isolated_trained_vs_zero"))
    trained_reference = _n3_metric(n3, ("contrasts", "trained_vs_reference"))
    tr_metrics = trained_reference.get("metrics") or {}
    brier = tr_metrics.get("brier_improvement") or {}
    mrr = tr_metrics.get("mrr_delta") or {}
    balanced = tr_metrics.get("balanced_accuracy") or {}
    reference_brier = _n3_metric(n3, ("profiles", "only-n3-reference", "risk_brier"))
    trained_brier = _n3_metric(n3, ("profiles", "only-n3-trained", "risk_brier"))
    reference_balanced = _n3_metric(
        n3, ("profiles", "only-n3-reference", "balanced_accuracy")
    )
    trained_balanced = _n3_metric(
        n3, ("profiles", "only-n3-trained", "balanced_accuracy")
    )
    context_names = (
        "p1_all_vs_isolated_trained",
        "without_n2_vs_isolated_trained",
        "without_n4_vs_isolated_trained",
    )
    for name in context_names:
        contrast = _n3_metric(n3, ("contrasts", name))
        if (
            _number(contrast.get("mean"), f"n3.{name}.mean") != 0.0
            or contrast.get("zero_count") != 12
        ):
            raise ClosureError(f"n3_context_dependence_detected:{name}")

    closure_gates = {
        "source_integrity": True,
        "campaign_integrity": True,
        "durable_closure": True,
        "certification_integrity": True,
        "canonical_parity": True,
        "authority_isolation": True,
        "n2_gate": False,
        "n3_limited_gate": True,
        "n4_gate": False,
        "n3_isolated_contribution": True,
        "n3_context_independence": True,
        "n3_trained_ranking_superiority": False,
        "n3_trained_brier_superiority": True,
        "n3_trained_mrr_superiority": False,
        "n3_trained_balanced_accuracy_superiority": False,
        "n3_trained_global_superiority": False,
        "p1_ready_to_close": True,
        "p1_closed": True,
        "p2_authorized": False,
        "live_authority": False,
        "staging_authorized": False,
        "promotion_authorized": False,
        "main_merge_authorized": False,
    }
    model = {
        "schema_version": SCHEMA_VERSION,
        "status": "CLOSED",
        "objective": "P1 cognitive attribution closure",
        "source": {
            "campaign_id": CAMPAIGN_ID,
            "experiment_commit": EXPERIMENT_COMMIT,
            "closure_fix_commit": CLOSURE_FIX_COMMIT,
            "evidence_publication_head": EVIDENCE_HEAD,
            "n3_attribution_head": N3_AUDIT_HEAD,
            "matrix_sha256": matrix_sha256,
            "audit_v2_sha256": audit_v2_sha256,
            "n3_attribution_v1_sha256": n3_sha256,
            "source_files_preserved": True,
        },
        "integrity": {
            "profiles": 10,
            "seeds_per_profile": 12,
            "steps_per_lane": 32,
            "warmup_visits_per_scenario": 2,
            "emitted_episodes": 3250,
            "matched_closures": 3250,
            "certification_rate": 1.0,
            "safety_violations": 0,
            "canonical_step_comparisons": 3456,
            "canonical_mismatches": 0,
            "canonical_behavior_identical": True,
        },
        "organ_results": {
            "N2": {
                "status": "FAILED",
                "retry_false_accepts": 0,
                "valid_corrections": 0,
                "final_false_rejections": 70,
                "interpretation": (
                    "The second-verification policy was safe but did not demonstrate utility."
                ),
            },
            "N3": {
                "status": "SUPPORTED_LIMITED",
                "isolated_contribution": "DEMONSTRATED",
                "context_dependence": "NOT_DETECTED",
                "brier_improvement": "DEMONSTRATED",
                "mrr_improvement": "NOT_DEMONSTRATED",
                "balanced_accuracy": "DEGRADED",
                "global_superiority": "NOT_DEMONSTRATED",
                "isolated_signal": {
                    "metric": "paired_binary_normalized_dcg_delta_v1",
                    "legacy_metric_name": "n3_ndcg_delta",
                    "mean": _number(isolated.get("mean"), "n3.isolated.mean"),
                    "ci95": isolated.get("ci95"),
                    "positive_seeds": isolated.get("positive_count"),
                    "zero_seeds": isolated.get("zero_count"),
                    "negative_seeds": isolated.get("negative_count"),
                    "exact_sign_flip_p": (isolated.get("exact_sign_flip_p_value") or {}).get(
                        "p_value"
                    ),
                },
                "context_contrasts": {
                    name: {"mean": 0.0, "zero_seeds": 12} for name in context_names
                },
                "trained_vs_reference": {
                    "ranking": {
                        "status": "INCONCLUSIVE",
                        "mean_delta": _number(
                            trained_reference.get("mean"), "n3.trained_reference.mean"
                        ),
                        "positive_seeds": trained_reference.get("positive_count"),
                        "zero_seeds": trained_reference.get("zero_count"),
                        "negative_seeds": trained_reference.get("negative_count"),
                        "exact_sign_flip_p": (
                            trained_reference.get("exact_sign_flip_p_value") or {}
                        ).get("p_value"),
                    },
                    "brier": {
                        "status": "SUPPORTED",
                        "reference_mean": _number(
                            reference_brier.get("mean"), "n3.reference_brier"
                        ),
                        "trained_mean": _number(
                            trained_brier.get("mean"), "n3.trained_brier"
                        ),
                        "paired_improvement": _number(
                            brier.get("mean"), "n3.brier_improvement"
                        ),
                        "positive_seeds": brier.get("positive_count"),
                        "exact_sign_flip_p": (
                            brier.get("exact_sign_flip_p_value") or {}
                        ).get("p_value"),
                        "decomposition": {
                            "reliability": "not_recomputable",
                            "resolution": "not_recomputable",
                            "uncertainty": "not_recomputable",
                        },
                    },
                    "mrr": {
                        "status": "NOT_SUPPORTED",
                        "mean_delta": _number(mrr.get("mean"), "n3.mrr_delta"),
                        "zero_seeds": mrr.get("zero_count"),
                    },
                    "balanced_accuracy": {
                        "status": "REFUTED",
                        "reference_mean": _number(
                            reference_balanced.get("mean"), "n3.reference_balanced"
                        ),
                        "trained_mean": _number(
                            trained_balanced.get("mean"), "n3.trained_balanced"
                        ),
                        "paired_delta": _number(
                            balanced.get("mean"), "n3.balanced_delta"
                        ),
                        "negative_seeds": balanced.get("negative_count"),
                        "exact_sign_flip_p": (
                            balanced.get("exact_sign_flip_p_value") or {}
                        ).get("p_value"),
                    },
                    "global_superiority": "NOT_DEMONSTRATED",
                },
            },
            "N4": {
                "status": "FAILED",
                "coverage_observed": coverage,
                "coverage_frozen_rounded": 0.917,
                "top1_accuracy": top1,
                "incremental_value_vs_causal_prior": "not_demonstrated",
                "isolated_coverage": isolated_coverage,
                "interpretation": (
                    "N4 did not demonstrate incremental value over the causal prior "
                    "and remains rejected."
                ),
            },
        },
        "n3_semantic_resolution": {
            "deprecated_source_gate": "trained_vs_reference",
            "deprecated_source_gate_status": "deprecated_ambiguous_source_gate",
            "replacement_subgates": {
                "trained_ranking_vs_reference": "INCONCLUSIVE",
                "trained_brier_vs_reference": "SUPPORTED",
                "trained_mrr_vs_reference": "NOT_SUPPORTED",
                "trained_balanced_accuracy_vs_reference": "REFUTED",
                "trained_global_superiority": "NOT_DEMONSTRATED",
            },
        },
        "closure_gates": closure_gates,
        "limitations": [
            "Conventional nDCG is not recomputable from the published aggregate.",
            "Brier reliability, resolution and uncertainty are not recomputable.",
            "The trained-reference ranking contrast is inconclusive (exact p=0.125).",
            "MRR did not improve.",
            "Trained balanced accuracy degraded relative to reference.",
            "No canonical decision, memory, scheduler, actuation or certification influence was demonstrated.",
            "One external reasoning-stress XPASS is recorded as non-blocking and unrelated to P1.",
        ],
        "qa_observations": {
            "external_xpass": {
                "node_id": (
                    "tests/reasoning_stress/test_temporal_hysteresis.py::"
                    "test_no_undesired_memory_effects"
                ),
                "xfail_reason": (
                    "Caracterización: histéresis ~0.02 (un paso discreto del sweep) "
                    "frente a una expectativa de ~0; brittleness de discretización, "
                    "no un bug del scheduler."
                ),
                "belongs_to_p1_scope": False,
                "closure_limitation": "recorded_non_blocking_external_xpass",
                "closure_impact": "none",
            }
        },
        "forbidden_inferences": list(FORBIDDEN_INFERENCES),
        "decisions": {
            "canonical_influence": "none",
            "live_authority": False,
            "staging_authorized": False,
            "promotion_authorized": False,
            "main_merge_authorized": False,
            "p2_authorized": False,
        },
        "invariants": {
            "source_matrix_preserved": True,
            "source_audit_v2_preserved": True,
            "source_n3_attribution_v1_preserved": True,
            "campaign_rerun": False,
            "runtime_modified": False,
            "training_executed": False,
            "external_reasoner_used": False,
            "raw_runtime_tree_used": False,
        },
        "runtime_modified": False,
        "campaign_rerun": False,
        "training_executed": False,
    }
    validate_closure_model(model)
    return model


def validate_closure_model(model: Mapping[str, Any]) -> None:
    reject_nonfinite(model)
    gates = model.get("closure_gates") or {}
    if "trained_vs_reference" in gates:
        raise ClosureError("ambiguous_trained_vs_reference_gate_forbidden")
    if model.get("status") != "CLOSED":
        raise ClosureError("p1_not_closed")
    organs = model.get("organ_results") or {}
    if (organs.get("N2") or {}).get("status") != "FAILED":
        raise ClosureError("n2_must_fail")
    if (organs.get("N3") or {}).get("status") != "SUPPORTED_LIMITED":
        raise ClosureError("n3_status_invalid")
    if (organs.get("N3") or {}).get("global_superiority") != "NOT_DEMONSTRATED":
        raise ClosureError("n3_global_superiority_forbidden")
    if (organs.get("N4") or {}).get("status") != "FAILED":
        raise ClosureError("n4_must_fail")
    decisions = model.get("decisions") or {}
    if decisions.get("canonical_influence") != "none" or any(
        decisions.get(key) is not False
        for key in (
            "live_authority",
            "staging_authorized",
            "promotion_authorized",
            "main_merge_authorized",
            "p2_authorized",
        )
    ):
        raise ClosureError("closure_authority_invalid")


def render_markdown(model: Mapping[str, Any]) -> str:
    n2 = model["organ_results"]["N2"]
    n3 = model["organ_results"]["N3"]
    n4 = model["organ_results"]["N4"]
    isolated = n3["isolated_signal"]
    tr = n3["trained_vs_reference"]
    forbidden = "\n".join(f"- {item}" for item in model["forbidden_inferences"])
    return f"""# P1 canonical closure

## 1. Identidad del experimento

- Campaign: `{CAMPAIGN_ID}`
- Experiment commit: `{EXPERIMENT_COMMIT}`
- Evidence publication HEAD: `{EVIDENCE_HEAD}`
- N3 attribution audit HEAD: `{N3_AUDIT_HEAD}`

## 2. Objetivo y alcance de P1

P1 midió atribución cognitiva en SHADOW. Este cierre es derivado, no repite la
campaña, no modifica runtime y no diseña ni autoriza P2.

## 3. Evidencia y cadena de commits

Se preservan `matrix.json`, `matrix.audit-v2.json` y
`n3-attribution.audit-v1.json`, ligados por SHA-256 en `SHA256SUMS`.

## 4. Integridad y reproducibilidad

Diez perfiles, doce seeds por perfil y 32 pasos por lane. Cerraron 3.250/3.250
episodios; certificación 1.0; cero violaciones de seguridad; 3.456 comparaciones
canónicas y cero mismatches. JSON y Markdown se generan offline y atómicamente.

## 5. Resultado N2

**FAILED.** `retry_false_accepts={n2['retry_false_accepts']}`,
`valid_corrections={n2['valid_corrections']}` y
`final_false_rejections={n2['final_false_rejections']}`. La política de segunda
verificación fue segura pero no demostró utilidad.

## 6. Resultado N3

**SUPPORTED_LIMITED.** La señal aislada
`paired_binary_normalized_dcg_delta_v1` tiene media `{isolated['mean']}`, IC95%
`{isolated['ci95']}`, 12 seeds positivas y p exacto
`{isolated['exact_sign_flip_p']}`. Los tres contrastes contextuales son cero en
las doce seeds.

N3 demuestra una contribución cognitiva limitada y reproducible dentro del
experimento P1 SHADOW.

La señal aislada mejora la métrica pareada de ranking interno frente al
retrieval canónico y permanece inalterada al habilitar o retirar N2 y N4.

El backend trained mejora sustancialmente el Brier score frente al backend
reference, pero no demuestra superioridad global: la ventaja de ranking
trained-reference es inconclusa, MRR no mejora y balanced accuracy empeora.

P1 no demuestra influencia sobre decisión, memoria, scheduler, actuación,
certificación ni comportamiento canónico.

## 7. Resultado N4

**FAILED.** Cobertura `{n4['coverage_frozen_rounded']}`, top-1
`{n4['top1_accuracy']}`, cobertura aislada `{n4['isolated_coverage']}`. N4 no
demostró valor incremental sobre el prior causal y permanece rechazado.

## 8. Resolución semántica trained vs reference

- Ranking: **{tr['ranking']['status']}**, delta `{tr['ranking']['mean_delta']}`,
  4 seeds positivas, 8 cero, p exacto `{tr['ranking']['exact_sign_flip_p']}`.
- Brier score: **{tr['brier']['status']}**, mejora emparejada
  `{tr['brier']['paired_improvement']}`.
- MRR: **{tr['mrr']['status']}**, delta `0.0`.
- Balanced accuracy: **{tr['balanced_accuracy']['status']}**, delta
  `{tr['balanced_accuracy']['paired_delta']}`.
- Superioridad global: **NOT_DEMONSTRATED**.

El gate fuente ambiguo `trained_vs_reference` queda deprecado y no aparece como
gate final.

## 9. Limitaciones científicas

- No puede recomputarse nDCG convencional.
- No puede descomponerse Brier en reliability, resolution y uncertainty.
- No hubo mejora de MRR y balanced accuracy empeoró.
- No hubo influencia canónica ni evaluación live/OOD suficiente.
- XPASS externo no bloqueante:
  `tests/reasoning_stress/test_temporal_hysteresis.py::test_no_undesired_memory_effects`;
  caracteriza histéresis/discretización del scheduler y no pertenece a P1.

## 10. Inferencias prohibidas

{forbidden}

## 11. Decisiones de autoridad

Influencia canónica: `none`. Autoridad live, staging, promoción, merge a `main` y
P2 permanecen explícitamente no autorizados.

## 12. Estado final de P1

`P1_STATUS=CLOSED`; `P1_N2=FAILED`; `P1_N3=SUPPORTED_LIMITED`;
`P1_N4=FAILED`.

## 13. Condición para seleccionar un nuevo objetivo

La selección de un nuevo objetivo requiere una decisión humana explícita. Este
cierre no selecciona, diseña ni inicia P2.

P1 queda CLOSED como experimento SHADOW de atribución cognitiva.

N2: FAILED.
N3: SUPPORTED_LIMITED.
N4: FAILED.

La contribución aislada de N3 queda demostrada dentro de las métricas y
condiciones de P1. La superioridad global del backend trained frente al
reference no queda demostrada.

Ningún resultado concede autoridad operativa, staging, promoción, merge a
main ni autorización de P2.

La selección de un nuevo objetivo requiere una decisión humana explícita.
"""


def _encoded_json(model: Mapping[str, Any]) -> bytes:
    return (json.dumps(model, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )


def atomic_write_pair(outputs: Mapping[Path, bytes]) -> None:
    temporaries: dict[Path, Path] = {}
    try:
        for destination, content in outputs.items():
            destination.parent.mkdir(parents=True, exist_ok=True)
            descriptor, name = tempfile.mkstemp(
                prefix=f".{destination.name}.", dir=destination.parent
            )
            temporary = Path(name)
            temporaries[destination] = temporary
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
        for destination, temporary in temporaries.items():
            os.replace(temporary, destination)
    finally:
        for temporary in temporaries.values():
            if temporary.exists():
                temporary.unlink()


def run(
    *,
    matrix_path: Path,
    audit_v2_path: Path,
    n3_attribution_path: Path,
    sha256s_path: Path,
    json_output: Path,
    markdown_output: Path,
) -> dict[str, Any]:
    validate_hashes(
        matrix_path=matrix_path,
        audit_v2_path=audit_v2_path,
        n3_attribution_path=n3_attribution_path,
        sha256s_path=sha256s_path,
    )
    sources = (matrix_path, audit_v2_path, n3_attribution_path)
    before = {path: sha256_file(path) for path in sources}
    model = build_closure_model(
        load_json(matrix_path),
        load_json(audit_v2_path),
        load_json(n3_attribution_path),
        matrix_sha256=before[matrix_path],
        audit_v2_sha256=before[audit_v2_path],
        n3_sha256=before[n3_attribution_path],
    )
    markdown = render_markdown(model)
    atomic_write_pair(
        {json_output: _encoded_json(model), markdown_output: markdown.encode("utf-8")}
    )
    if before != {path: sha256_file(path) for path in sources}:
        raise ClosureError("source_files_changed_during_closure")
    return model


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", required=True, type=Path)
    parser.add_argument("--audit-v2", required=True, type=Path)
    parser.add_argument("--n3-attribution", required=True, type=Path)
    parser.add_argument("--sha256s", required=True, type=Path)
    parser.add_argument("--json-output", required=True, type=Path)
    parser.add_argument("--markdown-output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        model = run(
            matrix_path=args.matrix,
            audit_v2_path=args.audit_v2,
            n3_attribution_path=args.n3_attribution,
            sha256s_path=args.sha256s,
            json_output=args.json_output,
            markdown_output=args.markdown_output,
        )
    except ClosureError as exc:
        parser.exit(2, f"source_integrity_failed:{exc}\n")
    print(json.dumps({"status": model["status"], "json": str(args.json_output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
